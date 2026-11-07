"""Hazard Agent — fuses YOLO detections with Gemini's hazard list, scores
severity by type, position and distance, and phrases warnings.

Severity policy (safety first, but don't cry wolf):
  CRITICAL  in the walking path within ~2.5 m, any fall risk (stairs down,
            wet floor, curb) ahead within 3 m, vehicle ahead within 8 m.
  WARNING   in path within 5 m, or beside the path within 2 m.
  INFO      everything else (folded into the scene description)."""
from __future__ import annotations

import re

from ..schemas import Detection, Hazard, Priority, SceneAnalysis, Utterance
from ..utils.geo import direction_phrase, object_phrase, spoken_distance

OBSTACLES = {"chair", "bench", "couch", "dining table", "potted plant", "suitcase", "backpack", "handbag",
             "bicycle", "fire hydrant", "dog", "bed", "skateboard", "umbrella", "parking meter",
             "refrigerator", "oven", "sink", "toilet", "tv"}
VEHICLES = {"car", "bus", "truck", "motorcycle"}
SPOKEN_NAMES = {"dining table": "table", "potted plant": "plant pot", "tv": "screen", "handbag": "bag",
                "backpack": "bag"}
FALL_RISK = {"stairs", "wet floor", "curb", "uneven surface", "hole"}

# Map Gemini's free-text hazard types onto a small vocabulary.
_CANON = [
    (r"wet|spill|puddle|slippery", "wet floor"),
    (r"stair|step", "stairs"),
    (r"glass", "glass door"),
    (r"door", "open door"),
    (r"construct|barrier|cone|scaffold|tape", "construction barrier"),
    (r"curb|kerb", "curb"),
    (r"uneven|pothole|crack|broken|ramp", "uneven surface"),
    (r"hole|trench", "hole"),
    (r"low.?hang|overhead|branch|sign at head", "low-hanging object"),
    (r"chair|stool", "chair"),
    (r"table|desk", "table"),
    (r"bike|bicycle|scooter", "bicycle"),
    (r"car|vehicle|truck|bus|van", "vehicle"),
    (r"people|person|crowd|pedestrian", "people"),
    (r"trolley|cart|box|bag|luggage|obstacle|object|bin|pole|bollard|post", "obstacle"),
]


def canonical_type(t: str) -> str:
    t = t.lower()
    for pat, name in _CANON:
        if re.search(pat, t):
            return name
    return t.strip() or "obstacle"


def side_phrase(side: str) -> str:
    return {"left": "on your left", "right": "on your right"}.get(side, "ahead")


class HazardAgent:
    name = "hazard"

    def assess(self, detections: list[Detection], scene: SceneAnalysis | None) -> list[Hazard]:
        hazards: list[Hazard] = []
        for d in detections:
            dist = d.distance_m if d.distance_m is not None else 99.0
            in_path = d.side == "ahead"
            if d.label in VEHICLES:
                if in_path and dist <= 8:
                    sev = Priority.CRITICAL
                elif dist <= 8:
                    sev = Priority.WARNING
                else:
                    continue
                typ = "vehicle"
            elif d.label in OBSTACLES:
                if in_path and dist <= 2.5:
                    sev = Priority.CRITICAL
                elif (in_path and dist <= 5) or dist <= 2:
                    sev = Priority.WARNING
                else:
                    continue
                typ = SPOKEN_NAMES.get(d.label, d.label)
            elif d.label == "person":
                if in_path and dist <= 1.5:
                    sev, typ = Priority.WARNING, "person"
                else:
                    continue  # people further away go in the crowd summary
            else:
                continue
            hazards.append(Hazard(type=typ, severity=sev, side=d.side, clock=d.clock,
                                  distance_m=d.distance_m, description="", source="yolo"))

        for g in (scene.hazards if scene else []):
            typ = canonical_type(g.type)
            if typ == "people":
                continue
            sev = g.severity
            if typ in FALL_RISK and g.side == "ahead" and (g.distance_m is None or g.distance_m <= 3):
                sev = Priority.CRITICAL
            match = next((h for h in hazards if h.side == g.side and
                          (h.type == typ or (typ in ("obstacle", "chair", "table") and h.source == "yolo"))), None)
            if match:  # fuse: YOLO's geometry, Gemini's semantics
                match.severity = max(match.severity, sev)
                match.description = g.description or match.description
                match.source = "fused"
                if typ not in ("obstacle",):
                    match.type = typ
                continue
            hazards.append(Hazard(type=typ, severity=sev, side=g.side, clock=g.clock,
                                  distance_m=g.distance_m, description=g.description, source="gemini"))

        hazards.sort(key=lambda h: (-int(h.severity), h.distance_m if h.distance_m is not None else 50))
        return hazards

    @staticmethod
    def people_summary(detections: list[Detection]) -> Utterance | None:
        people = sorted((d for d in detections if d.label == "person" and (d.distance_m or 99) <= 10),
                        key=lambda d: d.distance_m or 99)
        if not people:
            return None
        n, p = len(people), people[0]
        where = direction_phrase(p.clock, p.side) + (f", {spoken_distance(p.distance_m)}" if p.distance_m else "")
        if n == 1:
            text = f"One person {where}."                              # "One person at 11 o'clock, 3.2 meters."
        else:
            count = {2: "Two", 3: "Three", 4: "Four", 5: "Five"}.get(n, str(n))
            text = f"{count} people. Nearest {where}."                 # "Three people. Nearest at 12 o'clock, 2.1 meters."
        return Utterance(text=text, priority=Priority.INFO, category="scene", key=f"people:{n}")

    @staticmethod
    def measured_objects(detections: list[Detection], limit: int = 3, max_m: float = 10.0) -> Utterance | None:
        """The camera's measured facts, nearest first, in one fixed format:
        'Person at 12 o'clock, 1.5 meters. Chair at 1 o'clock, 2.8 meters.'
        Spoken verbatim before any AI description so numbers are never reworded."""
        objs = sorted((d for d in detections if d.distance_m is not None and d.distance_m <= max_m),
                      key=lambda d: d.distance_m)
        if not objs:
            return None
        text = ". ".join(object_phrase(SPOKEN_NAMES.get(d.label, d.label), d.clock, d.side, d.distance_m)
                         for d in objs[:limit]) + "."
        return Utterance(text=text, priority=Priority.GUIDANCE, category="scene", key=f"measured:{text}")

    @staticmethod
    def to_utterances(hazards: list[Hazard], limit: int = 2) -> list[Utterance]:
        out = []
        for h in hazards:
            if h.severity < Priority.WARNING:
                continue
            if h.type == "stairs":
                what = "Stairs" + (" going down" if "down" in h.description.lower() else
                                   " going up" if "up" in h.description.lower() else "")
            elif h.type == "open door":
                what = "Open door"
            else:
                what = h.type
            # Fixed format, as before: "Stop! Chair at 12 o'clock, 1.2 meters."
            phrase = object_phrase(what, h.clock, h.side, h.distance_m)
            text = f"{'Stop!' if h.severity == Priority.CRITICAL else 'Caution!'} {phrase}."
            out.append(Utterance(
                text=text, priority=h.severity, category="hazard",
                interrupt=h.severity == Priority.CRITICAL,
                haptic="warning" if h.severity == Priority.CRITICAL else "light",
                key=f"hazard:{h.type}:{h.side}",
            ))
            if len(out) >= limit:
                break
        return out
