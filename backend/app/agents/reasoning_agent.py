"""Reasoning Agent — the brain. Combines GPS, vision, OCR, hazards and the
navigation goal into decisions and words:

  * floor reasoning      443B → floor 4; elevator display "2" → you're on 2 → find the elevator
  * indoor wayfinding    sign "Rooms 440-460 →"; room numbers getting closer / further
  * entrance check       building-name OCR vs. Places name; entrance landmark position
  * arrival verification OCR room-plate match + Gemini confirmation across frames
  * speech governance    priority ordering, de-duplication, cooldowns, max utterances
  * assistant (bonus)    free-form questions answered from the fused world state
"""
from __future__ import annotations

import json
import re
import time

from ..config import Settings
from ..schemas import (ArrivalStatus, Landmark, OCRText, Phase, Priority, SceneAnalysis, SessionState,
                       Utterance)
from ..services.gemini import GeminiService
from ..utils.floor import floor_phrase, infer_floor_from_room, parse_floor_indicator
from ..utils.geo import direction_phrase, spoken_distance
from ..utils.text import canonical_room, fuzzy_contains, normalize, room_match_score, room_number_value
from .voice_agent import TARGETS

LANDMARK_SYNONYMS = {k: v + {"reception": ["desk", "concierge", "security"], "entrance": ["entry", "doorway", "door"],
                             "exit": ["emergency exit"], "elevator": ["elevator door", "lift lobby"]}.get(k, [])
                     for k, v in TARGETS.items()}
SCENE_FRESH_S = 15.0

ASSISTANT_SYSTEM = """You are VisionGuide, a calm, warm, concise navigation companion for a blind or low-vision person.
You receive a JSON snapshot of what the phone currently knows: GPS/address, route progress, latest camera scene,
OCR text, hazards, the user's goal and floor reasoning. Answer the user's question using ONLY that snapshot.
Rules:
- Max 3 short spoken sentences. No markdown, lists, emojis or URLs.
- Safety first: if a hazard is relevant, say it first.
- Use clock directions (12 o'clock = straight ahead) and metres.
- If the scene data is older than 15 seconds or missing, say you need a fresh look and ask them to hold the phone up.
- Never claim it is safe to cross a road; advise using traffic sounds, their cane or dog, or asking someone nearby.
- If you are unsure, say so plainly."""


def landmark_phrase(lm: Landmark, name: str | None = None) -> str:
    what = name or lm.type
    dist = spoken_distance(lm.distance_m)
    return f"The {what} is {direction_phrase(lm.clock, lm.side)}" + (f", {dist}" if dist else "") + "."


def find_landmark(scene: SceneAnalysis | None, kind: str) -> Landmark | None:
    if not scene:
        return None
    words = [kind] + LANDMARK_SYNONYMS.get(kind, [])
    hits = [lm for lm in scene.landmarks
            if any(re.search(rf"\b{re.escape(w)}", f"{lm.type} {lm.description}".lower()) for w in words)]
    if not hits:
        return None
    exact = [h for h in hits if h.type == kind]
    pool = exact or hits
    return min(pool, key=lambda l: l.distance_m if l.distance_m is not None else 50)


class SpeechGovernor:
    """Decides what actually gets spoken: a blind user can't skim, so every
    extra sentence costs attention. Highest priority first, no repeats inside a
    cooldown, at most N per tick."""

    def __init__(self, settings: Settings):
        self.s = settings

    def select(self, session: SessionState, utts: list[Utterance], *, force: bool = False, limit: int = 2) -> list[Utterance]:
        now = time.time()
        chosen: list[Utterance] = []
        seen_keys: set[str] = set()
        for u in sorted(utts, key=lambda u: -int(u.priority)):
            key = u.key or f"{u.category}:{normalize(u.text)[:80]}"
            if key in seen_keys:
                continue
            cooldown = self.s.critical_cooldown_s if u.priority == Priority.CRITICAL else self.s.speech_cooldown_s
            if u.category == "scene":
                cooldown = max(cooldown, 20.0)
            if not force and now - session.spoken_at.get(key, 0) < cooldown:
                continue
            seen_keys.add(key)
            chosen.append(u)
            session.spoken_at[key] = now
            if len(chosen) >= limit:
                break
        # Trim the de-dup table so long sessions don't grow unbounded.
        if len(session.spoken_at) > 300:
            session.spoken_at = dict(sorted(session.spoken_at.items(), key=lambda kv: kv[1])[-150:])
        return chosen


class ReasoningAgent:
    name = "reasoning"

    def __init__(self, gemini: GeminiService, settings: Settings):
        self.gemini = gemini
        self.s = settings
        self.governor = SpeechGovernor(settings)

    # ------------------------------------------------------------ floors
    def update_floor(self, session: SessionState, ocr: list[OCRText], scene: SceneAnalysis | None) -> list[Utterance]:
        observed: int | None = None
        if scene and scene.floor_indicator:
            observed = parse_floor_indicator(scene.floor_indicator)
            if observed is None and scene.floor_indicator.strip().lstrip("-").isdigit():
                observed = int(scene.floor_indicator.strip())
        if observed is None:
            floors = sorted((o for o in ocr if o.kind == "floor" and o.value is not None),
                            key=lambda o: -o.confidence)
            if floors:
                observed = int(floors[0].value)
        inferred_from_room = False
        if observed is None and session.destination and session.destination.room:
            # Infer from nearby room plates: seeing 237 and 241 means we're on floor 2.
            rooms = [o for o in ocr if o.kind == "room" and o.value and o.confidence >= 0.5]
            floors = {infer_floor_from_room(o.value).floor for o in rooms} - {None}
            if len(floors) == 1:
                observed, inferred_from_room = floors.pop(), True

        if observed is None or observed == session.current_floor:
            return []
        session.current_floor = observed
        dest = session.destination
        out: list[Utterance] = []
        basis = "based on the room numbers around you" if inferred_from_room else "based on the floor sign"
        if dest and dest.floor_hint is not None:
            target = f"Room {dest.room}" if dest.room else "your destination"
            if observed == dest.floor_hint:
                if session.subgoal in ("elevator", "stairs"):
                    session.subgoal = None
                out.append(Utterance(text=f"You're on {floor_phrase(observed)}, {basis}. That's the right floor for {target}.",
                                     priority=Priority.GUIDANCE, category="nav", haptic="success", key=f"floor:{observed}"))
            else:
                session.subgoal = session.subgoal or "elevator"
                direction = "up" if dest.floor_hint > observed else "down"
                out.append(Utterance(
                    text=f"You're on {floor_phrase(observed)}, {basis}. {target} is likely on {floor_phrase(dest.floor_hint)}, "
                         f"so we need to go {direction}. Let's find the elevator.",
                    priority=Priority.GUIDANCE, category="nav", key=f"floor:{observed}"))
        else:
            out.append(Utterance(text=f"You're on {floor_phrase(observed)}.", priority=Priority.INFO,
                                 category="scene", key=f"floor:{observed}"))
        return out

    # ------------------------------------------------------------ entrance
    def approach_guidance(self, session: SessionState, ocr: list[OCRText], scene: SceneAnalysis | None) -> list[Utterance]:
        dest = session.destination
        out: list[Utterance] = []
        if not dest:
            return out
        names = [n for n in (dest.sign_text, dest.place_name) if n]
        sign_match = None
        for o in ocr:
            if any(fuzzy_contains(o.text, n) >= 0.8 for n in names):
                sign_match = o
                break
        entrance = find_landmark(scene, "entrance")
        if sign_match and entrance:
            out.append(Utterance(text=f"I can see the sign '{sign_match.text}'. This is the right building. "
                                      + landmark_phrase(entrance, "entrance"),
                                 priority=Priority.GUIDANCE, category="nav", haptic="success", key="approach:verified"))
        elif sign_match:
            out.append(Utterance(text=f"I can see the sign '{sign_match.text}' at {sign_match.clock} o'clock. "
                                      f"That matches your destination. Now looking for the door.",
                                 priority=Priority.GUIDANCE, category="nav", key="approach:sign"))
        elif entrance:
            out.append(Utterance(text=landmark_phrase(entrance, "entrance") + " I can't confirm the building name yet.",
                                 priority=Priority.GUIDANCE, category="nav", key="approach:entrance"))
        elif scene:
            out.append(Utterance(text="No entrance in view. Turn slowly to scan the front of the building.",
                                 priority=Priority.INFO, category="nav", key="approach:scan"))
        return out

    # ------------------------------------------------------------ indoor
    def indoor_guidance(self, session: SessionState, ocr: list[OCRText], scene: SceneAnalysis | None) -> list[Utterance]:
        out: list[Utterance] = []
        dest = session.destination

        # 1. Sub-goal (elevator to change floor, or a user "find X" request)
        if session.subgoal:
            lm = find_landmark(scene, session.subgoal)
            if lm:
                close = (lm.distance_m or 9) <= 1.5
                changing_floor = (session.subgoal in ("elevator", "stairs") and dest and dest.floor_hint is not None
                                  and session.current_floor != dest.floor_hint)
                if close and changing_floor:
                    out.append(Utterance(text=f"The {session.subgoal} is right in front of you. Go to {floor_phrase(dest.floor_hint)}. "
                                              f"I'll read the floor display to confirm.",
                                         priority=Priority.GUIDANCE, category="nav", haptic="success", key="subgoal:inside"))
                elif close:
                    out.append(Utterance(text=f"You've reached the {session.subgoal}. It's right in front of you.",
                                         priority=Priority.GUIDANCE, category="arrival", haptic="success",
                                         key=f"subgoal:reached:{session.subgoal}"))
                    session.subgoal = None
                else:
                    out.append(Utterance(text=landmark_phrase(lm, session.subgoal), priority=Priority.GUIDANCE,
                                         category="nav", key=f"subgoal:{session.subgoal}:{lm.clock}"))
            elif scene:
                out.append(Utterance(text=f"No {session.subgoal} in view yet. Turn slowly left and right.",
                                     priority=Priority.INFO, category="nav", key=f"subgoal:scan:{session.subgoal}"))

        # 2. Room-number wayfinding
        if dest and dest.room:
            target_num = room_number_value(dest.room)
            for o in ocr:
                if o.kind == "direction" and o.room_range and target_num is not None:
                    lo, hi = o.room_range
                    if lo <= target_num <= hi:
                        way = o.value or f"at {o.clock} o'clock"
                        out.append(Utterance(text=f"Sign says rooms {lo} to {hi}, {way}. Room {dest.room} is that way.",
                                             priority=Priority.GUIDANCE, category="nav", key=f"range:{lo}-{hi}"))
            seen = [o for o in ocr if o.kind == "room" and o.value and room_match_score(o.value, dest.room) < 0.85]
            if seen and target_num is not None:
                nearest = min(seen, key=lambda o: abs((room_number_value(o.value) or 0) - target_num))
                n = room_number_value(nearest.value)
                if n is not None:
                    gap = abs(n - target_num)
                    same_base = canonical_room(nearest.value)[:-1] == canonical_room(dest.room)[:-1] and gap == 0
                    side = {"left": "on your left", "right": "on your right"}.get(nearest.side, "ahead")
                    if same_base:
                        text = f"This is Room {nearest.value}, not {dest.room}. It should be right next to it."
                    elif session.best_room_gap is not None and gap > session.best_room_gap + 2:
                        text = f"Room {nearest.value} {side}. The numbers are moving away from {dest.room}. Try turning around."
                    elif gap <= 10:
                        text = f"Room {nearest.value} {side}. {dest.room} should be very close."
                    else:
                        text = f"Room {nearest.value} {side}. Keep going."
                    session.best_room_gap = gap if session.best_room_gap is None else min(session.best_room_gap, gap)
                    out.append(Utterance(text=text, priority=Priority.GUIDANCE, category="nav", key=f"room:{nearest.value}"))

        # 3. Generic target (reception, exit, ...)
        if dest and dest.target_kind and not dest.room:
            lm = find_landmark(scene, dest.target_kind)
            if lm:
                out.append(Utterance(text=landmark_phrase(lm, dest.target_kind), priority=Priority.GUIDANCE,
                                     category="nav", key=f"target:{dest.target_kind}:{lm.clock}"))

        # 4. Gemini's own suggestion as a fallback
        if not out and scene and scene.suggested_guidance:
            out.append(Utterance(text=scene.suggested_guidance, priority=Priority.INFO, category="nav"))
        return out

    # ------------------------------------------------------------ arrival
    def verify_arrival(self, session: SessionState, ocr: list[OCRText], scene: SceneAnalysis | None) -> list[Utterance]:
        dest = session.destination
        if not dest or session.phase not in (Phase.INDOOR, Phase.APPROACH):
            return []
        target_label = (f"Room {dest.room}" if dest.room else dest.sign_text or dest.target_kind or dest.place_name or "destination")
        evidence: list[str] = []
        score, where = 0.0, None

        if dest.room:
            for o in ocr:
                if o.kind not in ("room", "other"):
                    continue
                m = room_match_score(o.value or o.text, dest.room)
                s = m * (0.7 + 0.3 * o.confidence)
                if m >= 0.85 and s > score:
                    score, where = s, o
            if where:
                evidence.append(f"OCR read '{where.text}' ({where.confidence:.0%})")
        elif dest.sign_text:
            for o in ocr:
                f = fuzzy_contains(o.text, dest.sign_text)
                if f >= 0.85 and f * (0.7 + 0.3 * o.confidence) > score:
                    score, where = f * (0.7 + 0.3 * o.confidence), o
            if where:
                evidence.append(f"Sign reads '{where.text}'")
        elif dest.target_kind:
            lm = find_landmark(scene, dest.target_kind)
            if lm and (lm.distance_m or 9) <= 2.5:
                score = 0.9
                evidence.append(f"{dest.target_kind} {spoken_distance(lm.distance_m)} away")

        if scene and scene.destination_visible:
            score = min(1.0, score + 0.15) if score else 0.6
            evidence.append(f"Vision: {scene.destination_evidence or 'destination visible'}")

        prev_state = session.arrival.state
        if score >= 0.6:
            session.candidate_streak += 1
        else:
            session.candidate_streak = 0
        confirmed = score >= 0.95 or (score >= 0.6 and session.candidate_streak >= 2)
        state = "confirmed" if confirmed else "candidate" if score >= 0.6 else "searching"
        session.arrival = ArrivalStatus(state=state, target=target_label, evidence=evidence, confidence=round(score, 2))

        if state == "confirmed" and prev_state != "confirmed":
            session.phase = Phase.ARRIVED
            session.subgoal = None
            loc = f" It's at {where.clock} o'clock." if where and where.bbox else ""
            return [Utterance(text=f"You have arrived at {target_label}.{loc}", priority=Priority.WARNING,
                              category="arrival", interrupt=True, haptic="success", key="arrival:confirmed")]
        if state == "candidate" and prev_state == "searching":
            loc = f" at {where.clock} o'clock" if where and where.bbox else ""
            return [Utterance(text=f"I think I see {target_label}{loc}. Move a little closer so I can confirm.",
                              priority=Priority.GUIDANCE, category="arrival", haptic="light", key="arrival:candidate")]
        return []

    # ------------------------------------------------------------ world model snapshot
    def snapshot(self, session: SessionState, address: str | None, progress: str | None) -> dict:
        age = time.time() - session.last_scene_at if session.last_scene_at else None
        sc = session.last_scene
        d = session.destination
        return {
            "phase": session.phase.value,
            "address": address,
            "gps": session.position.model_dump() if session.position else None,
            "heading_deg": session.heading,
            "route_progress": progress,
            "destination": d.model_dump(exclude_none=True) if d else None,
            "current_floor": session.current_floor,
            "subgoal": session.subgoal,
            "arrival": session.arrival.model_dump(),
            "scene_age_seconds": round(age, 1) if age is not None else None,
            "scene": {
                "summary": sc.summary, "environment": sc.environment,
                "landmarks": [l.model_dump(exclude_none=True) for l in sc.landmarks],
            } if sc else None,
            "hazards": [h.model_dump(exclude_none=True) for h in session.last_hazards],
            "visible_text": [o.text for o in session.last_ocr][:15],
        }

    def where_am_i_text(self, session: SessionState, address: str | None, progress: str | None) -> str:
        parts = []
        indoors = session.phase in (Phase.INDOOR, Phase.ARRIVED) or (session.last_scene and session.last_scene.environment == "indoor")
        if address and not indoors:
            parts.append(f"You're near {address}.")
        elif session.destination and session.destination.place_name and indoors:
            parts.append(f"You're inside {session.destination.place_name}.")
        if session.current_floor is not None:
            parts.append(f"You're on {floor_phrase(session.current_floor)}.")
        fresh = session.last_scene and time.time() - session.last_scene_at < SCENE_FRESH_S
        if fresh and session.last_scene.summary:
            parts.append(session.last_scene.summary)
        if progress:
            parts.append(progress)
        elif session.destination and session.phase == Phase.INDOOR:
            d = session.destination
            goal = f"Room {d.room}" if d.room else d.sign_text or d.target_kind
            hint = f", likely on {floor_phrase(d.floor_hint)}" if d.floor_hint is not None else ""
            parts.append(f"You're looking for {goal}{hint}.")
        if not parts:
            parts.append("I don't have a location fix yet. Hold the phone up so I can look around.")
        return " ".join(parts)

    async def answer(self, session: SessionState, question: str, address: str | None, progress: str | None) -> str:
        snap = self.snapshot(session, address, progress)
        history = "\n".join(f"{m['role']}: {m['text']}" for m in session.chat_history[-6:])
        prompt = f"World snapshot:\n{json.dumps(snap, default=str)}\n\nRecent conversation:\n{history}\n\nUser: {question}"
        reply = await self.gemini.text(ASSISTANT_SYSTEM, prompt, max_tokens=180)
        if not reply:  # offline fallback: deterministic answer from the same state
            reply = self.where_am_i_text(session, address, progress)
            if session.last_hazards:
                h = session.last_hazards[0]
                reply = f"Careful, {h.type} {h.side}. " + reply
        reply = re.sub(r"[*_#`]", "", reply).strip()
        session.chat_history.extend([{"role": "user", "text": question}, {"role": "assistant", "text": reply}])
        session.chat_history = session.chat_history[-12:]
        return reply
