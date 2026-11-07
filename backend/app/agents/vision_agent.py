"""Vision Agent — turns a camera frame into structured perception.

Two complementary paths, run concurrently by the orchestrator:
  * YOLOv8n on the server CPU (~40-80 ms): fast, metric-ish boxes for people,
    furniture and vehicles → distances and clock positions.
  * Gemini Flash (GEMINI_MODEL, ~1-2 s): open-vocabulary understanding — entrances,
    reception desks, elevators, corridors, wet floors, stairs, open doors,
    building names — plus a spoken summary, conditioned on the user's goal."""
from __future__ import annotations

from PIL import Image

from ..schemas import Detection, Hazard, Landmark, Phase, Priority, SceneAnalysis, SessionState
from ..services.gemini import GeminiService
from ..services.local_models import YoloDetector

SIDE_CLOCK = {"left": 11, "ahead": 12, "right": 1}
SEVERITY = {"high": Priority.CRITICAL, "medium": Priority.WARNING, "low": Priority.INFO}


class VisionAgent:
    name = "vision"

    def __init__(self, gemini: GeminiService, detector: YoloDetector):
        self.gemini = gemini
        self.detector = detector

    def detect(self, img: Image.Image) -> list[Detection]:
        return self.detector.detect(img)

    @staticmethod
    def build_context(s: SessionState, mode: str, focus: str | None = None) -> str:
        parts = [f"Navigation phase: {s.phase.value}."]
        d = s.destination
        if d:
            if d.room:
                parts.append(f"Destination: Room {d.room}" + (f" (likely floor {d.floor_hint})" if d.floor_hint is not None else "") + ".")
            if d.sign_text:
                parts.append(f"Destination sign text: '{d.sign_text}'.")
            if d.target_kind:
                parts.append(f"Looking for: {d.target_kind}.")
            if d.place_name and s.phase in (Phase.OUTDOOR, Phase.APPROACH):
                parts.append(f"Building: {d.place_name}. Find its correct entrance.")
        if s.subgoal:
            parts.append(f"Current sub-goal: find the {s.subgoal}.")
        if s.current_floor is not None:
            parts.append(f"User is believed to be on floor {s.current_floor}.")
        if mode == "read":
            parts.append("The user asked you to READ all visible text. Put every legible text in sign_texts.")
        elif mode == "describe":
            parts.append("The user asked what is in front of them: give a complete but short summary.")
        if focus:
            parts.append(f"The user specifically wants to know: {focus}")
        return " ".join(parts)

    async def understand(self, jpeg: bytes, s: SessionState, mode: str = "auto",
                         focus: str | None = None) -> SceneAnalysis | None:
        g = await self.gemini.analyze_scene(jpeg, self.build_context(s, mode, focus))
        if g is None:
            return None
        return SceneAnalysis(
            summary=g.summary.strip(),
            environment=g.environment,
            landmarks=[Landmark(type=l.type.lower().strip(), side=l.side, clock=SIDE_CLOCK[l.side],
                                distance_m=l.distance_m, description=l.description) for l in g.landmarks],
            hazards=[Hazard(type=h.type.lower().strip(), severity=SEVERITY[h.severity], side=h.side,
                            clock=SIDE_CLOCK[h.side], distance_m=h.distance_m, description=h.description,
                            source="gemini") for h in g.hazards],
            sign_texts=[t for t in g.sign_texts if t.strip()],
            floor_indicator=g.floor_indicator,
            destination_visible=g.destination_visible,
            destination_evidence=g.destination_evidence,
            suggested_guidance=g.suggested_guidance,
        )
