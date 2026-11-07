"""Orchestrator — runs the agents for each event and owns the phase machine.

    IDLE ──"take me to X"──▶ OUTDOOR ──GPS within 30 m──▶ APPROACH ──scene indoor──▶ INDOOR ──room plate confirmed──▶ ARRIVED
      ▲          │ (room-only request skips straight to INDOOR)                                                       │
      └──────────┴────────────────────────────── "stop" ◀─────────────────────────────────────────────────────────────┘

Three event types arrive from the phone:
  position  (every ~2 s)  → Navigation Agent
  frame     (every ~3 s)  → Vision + OCR + Hazard (parallel) → Reasoning → Speech governor
  voice     (on demand)   → Voice Agent (Whisper) → stop/repeat/help answered instantly;
                            everything else → Brain (Gemini tool-calling, agents as tools),
                            falling back to the rule-based dispatcher if the brain fails or is off
"""
from __future__ import annotations

import asyncio
import logging
import time

from ..config import Settings
from ..db import Database, SessionStore
from ..schemas import (AgentResponse, ArrivalStatus, Destination, FrameResponse, Intent, LatLng, Phase,
                       Priority, SessionState, Utterance)
from ..utils.floor import floor_phrase
from ..utils.image import load_image
from .area_agent import AreaAgent
from .brain import Brain, BrainError
from ..services.buildings import BuildingService
from ..utils.geo import object_phrase
from .hazard_agent import SPOKEN_NAMES, VEHICLES, HazardAgent
from .perception import Perception
from .navigation_agent import NavigationAgent
from .ocr_agent import OCRAgent
from .reasoning_agent import ReasoningAgent, find_landmark, landmark_phrase
from .vision_agent import VisionAgent
from .voice_agent import HELP_TEXT, VoiceAgent

log = logging.getLogger("visionguide.orchestrator")


class Orchestrator:
    def __init__(self, settings: Settings, store: SessionStore, db: Database, *, navigation: NavigationAgent,
                 vision: VisionAgent, ocr: OCRAgent, hazard: HazardAgent, voice: VoiceAgent, reasoning: ReasoningAgent,
                 area: AreaAgent | None = None, buildings: BuildingService | None = None):
        self.s, self.store, self.db = settings, store, db
        self.navigation, self.vision, self.ocr = navigation, vision, ocr
        self.hazard, self.voice, self.reasoning = hazard, voice, reasoning
        self.area = area
        self.buildings = buildings
        self.brain: Brain | None = Brain(self)

    # ------------------------------------------------------------------ helpers
    def _finish(self, session: SessionState, utts: list[Utterance], **kw) -> dict:
        if utts:
            session.last_utterances = utts
            self.db.log(session.id, "utterance", [u.text for u in utts])
        self.store.save(session)
        return dict(utterances=utts, phase=session.phase, step_index=session.step_index if session.route else None,
                    arrival=session.arrival, destination=session.destination, **kw)

    @staticmethod
    def _say(text: str, priority: Priority = Priority.GUIDANCE, category: str = "answer", **kw) -> Utterance:
        return Utterance(text=text, priority=priority, category=category, **kw)

    # ------------------------------------------------------------------ sessions
    def create_session(self) -> tuple[SessionState, list[Utterance]]:
        s = self.store.create()
        return s, [self._say("VisionGuide is ready. Tap the screen and tell me where you want to go.", category="system")]

    # ------------------------------------------------------------------ GPS
    async def handle_position(self, session: SessionState, lat: float, lng: float, heading: float | None,
                              accuracy: float | None, floor_change: int | None = None) -> AgentResponse:
        session.position = LatLng(lat=lat, lng=lng)
        session.heading = heading if heading is not None and heading >= 0 else session.heading
        session.position_accuracy_m = accuracy
        if self.area:
            self.area.maybe_refresh(session)  # background: never delays this response
        self._check_building(session)
        floor_utts = self._on_floor_change(session, floor_change) if floor_change else []
        utts, needs_reroute = self.navigation.on_position(session)
        utts = floor_utts + utts
        if needs_reroute:
            utts += await self.navigation.reroute(session)
            self.db.log(session.id, "route", {"reroute": True})
        utts = self.reasoning.governor.select(session, utts)
        if session.pending_utterances:  # e.g. "map of this area is ready", from background agents
            utts += session.pending_utterances
            session.pending_utterances = []
        return AgentResponse(**self._finish(session, utts))

    def _check_building(self, session: SessionState) -> None:
        """Entering / leaving a building we have floor plans for."""
        if not self.buildings:
            return
        b = self.buildings.near(session.position)
        if b and session.building_id != b.id:
            session.building_id = b.id
            self.db.log(session.id, "building", {"entered": b.id})
            session.pending_utterances.append(self._say(
                f"You're at {b.name}. I have its floor plans for all {len(b.floors)} levels, so you can ask "
                f"where any space is.", Priority.INFO, "info", key=f"building:{b.id}"))
        elif not b and session.building_id and not self.buildings.near(session.position, extra_m=150):
            session.building_id = None  # well clear of it (hysteresis against GPS jitter at the edge)

    def _resolve_building(self, session: SessionState, dest: Destination) -> None:
        """Use our own floor plans: 'the library in Marshgate' -> Library, Level 02, at Marshgate;
        'take me to Marshgate' -> the building's exact location (no place search needed)."""
        if not self.buildings:
            return
        named = self.buildings.by_name(dest.place_query) or self.buildings.by_name(dest.raw)
        b = named or (self.buildings.buildings.get(session.building_id) if session.building_id else None)
        if not b:
            return
        if dest.sign_text:
            space_q = dest.sign_text
        elif dest.place_query:
            space_q = self.buildings.strip_building(dest.place_query, b)
        else:
            space_q = None if (dest.room or dest.target_kind) else self.buildings.strip_building(dest.raw, b)
        hits = self.buildings.find_space(space_q, b) if space_q else []
        if not named and not hits:
            return  # at Marshgate but asking for somewhere else ("Stratford station"): leave it alone
        if hits:
            name = hits[0]["name"]
            levels = sorted({h["level"] for h in hits if h["name"] == name})
            dest.sign_text, dest.target_kind, dest.floor_hint = name, None, levels[0]
            where = " and ".join(b.level_label(lv) for lv in levels)
            dest.floor_explanation = f"The {name} is on {where}."
        dest.place_name, dest.place_location, dest.place_address = b.name, b.location, b.address
        # Already at the building: go straight to indoor guidance. Otherwise walk there first.
        dest.place_query = None if session.building_id == b.id else b.name

    def _on_floor_change(self, session: SessionState, change: int) -> list[Utterance]:
        """Barometer on the phone detected a floor change (lift or stairs)."""
        self.db.log(session.id, "floor_change", {"change": change, "from": session.current_floor})
        if session.current_floor is None:
            return []  # the phone already said "up one floor"; without a known start floor we can't name it
        session.current_floor += change
        d = session.destination
        if d and d.floor_hint is not None and session.phase in (Phase.INDOOR, Phase.APPROACH):
            target = f"Room {d.room}" if d.room else "your destination"
            if session.current_floor == d.floor_hint:
                if session.subgoal in ("elevator", "stairs"):
                    session.subgoal = None
                return [self._say(f"You're now on {floor_phrase(session.current_floor)}. That's the right floor for {target}.",
                                  Priority.WARNING, "nav", haptic="success", key=f"floor:{session.current_floor}")]
            way = "up" if d.floor_hint > session.current_floor else "down"
            n = abs(d.floor_hint - session.current_floor)
            return [self._say(f"You're now on {floor_phrase(session.current_floor)}. {target} is {n} floor"
                              f"{'s' if n > 1 else ''} {way}.", Priority.GUIDANCE, "nav",
                              key=f"floor:{session.current_floor}")]
        return [self._say(f"You're now on {floor_phrase(session.current_floor)}.", Priority.INFO, "nav",
                          key=f"floor:{session.current_floor}")]

    # ------------------------------------------------------------------ camera
    async def perceive(self, session: SessionState, image_bytes: bytes, mode: str = "auto",
                       focus: str | None = None) -> Perception:
        """Vision (Gemini Flash), YOLO and OCR concurrently, then hazard fusion.
        Shared by the live frame loop and the brain's look_around tool."""
        img, jpeg = load_image(image_bytes)
        timings: dict[str, float] = {}

        async def timed(name, coro):
            t = time.perf_counter()
            try:
                return await coro
            finally:
                timings[name] = round((time.perf_counter() - t) * 1000, 1)

        scene, detections, ocr = await asyncio.gather(
            timed("gemini", self.vision.understand(jpeg, session, mode, focus)),
            timed("yolo", asyncio.to_thread(self.vision.detect, img)),
            timed("ocr", asyncio.to_thread(self.ocr.read, img)),
        )
        if scene:
            ocr = self.ocr.merge_gemini_signs(ocr, scene.sign_texts)
            session.last_scene, session.last_scene_at = scene, time.time()
        hazards = self.hazard.assess(detections, scene)
        session.last_ocr, session.last_hazards = ocr, hazards
        return Perception(scene=scene, detections=detections, ocr=ocr, hazards=hazards, timings=timings)

    async def handle_frame(self, session: SessionState, image_bytes: bytes, mode: str = "auto") -> FrameResponse:
        t0 = time.perf_counter()
        p = await self.perceive(session, image_bytes, mode)
        scene, detections, ocr, hazards, timings = p.scene, p.detections, p.ocr, p.hazards, p.timings

        candidates: list[Utterance] = self.hazard.to_utterances(hazards)
        candidates += self.reasoning.update_floor(session, ocr, scene)

        # Phase-specific reasoning
        if session.phase == Phase.APPROACH:
            if scene and scene.environment == "indoor":
                session.phase = Phase.INDOOR
                d = session.destination
                if d and not (d.room or d.sign_text or d.target_kind):
                    session.phase = Phase.ARRIVED
                    candidates.append(self._say(f"You're inside {d.place_name or 'the building'}. You've arrived.",
                                                Priority.WARNING, "arrival", haptic="success", interrupt=True))
                else:
                    goal = f"Room {d.room}" if d and d.room else (d.sign_text or d.target_kind) if d else "your destination"
                    hint = f" It's likely on {floor_phrase(d.floor_hint)}." if d and d.floor_hint is not None else ""
                    candidates.append(self._say(f"You're inside. Now looking for {goal}.{hint}", category="nav",
                                                haptic="light", key="phase:indoor"))
            else:
                candidates += self.reasoning.approach_guidance(session, ocr, scene)
        if session.phase in (Phase.INDOOR, Phase.APPROACH):
            candidates += self.reasoning.verify_arrival(session, ocr, scene)
        if session.phase == Phase.INDOOR:
            candidates += self.reasoning.indoor_guidance(session, ocr, scene)
        elif session.phase == Phase.IDLE and session.subgoal:
            candidates += self.reasoning.indoor_guidance(session, ocr, scene)

        people = self.hazard.people_summary(detections)
        force, limit = False, 2
        if mode == "describe":
            force, limit = True, 4
            measured = self.hazard.measured_objects(detections)  # exact YOLO facts first, verbatim
            if measured:
                candidates.append(measured)
            summary = scene.summary if scene else ("" if measured else self._fallback_summary(detections))
            if summary or not measured:
                candidates.append(self._say(summary or "I can't make out much. Try holding the phone a little higher.",
                                            Priority.GUIDANCE, "scene"))
        elif mode == "read":
            force, limit = True, 3
            texts = [o.text for o in ocr if o.confidence >= 0.4]
            txt = ("I can read: " + ". ".join(dict.fromkeys(texts)) + ".") if texts else "I can't read any text right now. Move closer to the sign."
            candidates.append(self._say(txt, Priority.GUIDANCE, "scene"))
        else:
            if people:
                candidates.append(people)
            if scene and scene.summary and session.phase == Phase.IDLE and not session.subgoal:
                candidates.append(self._say(scene.summary, Priority.INFO, "scene"))

        utts = self.reasoning.governor.select(session, candidates, force=force, limit=limit)
        timings["total"] = round((time.perf_counter() - t0) * 1000, 1)
        self.db.log(session.id, "frame", {"mode": mode, "hazards": [h.type for h in hazards],
                                          "ocr": [o.text for o in ocr][:10], "timings": timings,
                                          "arrival": session.arrival.state})
        return FrameResponse(**self._finish(session, utts), scene=scene, detections=detections, ocr=ocr,
                             hazards=hazards, timings_ms=timings)

    async def handle_fast_frame(self, session: SessionState, image_bytes: bytes) -> FrameResponse:
        """YOLO-only safety pass (~0.1-0.3 s): runs every second while walking, independent of the
        slower Gemini + OCR frame. Warns about hazards and about objects in the path getting closer."""
        t0 = time.perf_counter()
        img, _ = load_image(image_bytes, max_side=640)
        detections = await asyncio.to_thread(self.vision.detect, img)
        hazards = self.hazard.assess(detections, None)
        candidates = self.hazard.to_utterances(hazards)
        candidates += self._closing_in(session, detections, hazards)
        session.last_hazards = hazards
        utts = self.reasoning.governor.select(session, candidates, limit=2)
        timings = {"yolo": round((time.perf_counter() - t0) * 1000, 1)}
        return FrameResponse(**self._finish(session, utts), detections=detections, hazards=hazards,
                             timings_ms=timings)

    def _closing_in(self, session: SessionState, detections, hazards) -> list[Utterance]:
        """Compare with the previous fast frame: an object in the walking path whose distance dropped
        by >= 0.6 m within 2.5 s is getting closer (it moves toward the user, or the user toward it)."""
        now = time.time()
        prev, prev_at = session.fast_track, session.fast_track_at
        session.fast_track = [{"label": d.label, "clock": d.clock, "d": d.distance_m}
                              for d in detections if d.distance_m is not None]
        session.fast_track_at = now
        if not prev or now - prev_at > 2.5:
            return []
        warned = {(h.type, h.clock) for h in hazards if h.severity >= Priority.WARNING}
        out = []
        for d in detections:
            if d.distance_m is None or d.distance_m > 6 or d.clock not in (11, 12, 1):
                continue
            name = SPOKEN_NAMES.get(d.label, d.label)
            if (name, d.clock) in warned or (("vehicle" if d.label in VEHICLES else name), d.clock) in warned:
                continue
            before = [p["d"] for p in prev if p["label"] == d.label and abs((p["clock"] or 12) - (d.clock or 12)) <= 1]
            if before and min(before, key=lambda x: abs(x - d.distance_m)) - d.distance_m >= 0.6:
                out.append(self._say(f"{object_phrase(name, d.clock, d.side, d.distance_m)}, getting closer.",
                                     Priority.WARNING, "hazard", haptic="light",
                                     key=f"closer:{d.label}:{d.clock}"))
        return out[:1]

    @staticmethod
    def _fallback_summary(detections) -> str:
        if not detections:
            return ""
        labels = {}
        for d in detections:
            labels[d.label] = labels.get(d.label, 0) + 1
        parts = [f"{n} {label}{'s' if n > 1 and not label.endswith('s') else ''}" for label, n in labels.items()]
        return "I can see " + ", ".join(parts) + "."

    # ------------------------------------------------------------------ voice / text commands
    async def handle_audio(self, session: SessionState, audio: bytes, filename: str,
                           frame: bytes | None = None) -> AgentResponse:
        try:
            transcript = await self.voice.transcribe(audio, filename)
        except Exception as e:  # noqa: BLE001
            log.warning("Transcription failed: %s", e)
            return AgentResponse(**self._finish(session, [self._say("Sorry, I didn't catch that. Please try again.",
                                                                    category="system")]))
        if not transcript:
            return AgentResponse(**self._finish(session, [self._say("I didn't hear anything. Tap and try again.",
                                                                    category="system")]))
        return await self.handle_text(session, transcript, frame)

    # Never wait on an LLM for these.
    INSTANT_INTENTS = {"stop", "repeat", "help"}

    async def handle_text(self, session: SessionState, text: str, frame: bytes | None = None) -> AgentResponse:
        intent = self.voice.understand(text)
        self.db.log(session.id, "command", {"text": text, "intent": intent.model_dump(exclude_none=True)})

        if self.brain and self.brain.enabled and intent.name not in self.INSTANT_INTENTS:
            try:
                turn, reply = await self.brain.run(session, text, frame)
            except BrainError as e:
                log.warning("Brain failed (%s); falling back to rules", e)
                self.db.log(session.id, "brain_fallback", {"error": str(e)})
            else:
                utts: list[Utterance] = []
                if turn.perception:  # safety layer: critical hazards are spoken first, whatever the model said
                    crit = [u for u in self.hazard.to_utterances(turn.perception.hazards) if u.priority == Priority.CRITICAL]
                    utts += self.reasoning.governor.select(session, crit)
                    # "What's ahead?": the measured objects are spoken verbatim, so the model can't round or drop them.
                    if intent.name == "describe":
                        measured = self.hazard.measured_objects(turn.perception.detections)
                        if measured:
                            utts.append(measured)
                utts.append(self._say(reply))
                session.chat_history.extend([{"role": "user", "text": text}, {"role": "assistant", "text": reply}])
                session.chat_history = session.chat_history[-12:]
                self.db.log(session.id, "brain", {"trace": turn.trace})
                return AgentResponse(**self._finish(session, utts, transcript=text, intent=intent,
                                                    client_action=turn.client_action, tool_trace=turn.trace,
                                                    handled_by="brain"))

        if frame and intent.name in ("describe", "read_text", "find", "where_am_i"):
            # Rules mode with a frame attached: answer from it directly instead of asking for another.
            utts, _ = await self._dispatch(session, intent)
            mode = {"describe": "describe", "read_text": "read"}.get(intent.name, "auto")
            fr = await self.handle_frame(session, frame, mode)
            return AgentResponse(**self._finish(session, utts + fr.utterances, transcript=text, intent=intent,
                                                handled_by="rules"))
        utts, action = await self._dispatch(session, intent)
        return AgentResponse(**self._finish(session, utts, transcript=text, intent=intent, client_action=action,
                                            handled_by="rules"))

    async def _dispatch(self, session: SessionState, intent: Intent) -> tuple[list[Utterance], str | None]:
        n = intent.name
        if n == "stop":
            session.phase, session.destination, session.route, session.subgoal = Phase.IDLE, None, [], None
            session.arrival, session.candidate_streak, session.best_room_gap = ArrivalStatus(), 0, None
            return [self._say("Navigation stopped. I'm still here if you need me.")], None
        if n == "repeat":
            return (session.last_utterances or [self._say("Nothing to repeat yet.")]), None
        if n == "help":
            return [self._say(HELP_TEXT)], None
        if n == "where_am_i":
            cached = session.area.address if session.area and session.area.address else None
            address = cached or await self.navigation.address(session.position)
            session.last_address = address or session.last_address
            text = self.reasoning.where_am_i_text(session, session.last_address, self.navigation.progress_text(session))
            stale = not session.last_scene_at or time.time() - session.last_scene_at > 15
            return [self._say(text)], ("capture_frame" if stale else None)
        if n == "describe":
            return [], "capture_frame"
        if n == "read_text":
            return [], "capture_frame_read"
        if n == "find":
            return self._start_find(session, intent.target_kind), "capture_frame"
        if n == "navigate":
            return await self._start_navigation(session, intent.destination), None
        # chat → conversational assistant
        address = session.last_address
        if session.position and not address:
            address = session.last_address = await self.navigation.address(session.position)
        reply = await self.reasoning.answer(session, intent.text, address, self.navigation.progress_text(session))
        return [self._say(reply)], None

    def _start_find(self, session: SessionState, kind: str | None) -> list[Utterance]:
        kind = kind or "entrance"
        session.subgoal = kind
        if session.phase in (Phase.IDLE, Phase.ARRIVED):
            session.phase = Phase.IDLE  # find works without a full navigation goal
        lm = find_landmark(session.last_scene, kind) if time.time() - session.last_scene_at < 6 else None
        if lm:
            return [self._say(landmark_phrase(lm, kind))]
        return [self._say(f"Looking for the {kind}. Hold the phone at chest height and turn slowly.")]

    async def _start_navigation(self, session: SessionState, dest: Destination | None) -> list[Utterance]:
        if not dest or not (dest.room or dest.place_query or dest.sign_text or dest.target_kind):
            return [self._say("Where would you like to go? For example, say: take me to Room 443B.")]
        self._resolve_building(session, dest)
        # Keep the building we're already in if the user only names a room.
        prev = session.destination
        if not dest.place_query and prev and prev.place_name and session.phase in (Phase.APPROACH, Phase.INDOOR, Phase.ARRIVED):
            dest.place_name, dest.place_location = prev.place_name, prev.place_location
        session.destination, session.subgoal = dest, None
        session.arrival, session.candidate_streak, session.best_room_gap = ArrivalStatus(), 0, None
        self.db.log(session.id, "route", dest.model_dump(exclude_none=True))

        utts: list[Utterance] = []
        if dest.place_query:
            utts += await self.navigation.start_route(session)
            if session.phase not in (Phase.OUTDOOR, Phase.APPROACH):  # routing failed → still help indoors
                session.phase = Phase.INDOOR
            if self.area:  # fetch entrance / accessibility notes for the destination in the background
                self.area.maybe_destination(session)
        else:
            session.phase = Phase.INDOOR
            goal = f"Room {dest.room}" if dest.room else dest.sign_text or f"the {dest.target_kind}"
            utts.append(self._say(f"Okay, finding {goal}.", category="nav"))
        if (dest.room or dest.sign_text) and dest.floor_explanation:
            floor_txt = dest.floor_explanation
            if session.current_floor is not None and dest.floor_hint is not None and session.phase == Phase.INDOOR:
                if session.current_floor != dest.floor_hint:
                    session.subgoal = "elevator"
                    floor_txt += f" You're on {floor_phrase(session.current_floor)}, so let's find the elevator."
                else:
                    floor_txt += " You're already on that floor."
            utts.append(self._say(floor_txt, category="nav"))
        if session.phase == Phase.INDOOR:
            utts.append(self._say("Hold your phone at chest height, camera forward. I'll read signs and room numbers as you walk.",
                                  Priority.INFO, "nav"))
        return utts
