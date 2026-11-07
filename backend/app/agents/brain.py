"""Brain: a tool-calling Gemini model (default Gemini 4 Argon) that handles
every voice turn. The specialist agents are exposed to it as tools, and the
model decides which to call, in what order, and how to phrase the answer.

    User: "Take me to Room 443B"
      model → infer_floor(room="443B")
      model → get_current_location()
      model → look_around()            ← runs on the frame captured with the voice command
      model → get_route(room="443B")
      model → "Room 443B should be on floor 4. You're on floor 2, so let's take the elevator.
               It's at 11 o'clock, 4 meters. Caution! Chair at 12 o'clock, 1.6 meters."

What the brain deliberately does NOT own:
  * the per-frame hazard loop: that runs every ~3 s whether or not anyone
    speaks, with deterministic severity rules, and can interrupt the brain;
  * stop / repeat / help: answered instantly without an LLM round trip;
  * any critical hazard seen during a turn is spoken first by the orchestrator,
    even if the model forgets to mention it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from google.genai import types

from ..schemas import Destination, Phase, Priority, SessionState
from ..utils.floor import floor_phrase, infer_floor_from_room
from ..utils.text import canonical_room
from ..utils.geo import object_phrase
from .area_agent import _plain
from .hazard_agent import SPOKEN_NAMES
from .perception import Perception
from .reasoning_agent import find_landmark

if TYPE_CHECKING:
    from .orchestrator import Orchestrator

log = logging.getLogger("visionguide.brain")

SCENE_FRESH_S = 10.0

BRAIN_SYSTEM = """You are VisionGuide, the voice of a navigation assistant for a blind or low-vision person.
They hold a phone at chest height with the camera facing forward. Each turn gives you their words,
a JSON status snapshot and saved memories. Use your tools to find out what you need, then reply.

How to work:
- Surroundings, "what's ahead", "is it safe", "where is X", "am I there": call look_around (and read_signs
  when text matters). Never describe the scene from memory.
- Going somewhere: call get_route with structured fields. A room alone ("Room 443B") with no building
  means they are already inside. Resolve "my office", "home" and similar from memories. If unclear, ask one
  short question instead of guessing.
- If the destination room is on another floor than current_floor, say so and guide them to the elevator
  (find_landmark kind="elevator").
- When the user tells you a lasting fact or preference ("my office is 443B in Bush House", "I prefer stairs"),
  call save_memory.
- Call independent tools together in one step where you can; every step adds delay for the user.

Buildings with our own floor plans (see status.building, present at or on the way to such a building):
- It lists every level's named spaces (Level 00 = ground floor, Level N = floor N) and detailed notes for the current
  and destination levels. Answer "where is X / which floor / what's on level N" from it, naming the level exactly.
- To go to a space there, call get_route with place=<building name> and sign_text=<space name exactly as listed>;
  the level is filled in from the plans. If a space exists on several levels (e.g. Library on 02 and 03), say so.
- Describe where a space is using the landmarks in the notes (atrium, stair/lift cores, neighbouring spaces).
  "Top/left of the page" in the notes is plan orientation, NOT the user's left or right: never turn it into
  left/right/clock directions unless the camera or a sign confirms which way they face.
- Use building_lookup for a space not in the summary; use consult_floor_plan only when the notes can't answer
  (it looks at the plan image and is slower). The plans show named spaces, not room numbers.

Area map (prefetched in the background, see status.area):
- status.area holds the user's surroundings: street address, a Google Maps briefing, destination entrance notes,
  and the nearest places with distance in metres, clock direction (when the compass heading is known) and compass
  direction. Answer questions about nearby places, streets, stations, stops, crossings, shops and entrances FROM IT
  directly, quoting names and distances exactly as given (round to the nearest 10 metres when speaking).
- If it doesn't contain what was asked, call nearby_places with a query; if that finds nothing, call ask_maps.
- If status.area.status is "loading", say the map of the area is still loading and answer what you can.
  If it is "partial", a place missing from the map may still exist: call nearby_places, then ask_maps, before
  saying something isn't nearby.
- Never invent a place, street, distance or direction that is not in the map data or a tool result.

How to speak (your text is read aloud by text-to-speech):
- At most 3 short sentences. Plain spoken English. No markdown, lists, emojis or URLs.
- Safety first: CRITICAL hazards are spoken by the app, word for word, right before your reply - do not repeat
  them. If a WARNING hazard is reported, say its "say" phrase in your first sentence.
- Use clock directions (12 o'clock = straight ahead, 3 = right, 9 = left) and metres.
- PRECISION: when a tool result gives a "say" phrase (e.g. "Chair at 1 o'clock, 2.4 meters"), use it word for
  word. Never round, re-estimate or drop a measured distance or clock position, never write "about",
  "around", "roughly", "nearby" or "very close" for a measured value, and keep sign text, room numbers and
  place names exactly as given. Only for map distances above 10 metres may you round to the nearest 10.
- For "what's ahead / what's in front of me / describe" questions the app already speaks the measured object
  list before your reply: don't repeat that list; add what it lacks (layout, doors, signs, the way forward).
- Say "likely" for inferences (floor from room number). Never invent text, rooms or objects you were not given.
- Never say it is safe to cross a road; suggest listening to traffic, using their cane or dog, or asking someone.
- If you have no fresh camera view, say so and ask them to hold the phone up."""


def _schema(props: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props or {}, "required": required or []}


DECLARATIONS = [
    types.FunctionDeclaration(
        name="get_current_location",
        description="User's GPS position, street address, compass heading, current floor (if known), "
                    "navigation phase, active destination and sub-goal.",
        parameters_json_schema=_schema()),
    types.FunctionDeclaration(
        name="get_route",
        description="Set the destination and plan the route. Outdoor walking directions are computed if a "
                    "building/place is given; otherwise indoor guidance starts. Returns route summary, first steps "
                    "and floor inference.",
        parameters_json_schema=_schema({
            "place": {"type": "string", "description": "Building or place name with city if known, e.g. 'Bush House, London'. Omit if the user is already inside."},
            "room": {"type": "string", "description": "Room or office code, e.g. '443B'."},
            "sign_text": {"type": "string", "description": "Non-room sign or named space to find, e.g. 'Gate 4', "
                                                          "'Building C', or a space from our floor plans like 'Library'."},
            "target_kind": {"type": "string", "enum": ["elevator", "stairs", "escalator", "reception", "exit",
                                                       "entrance", "restroom", "door"],
                            "description": "A type of place inside the building."},
        })),
    types.FunctionDeclaration(
        name="get_route_progress",
        description="Current turn-by-turn step, next instruction, distance and clock direction to the destination.",
        parameters_json_schema=_schema()),
    types.FunctionDeclaration(
        name="look_around",
        description="Analyse the camera view the user is facing right now: scene summary, landmarks with clock "
                    "direction and distance, hazards, people, visible text, indoor/outdoor, floor indicator.",
        parameters_json_schema=_schema({
            "focus": {"type": "string", "description": "Optional specific question to look for, e.g. 'is there a free seat'."},
        })),
    types.FunctionDeclaration(
        name="detect_hazards",
        description="Hazards in or near the walking path from the current camera view (fused object detection + "
                    "vision), sorted by severity, with side/clock direction and distance.",
        parameters_json_schema=_schema()),
    types.FunctionDeclaration(
        name="read_signs",
        description="All readable text in the current camera view (OCR), classified as room, floor, gate, "
                    "building, direction or other, with clock direction.",
        parameters_json_schema=_schema()),
    types.FunctionDeclaration(
        name="infer_floor",
        description="Infer which floor a room is on from its number (e.g. 443B → floor 4) and compare with the "
                    "user's current floor.",
        parameters_json_schema=_schema({"room": {"type": "string"}}, ["room"])),
    types.FunctionDeclaration(
        name="find_landmark",
        description="Start looking for a landmark (elevator, stairs, reception, exit, entrance, restroom, door). "
                    "The live camera loop will keep announcing it. Returns its position if visible now.",
        parameters_json_schema=_schema({"kind": {"type": "string", "enum": ["elevator", "stairs", "escalator", "reception",
                                                                            "exit", "entrance", "restroom", "door"]}},
                                       ["kind"])),
    types.FunctionDeclaration(
        name="check_arrival",
        description="Whether the user has reached the destination, using room-plate OCR and vision evidence.",
        parameters_json_schema=_schema()),
    types.FunctionDeclaration(
        name="save_memory",
        description="Remember a lasting fact or preference about the user for future sessions.",
        parameters_json_schema=_schema({
            "key": {"type": "string", "description": "Short snake_case key, e.g. 'office', 'mobility_aid'."},
            "value": {"type": "string"},
        }, ["key", "value"])),
    types.FunctionDeclaration(
        name="building_lookup",
        description="Search our floor plans for a named space (e.g. 'library', 'cafe', 'lecture theatre', "
                    "'computer lab'). Returns building, level and where on the level it is. Instant.",
        parameters_json_schema=_schema({
            "query": {"type": "string"},
            "building": {"type": "string", "description": "Building name if the user named one."},
        }, ["query"])),
    types.FunctionDeclaration(
        name="consult_floor_plan",
        description="Look at the actual floor-plan image of one level to answer a wayfinding question the notes "
                    "can't (e.g. 'is the lecture theatre next to the stairs?'). Slower (~5-10 s).",
        parameters_json_schema=_schema({
            "level": {"type": "integer", "description": "Level number, 0 = ground."},
            "question": {"type": "string"},
            "building": {"type": "string"},
        }, ["level", "question"])),
    types.FunctionDeclaration(
        name="nearby_places",
        description="Search the prefetched map of the user's surroundings (Google Places + OpenStreetMap) for places "
                    "matching a name or kind, e.g. query='pharmacy', 'Holborn', 'bus stop', 'crossing', 'entrance'. "
                    "Returns matches sorted by distance with metres, clock and compass direction. Instant.",
        parameters_json_schema=_schema({
            "query": {"type": "string", "description": "Name or kind of place to look for."},
        }, ["query"])),
    types.FunctionDeclaration(
        name="ask_maps",
        description="Ask Google Maps (via Gemini grounding) a question about places around the user or their "
                    "destination: opening hours, accessible entrances, which station is closest, etc. Slower (~5-10 s); "
                    "use only when the area map and nearby_places don't answer it.",
        parameters_json_schema=_schema({
            "question": {"type": "string"},
        }, ["question"])),
    types.FunctionDeclaration(
        name="stop_navigation",
        description="Cancel the current destination and guidance.",
        parameters_json_schema=_schema()),
]


class BrainError(RuntimeError):
    pass


class Turn:
    """Per-turn state shared by the tool implementations."""

    def __init__(self, orch: "Orchestrator", session: SessionState, frame: bytes | None):
        self.o, self.s, self.frame = orch, session, frame
        self.perception: Perception | None = None
        self._perceive_task: asyncio.Task | None = None
        self.client_action: str | None = None
        self.trace: list[dict] = []
        if frame:  # speculative: most turns need the camera, so start it alongside the first model call
            self._perceive_task = asyncio.create_task(orch.perceive(session, frame, "describe"))

    async def get_perception(self, focus: str | None = None) -> Perception | None:
        if not self.perception and self._perceive_task:
            self.perception = await self._perceive_task
            # Floor signs / elevator displays in view update the world model before any tool reads it.
            self.o.reasoning.update_floor(self.s, self.perception.ocr, self.perception.scene)
        if self.perception and focus and self.frame:  # a focused question deserves a focused look
            self.perception = await self.o.perceive(self.s, self.frame, "describe", focus)
        return self.perception

    def cancel(self) -> None:
        if self._perceive_task and not self._perceive_task.done():
            self._perceive_task.cancel()


def _hz(h) -> dict:
    return {"type": h.type, "severity": Priority(h.severity).name, "side": h.side, "clock": h.clock,
            "distance_m": h.distance_m, "detail": h.description or None,
            "say": object_phrase(h.type, h.clock, h.side, h.distance_m)}


class Brain:
    name = "brain"

    def __init__(self, orch: "Orchestrator"):
        self.o = orch
        self.s = orch.s
        self.gemini = orch.vision.gemini
        self.tools: dict[str, Callable[[Turn, dict], Awaitable[Any]]] = {
            "get_current_location": self.t_location,
            "get_route": self.t_route,
            "get_route_progress": self.t_progress,
            "look_around": self.t_look,
            "detect_hazards": self.t_hazards,
            "read_signs": self.t_signs,
            "infer_floor": self.t_floor,
            "find_landmark": self.t_find,
            "check_arrival": self.t_arrival,
            "save_memory": self.t_save_memory,
            "building_lookup": self.t_building_lookup,
            "consult_floor_plan": self.t_consult_plan,
            "nearby_places": self.t_nearby,
            "ask_maps": self.t_ask_maps,
            "stop_navigation": self.t_stop,
        }

    @property
    def enabled(self) -> bool:
        return self.s.agent_mode == "brain" and self.gemini.enabled

    # ------------------------------------------------------------------ loop
    async def run(self, session: SessionState, text: str, frame: bytes | None) -> tuple[Turn, str]:
        turn = Turn(self.o, session, frame)
        try:
            reply = await asyncio.wait_for(self._loop(turn, text), timeout=self.s.brain_timeout_s)
        except asyncio.TimeoutError as e:
            turn.cancel()
            raise BrainError("brain timed out") from e
        except BrainError:
            turn.cancel()
            raise
        except Exception as e:  # noqa: BLE001
            turn.cancel()
            raise BrainError(str(e)) from e
        if turn.perception is None and turn._perceive_task and turn._perceive_task.done():
            turn.perception = turn._perceive_task.result()
        return turn, reply

    async def _loop(self, turn: Turn, text: str) -> str:
        context = {
            "status": self._status(turn.s),
            "area": self.o.area.summary(turn.s) if self.o.area else None,
            "building": self._building_context(turn.s),
            "memories": self.o.db.memories(),
            "camera_frame_attached": turn.frame is not None,
        }
        history = [m for m in turn.s.chat_history[-6:]]
        prompt = (f"Status: {json.dumps(context, default=str)}\n"
                  + (f"Recent conversation: {json.dumps(history)}\n" if history else "")
                  + f"User said: {text}")
        contents: list = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]

        for step in range(self.s.brain_max_steps):
            t = time.perf_counter()
            resp = await self.gemini.generate_with_tools(
                system=BRAIN_SYSTEM, contents=contents, declarations=DECLARATIONS,
                timeout=self.s.brain_timeout_s)
            turn.trace.append({"step": step, "model_ms": round((time.perf_counter() - t) * 1000)})
            calls = resp.function_calls or []
            if not calls:
                reply = (resp.text or "").strip()
                if not reply:
                    raise BrainError("empty reply")
                return reply
            # Keep the model's own turn verbatim: Gemini 3+ attaches thought
            # signatures that must be sent back with the function responses.
            contents.append(resp.candidates[0].content)
            results = await asyncio.gather(*(self._call(turn, c) for c in calls))
            contents.append(types.Content(role="user", parts=[
                types.Part(function_response=types.FunctionResponse(id=c.id, name=c.name, response={"result": r}))
                for c, r in zip(calls, results)
            ]))
        raise BrainError("too many tool steps")

    async def _call(self, turn: Turn, call) -> Any:
        fn = self.tools.get(call.name)
        t = time.perf_counter()
        try:
            result = await fn(turn, dict(call.args or {})) if fn else {"error": f"unknown tool {call.name}"}
        except Exception as e:  # noqa: BLE001 - a broken tool should not end the turn
            log.warning("tool %s failed: %s", call.name, e)
            result = {"error": str(e)}
        turn.trace.append({"tool": call.name, "args": dict(call.args or {}), "ms": round((time.perf_counter() - t) * 1000)})
        return result

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _status(s: SessionState) -> dict:
        d = s.destination
        return {
            "phase": s.phase.value,
            "destination": d.model_dump(exclude_none=True, include={"room", "sign_text", "target_kind", "place_name", "floor_hint"}) if d else None,
            "current_floor": s.current_floor,
            "subgoal": s.subgoal,
            "has_gps": s.position is not None,
            "arrival": s.arrival.state,
        }

    async def _need_view(self, turn: Turn, focus: str | None = None) -> Perception | dict:
        p = await turn.get_perception(focus)
        if p:
            return p
        if turn.s.last_scene and time.time() - turn.s.last_scene_at < SCENE_FRESH_S:
            return Perception(scene=turn.s.last_scene, detections=[], ocr=turn.s.last_ocr,
                              hazards=turn.s.last_hazards)
        turn.client_action = "capture_frame"
        return {"status": "no_fresh_image",
                "advice": "No recent camera view. Ask the user to hold the phone up at chest height; a frame is being captured."}

    # ------------------------------------------------------------------ tools
    async def t_location(self, turn: Turn, a: dict) -> dict:
        s = turn.s
        if s.position and not s.last_address:
            s.last_address = await self.o.navigation.address(s.position)
        out = self._status(s)
        out.update({
            "gps": s.position.model_dump() if s.position else None,
            "accuracy_m": s.position_accuracy_m,
            "heading_deg": s.heading,
            "address": s.last_address,
            "environment": s.last_scene.environment if s.last_scene else "unknown",
        })
        return out

    async def t_route(self, turn: Turn, a: dict) -> dict:
        await turn.get_perception()
        room = canonical_room(a["room"]) if a.get("room") else None
        dest = Destination(raw=" ".join(str(v) for v in a.values()), room=room, sign_text=a.get("sign_text"),
                           target_kind=a.get("target_kind"), place_query=a.get("place"))
        if room:
            inf = infer_floor_from_room(room, self.s.ground_floor_is_one)
            dest.floor_hint, dest.floor_explanation = inf.floor, inf.explanation
        notes = await self.o._start_navigation(turn.s, dest)
        s = turn.s
        return {
            "phase": s.phase.value,
            "place_found": dest.place_name,
            "address": dest.place_address,
            "first_steps": [{"instruction": st.instruction, "distance_m": st.distance_m} for st in s.route[:3]],
            "floor_hint": dest.floor_hint,
            "current_floor": s.current_floor,
            "subgoal": s.subgoal,
            "notes": [u.text for u in notes],
        }

    async def t_progress(self, turn: Turn, a: dict) -> dict:
        s = turn.s
        if s.phase != Phase.OUTDOOR or not s.route:
            return {"phase": s.phase.value, "message": "No outdoor route active."}
        return {"phase": s.phase.value, "step_index": s.step_index, "total_steps": len(s.route),
                "progress": self.o.navigation.progress_text(s),
                "next_instruction": s.route[s.step_index + 1].instruction if s.step_index + 1 < len(s.route) else None}

    async def t_look(self, turn: Turn, a: dict) -> dict:
        p = await self._need_view(turn, a.get("focus"))
        if isinstance(p, dict):
            return p
        sc = p.scene
        people = [d for d in p.detections if d.label == "person"]
        measured = sorted((d for d in p.detections if d.distance_m is not None), key=lambda d: d.distance_m)
        return {
            "summary": sc.summary if sc else None,
            "environment": sc.environment if sc else "unknown",
            # Camera-measured (YOLO) objects, nearest first: the most precise facts available.
            "measured_objects": [{"object": d.label, "clock": d.clock, "distance_m": d.distance_m,
                                  "say": object_phrase(SPOKEN_NAMES.get(d.label, d.label), d.clock, d.side, d.distance_m)}
                                 for d in measured[:6]],
            "landmarks": [{"type": l.type, "clock": l.clock, "distance_m": l.distance_m, "detail": l.description or None,
                           "say": object_phrase(l.type, l.clock, l.side, l.distance_m)}
                          for l in (sc.landmarks if sc else [])],
            "hazards": [_hz(h) for h in p.hazards],
            "people": {"count": len(people), "nearest_m": min((d.distance_m or 99 for d in people), default=None)},
            "visible_text": [o.text for o in p.ocr][:12],
            "floor_indicator": sc.floor_indicator if sc else None,
            "vision_model_suggestion": sc.suggested_guidance if sc else None,
        }

    async def t_hazards(self, turn: Turn, a: dict) -> dict:
        p = await self._need_view(turn)
        if isinstance(p, dict):
            return p
        return {"hazards": [_hz(h) for h in p.hazards], "path_clear": not any(h.severity >= Priority.WARNING for h in p.hazards)}

    async def t_signs(self, turn: Turn, a: dict) -> dict:
        p = await self._need_view(turn)
        if isinstance(p, dict):
            return p
        return {"texts": [{"text": o.text, "kind": o.kind, "value": o.value, "clock": o.clock,
                           "room_range": o.room_range} for o in p.ocr]}

    async def t_floor(self, turn: Turn, a: dict) -> dict:
        await turn.get_perception()  # current floor may come from the attached frame
        inf = infer_floor_from_room(a.get("room", ""), self.s.ground_floor_is_one)
        cur = turn.s.current_floor
        return {"room": canonical_room(a.get("room", "")), "likely_floor": inf.floor, "confidence": inf.confidence,
                "explanation": inf.explanation, "current_floor": cur,
                "needs_floor_change": None if cur is None or inf.floor is None else cur != inf.floor,
                "current_floor_phrase": floor_phrase(cur) if cur is not None else "unknown"}

    async def t_find(self, turn: Turn, a: dict) -> dict:
        kind = a.get("kind", "entrance")
        turn.s.subgoal = kind
        p = await turn.get_perception()
        scene = p.scene if p else (turn.s.last_scene if time.time() - turn.s.last_scene_at < SCENE_FRESH_S else None)
        lm = find_landmark(scene, kind)
        if lm:
            return {"visible": True, "clock": lm.clock, "distance_m": lm.distance_m, "detail": lm.description or None}
        if not p:
            turn.client_action = "capture_frame"
        return {"visible": False, "note": "Not in view. The live camera loop will announce it when it appears."}

    async def t_arrival(self, turn: Turn, a: dict) -> dict:
        p = await turn.get_perception()
        if p:
            self.o.reasoning.verify_arrival(turn.s, p.ocr, p.scene)
        return turn.s.arrival.model_dump()

    async def t_save_memory(self, turn: Turn, a: dict) -> dict:
        self.o.db.save_memory(a["key"], a["value"])
        return {"saved": True}

    def _building(self, s, name: str | None = None):
        svc = getattr(self.o, "buildings", None)
        if not svc:
            return None
        if name and (b := svc.by_name(name)):
            return b
        if s.building_id and s.building_id in svc.buildings:
            return svc.buildings[s.building_id]
        d = s.destination
        return svc.by_name(d.place_name) if d and d.place_name else None

    def _building_context(self, s) -> dict | None:
        b = self._building(s)
        if not b:
            return None
        levels = [lv for lv in (s.current_floor, s.destination.floor_hint if s.destination else None) if lv is not None]
        ctx = self.o.buildings.summary(b, levels=list(dict.fromkeys(levels)))
        ctx["user_is_here"] = s.building_id == b.id
        ctx["current_level"] = s.current_floor
        return ctx

    async def t_building_lookup(self, turn: Turn, a: dict) -> dict:
        svc = getattr(self.o, "buildings", None)
        if not svc or not svc.buildings:
            return {"error": "no building floor plans loaded"}
        b = self._building(turn.s, a.get("building"))
        hits = svc.find_space(a.get("query", ""), b) or (svc.find_space(a.get("query", "")) if b else [])
        return {"query": a.get("query"), "matches": hits,
                "note": None if hits else "Not on our floor plans (they show named spaces, not room numbers)."}

    async def t_consult_plan(self, turn: Turn, a: dict) -> dict:
        svc = getattr(self.o, "buildings", None)
        b = self._building(turn.s, a.get("building"))
        if not svc or not b:
            return {"error": "no floor plan for this building"}
        level = int(a.get("level", turn.s.current_floor or 0))
        ctx = f"The user is currently on level {turn.s.current_floor}." if turn.s.current_floor is not None else ""
        answer = await svc.consult(b, level, a.get("question", ""), ctx)
        return {"building": b.name, "level": b.level_label(level), "answer": _plain(answer)} if answer else \
            {"error": f"couldn't read the plan for {b.level_label(level)}"}

    async def t_nearby(self, turn: Turn, a: dict) -> dict:
        area = self.o.area
        if not area:
            return {"error": "area map disabled"}
        if turn.s.area is None:
            await area.wait(turn.s, 8.0)  # first fetch may still be running
        if turn.s.area is None:
            return {"status": "no_map_yet", "has_gps": turn.s.position is not None}
        q = a.get("query", "")
        hits = area.lookup(turn.s, query=q) or area.lookup(turn.s, kind=q)
        return {"query": q, "matches": hits, "searched_places": len(turn.s.area.places)}

    async def t_ask_maps(self, turn: Turn, a: dict) -> dict:
        s = turn.s
        target = s.position or (s.destination.place_location if s.destination else None)
        if not target:
            return {"error": "no location yet; ask the user to allow location access"}
        fn = getattr(self.gemini, "maps_grounded", None)
        if not fn:
            return {"error": "Google Maps grounding unavailable"}
        where = f" The user is near {s.last_address}." if s.last_address else ""
        answer, sources = await fn(f"{a.get('question', '')}{where} Answer in at most 60 words, facts only.",
                                   target.lat, target.lng, timeout=15.0)
        answer = _plain(answer)
        return {"answer": answer, "google_maps_places": sources[:6]} if answer else {"error": "no answer from Google Maps"}

    async def t_stop(self, turn: Turn, a: dict) -> dict:
        await self.o._dispatch(turn.s, self.o.voice.understand("stop"))
        return {"stopped": True}
