"""Pydantic models shared by agents, API and storage."""
from __future__ import annotations

import time
import uuid
from enum import Enum, IntEnum
from typing import Literal

from pydantic import BaseModel, Field


class Priority(IntEnum):
    INFO = 0       # ambient description
    GUIDANCE = 1   # navigation instruction
    WARNING = 2    # hazard worth knowing
    CRITICAL = 3   # hazard directly in path / imminent


class Phase(str, Enum):
    IDLE = "idle"            # no goal; answers questions, describes scene
    OUTDOOR = "outdoor"      # GPS turn-by-turn to the building
    APPROACH = "approach"    # near the building: camera finds the right entrance
    INDOOR = "indoor"        # camera + OCR find the room / target
    ARRIVED = "arrived"


class Utterance(BaseModel):
    text: str
    priority: Priority = Priority.INFO
    category: str = "info"          # hazard | nav | arrival | scene | answer | system
    interrupt: bool = False         # client should cut off current speech
    haptic: Literal["none", "light", "warning", "success"] = "none"
    key: str | None = Field(default=None, exclude=True)  # de-duplication key for the speech governor


class LatLng(BaseModel):
    lat: float
    lng: float


# ---------------------------------------------------------------- perception
class Detection(BaseModel):
    label: str
    confidence: float
    bbox: list[float]               # normalised x1,y1,x2,y2
    clock: int                      # 11, 12, 1 ... relative to camera
    side: str                       # left | ahead | right
    distance_m: float | None = None
    source: Literal["yolo", "gemini"] = "yolo"


class OCRText(BaseModel):
    text: str
    confidence: float
    bbox: list[float]
    clock: int
    side: str
    kind: Literal["room", "floor", "gate", "building", "direction", "other"] = "other"
    value: str | None = None        # canonical room code, floor number, gate number...
    room_range: list[int] | None = None


class Hazard(BaseModel):
    type: str
    severity: Priority
    side: str = "ahead"
    clock: int | None = None
    distance_m: float | None = None
    description: str = ""
    source: Literal["yolo", "gemini", "fused"] = "gemini"


class Landmark(BaseModel):
    type: str                       # entrance, exit, reception, elevator, stairs, escalator, corridor, door, sign, restroom
    side: str = "ahead"
    clock: int | None = None
    distance_m: float | None = None
    description: str = ""


class SceneAnalysis(BaseModel):
    summary: str = ""
    environment: Literal["indoor", "outdoor", "unknown"] = "unknown"
    landmarks: list[Landmark] = Field(default_factory=list)
    hazards: list[Hazard] = Field(default_factory=list)
    sign_texts: list[str] = Field(default_factory=list)
    floor_indicator: str | None = None
    destination_visible: bool = False
    destination_evidence: str | None = None
    suggested_guidance: str | None = None


class ArrivalStatus(BaseModel):
    state: Literal["searching", "candidate", "confirmed"] = "searching"
    target: str | None = None
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.0


# ---------------------------------------------------------------- navigation
class RouteStep(BaseModel):
    instruction: str
    maneuver: str | None = None
    distance_m: float
    start: LatLng
    end: LatLng
    announced_pre: bool = False


class Destination(BaseModel):
    raw: str
    room: str | None = None                 # "443B"
    sign_text: str | None = None            # non-room text to find on signage: "Gate 4", "Building C"
    target_kind: str | None = None          # elevator | reception | stairs | exit | restroom | entrance
    place_query: str | None = None          # "Bush House, King's College London"
    place_name: str | None = None
    place_address: str | None = None
    place_location: LatLng | None = None
    floor_hint: int | None = None
    floor_explanation: str | None = None


class Intent(BaseModel):
    name: Literal["navigate", "find", "where_am_i", "describe", "read_text",
                  "repeat", "stop", "help", "chat"]
    destination: Destination | None = None
    target_kind: str | None = None
    text: str = ""


# ---------------------------------------------------------------- area map
class NearbyPlace(BaseModel):
    name: str
    kind: str                               # e.g. "subway_entrance", "pharmacy", "crossing (signals)"
    lat: float
    lng: float
    source: str                             # "google_places" | "osm"
    detail: str | None = None               # e.g. "wheelchair: yes", street address


class AreaContext(BaseModel):
    """Background-prefetched map of the user's surroundings; the brain answers from it."""
    center: LatLng
    fetched_at: float
    address: str | None = None
    places: list[NearbyPlace] = Field(default_factory=list)
    briefing: str | None = None             # Gemini + Google Maps grounded summary of the area
    briefing_sources: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    destination_brief: str | None = None    # entrances / accessibility notes for the active destination
    destination_key: str | None = None
    complete: bool = False                  # every source answered; incomplete maps are retried sooner


# ---------------------------------------------------------------- session
class SessionState(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = Field(default_factory=time.time)
    phase: Phase = Phase.IDLE
    destination: Destination | None = None
    route: list[RouteStep] = Field(default_factory=list)
    route_polyline: list[tuple[float, float]] = Field(default_factory=list)
    step_index: int = 0
    last_reroute_at: float = 0.0
    position: LatLng | None = None
    heading: float | None = None
    position_accuracy_m: float | None = None
    current_floor: int | None = None
    subgoal: str | None = None              # e.g. "elevator" while heading to a room on another floor
    last_scene: SceneAnalysis | None = None
    last_scene_at: float = 0.0
    last_ocr: list[OCRText] = Field(default_factory=list)
    last_hazards: list[Hazard] = Field(default_factory=list)
    last_address: str | None = None
    arrival: ArrivalStatus = Field(default_factory=ArrivalStatus)
    candidate_streak: int = 0
    best_room_gap: int | None = None        # |seen room number - target| (getting closer?)
    spoken_at: dict[str, float] = Field(default_factory=dict)
    last_utterances: list[Utterance] = Field(default_factory=list)
    chat_history: list[dict] = Field(default_factory=list)
    area: AreaContext | None = None
    building_id: str | None = None   # a building we have floor plans for, when the user is at it
    fast_track: list[dict] = Field(default_factory=list)   # last fast-frame detections (label, clock, distance)
    fast_track_at: float = 0.0
    pending_utterances: list[Utterance] = Field(default_factory=list)  # from background agents


# ---------------------------------------------------------------- API I/O
class CreateSessionResponse(BaseModel):
    session_id: str
    utterances: list[Utterance]


class PositionUpdate(BaseModel):
    lat: float
    lng: float
    heading: float | None = None
    accuracy_m: float | None = None
    floor_change: int | None = None   # from the phone barometer: +1 = went up one floor since the last report


class CommandRequest(BaseModel):
    text: str


class ChatRequest(BaseModel):
    message: str


class AgentResponse(BaseModel):
    utterances: list[Utterance] = Field(default_factory=list)
    phase: Phase
    step_index: int | None = None
    arrival: ArrivalStatus | None = None
    transcript: str | None = None
    intent: Intent | None = None
    destination: Destination | None = None
    client_action: Literal["capture_frame", "capture_frame_read"] | None = None  # ask app to grab a frame now
    tool_trace: list[dict] | None = None    # brain mode: which tools the model called, with timings
    handled_by: Literal["brain", "rules"] | None = None


class FrameResponse(AgentResponse):
    scene: SceneAnalysis | None = None
    detections: list[Detection] = Field(default_factory=list)
    ocr: list[OCRText] = Field(default_factory=list)
    hazards: list[Hazard] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
