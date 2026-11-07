"""VisionGuide API.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agents.area_agent import AreaAgent
from .agents.hazard_agent import HazardAgent
from .agents.navigation_agent import NavigationAgent
from .agents.ocr_agent import OCRAgent
from .agents.orchestrator import Orchestrator
from .agents.reasoning_agent import ReasoningAgent
from .agents.vision_agent import VisionAgent
from .agents.voice_agent import VoiceAgent
from .config import Settings, get_settings
from .db import Database, SessionStore
from .schemas import (AgentResponse, ChatRequest, CommandRequest, CreateSessionResponse, FrameResponse,
                      PositionUpdate, SessionState, Utterance)
from .services.buildings import BuildingService
from .services.gemini import GeminiService
from .services.local_models import OcrEngine, YoloDetector
from .services.maps import MapsService
from .services.speech import WhisperService

MAX_UPLOAD_BYTES = 8 * 1024 * 1024


def build_orchestrator(settings: Settings, *, gemini=None, maps=None, detector=None, ocr_engine=None, whisper=None,
                       db: Database | None = None, buildings: BuildingService | None = None) -> Orchestrator:
    """Wire services → agents → orchestrator. Tests inject fakes here."""
    db = db or Database(settings.db_path)
    gemini = gemini or GeminiService(settings)
    maps = maps or MapsService(settings)
    detector = detector or YoloDetector(settings)
    ocr_engine = ocr_engine or OcrEngine(settings)
    whisper = whisper or WhisperService(settings, gemini)
    store = SessionStore(db)
    buildings = buildings or BuildingService(settings, gemini)
    return Orchestrator(
        settings, store, db,
        area=AreaAgent(settings, maps, gemini, store, db),
        buildings=buildings,
        navigation=NavigationAgent(maps, settings),
        vision=VisionAgent(gemini, detector),
        ocr=OCRAgent(ocr_engine, settings.camera_hfov_deg),
        hazard=HazardAgent(),
        voice=VoiceAgent(whisper, db),
        reasoning=ReasoningAgent(gemini, settings),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # httpx logs full request URLs at INFO, and Google Geocoding/legacy APIs carry the API key in the URL.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if not hasattr(app.state, "orchestrator"):
        app.state.orchestrator = build_orchestrator(settings)
    orch: Orchestrator = app.state.orchestrator
    # Warm the local models in the background so the first frame isn't slow.
    asyncio.get_running_loop().run_in_executor(None, orch.vision.detector.load)
    asyncio.get_running_loop().run_in_executor(None, orch.ocr.engine.load)
    if orch.buildings:
        orch.buildings.prepare_notes_background()  # one-time Gemini read of any new floor plans
    yield
    maps = orch.navigation.maps
    if hasattr(maps, "close"):
        await maps.close()


app = FastAPI(title="VisionGuide API", version="1.0.0", lifespan=lifespan,
              description="Multi-agent navigation assistant for blind and low-vision users.")
_origins = [o.strip() for o in get_settings().cors_origins.split(",")]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["*"], allow_headers=["*"])
# Floor plans and photos, viewable from the phone: /buildings/marshgate/images/level-02.jpg
if Path(get_settings().buildings_dir).is_dir():
    app.mount("/buildings", StaticFiles(directory=get_settings().buildings_dir), name="buildings")


def orch() -> Orchestrator:
    return app.state.orchestrator


def session_or_404(session_id: str, o: Orchestrator = Depends(orch)) -> SessionState:
    s = o.store.get(session_id)
    if not s:
        raise HTTPException(404, "Unknown session")
    return s


async def _read_upload(f: UploadFile) -> bytes:
    data = await f.read()
    if not data:
        raise HTTPException(400, "Empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Upload too large (max 8 MB)")
    return data


@app.get("/health")
def health(o: Orchestrator = Depends(orch)):
    return {
        "status": "ok",
        "agent_mode": "brain" if o.brain and o.brain.enabled else "rules",
        "brain_model": o.s.brain_model,
        "vision_model": o.s.gemini_model,
        "gemini": o.vision.gemini.enabled,
        "maps": o.navigation.maps.enabled,
        "map_sources": o.navigation.maps.status() if hasattr(o.navigation.maps, "status") else None,
        "area_prefetch": o.s.area_prefetch,
        "buildings": [b.name for b in o.buildings.buildings.values()] if o.buildings else [],
        "whisper": o.voice.whisper.enabled,
        "speech_to_text": getattr(o.voice.whisper, "engine", None),
        "yolo": o.vision.detector.enabled,
        "ocr": o.ocr.engine.enabled,
    }


@app.post("/v1/sessions", response_model=CreateSessionResponse)
def create_session(o: Orchestrator = Depends(orch)):
    s, utts = o.create_session()
    return CreateSessionResponse(session_id=s.id, utterances=utts)


@app.get("/v1/sessions/{session_id}", response_model=SessionState)
def get_session(s: SessionState = Depends(session_or_404)):
    return s


@app.get("/v1/buildings")
def list_buildings(o: Orchestrator = Depends(orch)):
    """Buildings we have floor plans for: levels, named spaces, plan image URLs, notes status."""
    return o.buildings.info() if o.buildings else []


@app.get("/v1/sessions/{session_id}/area")
def get_area(s: SessionState = Depends(session_or_404), o: Orchestrator = Depends(orch)):
    """The background-prefetched map of the user's surroundings, as the brain sees it."""
    return o.area.summary(s, limit=40) if o.area else {"status": "disabled"}


@app.post("/v1/sessions/{session_id}/position", response_model=AgentResponse)
async def update_position(body: PositionUpdate, s: SessionState = Depends(session_or_404), o: Orchestrator = Depends(orch)):
    return await o.handle_position(s, body.lat, body.lng, body.heading, body.accuracy_m, body.floor_change)


@app.post("/v1/sessions/{session_id}/frame", response_model=FrameResponse)
async def analyze_frame(
    image: UploadFile = File(..., description="JPEG from the phone camera"),
    mode: str = Form("auto", pattern="^(auto|describe|read)$"),
    lat: float | None = Form(None), lng: float | None = Form(None), heading: float | None = Form(None),
    s: SessionState = Depends(session_or_404), o: Orchestrator = Depends(orch),
):
    data = await _read_upload(image)
    nav_utts = []
    if lat is not None and lng is not None:
        nav_utts = (await o.handle_position(s, lat, lng, heading, None)).utterances
    try:
        resp = await o.handle_frame(s, data, mode)
    except OSError:
        raise HTTPException(400, "Could not decode image")
    # Turn instructions triggered by the piggy-backed GPS fix must not be lost;
    # keep safety-critical items first.
    resp.utterances = sorted(nav_utts + resp.utterances, key=lambda u: -int(u.priority))
    return resp


@app.post("/v1/sessions/{session_id}/fast_frame", response_model=FrameResponse)
async def fast_frame(image: UploadFile = File(..., description="Small JPEG for the YOLO-only safety pass"),
                     s: SessionState = Depends(session_or_404), o: Orchestrator = Depends(orch)):
    """~0.1-0.3 s hazard check (no Gemini, no OCR) for the live loop while walking."""
    data = await _read_upload(image)
    try:
        return await o.handle_fast_frame(s, data)
    except OSError:
        raise HTTPException(400, "Could not decode image")


@app.post("/v1/sessions/{session_id}/voice", response_model=AgentResponse)
async def voice_command(audio: UploadFile = File(...),
                        image: UploadFile | None = File(None, description="Optional camera frame taken with the command"),
                        s: SessionState = Depends(session_or_404), o: Orchestrator = Depends(orch)):
    frame = await _read_upload(image) if image is not None else None
    return await o.handle_audio(s, await _read_upload(audio), audio.filename or "speech.m4a", frame)


@app.post("/v1/sessions/{session_id}/ask", response_model=AgentResponse)
async def ask(text: str = Form(...), image: UploadFile | None = File(None), s: SessionState = Depends(session_or_404),
              o: Orchestrator = Depends(orch)):
    """Typed command + optional camera frame (same path as /voice, minus Whisper). Handy for demos and tests."""
    frame = await _read_upload(image) if image is not None else None
    return await o.handle_text(s, text, frame)


@app.post("/v1/sessions/{session_id}/command", response_model=AgentResponse)
async def text_command(body: CommandRequest, s: SessionState = Depends(session_or_404), o: Orchestrator = Depends(orch)):
    return await o.handle_text(s, body.text)


@app.post("/v1/sessions/{session_id}/chat", response_model=AgentResponse)
async def chat(body: ChatRequest, s: SessionState = Depends(session_or_404), o: Orchestrator = Depends(orch)):
    """Bonus: free-form conversational assistant over the fused world state."""
    address = s.last_address or await o.navigation.address(s.position)
    s.last_address = address
    reply = await o.reasoning.answer(s, body.message, address, o.navigation.progress_text(s))
    return AgentResponse(**o._finish(s, [Utterance(text=reply, category="answer")], transcript=body.message))


@app.get("/v1/memories")
def memories(o: Orchestrator = Depends(orch)):
    return o.db.memories()


@app.get("/v1/sessions/{session_id}/events")
def session_events(session_id: str, limit: int = 100, s: SessionState = Depends(session_or_404),
                   o: Orchestrator = Depends(orch)):
    return o.db.events(session_id, limit)


class SavedPlace(BaseModel):
    label: str
    query: str
    room: str | None = None


@app.put("/v1/places/saved")
def save_place(body: SavedPlace, o: Orchestrator = Depends(orch)):
    """Save a shortcut so the user can say 'take me to work'."""
    o.db.upsert_saved_place(body.label, body.query, body.room)
    return {"ok": True}
