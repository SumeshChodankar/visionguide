"""Fakes for every external dependency so the full agent pipeline can be
exercised offline (no API keys, no torch)."""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.db import Database
from app.schemas import Detection, LatLng, RouteStep
from app.services.gemini import GLandmark, GHazard, GScene
from app.services.maps import Route


class FakeGemini:
    enabled = True

    def __init__(self):
        self.next_scene: GScene | None = None
        self.next_text: str | None = None
        self.script = []

    async def analyze_scene(self, jpeg, context):
        self.last_context = context
        return self.next_scene

    # Brain: a scripted sequence of model turns. Each item is a response object
    # or a callable(contents) -> response, so a step can read earlier tool results.
    script: list
    brain_calls = 0

    async def generate_with_tools(self, *, system, contents, declarations, timeout):
        self.brain_calls += 1
        self.brain_contents = contents
        if not self.script:
            raise RuntimeError("model unavailable")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step(contents) if callable(step) else step

    async def text(self, system, prompt, **kw):
        self.last_prompt = prompt
        return self.next_text


# A straight walk north along a street, then a right turn: 3 steps.
A, B, C, D = (51.5000, -0.1200), (51.5010, -0.1200), (51.5010, -0.1185), (51.5012, -0.1185)


class FakeMaps:
    enabled = True

    async def search_place(self, query, near):
        return {"name": "Bush House", "address": "30 Aldwych, London", "location": LatLng(lat=D[0], lng=D[1])}

    async def walking_route(self, origin, dest):
        steps = [
            RouteStep(instruction="Head north on Kingsway", maneuver="DEPART", distance_m=111,
                      start=LatLng(lat=A[0], lng=A[1]), end=LatLng(lat=B[0], lng=B[1])),
            RouteStep(instruction="Turn right onto Aldwych", maneuver="TURN_RIGHT", distance_m=104,
                      start=LatLng(lat=B[0], lng=B[1]), end=LatLng(lat=C[0], lng=C[1])),
            RouteStep(instruction="Turn left. Destination will be on the right", maneuver="TURN_LEFT", distance_m=22,
                      start=LatLng(lat=C[0], lng=C[1]), end=LatLng(lat=D[0], lng=D[1])),
        ]
        return Route(steps, [A, B, C, D], 237, 200)

    async def reverse_geocode(self, pos):
        return "Kingsway, London WC2B"


class FakeDetector:
    enabled = True

    def __init__(self):
        self.next: list[Detection] = []

    def load(self):
        pass

    def detect(self, img):
        return self.next


class FakeOcr:
    enabled = True

    def __init__(self):
        self.next: list = []

    def load(self):
        pass

    def read(self, img):
        return self.next


class FakeWhisper:
    enabled = True
    next_text = "Take me to Room 443B in Bush House"

    async def transcribe(self, audio, filename):
        return self.next_text


@pytest.fixture
def fakes(tmp_path):
    return dict(gemini=FakeGemini(), maps=FakeMaps(), detector=FakeDetector(), ocr_engine=FakeOcr(),
                whisper=FakeWhisper(), db=Database(str(tmp_path / "t.db")))


@pytest.fixture
def client(fakes):
    from app import main

    settings = Settings(_env_file=None, speech_cooldown_s=10, critical_cooldown_s=4,
                        reroute_min_interval_s=0, agent_mode="rules")
    main.app.state.orchestrator = main.build_orchestrator(settings, **fakes)
    with TestClient(main.app) as c:
        yield c
    del main.app.state.orchestrator


@pytest.fixture
def jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (120, 120, 120)).save(buf, format="JPEG")
    return buf.getvalue()


def scene(**kw) -> GScene:
    base = dict(summary="A corridor.", environment="indoor", landmarks=[], hazards=[], sign_texts=[])
    base.update(kw)
    return GScene(**base)


__all__ = ["scene", "GLandmark", "GHazard", "A", "B", "C", "D"]
