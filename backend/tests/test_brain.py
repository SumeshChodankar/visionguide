"""Brain mode: Gemini (Argon) drives the agents through tool calls.
The model is scripted, so these tests check the loop, the tools, and the
safety guarantees around the model, not the model's own judgement."""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from google.genai import types

from app import main
from app.config import Settings
from app.schemas import Detection
from conftest import GLandmark, scene


def calls(*pairs):
    fcs = [SimpleNamespace(id=f"c{i}", name=n, args=a) for i, (n, a) in enumerate(pairs)]
    content = types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(id=f.id, name=f.name, args=f.args)) for f in fcs])
    return SimpleNamespace(function_calls=fcs, candidates=[SimpleNamespace(content=content)], text=None)


def say(text):
    return SimpleNamespace(function_calls=None, candidates=[], text=text)


def tool_results(contents) -> dict:
    """All function responses the model has been sent so far, by tool name."""
    out = {}
    for c in contents:
        for p in getattr(c, "parts", None) or []:
            if getattr(p, "function_response", None):
                out[p.function_response.name] = p.function_response.response["result"]
    return out


@pytest.fixture
def bclient(fakes):
    settings = Settings(_env_file=None, agent_mode="brain", reroute_min_interval_s=0)
    main.app.state.orchestrator = main.build_orchestrator(settings, **fakes)
    with TestClient(main.app) as c:
        yield c
    del main.app.state.orchestrator


def voice(client, sid, jpeg=None):
    files = {"audio": ("a.m4a", b"fake", "audio/m4a")}
    if jpeg:
        files["image"] = ("f.jpg", jpeg, "image/jpeg")
    r = client.post(f"/v1/sessions/{sid}/voice", files=files)
    assert r.status_code == 200, r.text
    return r.json()


def test_take_me_to_room_with_tools(bclient, fakes, jpeg):
    g = fakes["gemini"]
    fakes["whisper"].next_text = "Take me to Room 443B"
    # Camera frame sent with the voice command: elevator display says 2, elevator on the left, chair ahead.
    g.next_scene = scene(floor_indicator="2", summary="Lift lobby.",
                         landmarks=[GLandmark(type="elevator", side="left", distance_m=4)])
    fakes["detector"].next = [Detection(label="chair", confidence=0.9, bbox=[0.4, 0.4, 0.6, 0.95], clock=12,
                                        side="ahead", distance_m=1.4)]

    def compose(contents):
        r = tool_results(contents)
        assert r["infer_floor"]["likely_floor"] == 4 and r["infer_floor"]["needs_floor_change"] is True
        assert r["get_route"]["phase"] == "indoor" and r["get_route"]["subgoal"] == "elevator"
        assert r["look_around"]["hazards"][0]["severity"] == "CRITICAL"
        elev = r["find_landmark"]
        return say(f"Room 443B is likely on floor 4 and you're on floor 2. "
                   f"The elevator is at {elev['clock']} o'clock, about {round(elev['distance_m'])} metres.")

    g.script = [
        calls(("look_around", {}), ("infer_floor", {"room": "443B"})),
        calls(("get_route", {"room": "443B"}), ("find_landmark", {"kind": "elevator"})),
        compose,
    ]
    sid = bclient.post("/v1/sessions").json()["session_id"]
    body = voice(bclient, sid, jpeg)

    assert body["handled_by"] == "brain"
    texts = [u["text"] for u in body["utterances"]]
    # Safety layer: the critical hazard is spoken first even though the model didn't mention it.
    assert texts[0].startswith("Stop! Chair at 12 o'clock, ") and body["utterances"][0]["interrupt"]
    assert texts[-1] == "Room 443B is likely on floor 4 and you're on floor 2. The elevator is at 11 o'clock, about 4 metres."
    tools = [t["tool"] for t in body["tool_trace"] if "tool" in t]
    assert tools == ["look_around", "infer_floor", "get_route", "find_landmark"]
    assert body["phase"] == "indoor" and body["destination"]["room"] == "443B"
    # Gemini 3 thought signatures: the model's own turn must be echoed back verbatim.
    assert any(getattr(c, "role", None) == "model" for c in g.brain_contents)


def test_memory_and_status_prompt(bclient, fakes):
    g = fakes["gemini"]
    sid = bclient.post("/v1/sessions").json()["session_id"]
    fakes["whisper"].next_text = "My office is Room 443B in Bush House"
    g.script = [calls(("save_memory", {"key": "office", "value": "Room 443B in Bush House"})), say("Got it, I'll remember that.")]
    assert voice(bclient, sid)["utterances"][-1]["text"] == "Got it, I'll remember that."
    assert bclient.get("/v1/memories").json() == {"office": "Room 443B in Bush House"}

    fakes["whisper"].next_text = "Take me to my office"
    g.script = [say("Okay.")]
    voice(bclient, sid)
    prompt = g.brain_contents[0].parts[0].text
    assert "Room 443B in Bush House" in prompt and "Take me to my office" in prompt


def test_no_frame_asks_for_one(bclient, fakes):
    g = fakes["gemini"]
    fakes["whisper"].next_text = "What's in front of me?"
    g.script = [calls(("look_around", {})),
                lambda c: say("I can't see right now. Hold the phone up." if tool_results(c)["look_around"]["status"] == "no_fresh_image" else "?")]
    sid = bclient.post("/v1/sessions").json()["session_id"]
    body = voice(bclient, sid)
    assert body["client_action"] == "capture_frame" and "Hold the phone up" in body["utterances"][-1]["text"]


def test_outdoor_route_tool(bclient, fakes):
    g = fakes["gemini"]
    sid = bclient.post("/v1/sessions").json()["session_id"]
    bclient.post(f"/v1/sessions/{sid}/position", json={"lat": 51.5, "lng": -0.12})
    g.script = [calls(("get_route", {"place": "Bush House, London", "room": "443B"})),
                lambda c: say(f"Starting. {tool_results(c)['get_route']['first_steps'][0]['instruction']}.")]
    r = bclient.post(f"/v1/sessions/{sid}/ask", data={"text": "Take me to Room 443B in Bush House"}).json()
    assert r["phase"] == "outdoor" and r["utterances"][-1]["text"] == "Starting. Head north on Kingsway."


def test_instant_intents_skip_the_model(bclient, fakes):
    sid = bclient.post("/v1/sessions").json()["session_id"]
    r = bclient.post(f"/v1/sessions/{sid}/command", json={"text": "stop"}).json()
    assert r["handled_by"] == "rules" and fakes["gemini"].brain_calls == 0


def test_brain_failure_falls_back_to_rules(bclient, fakes):
    fakes["gemini"].script = [RuntimeError("503 overloaded")]
    sid = bclient.post("/v1/sessions").json()["session_id"]
    r = bclient.post(f"/v1/sessions/{sid}/command", json={"text": "Take me to Room 205A"}).json()
    assert r["handled_by"] == "rules" and r["phase"] == "indoor" and "floor 2" in json.dumps(r["utterances"])


def test_runaway_tool_loop_is_capped(bclient, fakes):
    fakes["gemini"].script = [calls(("get_current_location", {}))] * 10
    sid = bclient.post("/v1/sessions").json()["session_id"]
    r = bclient.post(f"/v1/sessions/{sid}/command", json={"text": "Where am I?"}).json()
    assert r["handled_by"] == "rules" and fakes["gemini"].brain_calls == 6


def test_health_reports_mode(bclient):
    h = bclient.get("/health").json()
    assert h["agent_mode"] == "brain" and h["brain_model"] == "gemini-4-argon"


def test_thinking_config_by_generation():
    from app.services.gemini import GeminiService
    g = GeminiService(Settings(_env_file=None))
    assert g._thinking("gemini-2.5-flash", "minimal").thinking_budget == 0
    assert g._thinking("gemini-4-argon", "low").thinking_level == types.ThinkingLevel.LOW
