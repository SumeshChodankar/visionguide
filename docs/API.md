# VisionGuide API

Base URL: `http://<host>:8000`. Interactive docs: `/docs` (Swagger). All bodies are JSON unless marked multipart.

Every agent endpoint returns an **AgentResponse**. The app only has to do three things with it: speak `utterances` in order, show `phase`, and run `client_action` when present.

| Method | Path | Purpose | Owner agent |
|---|---|---|---|
| GET | `/health` | Which components are live (gemini, maps, whisper, yolo, ocr) | — |
| POST | `/v1/sessions` | Start a session, get a greeting | Orchestrator |
| GET | `/v1/sessions/{id}` | Full session state (debug / judge view) | Orchestrator |
| POST | `/v1/sessions/{id}/position` | GPS + compass update (~every 2 s) | Navigation |
| POST | `/v1/sessions/{id}/frame` | Camera frame (multipart, ~every 3 s) | Vision + OCR + Hazard → Reasoning |
| POST | `/v1/sessions/{id}/voice` | Recorded voice command + optional camera frame (multipart) | Voice → Brain (or rules) |
| POST | `/v1/sessions/{id}/ask` | Typed command + optional camera frame (multipart) | Brain (or rules) |
| POST | `/v1/sessions/{id}/command` | Typed command, JSON `{"text": ...}`, no frame | Brain (or rules) |
| POST | `/v1/sessions/{id}/chat` | Free-form question to the assistant | Reasoning |
| GET | `/v1/sessions/{id}/events` | Event log: commands, utterances, frames, brain traces | — |
| GET | `/v1/memories` | What the brain has saved with `save_memory` | — |
| PUT | `/v1/places/saved` | Save a shortcut: `{"label":"work","query":"Bush House London","room":"443B"}` | — |

## Common types

```ts
Utterance   { text, priority: 0 INFO | 1 GUIDANCE | 2 WARNING | 3 CRITICAL,
              category: hazard|nav|arrival|scene|answer|system,
              interrupt: bool,               // cut off current speech
              haptic: none|light|warning|success }
Phase       idle | outdoor | approach | indoor | arrived
ArrivalStatus { state: searching|candidate|confirmed, target, evidence[], confidence }
AgentResponse { utterances[], phase, step_index, arrival, transcript, intent, destination,
                client_action: null | "capture_frame" | "capture_frame_read",
                handled_by: "brain" | "rules",
                tool_trace: [{step, model_ms} | {tool, args, ms}] }   // brain turns only
```

## Brain mode (default)

With `AGENT_MODE=brain` and a Gemini key, `/voice`, `/ask` and `/command` go to the tool-calling brain (Gemini 4 Argon). "Stop", "repeat" and "help" are always answered instantly by rules. Example response for *"Take me to Room 443B"* sent with a camera frame:

```json
{
  "utterances": [
    {"text": "Stop. Chair ahead, about 1.4 meters.", "priority": 3, "category": "hazard", "interrupt": true, "haptic": "warning"},
    {"text": "Room 443B is likely on floor 4 and you're on floor 2. The elevator is at 11 o'clock, about 4 metres.", "priority": 1, "category": "answer"}
  ],
  "phase": "indoor",
  "handled_by": "brain",
  "tool_trace": [
    {"step": 0, "model_ms": 1810},
    {"tool": "look_around", "args": {}, "ms": 240},
    {"tool": "infer_floor", "args": {"room": "443B"}, "ms": 0},
    {"step": 1, "model_ms": 1540},
    {"tool": "get_route", "args": {"room": "443B"}, "ms": 2},
    {"tool": "find_landmark", "args": {"kind": "elevator"}, "ms": 0},
    {"step": 2, "model_ms": 1320}
  ],
  "destination": {"room": "443B", "floor_hint": 4, "...": "..."},
  "client_action": null
}
```
The first utterance comes from the safety layer, not the model: critical hazards in the attached frame are always spoken first. (Timings above are illustrative; real numbers depend on the model and network.)

If the brain times out, errors or exceeds `BRAIN_MAX_STEPS`, the same request is handled by the rule-based dispatcher and `handled_by` is `"rules"`.

`client_action` lets the server ask the phone for a fresh look: "What is in front of me?" returns `capture_frame`, and the app immediately posts a frame with `mode=describe`.

## POST /v1/sessions/{id}/command

```bash
curl -X POST localhost:8000/v1/sessions/$SID/command -H 'content-type: application/json' \
     -d '{"text":"Take me to Room 443B in Bush House"}'
```

```json
{
  "utterances": [
    {"text": "Starting route to Bush House, about 240 meters, about 3 minutes walking. Head north on Kingsway, for about 110 meters.",
     "priority": 1, "category": "nav", "interrupt": false, "haptic": "light"},
    {"text": "Room 443B is likely on floor 4, because the number starts with 4.",
     "priority": 1, "category": "nav", "interrupt": false, "haptic": "none"}
  ],
  "phase": "outdoor",
  "step_index": 0,
  "arrival": {"state": "searching", "target": null, "evidence": [], "confidence": 0.0},
  "transcript": "Take me to Room 443B in Bush House",
  "intent": {"name": "navigate", "destination": {"...": "same as below"}, "text": "..."},
  "destination": {
    "raw": "Room 443B in Bush House", "room": "443B", "place_query": "Bush House",
    "place_name": "Bush House", "place_address": "30 Aldwych, London",
    "place_location": {"lat": 51.5012, "lng": -0.1185},
    "floor_hint": 4, "floor_explanation": "Room 443B is likely on floor 4, because the number starts with 4."
  },
  "client_action": null
}
```

Supported intents:

| Say | Intent | What happens |
|---|---|---|
| "Take me to Room 443B in Bush House" | `navigate` | Places search → walking route → OUTDOOR; floor hint spoken |
| "Take me to Room 205A" | `navigate` | No building named → INDOOR immediately |
| "Take me to Gate 4" / "Building C at Imperial College" | `navigate` | `sign_text` matched against OCR for arrival |
| "Take me to work" | `navigate` | Saved place lookup |
| "Find elevator" / "Guide me to reception" / "Where's the toilet" | `find` | sets `subgoal`, returns `capture_frame` |
| "Where am I?" | `where_am_i` | reverse geocode + floor + fresh scene + route progress |
| "What is in front of me?" | `describe` | `capture_frame` → full description |
| "Read the sign" | `read_text` | `capture_frame_read` → reads all text |
| "Repeat" / "Stop" / "Help" | … | |
| anything else ("Is it busy here?") | `chat` | Gemini assistant over the world snapshot |

## POST /v1/sessions/{id}/frame (multipart)

| Field | Type | Notes |
|---|---|---|
| `image` | file | JPEG, ≤ 8 MB (app sends ~960 px wide) |
| `mode` | `auto` \| `describe` \| `read` | `auto` = governed live loop; `describe`/`read` force speech |
| `lat`, `lng`, `heading` | float, optional | piggy-backed position update |

```bash
curl -X POST localhost:8000/v1/sessions/$SID/frame -F image=@corridor.jpg -F mode=auto
```

Response = AgentResponse plus perception detail:

```json
{
  "utterances": [
    {"text": "Caution, construction barrier on your left, about 3 meters.", "priority": 2, "category": "hazard", "interrupt": false, "haptic": "light"},
    {"text": "I can see the sign 'BUSH HOUSE'. This is the right building. The entrance is at 1 o'clock, about 6 meters.", "priority": 1, "category": "nav", "interrupt": false, "haptic": "success"}
  ],
  "phase": "approach",
  "arrival": {"state": "searching", "target": "Room 443B", "evidence": [], "confidence": 0.0},
  "scene": {
    "summary": "Glass doors with a sign above.", "environment": "outdoor",
    "landmarks": [{"type": "entrance", "side": "right", "clock": 1, "distance_m": 6.0, "description": "glass doors"}],
    "hazards": [{"type": "construction barrier", "severity": 2, "side": "left", "clock": 11, "distance_m": 3.0, "source": "gemini"}],
    "sign_texts": ["BUSH HOUSE"], "floor_indicator": null, "destination_visible": false, "suggested_guidance": null
  },
  "detections": [{"label": "person", "confidence": 0.88, "bbox": [0.1, 0.2, 0.3, 0.9], "clock": 11, "side": "left", "distance_m": 2.4, "source": "yolo"}],
  "ocr": [{"text": "BUSH HOUSE", "confidence": 0.7, "clock": 12, "side": "ahead", "kind": "building", "value": "BUSH HOUSE"}],
  "hazards": [{"type": "construction barrier", "severity": 2, "side": "left", "distance_m": 3.0, "source": "gemini"}],
  "timings_ms": {"gemini": 1430.2, "yolo": 61.0, "ocr": 712.4, "total": 1460.9}
}
```

## POST /v1/sessions/{id}/position

```json
{"lat": 51.50075, "lng": -0.1200, "heading": 2.5, "accuracy_m": 6}
```
→ AgentResponse, e.g. `"In 25 meters, turn right onto Aldwych."`, then `"Now, turn right onto Aldwych. Then continue about 100 meters."`, then on arrival near the building `"You're near Bush House. It's at 2 o'clock. Hold your phone at chest height…"` with `phase: "approach"`.

## POST /v1/sessions/{id}/voice (multipart)

| Field | Notes |
|---|---|
| `audio` | m4a / wav / webm (≤ 8 MB), transcribed by Whisper |
| `image` | optional JPEG of what the user is facing; used by `look_around` / `detect_hazards` / `read_signs` |

`transcript` is returned so the UI can show what was heard.

## POST /v1/sessions/{id}/ask (multipart)

Fields `text` and optional `image`. Same handling as `/voice` without Whisper. The app's quick buttons use it, and it's handy for demos:
```bash
curl -X POST localhost:8000/v1/sessions/$SID/ask -F text="Take me to Room 443B" -F image=@lobby.jpg
```

## POST /v1/sessions/{id}/chat

```json
{"message": "Is there anything I could trip over?"}
```
→ `{"utterances":[{"text":"Yes, there's a chair directly ahead, about 2 meters. Step to your right to pass it.", ...}], ...}`

The assistant receives a JSON snapshot: address, GPS, heading, route progress, destination and floor hint, current floor, sub-goal, arrival status, scene age, landmarks, hazards and visible text. If Gemini is unavailable it falls back to a deterministic summary of the same state.

## Errors

| Code | When |
|---|---|
| 400 | Empty upload, undecodable image |
| 404 | Unknown session id |
| 413 | Upload larger than 8 MB |
| 422 | Validation error (bad `mode`, missing field) |

External API failures never surface as 5xx. They become a spoken fallback, e.g. "I couldn't get walking directions right now. I can still guide you with the camera."
