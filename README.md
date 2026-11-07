# VisionGuide 🦯

**From the street to the exact door, for blind and low-vision travellers.**

VisionGuide combines GPS walking directions, live camera understanding, sign and room-number reading, hazard detection and a voice assistant. It guides a user from where they are to a specific indoor destination: *"Take me to Room 443B in Bush House."*

```
"Take me to Room 443B in Bush House"
  → "Starting route to Bush House, about 240 meters… Room 443B is likely on floor 4."
  → "In 25 meters, turn right onto Aldwych."
  → "I can see the sign 'BUSH HOUSE'. This is the right building. The entrance is at 1 o'clock, about 6 meters."
  → "You're on floor 2. Room 443B is likely on floor 4, so we need to go up. Let's find the elevator."
  → "Stop. Chair ahead, about 2 meters."
  → "Sign says rooms 440 to 460, to the right. Room 443B is that way."
  → "This is Room 443A, not 443B. It should be right next to it."
  → "You have arrived at Room 443B."
```

## Stack

| Layer | Tech |
|---|---|
| App | React Native (Expo SDK 58): camera, GPS + compass, tap-to-talk, on-device TTS, haptics |
| API | FastAPI + SQLite |
| Brain | **Gemini 4 Argon** with tool calling: the agents are its tools |
| Vision | Gemini Flash (structured scene JSON per frame) + YOLOv8n (objects and distances) |
| OCR | EasyOCR + Gemini sign transcription |
| Maps | Google Routes API (walking), Places API (New), Geocoding API |
| Speech | Whisper (STT), expo-speech / screen reader (TTS) |

## Architecture: a brain and a safety layer

- **🧠 Brain: Gemini 4 Argon, tool-calling.** Every voice turn goes to Argon, which calls the agents as tools (`look_around`, `get_route`, `infer_floor`, `find_landmark`, `detect_hazards`, `read_signs`, `check_arrival`, `save_memory`…) and phrases the answer. The camera frame taken with the command is analysed while Argon is still thinking.
- **🛡️ Safety layer: always on.** Every ~3 s a Flash model, YOLOv8n and EasyOCR run in parallel through deterministic hazard, floor and arrival rules. It speaks on its own, interrupts for critical hazards, and its critical warnings are spoken before any brain answer.
- **Six agents:** **Navigation** (GPS route and turns) · **Vision** (scene, landmarks) · **OCR** (rooms, floors, gates, signs) · **Hazard** (fused, prioritised warnings) · **Voice** (STT, instant stop/repeat/help) · **Reasoning** (floor inference, wayfinding, arrival verification, speech governor), coordinated by an **Orchestrator** phase machine: `idle → outdoor → approach → indoor → arrived`.
- **Fallbacks:** if Argon times out, errors or loops, the turn is handled by the rule-based parser. Set `AGENT_MODE=rules` to run without any LLM in the voice path.

## Quick start

```bash
# 1. Backend
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env   # add GEMINI_API_KEY, OPENAI_API_KEY, GOOGLE_MAPS_API_KEY (BRAIN_MODEL defaults to gemini-4-argon)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 2. App (phone with Expo Go on the same Wi-Fi)
cd mobile
npm install
cp .env.example .env   # EXPO_PUBLIC_API_URL=http://<your-laptop-LAN-IP>:8000
npx expo start
```

Tests run without any API keys: `cd backend && pip install pytest && pytest -q` (63 tests, including the full outdoor-to-room journey and the Argon tool loop driven by a scripted model).

## Repository map

```
visionguide/
├── backend/
│   ├── app/
│   │   ├── main.py                 FastAPI routes + dependency wiring
│   │   ├── config.py               env settings
│   │   ├── schemas.py              shared pydantic models
│   │   ├── db.py                   SQLite sessions / events / saved places
│   │   ├── agents/
│   │   │   ├── brain.py            Gemini 4 Argon tool loop: agents exposed as tools
│   │   │   ├── orchestrator.py     phase machine, parallel perception, brain/rules dispatch
│   │   │   ├── navigation_agent.py Places + Routes, GPS tracking, re-routing
│   │   │   ├── vision_agent.py     Gemini scene + YOLO
│   │   │   ├── ocr_agent.py        EasyOCR + sign classification
│   │   │   ├── hazard_agent.py     fusion + severity + phrasing
│   │   │   ├── voice_agent.py      Whisper + intent parser
│   │   │   └── reasoning_agent.py  floors, wayfinding, arrival, governor, assistant
│   │   ├── services/               gemini · maps · local_models (YOLO/EasyOCR) · speech (Whisper)
│   │   └── utils/                  geo (clock directions, polylines) · floor · text · image
│   ├── tests/                      offline fakes + unit + end-to-end tests
│   ├── scripts/try_frame.py        run photos through the pipeline from a laptop
│   ├── requirements.txt · Dockerfile · .env.example
├── mobile/
│   ├── App.tsx
│   └── src/
│       ├── screens/MainScreen.tsx  accessible single-screen UI
│       ├── hooks/                  useSceneLoop · useLocationTracking · useVoiceCommand
│       ├── speech.ts               priority speech queue + haptics
│       ├── api.ts · types.ts · config.ts
└── docs/
    ├── ARCHITECTURE.md             diagrams, agents, orchestration, DB schema, decisions
    ├── API.md                      endpoints with real request/response examples
    ├── DEPLOYMENT.md               keys, env vars, local / cloud deploy, troubleshooting
    ├── BUILD_PLAN.md               5-hour team plan, file-by-file order, cut list
    └── DEMO_SCRIPT.md              4-minute judge demo + Q&A prep
```

## Safety note

VisionGuide is an orientation aid, not a mobility aid. It adds information to a white cane or guide dog and doesn't replace them. Its typical 2–3 second perception latency is fine for wayfinding but not for last-moment collision avoidance. It never tells a user that it is safe to cross a road.
