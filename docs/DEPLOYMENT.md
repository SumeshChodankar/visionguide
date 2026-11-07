# Deployment & Setup Guide

## 1. Get API keys (15 min, do this first)

| Key | Where | Enable / notes |
|---|---|---|
| `GEMINI_API_KEY` | https://aistudio.google.com/apikey | Free tier is enough for a demo. |
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys | Whisper (`whisper-1`). Add a little credit. |
| `GOOGLE_MAPS_API_KEY` | Google Cloud Console → APIs & Services | Enable **Routes API**, **Places API (New)**, **Geocoding API**. Billing account required (monthly free credit covers a demo). Restrict the key to those 3 APIs. |

> The legacy *Directions API* can't be enabled on Cloud projects created after March 2025, so VisionGuide uses its successor, the Routes API. If you're on an older project that already has Directions enabled, you may set `USE_LEGACY_DIRECTIONS=true`.

## 2. Environment variables

`backend/.env` (copy from `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Brain + vision. Without it: YOLO/OCR only, rules mode, deterministic answers. |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Fast per-frame perception model. Any Flash model your key can use. |
| `VISION_THINKING_LEVEL` | `minimal` | Thinking for the per-frame call (Gemini 2.x models automatically use `thinking_budget=0`). |
| `AGENT_MODE` | `brain` | `brain` = tool-calling Gemini handles voice turns; `rules` = deterministic parser only |
| `BRAIN_MODEL` | `gemini-4-argon` | Tool-calling model for voice turns |
| `BRAIN_THINKING_LEVEL` | `low` | `minimal`/`low`/`medium`/`high`: higher means better planning and slower replies |
| `BRAIN_MAX_STEPS` | `6` | Max model↔tool round trips per turn before falling back to rules |
| `BRAIN_TIMEOUT_S` | `25` | Whole-turn timeout before falling back to rules |
| `OPENAI_API_KEY` | — | Whisper. Without it: use `/command` with typed text. |
| `WHISPER_MODEL` | `whisper-1` | |
| `GOOGLE_MAPS_API_KEY` | — | Outdoor routing. Without it: indoor mode still works. |
| `USE_LEGACY_DIRECTIONS` | `false` | Use the old Directions API instead of Routes API |
| `ENABLE_YOLO` / `YOLO_MODEL` / `YOLO_CONF` | `true` / `yolov8n.pt` / `0.35` | Object detection |
| `ENABLE_OCR` / `OCR_LANGUAGES` / `OCR_GPU` | `true` / `en` / `false` | EasyOCR. Comma-separate languages (`en,fr`). |
| `SPEECH_COOLDOWN_S` | `10` | Don't repeat the same sentence sooner |
| `CRITICAL_COOLDOWN_S` | `4` | Hazard repeat interval |
| `ARRIVAL_RADIUS_M` | `30` | GPS distance at which outdoor → entrance search |
| `STEP_ADVANCE_RADIUS_M` | `15` | Distance to a step's end that counts as "reached" |
| `OFFROUTE_THRESHOLD_M` | `45` | Distance from route polyline that triggers re-route |
| `GROUND_FLOOR_IS_ONE` | `true` | US numbering for 2-digit rooms |
| `DB_PATH` | `visionguide.db` | SQLite file |
| `CORS_ORIGINS` | `*` | |

`mobile/.env` (copy from `.env.example`):

| Variable | Example | Purpose |
|---|---|---|
| `EXPO_PUBLIC_API_URL` | `http://192.168.1.23:8000` | Backend URL reachable **from the phone** |
| `EXPO_PUBLIC_SCENE_INTERVAL_MS` | `3000` | Camera analysis cadence |

## 3. Run locally (laptop + phone on the same Wi-Fi)

```bash
# Backend
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
# CPU-only torch first (much smaller than the default CUDA wheel):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env            # fill in keys
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# First start downloads yolov8n.pt (~6 MB) and EasyOCR weights (~100 MB).
curl localhost:8000/health
```

```bash
# Mobile
cd mobile
npm install
cp .env.example .env            # set EXPO_PUBLIC_API_URL to your laptop's LAN IP
npx expo start                  # scan the QR code with Expo Go (iOS/Android)
```

Find your LAN IP: macOS `ipconfig getifaddr en0`, Linux `hostname -I`, Windows `ipconfig`.

**Venue Wi-Fi blocks device-to-device traffic?** (Common at hackathons.) Expose the backend publicly and point the app at it:
```bash
cloudflared tunnel --url http://localhost:8000     # or: ngrok http 8000
# then EXPO_PUBLIC_API_URL=https://<random>.trycloudflare.com
npx expo start --tunnel                            # tunnels the JS bundle too
```

**Test without a phone:**
```bash
python backend/scripts/try_frame.py ~/photos/corridor.jpg --goal "Take me to Room 443B"
curl -X POST localhost:8000/v1/sessions/$SID/command -H 'content-type: application/json' -d '{"text":"Find elevator"}'
```

**Run the tests (no keys needed):**
```bash
cd backend && pip install pytest && pytest -q        # 63 tests
cd mobile && npm run typecheck
```

## 4. Deploy the backend to the cloud

The image is ~2.5 GB with CPU torch + EasyOCR weights baked in. Give it **≥ 2 GB RAM (4 GB recommended)** and **1 worker** (models are loaded per process).

### Google Cloud Run (same project as your Maps key)
```bash
cd backend
gcloud run deploy visionguide --source . --region europe-west2 \
  --memory 4Gi --cpu 2 --min-instances 1 --timeout 60 --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=...,OPENAI_API_KEY=...,GOOGLE_MAPS_API_KEY=...
```
`--min-instances 1` avoids a 30–60 s cold start during judging. SQLite on Cloud Run is per-instance and ephemeral, which is fine for a demo. For persistence, mount a volume or swap to Cloud SQL.

### Render / Railway / Fly.io
Point the service at `backend/` with the Dockerfile. Set the env vars in the dashboard and choose an instance with ≥ 2 GB RAM. The container listens on `$PORT`.

### Lightweight deploy (no torch)
Set `ENABLE_YOLO=false ENABLE_OCR=false` and skip the last two lines of `requirements.txt`. The image drops to ~300 MB and Gemini handles detection and sign reading alone. Hazard distances are then Gemini estimates.

## 5. Build a standalone app (optional)
Expo Go is enough for the demo. For an installable build:
```bash
npm i -g eas-cli && eas login
eas build -p android --profile preview      # APK you can sideload
```
For production, serve the backend over HTTPS and remove `NSAllowsArbitraryLoads` / `usesCleartextTraffic` from `app.json`.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| App says it can't reach the server | Phone and laptop must share Wi-Fi; use the LAN IP, not `localhost`; check the firewall allows port 8000; or use a tunnel. |
| `/health` shows `"yolo": false` | torch/ultralytics not installed or failed to load; check the logs. The app still works via Gemini. |
| Speech plays through the earpiece on iPhone | Fixed by switching the audio mode back after recording (`useVoiceCommand.ts`). Make sure silent mode isn't muting the phone. |
| Frames take > 4 s | Set `ENABLE_OCR=false` (Gemini reads signs) or use a GPU. Lower `frameWidth` in `mobile/src/config.ts`. |
| `PERMISSION_DENIED` from Routes or Places | Enable **Routes API** and **Places API (New)** (not the legacy ones) for the key's project. |
| Voice replies feel slow in brain mode | Check `tool_trace` in the response: `model_ms` per step. Lower `BRAIN_THINKING_LEVEL` to `minimal`, try a Flash model as `BRAIN_MODEL`, or use `AGENT_MODE=rules` for the demo. The safety layer isn't affected either way. |
| `/health` shows `"agent_mode": "rules"` | No `GEMINI_API_KEY`, or `AGENT_MODE=rules`. |
| Gemini returns empty scenes | Check the key on `/health`; look for `Gemini scene analysis failed` in the logs (quota or region). Raise `GEMINI_TIMEOUT_S` on slow networks. |
