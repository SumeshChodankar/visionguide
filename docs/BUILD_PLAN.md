# 5-Hour Build Plan

The code in this repo is complete. This plan is for a team rebuilding it from scratch, or for splitting up the work of setting it up, tuning it and rehearsing. It assumes **4 people**. With 3, merge the Mobile and Demo roles.

| Role | Owns |
|---|---|
| **A — Backend core** | config, schemas, db, services (Gemini, Maps, YOLO/EasyOCR, Whisper), `main.py` |
| **B — Agents** | voice intents, OCR classification, floor/arrival reasoning, hazard fusion, orchestrator, tests |
| **C — Mobile** | Expo app: camera loop, GPS, voice, speech queue, accessible UI |
| **D — Demo & prompts** | API keys, signs and props, test photos, Gemini prompt tuning, demo script, slides |

## Timeline

| Time | A — Backend core | B — Agents | C — Mobile | D — Demo |
|---|---|---|---|---|
| **0:00–0:20** | Repo, venv, `requirements.txt`, CPU torch | Read ARCHITECTURE.md, agree on `schemas.py` with A | `create-expo-app`, install expo modules, `app.json` permissions | Create Gemini/OpenAI/Maps keys, enable the 3 Maps APIs, share `.env` |
| **0:20–1:20** | `config.py`, `schemas.py`, `db.py`, `services/gemini.py` (structured scene JSON), `services/maps.py` | `utils/geo.py`, `utils/text.py`, `utils/floor.py` + `test_logic.py` (TDD: 443B→4) | `api.ts`, `speech.ts` priority queue, `MainScreen` layout with big buttons | Print signs: "Room 443B", "Room 441", "Rooms 440–460 →", "Level 4", "Level 2", building name; take 20 test photos |
| **1:20–2:20** | `services/local_models.py` (YOLO + EasyOCR), `services/speech.py`, `main.py` endpoints, `/health` | `voice_agent.py` intents, `ocr_agent.py` classify, `hazard_agent.py` | `useVoiceCommand` (record → /voice), `useLocationTracking` | Run `scripts/try_frame.py` on photos; tune `SCENE_SYSTEM` prompt in gemini.py |
| **2:20–3:20** | `brain.py`: tool declarations wrapping the agents, Argon tool loop, fallback; latency check with `tool_trace` | `reasoning_agent.py` (floor, arrival, indoor guidance, governor), `orchestrator.py`, `test_flow.py` | `useSceneLoop` (capture → resize → /frame), `client_action` handling | Walk the route outdoors once; note GPS behaviour at the building |
| **3:20–4:00** | Integration: phone ↔ backend end-to-end | Fix bugs from real frames; tune cooldowns and thresholds | VoiceOver/TalkBack pass: labels, magic tap, caption contrast | Draft the demo script; set up the staged corridor |
| **4:00–4:30** | **Feature freeze.** Rehearse ×2 with a blindfold | | | |
| **4:30–5:00** | Backup video recording, slides, README polish, buffer | | | |

**Merge checkpoints:** 1:20 (schemas frozen), 2:20 (backend runs end-to-end with fakes), 3:20 (phone talks to backend), 4:00 (freeze).

## File-by-file implementation order

| # | File | Lines | Depends on | Done when |
|---|---|---|---|---|
| 1 | `backend/app/config.py` | 55 | — | settings load from `.env` |
| 2 | `backend/app/schemas.py` | 200 | — | all agents import from it |
| 3 | `backend/app/utils/geo.py` | 120 | — | clock directions, polyline distance tested |
| 4 | `backend/app/utils/text.py` | 70 | — | `room_match_score("4438","443B") ≥ 0.85` |
| 5 | `backend/app/utils/floor.py` | 110 | 4 | `443B → 4`, `Level 3 → 3` |
| 6 | `backend/app/db.py` | 120 | 2 | sessions persist across restart |
| 7 | `backend/app/services/gemini.py` | 150 | 1 | returns `GScene` for a real photo |
| 8 | `backend/app/services/maps.py` | 145 | 1, 3 | walking route returned for 2 real points |
| 9 | `backend/app/services/local_models.py` | 120 | 1, 3 | YOLO boxes with distances; EasyOCR reads a printed sign |
| 10 | `backend/app/services/speech.py` | 40 | 1 | Whisper transcribes an m4a |
| 11 | `backend/app/agents/voice_agent.py` | 130 | 2, 5 | the 5 required commands parse |
| 12 | `backend/app/agents/ocr_agent.py` | 90 | 9, 5 | room / floor / gate / building / direction classified |
| 13 | `backend/app/agents/vision_agent.py` | 70 | 7, 9 | context includes the goal |
| 14 | `backend/app/agents/hazard_agent.py` | 155 | 2 | "Stop. Chair ahead" for chair < 2.5 m |
| 15 | `backend/app/agents/navigation_agent.py` | 130 | 8 | steps advance on simulated walk |
| 16 | `backend/app/agents/reasoning_agent.py` | 365 | 5, 4 | full journey test passes |
| 17 | `backend/app/agents/orchestrator.py` | 290 | 11–16 | rules mode works end-to-end |
| 17b | `backend/app/agents/brain.py` | 390 | 17 | scripted-model tests pass; one real Argon turn calls ≥ 2 tools |
| 18 | `backend/app/main.py` | 180 | 17 | `/docs` works, `/health` green |
| 19 | `backend/tests/*` | 360 | all | `pytest` green |
| 20 | `mobile/src/types.ts`, `config.ts`, `api.ts` | 140 | 18 | |
| 21 | `mobile/src/speech.ts` | 125 | 20 | critical interrupts info |
| 22 | `mobile/src/hooks/useLocationTracking.ts` | 80 | 20 | positions posted |
| 23 | `mobile/src/hooks/useVoiceCommand.ts` | 75 | 20 | tap → speak → response spoken |
| 24 | `mobile/src/hooks/useSceneLoop.ts` | 75 | 20 | frames every 3 s, no backlog |
| 25 | `mobile/src/screens/MainScreen.tsx`, `App.tsx` | 220 | 21–24 | usable with eyes closed |

## Cut list (if you're behind at 3:20, drop in this order)

1. Saved places, memories and the `/events` endpoint
1b. Brain mode (`AGENT_MODE=rules`): the rule parser covers every demo command. Cut it only if the Argon integration isn't stable by 3:20.
2. Off-route re-routing (keep step advance)
3. EasyOCR (`ENABLE_OCR=false`; Gemini reads signs)
4. YOLO (`ENABLE_YOLO=false`; Gemini reports hazards)
5. Whisper (use the three quick buttons and typed `/command`)

**Never cut:** hazard priority and interrupt, floor reasoning, arrival confirmation. These are what the demo is judged on.
