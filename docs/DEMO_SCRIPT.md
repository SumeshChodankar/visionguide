# Demo Script for Judges (4 minutes + Q&A)

## Cast and props

- **Narrator** (talks to the judges) and **Traveller** (blindfolded or wearing a sleep mask, holding the phone at chest height, with a white cane if you can borrow one). Optionally a **Spotter** walking beside the Traveller for safety.
- Phone volume at maximum; mirror the phone screen to the projector (QuickTime on iOS or `scrcpy` on Android) so judges see captions and badges.
- Printed A4 signs, large bold font:
  `BUSH HOUSE` (use your venue's building name) · `Level 2` · `Level 4` · `Rooms 440–460 →` · `Room 441` · `Room 443A` · `Room 443B`
- A chair, a yellow "wet floor" sign (or a printed one) and a doorway or a cardboard "elevator" door.
- **Staging:** a corridor or row of tables. Station 1: "Level 2" sign by an "elevator" spot. Station 2: "Level 4" sign. Station 3: "Rooms 440–460 →" arrow sign. Station 4: chair in the walking path. Station 5: `Room 443A`, then `Room 443B` a few steps further.

## Before judges arrive (checklist)

- [ ] `curl $API/health` → all `true`, `"agent_mode": "brain"`
- [ ] Run one brain turn and check `tool_trace` `model_ms`. If replies take longer than ~6 s, set `BRAIN_THINKING_LEVEL=minimal`.
- [ ] Phone on charger until the last minute; Expo Go open; app greeting heard
- [ ] Outdoor leg pre-recorded as a 30-second screen recording with audio (indoor demos rarely get a GPS fix)
- [ ] Backup: `python scripts/try_frame.py demo_photos/ --goal "Take me to Room 443B"` works from the laptop
- [ ] Do Not Disturb on; screen auto-lock is handled by the app (keep-awake)

## Script

**0:00 — Hook (Narrator, 20 s)**
> "There are hundreds of millions of people living with visual impairment. GPS apps get them to the right street, and then stop. The hardest part is the last 50 metres: which door, which floor, which room. VisionGuide covers that gap, from the pavement to the door of Room 443B."

**0:20 — Voice goal (Traveller, live)**
Traveller taps the big button: *"Take me to Room 443B in Bush House."*
App says: *"Starting route to Bush House, about 240 meters, about 3 minutes walking. Head north on… Room 443B is likely on floor 4, because the number starts with 4."*
> Narrator: "Whisper transcribes the request and Gemini 4 Argon takes over as the brain. Watch the grey line under the caption: it shows the tools Argon chose, look_around, infer_floor, get_route, find_landmark. Each tool is one of our agents."

**0:45 — Outdoor leg (play the 30 s recording)**
Show turn-by-turn ("In 25 meters, turn right onto…"), then *"You're near Bush House… I'll find the entrance"*, then the camera reading the building sign: *"I can see the sign 'BUSH HOUSE'. This is the right building. The entrance is at 1 o'clock, about 6 meters."*
> "Routes API walking directions, GPS tracking, and the camera confirming it's the right building and the right door. Directions are given as clock positions, the way orientation-and-mobility instructors teach them."

**1:20 — Wrong floor (live, Station 1)**
Traveller faces the "Level 2" sign.
App: *"You're on floor 2, based on the floor sign. Room 443B is likely on floor 4, so we need to go up. Let's find the elevator."* Then: *"The elevator is at 12 o'clock, about 3 meters."*
> "That's agentic reasoning: it read the floor, compared it to the goal, and set itself a sub-goal of finding the elevator."

**1:50 — Exit the elevator (Station 2)**
Traveller turns to "Level 4". App: *"You're on floor 4, based on the floor sign. That's the right floor for Room 443B."* (success vibration)

**2:05 — Signage (Station 3)**
App: *"Sign says rooms 440 to 460, to the right. Room 443B is that way."*

**2:20 — Hazard (Station 4)**
Traveller walks toward the chair. App cuts itself off: *"Stop. Chair ahead, about 2 meters."* (warning vibration). The wet-floor sign: *"Caution, wet floor on your left."*
> "Hazards always come first. YOLO gives fast boxes and distances, Gemini catches things YOLO can't see, like wet floors and stairs. Critical warnings interrupt whatever the app was saying."

**2:45 — Ask anything (live)**
Traveller: *"What is in front of me?"* → full description, including *"Two people around you"* (point at the judges).
Traveller: *"Is there anything I could trip over?"* → conversational answer from the assistant using GPS, vision, OCR and hazard state.

**3:10 — Arrival (Station 5)**
Traveller passes `Room 443A`: *"This is Room 443A, not 443B. It should be right next to it."*
Then faces `Room 443B`: *"I think I see Room 443B… Move a little closer."* → *"You have arrived at Room 443B."* (success vibration)
> "Arrival is verified, not assumed. The room plate has to be read in two frames, or confirmed by both OCR and vision."

**3:30 — Architecture (Narrator, 1 slide)**
Show the diagram from ARCHITECTURE.md:
> "Two layers. Gemini 4 Argon is the brain: it reasons and calls our six agents as tools. Under it, a safety layer of Gemini Flash, YOLO and OCR checks every frame and can interrupt the brain. If Argon is slow or down, the rule-based fallback still answers, and hazards never wait on a language model. It's built with Expo, FastAPI, Gemini 4 Argon, Gemini Flash, YOLOv8n, EasyOCR, Whisper and Google Maps."

**3:50 — Close**
> "VisionGuide gets you to the door, not just the street. Next: on-device detection for faster hazard alerts, and testing with blind users and mobility instructors."

## Likely judge questions

| Question | Answer |
|---|---|
| How fast is it? | About 2–3 s from capture to speech, dominated by Gemini. Hazards come from YOLO in parallel. It complements a cane or guide dog and doesn't replace them. On-device YOLO is the next step to get under 300 ms. |
| What if the network drops? | TTS is on the phone, so it keeps talking; the app tells the user it's offline. The backend degrades per component, and every external call has a timeout and a spoken fallback. |
| How do you know it's floor 4? | Room-numbering conventions (3-digit → first digit, 4-digit → first two, G = ground), confirmed by floor signs, elevator displays or nearby room plates. It says "likely" until it sees confirmation. |
| Hallucination risk? | Gemini is told never to invent text, and arrival needs agreement across frames or sources. Safety-critical wording uses deterministic templates, not free-form LLM text. |
| Privacy? | Frames are processed and discarded. Only text events are logged. No faces are identified, just "a person". |
| Why not let Argon do everything? | It does all the reasoning: every voice turn is Argon choosing tools. But a deep-reasoning model shouldn't sit on the safety path. Each tool call is another model round trip, and a hazard warning can't depend on the model deciding to look. So a fast safety layer runs every 3 seconds regardless, and its critical warnings are spoken before Argon's answer. |
| What if Argon is slow or down? | Each turn has a step cap and a timeout, then falls back to the rule-based parser, which handles every demo command. `AGENT_MODE=rules` switches it off entirely. |
| Why not just use Gemini Live? | Structured per-frame outputs let us fuse YOLO and OCR, keep deterministic safety rules, and verify arrival. Streaming is on the roadmap. |
| Does it work with VoiceOver? | Yes. Every control has a label and hint, magic tap toggles the microphone, and when a screen reader is on, speech goes through it. |
