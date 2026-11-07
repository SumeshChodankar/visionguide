# Running VisionGuide on an Android phone

The **backend** (AI, maps, floor plans) runs on the laptop. The **app** runs on the phone in **Expo Go**.
The phone talks to the laptop over Wi-Fi, so both must be on the same network.

```
Android phone (Expo Go)  ──Wi-Fi──▶  Laptop
  camera, GPS, compass,                backend  :8001  (FastAPI: YOLO, OCR, Gemini, maps, floor plans)
  steps, gyroscope, barometer,         Expo     :8081  (serves the app to Expo Go)
  speech, vibration
```

---

## Quick start: one command

After the one-time setup below, start everything with **one command** (or double-click `start-visionguide.cmd`):

```powershell
cd "C:\Users\sumes\projects_sumeshchodankar\UCL London Hackathon 2\visionguide\visionguide"
powershell -ExecutionPolicy Bypass -File scripts\start-visionguide.ps1
```

It stops anything left over, starts the backend, the Cloudflare tunnel and Expo (each in its own window),
writes the tunnel's new address into `mobile/.env`, keeps the access key in sync, and prints what to do on the
phone. Takes ~1.5 minutes. Works on UCL Wi-Fi, eduroam or mobile data.

| Option | What it does |
|---|---|
| *(none)* | Tunnel mode: phone on any network |
| `-Local` | Hotspot mode: phone and laptop on the same network, no tunnels |
| `-Stop` | Stops the backend, Expo and tunnels |

Then on the phone: **Settings → Apps → Expo Go → Force stop**, open Expo Go, scan the QR code in the
"VisionGuide Expo" window, and check the version at the end of the bottom status line.

---

## 1. One-time setup on the laptop

### 1a. Allow the backend through Windows Firewall

Windows blocks the backend's Python on public networks (hotspots count as public). Open **PowerShell as
Administrator** and run both:

```powershell
Get-NetFirewallApplicationFilter -Program "C:\Users\sumes\AppData\Local\Programs\Python\Python311\python.exe" | Get-NetFirewallRule | Disable-NetFirewallRule
New-NetFirewallRule -DisplayName "VisionGuide backend 8001" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8001 -Program "C:\Users\sumes\AppData\Local\Programs\Python\Python311\python.exe"
```

To undo after the hackathon: `Remove-NetFirewallRule -DisplayName "VisionGuide backend 8001"`, then the first
command again with `Enable-NetFirewallRule` at the end.

### 1b. Backend keys (`backend/.env`, never commit it)

| Setting | Value |
|---|---|
| `GEMINI_API_KEY` | your team's `hk_…` token |
| `GEMINI_BASE_URL` | `https://ai4a11y.lovable.app/api/public/gemini` |
| `GOOGLE_MAPS_API_KEY` | Maps key (Routes API works; enable **Places API (New)** and **Geocoding API** in Google Cloud for better place search) |
| `OPENAI_API_KEY` | optional; without it, speech is transcribed by Gemini |

---

## 2. Network

**UCLGuest blocks devices from reaching each other**, so use a phone hotspot:

1. Turn on a hotspot (on your Android phone, or another phone).
2. Connect the **laptop** to it.
3. If the hotspot is on a different phone, connect your Android phone to it too.

You don't need to edit any address: the app finds the backend on the same laptop that served it.
If the laptop changes networks, **restart Expo** (step 3) and scan the new QR code.

---

### 2b. On UCL Wi-Fi (or any network): tunnels instead of a hotspot

Campus Wi-Fi blocks the phone from reaching the laptop, but the laptop's internet there is much better than a
hotspot's. Tunnels let the phone reach the laptop through the internet, from any network (UCL Wi-Fi, mobile data):

1. `backend/.env` has `ACCESS_KEY=…` and `mobile/.env` has the same value as `EXPO_PUBLIC_ACCESS_KEY=…`.
   With a public URL, only the app (which sends the key) can use the backend.
2. Start the backend (step 3, terminal 1), then the backend tunnel in a third terminal:
   ```powershell
   cd "C:\Users\sumes\projects_sumeshchodankar\UCL London Hackathon 2\visionguide\visionguide"
   & "..\tools\cloudflared\cloudflared.exe" tunnel --no-autoupdate --url http://localhost:8001
   ```
   Copy the `https://….trycloudflare.com` address it prints into `mobile/.env` as `EXPO_PUBLIC_API_URL=…`.
   The address is new every time cloudflared restarts.
3. Start the app with Expo's tunnel (instead of terminal 2 below):
   ```powershell
   cd "C:\Users\sumes\projects_sumeshchodankar\UCL London Hackathon 2\visionguide\visionguide\mobile"
   npx expo start --tunnel ngrok --clear
   ```
   Scan the `exp://….exp.direct` QR code in Expo Go. (Plain `--tunnel` needs an Expo account login.)
4. Check: `https://….trycloudflare.com/health` opens in the phone's browser.

To go back to hotspot / local mode, remove `EXPO_PUBLIC_API_URL` from `mobile/.env` (keep the access key).

## 3. Start the servers (two terminals on the laptop)

```powershell
# Terminal 1: backend
cd "C:\Users\sumes\projects_sumeshchodankar\UCL London Hackathon 2\visionguide\visionguide\backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

```powershell
# Terminal 2: the app for Expo Go
cd "C:\Users\sumes\projects_sumeshchodankar\UCL London Hackathon 2\visionguide\visionguide\mobile"
npx expo start --clear
```

Ready when terminal 1 shows `EasyOCR loaded` and terminal 2 shows a QR code with `exp://<laptop-ip>:8081`.

Check from the laptop: open `http://<laptop-ip>:8001/health`. It should list `"buildings": ["UCL East Marshgate"]`.
To check from the phone, open the same address in Chrome on the phone. If it doesn't load, the firewall (1a) or
the network (2) is the problem.

---

## 4. On the phone

1. Install / update **Expo Go** from the Play Store (project uses Expo SDK 58).
2. Open Expo Go → **Scan QR code** → scan the QR in terminal 2.
3. Allow **Camera**, **Location (precise)**, **Microphone** and **Physical activity** (step counter).
4. You'll hear *"VisionGuide is ready…"*. If you hear *"I can't reach the VisionGuide server"*, see Troubleshooting.

---

## 5. Using it

| You do | What happens |
|---|---|
| **Tap the big yellow button**, speak, tap again | Your words go to the agent (Gemini transcribes). Answer is spoken word for word, then **repeated every 5 s until your next instruction** (paused while you walk). |
| **What's ahead / Where am I / Read sign** | Camera + agent answer. Measured objects first: *"Person at 12 o'clock, 2.6 meters."* |
| **Stop** | Stops talking and navigation. |
| **Live view** | Camera runs continuously: YOLO every ~1 s (hazards, *"…getting closer"*), Gemini + OCR every ~5 s (signs, rooms, scene). Turns on automatically while navigating. |
| Walk a route | Phone checks your compass against the route twice a second: *"Turn left. Your path is at 10 o'clock."*, *"Wrong way. Turn around."*, *"On track."* Vibrates while you're off course. |
| Take a lift / stairs | Barometer: *"Up one floor."* and *"You're now on floor 2. That's the right floor for the Library."* |

The line at the bottom of the camera view shows the sensors: `🧭 heading · 🚶 steps · 💡 light · ⛰ baro · course`.
⚡ is the fast-camera latency, 👁 the full-look latency.

**First time outdoors:** if asked, wave the phone in a figure eight to calibrate the compass.

---

## 6. Marshgate floor plans

Within ~120 m of UCL East Marshgate you'll hear *"You're at UCL East Marshgate. I have its floor plans for all 9
levels…"*. Then try:

- *"Where is the library?"* → Level 02 and Level 03, with where it is on each level
- *"What's on level 4?"*
- *"Is the café close to the reception?"*
- *"Take me to the lecture theatre"* → indoor guidance to Level 01 (lift advice if you're on another floor)
- From outside: *"Take me to the library in Marshgate"* → walking route to the building, then indoors

The plans show **named spaces, not room numbers** (no "443B"). Directions inside use landmarks from the plans
(atrium, stair/lift cores, neighbouring spaces); the camera and signs confirm the way.

Plan images can be viewed at `http://<laptop-ip>:8001/buildings/marshgate/images/level-02.jpg`;
what the agent knows is at `http://<laptop-ip>:8001/v1/buildings`.

---

## 7. Adding buildings or photos

```
backend/buildings/<id>/
  building.json        name, aliases, location {lat,lng}, radius_m, floors[{level, name, plan, key{number: name}}]
  images/level-00.jpg  one plan per level (photos of entrances/corridors can go here too)
  notes.json           generated automatically, don't edit
```

Copy `backend/buildings/marshgate/building.json` as a template. Type each plan's key **exactly** as printed.
Restart the backend: new or changed plans are read by Gemini once, in the background (~5-10 s per plan), and
saved to `notes.json`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| *"I can't reach the VisionGuide server"* | Phone and laptop on the same network? Firewall rule (1a) added? Backend running? Open `http://<laptop-ip>:8001/health` in the phone's Chrome. |
| QR code won't load the app | Phone on a different network, or the laptop changed networks: restart Expo and rescan. |
| No turn corrections while walking | Needs an active outdoor route (*"Take me to…"*) and a GPS fix. Corrections wait 2 s (4 s standing). Calibrate the compass. |
| Compass line shows `🧭 –` | Location permission denied, or the phone has no magnetometer. |
| "Thinking…" takes 20-40 s | Gemini through the hackathon proxy is slow for multi-step questions; simple questions take ~6-12 s. |
| No floor changes announced | Phone has no barometer (the status line has no `⛰`). |

## Known limits

- Distances from the camera are estimates (±~30%); YOLO measures people, vehicles and furniture, Gemini estimates
  curbs/stairs/doors (less consistent).
- GPS is poor indoors; inside, guidance relies on the camera, signs, floor plans and the barometer.
- The "Alex" wake word only works in the web version; Expo Go uses the tap button.
