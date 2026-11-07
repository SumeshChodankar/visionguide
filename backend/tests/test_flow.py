"""End-to-end: the judge demo, step by step, through the HTTP API."""
from app.schemas import Detection
from conftest import A, B, C, D, GHazard, GLandmark, scene


def texts(resp) -> str:
    return " | ".join(u["text"] for u in resp.json()["utterances"])


def frame(client, sid, jpeg, mode="auto"):
    r = client.post(f"/v1/sessions/{sid}/frame", files={"image": ("f.jpg", jpeg, "image/jpeg")}, data={"mode": mode})
    assert r.status_code == 200, r.text
    return r


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_full_journey_outdoor_to_room(client, fakes, jpeg):
    sid = client.post("/v1/sessions").json()["session_id"]
    pos = lambda lat, lng, h=0: client.post(f"/v1/sessions/{sid}/position", json={"lat": lat, "lng": lng, "heading": h, "accuracy_m": 5})
    pos(*A)

    # 1. Voice command → Whisper (fake) → route
    r = client.post(f"/v1/sessions/{sid}/voice", files={"audio": ("a.m4a", b"fake", "audio/m4a")})
    body = r.json()
    assert body["phase"] == "outdoor"
    assert "Starting route to Bush House" in texts(r)
    assert "floor 4" in texts(r)
    assert body["destination"]["room"] == "443B"

    # 2. Walk: pre-announce, then advance step
    assert "turn right onto Aldwych" in texts(pos(51.50075, -0.1200))
    r = pos(51.50095, -0.1200)
    assert "Now, turn right onto Aldwych" in texts(r) and r.json()["step_index"] == 1

    # 3. Off-route → reroute
    r = pos(51.5030, -0.1250)
    assert "off the route" in texts(r)

    # 4. Arrive near building → APPROACH
    r = pos(51.50115, -0.11852)
    assert r.json()["phase"] == "approach" and "find the entrance" in texts(r)

    # 5. Camera: building sign + entrance → verified
    fakes["gemini"].next_scene = scene(environment="outdoor", sign_texts=["BUSH HOUSE"],
                                       landmarks=[GLandmark(type="entrance", side="right", distance_m=6, description="glass doors")])
    r = frame(client, sid, jpeg)
    assert "right building" in texts(r) and "1 o'clock" in texts(r)

    # 6. Step inside → INDOOR. Elevator display shows floor 2.
    fakes["gemini"].next_scene = scene(floor_indicator="2", landmarks=[GLandmark(type="elevator", side="ahead", distance_m=3)])
    r = frame(client, sid, jpeg)
    t = texts(r)
    assert r.json()["phase"] == "indoor"
    assert "floor 2" in t and "go up" in t

    fakes["gemini"].next_scene = scene(landmarks=[GLandmark(type="elevator", side="ahead", distance_m=3)])
    assert "The elevator is at 12 o'clock, 3 meters." in texts(frame(client, sid, jpeg))

    # 7. Exit elevator on floor 4
    fakes["gemini"].next_scene = scene(sign_texts=["Level 4"])
    assert "right floor for Room 443B" in texts(frame(client, sid, jpeg))

    # 8. Signage + neighbouring rooms
    fakes["gemini"].next_scene = scene(sign_texts=["Rooms 440-460 →", "437"])
    t = texts(frame(client, sid, jpeg))
    assert "rooms 440 to 460, to the right" in t

    # 9. Room plate read by OCR in two frames → candidate then confirmed
    fakes["gemini"].next_scene = scene()
    fakes["ocr_engine"].next = [([0.6, 0.3, 0.8, 0.4], "443B", 0.8)]
    r = frame(client, sid, jpeg)
    assert r.json()["arrival"]["state"] == "candidate" and "I think I see Room 443B" in texts(r)
    r = frame(client, sid, jpeg)
    assert r.json()["phase"] == "arrived" and "You have arrived at Room 443B" in texts(r)
    assert r.json()["utterances"][0]["haptic"] == "success"

    events = client.get(f"/v1/sessions/{sid}/events").json()
    assert any(e["kind"] == "command" for e in events)


def test_hazard_is_critical_and_not_repeated(client, fakes, jpeg):
    sid = client.post("/v1/sessions").json()["session_id"]
    fakes["detector"].next = [Detection(label="chair", confidence=0.9, bbox=[0.4, 0.4, 0.6, 0.9], clock=12,
                                        side="ahead", distance_m=1.6)]
    fakes["gemini"].next_scene = scene(hazards=[GHazard(type="wet floor sign / spill", severity="medium", side="left", distance_m=2)])
    r = frame(client, sid, jpeg)
    u = r.json()["utterances"]
    assert u[0]["text"] == "Stop! Chair at 12 o'clock, 1.6 meters." and u[0]["interrupt"] is True
    assert any(x["text"] == "Caution! Wet floor at 11 o'clock, 2.0 meters." for x in u)
    # same frame again immediately: governor suppresses repeats
    assert all(not x["text"].startswith("Stop! Chair") for x in frame(client, sid, jpeg).json()["utterances"])


def test_find_elevator_and_describe(client, fakes, jpeg):
    sid = client.post("/v1/sessions").json()["session_id"]
    r = client.post(f"/v1/sessions/{sid}/command", json={"text": "Find elevator"})
    assert r.json()["client_action"] == "capture_frame"
    fakes["gemini"].next_scene = scene(summary="Lobby with a reception desk.",
                                       landmarks=[GLandmark(type="elevator", side="right", distance_m=4)])
    assert "elevator is at 1 o'clock" in texts(frame(client, sid, jpeg))

    r = client.post(f"/v1/sessions/{sid}/command", json={"text": "What is in front of me?"})
    assert r.json()["client_action"] == "capture_frame"
    fakes["detector"].next = [Detection(label="person", confidence=0.9, bbox=[0.1, 0.2, 0.3, 0.9], clock=11, side="left", distance_m=2.0),
                              Detection(label="person", confidence=0.9, bbox=[0.4, 0.2, 0.5, 0.7], clock=12, side="ahead", distance_m=3.0)]
    t = texts(frame(client, sid, jpeg, mode="describe"))
    assert "Lobby with a reception desk." in t
    # measured YOLO facts are spoken verbatim, nearest first, before the AI summary
    assert "Person at 11 o'clock, 2.0 meters. Person at 12 o'clock, 3 meters." in t
    assert t.index("Person at 11 o'clock") < t.index("Lobby with a reception desk.")


def test_room_only_goes_indoor_and_wrong_room(client, fakes, jpeg):
    sid = client.post("/v1/sessions").json()["session_id"]
    r = client.post(f"/v1/sessions/{sid}/command", json={"text": "Take me to Room 205A"})
    assert r.json()["phase"] == "indoor" and "floor 2" in texts(r)
    fakes["gemini"].next_scene = scene()
    fakes["ocr_engine"].next = [([0.1, 0.3, 0.2, 0.4], "205B", 0.9)]
    t = texts(frame(client, sid, jpeg))
    assert "This is Room 205B, not 205A" in t


def test_chat_uses_gemini_and_falls_back(client, fakes):
    sid = client.post("/v1/sessions").json()["session_id"]
    client.post(f"/v1/sessions/{sid}/position", json={"lat": A[0], "lng": A[1]})
    fakes["gemini"].next_text = "You're on Kingsway. The path ahead is clear."
    r = client.post(f"/v1/sessions/{sid}/command", json={"text": "Is the path clear?"})
    assert r.json()["intent"]["name"] == "chat" and "path ahead is clear" in texts(r)
    assert "Kingsway, London" in fakes["gemini"].last_prompt  # world snapshot was passed
    fakes["gemini"].next_text = None
    r = client.post(f"/v1/sessions/{sid}/chat", json={"message": "Where exactly am I?"})
    assert "Kingsway" in texts(r)


def test_where_am_i(client):
    sid = client.post("/v1/sessions").json()["session_id"]
    client.post(f"/v1/sessions/{sid}/position", json={"lat": B[0], "lng": B[1]})
    r = client.post(f"/v1/sessions/{sid}/command", json={"text": "Where am I?"})
    assert "You're near Kingsway, London WC2B." in texts(r)


def test_bad_inputs(client):
    assert client.post("/v1/sessions/nope/command", json={"text": "hi"}).status_code == 404
    sid = client.post("/v1/sessions").json()["session_id"]
    r = client.post(f"/v1/sessions/{sid}/frame", files={"image": ("f.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400


def test_frame_with_gps_returns_turn_instruction(client, jpeg):
    sid = client.post("/v1/sessions").json()["session_id"]
    client.post(f"/v1/sessions/{sid}/position", json={"lat": A[0], "lng": A[1]})
    client.post(f"/v1/sessions/{sid}/command", json={"text": "Take me to Bush House"})
    r = client.post(f"/v1/sessions/{sid}/frame", files={"image": ("f.jpg", jpeg, "image/jpeg")},
                    data={"lat": "51.50095", "lng": "-0.1200"})
    assert "Now, turn right onto Aldwych" in texts(r)


def test_ask_with_frame_in_rules_mode_describes_directly(client, fakes, jpeg):
    sid = client.post("/v1/sessions").json()["session_id"]
    fakes["gemini"].next_scene = scene(summary="A quiet corridor with doors on both sides.")
    r = client.post(f"/v1/sessions/{sid}/ask", data={"text": "What is in front of me?"},
                    files={"image": ("f.jpg", jpeg, "image/jpeg")}).json()
    assert r["handled_by"] == "rules" and r["client_action"] is None
    assert "quiet corridor" in texts_json(r)


def texts_json(body) -> str:
    return " | ".join(u["text"] for u in body["utterances"])


def test_fast_frame_is_yolo_only_and_notices_objects_getting_closer(client, fakes, jpeg):
    sid = client.post("/v1/sessions").json()["session_id"]
    fakes["gemini"].next_scene = scene(summary="SHOULD NOT BE USED")

    def fast(dist):
        fakes["detector"].next = [Detection(label="person", confidence=0.9, bbox=[0.45, 0.2, 0.55, 0.9], clock=12,
                                            side="ahead", distance_m=dist)]
        r = client.post(f"/v1/sessions/{sid}/fast_frame", files={"image": ("f.jpg", jpeg, "image/jpeg")})
        assert r.status_code == 200
        return r.json()

    first = fast(3.6)
    assert first["utterances"] == [] and "gemini" not in first["timings_ms"] and first["scene"] is None
    second = fast(2.7)                       # 0.9 m closer within the window
    assert [u["text"] for u in second["utterances"]] == ["Person at 12 o'clock, 2.7 meters, getting closer."]
    assert second["detections"][0]["distance_m"] == 2.7
    third = fast(2.7)                        # not closer: silent
    assert third["utterances"] == []


def test_barometer_floor_change_updates_floor_and_confirms_target(client):
    from app import main

    sid = client.post("/v1/sessions").json()["session_id"]
    client.post(f"/v1/sessions/{sid}/command", json={"text": "Take me to Room 443B"})
    main.app.state.orchestrator.store.get(sid).current_floor = 2   # e.g. read from a floor sign

    def up():
        r = client.post(f"/v1/sessions/{sid}/position", json={"lat": 51.5, "lng": -0.12, "floor_change": 1})
        return " | ".join(u["text"] for u in r.json()["utterances"])

    first = up()
    assert "floor 3" in first and "443B is 1 floor up" in first
    second = up()
    assert "floor 4" in second and "right floor for Room 443B" in second
    assert main.app.state.orchestrator.store.get(sid).current_floor == 4
