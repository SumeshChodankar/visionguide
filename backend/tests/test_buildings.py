"""Our own building data: backend/buildings/marshgate (real floor-plan keys) + one-time plan notes."""
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from app.config import Settings
from app.schemas import LatLng
from app.services.buildings import BuildingService

MARSHGATE = (51.537542, -0.011982)
BUILDINGS = str(Path(__file__).resolve().parents[1] / "buildings")


def svc() -> BuildingService:
    return BuildingService(Settings(_env_file=None, buildings_dir=BUILDINGS))


def names(hits):
    return [(h["name"], h["level"]) for h in hits]


def test_marshgate_loads_all_levels_with_exact_keys():
    b = svc().buildings["marshgate"]
    assert b.name == "UCL East Marshgate" and [f.level for f in b.floors] == list(range(9))
    assert b.floor(0).key["1"] == "UCL Reception" and b.floor(2).key["3"] == "Library"
    assert all(f.plan.exists() for f in b.floors)


def test_find_space_names_synonyms_and_multi_level():
    s = svc()
    assert names(s.find_space("the library")) == [("Library", 2), ("Library", 3)]
    assert names(s.find_space("where can I get a coffee"))[0] == ("Café", 0)
    assert names(s.find_space("lecture hall")) == [("Lecture Theatre", 1)]
    assert names(s.find_space("reception")) == [("UCL Reception", 0)]
    assert names(s.find_space("Institute of Making")) == [("Institute of Making", 0), ("Institute of Making", 1)]
    assert [lv for _, lv in names(s.find_space("specialist computer lab"))] == [1, 2, 6]
    assert s.find_space("Room 443B") == [] and s.find_space("stratford station") == []


def test_building_by_name_near_and_strip():
    s = svc()
    b = s.by_name("Take me to the library in Marshgate")
    assert b and b.id == "marshgate"
    assert s.strip_building("the library in UCL East Marshgate", b) == "the library"
    assert s.near(LatLng(lat=MARSHGATE[0] + 0.0005, lng=MARSHGATE[1])) is b        # ~55 m
    assert s.near(LatLng(lat=MARSHGATE[0] + 0.004, lng=MARSHGATE[1])) is None      # ~450 m
    assert s.by_name("Bush House") is None


def test_summary_has_every_level_and_detail_for_requested_ones():
    s = svc()
    out = s.summary(s.buildings["marshgate"], levels=[2])
    assert out["levels"]["Ground Level"][0] == "UCL Reception" and len(out["levels"]) == 9
    assert ("level_details" not in out) or ("Level 02" in out["level_details"])


def test_plan_notes_generated_once_and_redone_when_image_changes(tmp_path):
    root = tmp_path / "buildings"
    shutil.copytree(Path(BUILDINGS) / "marshgate", root / "test", ignore=shutil.ignore_patterns("notes.json"))
    cfg = json.loads((root / "test" / "building.json").read_text(encoding="utf-8"))
    cfg["id"], cfg["floors"] = "test", cfg["floors"][:2]
    (root / "test" / "building.json").write_text(json.dumps(cfg), encoding="utf-8")

    class G:
        enabled, calls = True, 0

        async def image_json(self, image, prompt, system, timeout=60.0):
            G.calls += 1
            return {"layout": "L", "spaces": {"1": "by the atrium"}, "circulation": "", "notes": ""}

    s = BuildingService(Settings(_env_file=None, buildings_dir=str(root)), G())
    assert asyncio.run(s.prepare_notes()) == 2 and G.calls == 2
    assert s.find_space("reception")[0]["where"] == "by the atrium"
    again = BuildingService(Settings(_env_file=None, buildings_dir=str(root)), G())
    assert again.missing_notes() == [] and asyncio.run(again.prepare_notes()) == 0 and G.calls == 2
    (root / "test" / "images" / "level-01.jpg").write_bytes(b"changed")       # a plan was replaced
    assert [f.level for _, f in BuildingService(Settings(_env_file=None, buildings_dir=str(root)), G()).missing_notes()] == [1]


# ---------------------------------------------------------------- through the API (rules mode, fakes)
def _client_with_buildings(client):
    from app import main
    o = main.app.state.orchestrator
    o.buildings = svc()
    return o


def test_arriving_at_marshgate_and_going_to_the_library(client):
    o = _client_with_buildings(client)
    sid = client.post("/v1/sessions").json()["session_id"]
    r = client.post(f"/v1/sessions/{sid}/position", json={"lat": MARSHGATE[0], "lng": MARSHGATE[1]}).json()
    assert any("You're at UCL East Marshgate. I have its floor plans for all 9 levels" in u["text"] for u in r["utterances"])
    o.store.get(sid).current_floor = 0
    r = client.post(f"/v1/sessions/{sid}/command", json={"text": "Take me to the library"}).json()
    texts = " | ".join(u["text"] for u in r["utterances"])
    assert r["phase"] == "indoor"
    assert r["destination"]["sign_text"] == "Library" and r["destination"]["floor_hint"] == 2
    assert "The Library is on Level 02 and Level 03." in texts and "find the elevator" in texts
    # barometer: up two floors -> right floor
    r = client.post(f"/v1/sessions/{sid}/position",
                    json={"lat": MARSHGATE[0], "lng": MARSHGATE[1], "floor_change": 2}).json()
    assert any("right floor for" in u["text"] for u in r["utterances"])


def test_named_building_from_outside_routes_there_without_place_search(client, fakes):
    _client_with_buildings(client)
    called = []
    orig = fakes["maps"].search_place

    async def spy(q, near):
        called.append(q)
        return await orig(q, near)

    fakes["maps"].search_place = spy
    sid = client.post("/v1/sessions").json()["session_id"]
    client.post(f"/v1/sessions/{sid}/position", json={"lat": 51.5000, "lng": -0.1200})
    r = client.post(f"/v1/sessions/{sid}/command", json={"text": "Take me to the library in Marshgate"}).json()
    assert r["phase"] == "outdoor" and called == []
    assert r["destination"]["place_name"] == "UCL East Marshgate" and r["destination"]["sign_text"] == "Library"


def test_other_places_are_not_hijacked_while_at_marshgate(client):
    _client_with_buildings(client)
    sid = client.post("/v1/sessions").json()["session_id"]
    client.post(f"/v1/sessions/{sid}/position", json={"lat": MARSHGATE[0], "lng": MARSHGATE[1]})
    r = client.post(f"/v1/sessions/{sid}/command", json={"text": "Take me to Stratford station"}).json()
    assert r["destination"]["sign_text"] is None and r["destination"]["place_name"] != "UCL East Marshgate"
