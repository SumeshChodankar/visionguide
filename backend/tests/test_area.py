"""Area Agent: background map prefetch and the brain tools that read it (offline fakes)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.agents.area_agent import AreaAgent
from app.config import Settings
from app.db import Database, SessionStore
from app.schemas import LatLng, NearbyPlace

HERE = LatLng(lat=51.5129, lng=-0.1171)
# ~110 m north, ~70 m east, ~400 m south of HERE
NORTH = (51.5139, -0.1171)
EAST = (51.5129, -0.1161)
FAR_SOUTH = (51.5093, -0.1171)


class AreaMaps:
    enabled = True

    def __init__(self):
        self.nearby_calls = 0

    async def reverse_geocode(self, pos):
        return "Bush House, 30 Aldwych, London"

    async def nearby(self, pos, radius):
        self.nearby_calls += 1
        return [
            NearbyPlace(name="Temple", kind="underground entrance", lat=FAR_SOUTH[0], lng=FAR_SOUTH[1], source="osm"),
            NearbyPlace(name="Boots", kind="pharmacy", lat=EAST[0], lng=EAST[1], source="google_places"),
            NearbyPlace(name="Aldwych crossing", kind="pedestrian crossing", lat=NORTH[0], lng=NORTH[1],
                        source="osm", detail="traffic signals, tactile paving"),
        ], ["google_places", "osm"]


class AreaGemini:
    enabled = True

    async def maps_grounded(self, prompt, lat, lng, *, timeout=30.0, system=None):
        return "You are on Aldwych outside Bush House. Temple station is about 400 m south.", ["Temple", "Bush House"]


def make(tmp_path, **kw):
    settings = Settings(_env_file=None, **kw)
    db = Database(str(tmp_path / "a.db"))
    store = SessionStore(db)
    maps, gem = AreaMaps(), AreaGemini()
    return AreaAgent(settings, maps, gem, store, db), store, maps


def test_prefetch_builds_map_and_announces_once(tmp_path):
    agent, store, maps = make(tmp_path)
    s = store.create()
    s.position, s.heading = HERE, 0.0  # facing north

    async def run():
        agent.maybe_refresh(s)
        await agent.wait(s, 5)

    asyncio.run(run())
    assert s.area and s.area.address.startswith("Bush House")
    assert "gemini_google_maps" in s.area.sources and "google_places" in s.area.sources
    assert [u.text for u in s.pending_utterances] == ["Map of this area is ready. You're near Bush House, 30 Aldwych, London."]

    summary = agent.summary(s)
    assert summary["status"] == "ready"
    assert summary["google_maps_briefing"].startswith("You are on Aldwych")
    names = [p["name"] for p in summary["nearest_places"]]
    assert names == ["Boots", "Aldwych crossing", "Temple"]           # sorted by distance
    boots, crossing = summary["nearest_places"][0], summary["nearest_places"][1]
    assert boots["clock"] == 3 and boots["compass"] == "east"          # east while facing north
    assert crossing["clock"] == 12 and 100 <= crossing["distance_m"] <= 120

    # A second fix nearby doesn't refetch or re-announce.
    s.pending_utterances = []
    asyncio.run(run())
    assert maps.nearby_calls == 1 and not s.pending_utterances


def test_refreshes_after_moving_far(tmp_path):
    agent, store, maps = make(tmp_path, area_refresh_m=150)
    s = store.create()
    s.position = HERE

    async def go(pos):
        s.position = pos
        agent.maybe_refresh(s)
        await agent.wait(s, 5)

    asyncio.run(go(HERE))
    asyncio.run(go(LatLng(lat=51.5129, lng=-0.1150)))   # ~145 m: still fresh
    assert maps.nearby_calls == 1
    asyncio.run(go(LatLng(lat=51.5129, lng=-0.1140)))   # ~215 m: refetch
    assert maps.nearby_calls == 2


def test_lookup_matches_name_kind_and_detail(tmp_path):
    agent, store, _ = make(tmp_path)
    s = store.create()
    s.position = HERE

    async def run():
        agent.maybe_refresh(s)
        await agent.wait(s, 5)

    asyncio.run(run())
    assert [p["name"] for p in agent.lookup(s, query="the nearest pharmacy")] == ["Boots"]
    assert [p["name"] for p in agent.lookup(s, query="tactile paving")] == ["Aldwych crossing"]
    assert [p["name"] for p in agent.lookup(s, query="Temple")] == ["Temple"]
    assert agent.lookup(s, query="hospital") == []


def test_summary_before_gps_and_while_loading(tmp_path):
    agent, store, _ = make(tmp_path)
    s = store.create()
    assert agent.summary(s) == {"status": "no_gps"}
    disabled, store2, _ = make(tmp_path, area_prefetch=False)
    s2 = store2.create()
    s2.position = HERE
    disabled.maybe_refresh(s2)
    assert s2.area is None


def test_failed_source_gives_partial_map_and_quick_retry(tmp_path):
    import app.agents.area_agent as area_mod

    agent, store, maps = make(tmp_path)

    async def broken_nearby(pos, radius):
        raise TimeoutError("overpass slow")

    maps.nearby = broken_nearby
    s = store.create()
    s.position = HERE

    async def run():
        agent.maybe_refresh(s)
        await agent.wait(s, 5)

    asyncio.run(run())
    assert s.area.briefing and not s.area.places and not s.area.complete
    assert agent.summary(s)["status"] == "partial"
    assert len(s.pending_utterances) == 1                      # briefing alone is enough to announce
    assert not agent.needs_refresh(s)                          # not immediately...
    s.area.fetched_at -= area_mod.AREA_RETRY_S + 1
    assert agent.needs_refresh(s)                              # ...but within a minute, not 15


def test_brain_nearby_places_tool(tmp_path):
    from app.agents.brain import Brain, Turn

    agent, store, _ = make(tmp_path)
    s = store.create()
    s.position, s.heading = HERE, 0.0
    orch = SimpleNamespace(s=Settings(_env_file=None), area=agent,
                           vision=SimpleNamespace(gemini=AreaGemini()))
    brain = Brain(orch)

    async def run():
        turn = Turn(orch, s, None)
        agent.maybe_refresh(s)  # tool waits for the in-flight fetch
        return await brain.t_nearby(turn, {"query": "pharmacy"}), await brain.t_ask_maps(turn, {"question": "Nearest station?"})

    near, maps_answer = asyncio.run(run())
    assert near["matches"][0]["name"] == "Boots" and near["searched_places"] == 3
    assert maps_answer["answer"].startswith("You are on Aldwych") and "Temple" in maps_answer["google_maps_places"]
