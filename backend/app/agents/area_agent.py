"""Area Agent: keeps a map of the user's surroundings ready before anyone asks.

Runs in the background whenever a GPS fix arrives and the cached map is
missing, stale (older than area_max_age_s) or far away (user moved more than
area_refresh_m). It gathers, in parallel:

  * the street address                 (Google Geocoding, else OpenStreetMap)
  * named places and street furniture  (Google Places nearby + OpenStreetMap:
                                        stations, stops, crossings, entrances, shops...)
  * a briefing grounded on Google Maps (Gemini + google_maps tool)

and, once a destination is set, entrance / accessibility notes for it.

The brain gets a compact, distance-sorted view of this map in every turn's
status (no extra tool round trip), plus nearby_places / ask_maps tools for
anything more specific. Nothing here ever blocks a request.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import TYPE_CHECKING

from ..config import Settings
from ..schemas import AreaContext, LatLng, NearbyPlace, Priority, SessionState, Utterance
from ..utils.geo import bearing_deg, clock_from_bearing, haversine_m

if TYPE_CHECKING:
    from ..db import Database, SessionStore
    from ..services.gemini import GeminiService
    from ..services.maps import MapsService

log = logging.getLogger("visionguide.area")

BRIEFING_SYSTEM = """You prepare a factual map briefing for a navigation assistant that guides a blind pedestrian.
Use Google Maps. Plain text, no markdown, no URLs, at most 140 words. Only state facts you found; never guess."""

BRIEFING_PROMPT = """The user is standing at latitude {lat:.6f}, longitude {lng:.6f}{addr}.
Describe ONLY the area within about {radius} metres of that exact point. Do not describe a different neighbourhood
or a nearby landmark area instead. Measure every distance from that point. Cover:
1. The street they are on and the nearest named building or landmark.
2. Nearest Underground/train stations and bus stops, with approximate walking distance and direction (north/south/east/west).
3. Nearest pedestrian crossings, especially signal-controlled ones.
4. Useful places close by: pharmacy, toilets, cafe, supermarket, bank or cash machine, hospital.
5. Anything a blind pedestrian should know: busy roads, construction, steps, step-free entrances."""

DEST_PROMPT = """A blind visitor is walking to {name}{addr}. Using Google Maps, in at most 70 words of plain text:
which street the main entrance is on, whether there is a step-free or accessible entrance and where it is,
and one landmark that helps confirm they are at the right door. Only state facts you found."""

AREA_RETRY_S = 60.0   # retry an incomplete map (a source failed or timed out) this soon

STOPWORDS ={"the", "a", "an", "nearest", "closest", "near", "nearby", "me", "is", "where", "any", "some", "to", "of", "for"}


def _plain(text: str | None) -> str | None:
    """Strip markdown the model adds despite instructions (it ends up in speech otherwise)."""
    if not text:
        return text
    text = re.sub(r"[*_`#]+", "", text)
    text = re.sub(r"^\s*(?:[-•]|\d+[.)])\s*", "", text, flags=re.M)
    return re.sub(r"\s*\n+\s*", " ", text).strip() or None


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


class AreaAgent:
    name = "area"

    def __init__(self, settings: Settings, maps: "MapsService", gemini: "GeminiService",
                 store: "SessionStore", db: "Database"):
        self.s, self.maps, self.gemini, self.store, self.db = settings, maps, gemini, store, db
        self._tasks: dict[str, asyncio.Task] = {}
        self._dest_tasks: dict[str, asyncio.Task] = {}

    # ------------------------------------------------------------------ triggers
    def needs_refresh(self, session: SessionState) -> bool:
        if not self.s.area_prefetch or not session.position:
            return False
        a = session.area
        if a is None:
            return True
        moved = haversine_m(a.center.lat, a.center.lng, session.position.lat, session.position.lng)
        max_age = self.s.area_max_age_s if a.complete else AREA_RETRY_S
        return moved > self.s.area_refresh_m or time.time() - a.fetched_at > max_age

    def maybe_refresh(self, session: SessionState) -> None:
        """Called on every GPS fix. Starts a background fetch if needed; returns at once."""
        if self.needs_refresh(session):
            t = self._tasks.get(session.id)
            if not t or t.done():
                self._tasks[session.id] = asyncio.create_task(self._build(session, session.position))
        self.maybe_destination(session)

    def maybe_destination(self, session: SessionState) -> None:
        d = session.destination
        if not self.s.area_prefetch or not d or not d.place_location or not d.place_name:
            return
        key = f"{d.place_name}|{d.place_location.lat:.5f}"
        if session.area and session.area.destination_key == key:
            return
        t = self._dest_tasks.get(session.id)
        if not t or t.done():
            self._dest_tasks[session.id] = asyncio.create_task(self._build_destination(session, key))

    async def wait(self, session: SessionState, timeout: float) -> None:
        """Let a turn wait briefly for an in-flight fetch (used when the map is still empty)."""
        t = self._tasks.get(session.id)
        if t and not t.done():
            try:
                await asyncio.wait_for(asyncio.shield(t), timeout)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ fetching
    async def _build(self, session: SessionState, center: LatLng) -> None:
        """Publishes in stages so the brain can use each part as soon as it lands:
        address (~1 s) → Google Maps briefing (~10-30 s) / nearby places (~5-40 s), whichever first."""
        t0 = time.perf_counter()
        announced = session.area is not None and bool(session.area.places or session.area.briefing)
        prev = session.area
        same_spot = prev is not None and haversine_m(prev.center.lat, prev.center.lng, center.lat, center.lng) < 50
        try:
            # Places (slowest) start at once; the address comes first because it anchors the
            # Google Maps briefing to the right spot (coordinates alone are only a hint to Gemini).
            nearby_task = asyncio.create_task(self._nearby(center))
            try:
                address = await self._address(center)
            except Exception:  # noqa: BLE001
                address = None
            area = AreaContext(
                center=center, fetched_at=time.time(), address=address,
                # Keep what we had for this spot until the new data lands (e.g. on a retry).
                places=prev.places if same_spot else [], briefing=prev.briefing if same_spot else None,
                briefing_sources=prev.briefing_sources if same_spot else [],
                sources=[s for s in (prev.sources if same_spot else []) if s != "gemini_google_maps"],
                destination_brief=prev.destination_brief if prev else None,
                destination_key=prev.destination_key if prev else None)
            session.area = area
            if address:
                session.last_address = address
            briefing_task = asyncio.create_task(self._briefing(center, address))
            got_places = got_briefing = False

            for fut in asyncio.as_completed([nearby_task, briefing_task]):
                try:
                    res = await fut
                except Exception as e:  # noqa: BLE001
                    log.warning("Area part failed: %s", e)
                    continue
                if isinstance(res[0], list):  # (places, sources)
                    places, srcs = res
                    if places:
                        got_places = True
                        area.places = places
                        area.sources = sorted(set(srcs) | ({"gemini_google_maps"} if area.briefing else set()))
                else:                          # (briefing text, cited places)
                    text, cited = _plain(res[0]), res[1]
                    if text:
                        got_briefing = True
                        area.briefing, area.briefing_sources = text, cited
                        area.sources = sorted(set(area.sources) | {"gemini_google_maps"})
                if not announced and (area.places or area.briefing):
                    announced = True
                    where = f" You're near {address}." if address else ""
                    session.pending_utterances.append(Utterance(
                        text=f"Map of this area is ready.{where}", priority=Priority.INFO, category="info",
                        key="area:ready"))
                self.store.save(session)

            # Missing parts (slow public servers, timeouts) are retried soon instead of in 15 minutes.
            # Complete = we hold both parts (fresh, or kept from an earlier fetch of this spot).
            area.complete = bool(area.places) and bool(area.briefing)
            if not (got_places and got_briefing) and area.complete:
                log.info("Area map for %s: a source failed this time; kept the earlier data for this spot", session.id)
            ms = round((time.perf_counter() - t0) * 1000)
            log.info("Area map for %s: %d places, briefing=%s, sources=%s (%d ms)%s",
                     session.id, len(area.places), bool(area.briefing), area.sources, ms,
                     "" if area.complete else " - incomplete, will retry")
            self.db.log(session.id, "area", {"places": len(area.places), "sources": area.sources, "ms": ms})
            self.store.save(session)
        except Exception as e:  # noqa: BLE001 - a background job must never crash the app
            log.warning("Area map build failed: %s", e)

    async def _address(self, center: LatLng) -> str | None:
        fn = getattr(self.maps, "reverse_geocode", None)
        return await fn(center) if fn else None

    async def _nearby(self, center: LatLng) -> tuple[list[NearbyPlace], list[str]]:
        fn = getattr(self.maps, "nearby", None)
        return await fn(center, self.s.area_radius_m) if fn else ([], [])

    async def _briefing(self, center: LatLng, address: str | None = None) -> tuple[str | None, list[str]]:
        fn = getattr(self.gemini, "maps_grounded", None)
        if not fn or not getattr(self.gemini, "enabled", False):
            return None, []
        addr = f", at {address}" if address else ""
        return await fn(BRIEFING_PROMPT.format(lat=center.lat, lng=center.lng, addr=addr, radius=self.s.area_radius_m),
                        center.lat, center.lng, system=BRIEFING_SYSTEM, timeout=45.0)

    async def _build_destination(self, session: SessionState, key: str) -> None:
        d = session.destination
        fn = getattr(self.gemini, "maps_grounded", None)
        if not d or not d.place_location or not fn or not getattr(self.gemini, "enabled", False):
            return
        addr = f" ({d.place_address})" if d.place_address else ""
        text, _ = await fn(DEST_PROMPT.format(name=d.place_name, addr=addr), d.place_location.lat,
                           d.place_location.lng, system=BRIEFING_SYSTEM, timeout=40.0)
        text = _plain(text)
        if text and session.area:
            session.area.destination_brief, session.area.destination_key = text, key
            self.store.save(session)
        elif text:  # no area yet: keep the note on a minimal area record
            session.area = AreaContext(center=d.place_location, fetched_at=0.0, destination_brief=text,
                                       destination_key=key)
            self.store.save(session)

    # ------------------------------------------------------------------ reading
    def _placed(self, session: SessionState, p: NearbyPlace) -> dict:
        out = {"name": p.name, "kind": p.kind}
        if session.position:
            pos = session.position
            out["distance_m"] = round(haversine_m(pos.lat, pos.lng, p.lat, p.lng))
            b = bearing_deg(pos.lat, pos.lng, p.lat, p.lng)
            clock = clock_from_bearing(b, session.heading)
            if clock:
                out["clock"] = clock
            out["compass"] = ["north", "north-east", "east", "south-east", "south", "south-west", "west",
                              "north-west"][round(b / 45) % 8]
        if p.detail:
            out["detail"] = p.detail
        return out

    def summary(self, session: SessionState, limit: int = 12) -> dict | None:
        """Compact map for the brain's per-turn status: nearest places, briefing, address."""
        a = session.area
        if a is None:
            pending = bool(self._tasks.get(session.id) and not self._tasks[session.id].done())
            return {"status": "loading" if pending else ("no_gps" if not session.position else "empty")}
        placed = sorted((self._placed(session, p) for p in a.places), key=lambda x: x.get("distance_m", 1e9))
        # Keep the list varied: at most 3 crossings so they don't crowd out stations and shops.
        picked, crossings = [], 0
        for p in placed:
            if p["kind"] == "pedestrian crossing":
                crossings += 1
                if crossings > 3:
                    continue
            picked.append(p)
            if len(picked) >= limit:
                break
        task = self._tasks.get(session.id)
        loading = bool(task and not task.done())
        return {
            # "partial": some parts still loading or failed - absence of a place is not proof it isn't there.
            "status": "ready" if a.complete else ("loading" if loading and not (a.places or a.briefing) else "partial"),
            "still_loading": loading,
            "age_s": round(time.time() - a.fetched_at) if a.fetched_at else None,
            "address": a.address,
            "google_maps_briefing": a.briefing,
            "destination_notes": a.destination_brief,
            "nearest_places": picked,
            "total_places_cached": len(a.places),
            "sources": a.sources,
        }

    def lookup(self, session: SessionState, query: str | None = None, kind: str | None = None,
               limit: int = 8) -> list[dict]:
        a = session.area
        if a is None:
            return []
        q, k = _words(query or ""), (kind or "").lower().strip()
        hits = []
        for p in a.places:
            hay = _words(f"{p.name} {p.kind} {p.detail or ''}")
            if k and k not in p.kind.lower() and not (_words(k) & hay):
                continue
            score = len(q & hay)
            if q and not score:
                continue
            hits.append((score, self._placed(session, p)))
        # Best word match first ("underground station" beats "charging station"), then nearest.
        hits.sort(key=lambda h: (-h[0], h[1].get("distance_m", 1e9)))
        return [h for _, h in hits[:limit]]
