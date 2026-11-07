"""Google Maps Platform: Places API (New) text + nearby search, Routes API
walking directions (with optional legacy Directions API), and reverse
geocoding.

Google is always tried first. If an API is not enabled on the key (403 /
REQUEST_DENIED), that API is skipped for a while and OpenStreetMap fills in:
Nominatim for place lookup and addresses, Overpass for nearby places. Routes
stay on Google's Routes API."""
from __future__ import annotations

import asyncio
import logging
import re
import time

import httpx

from ..config import Settings
from ..schemas import LatLng, NearbyPlace, RouteStep
from ..utils.geo import decode_polyline

log = logging.getLogger("visionguide.maps")

PLACES_URL = "https://places.googleapis.com/v1/places:searchText"
PLACES_NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"
NOMINATIM_URL = "https://nominatim.openstreetmap.org"
OVERPASS_URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter"]
OSM_HEADERS = {"User-Agent": "VisionGuide/1.0 (accessibility navigation prototype)"}
OSM_CACHE_S = 900.0      # nearby-places results are shared between sessions in the same ~100 m cell
GOOGLE_RETRY_S = 600.0   # re-check a disabled Google API this often (it may get enabled mid-demo)
ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
LEGACY_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

ROUTE_FIELDS = ",".join([
    "routes.distanceMeters",
    "routes.duration",
    "routes.polyline.encodedPolyline",
    "routes.legs.steps.distanceMeters",
    "routes.legs.steps.startLocation",
    "routes.legs.steps.endLocation",
    "routes.legs.steps.navigationInstruction",
])


class Route:
    def __init__(self, steps: list[RouteStep], polyline: list[tuple[float, float]], distance_m: float, duration_s: float):
        self.steps, self.polyline, self.distance_m, self.duration_s = steps, polyline, distance_m, duration_s


class MapsError(RuntimeError):
    pass


def _clean_instruction(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")       # legacy html_instructions
    text = text.replace("\n", ". ")
    return re.sub(r"\s+", " ", text).strip()


class MapsService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = bool(settings.google_maps_api_key)
        self.osm = settings.osm_fallback
        self._http = httpx.AsyncClient(timeout=8.0)
        self._google_off: dict[str, float] = {}   # api name -> time it was found disabled
        self._nominatim_lock = asyncio.Lock()      # Nominatim policy: max 1 request per second
        self._nominatim_last = 0.0
        self._osm_cache: dict[tuple, tuple[float, list[NearbyPlace]]] = {}

    async def close(self) -> None:
        await self._http.aclose()

    def _headers(self, field_mask: str) -> dict:
        return {"X-Goog-Api-Key": self.settings.google_maps_api_key, "X-Goog-FieldMask": field_mask,
                "Content-Type": "application/json"}

    # ------------------------------------------------------------ availability
    def google_ok(self, api: str) -> bool:
        if not self.enabled:
            return False
        off = self._google_off.get(api)
        return off is None or time.time() - off > GOOGLE_RETRY_S

    def _mark_off(self, api: str, why: str) -> None:
        if api not in self._google_off:
            log.warning("Google %s unavailable (%s); using OpenStreetMap fallback", api, why[:120])
        self._google_off[api] = time.time()

    def status(self) -> dict:
        """Which map sources are live right now (for /health and the brain)."""
        return {
            "google_routes": self.enabled and self.google_ok("routes"),
            "google_places": self.enabled and self.google_ok("places"),
            "google_geocoding": self.enabled and self.google_ok("geocoding"),
            "osm_fallback": self.osm,
        }

    # ------------------------------------------------------------ place lookup
    async def search_place(self, query: str, near: LatLng | None) -> dict | None:
        """Returns {name, address, location: LatLng} for the best match.
        Google Places (New) first; OpenStreetMap Nominatim if Places is unavailable."""
        if self.google_ok("places"):
            try:
                return await self._google_search_place(query, near)
            except MapsError as e:
                if not self.osm:
                    raise
                self._mark_off("places", str(e))
        if self.osm:
            return await self._nominatim_search(query, near)
        raise MapsError("No place search available: enable Places API (New) or OSM fallback")

    async def _google_search_place(self, query: str, near: LatLng | None) -> dict | None:
        body: dict = {"textQuery": query, "maxResultCount": 3}
        if near:
            body["locationBias"] = {"circle": {"center": {"latitude": near.lat, "longitude": near.lng},
                                               "radius": 10000.0}}
        r = await self._http.post(PLACES_URL, json=body,
                                  headers=self._headers("places.displayName,places.formattedAddress,places.location"))
        if r.status_code != 200:
            raise MapsError(f"Places search failed ({r.status_code}): {r.text[:200]}")
        places = r.json().get("places", [])
        if not places:
            return None
        p = places[0]
        return {
            "name": p.get("displayName", {}).get("text", query),
            "address": p.get("formattedAddress"),
            "location": LatLng(lat=p["location"]["latitude"], lng=p["location"]["longitude"]),
        }

    async def walking_route(self, origin: LatLng, dest: LatLng) -> Route:
        if not self.enabled:
            raise MapsError("Google Maps API key not configured")
        if self.settings.use_legacy_directions:
            return await self._legacy_route(origin, dest)
        body = {
            "origin": {"location": {"latLng": {"latitude": origin.lat, "longitude": origin.lng}}},
            "destination": {"location": {"latLng": {"latitude": dest.lat, "longitude": dest.lng}}},
            "travelMode": "WALK",
            "languageCode": "en-US",
            "units": "METRIC",
        }
        r = await self._http.post(ROUTES_URL, json=body, headers=self._headers(ROUTE_FIELDS))
        if r.status_code != 200:
            raise MapsError(f"Routes API failed ({r.status_code}): {r.text[:200]}")
        routes = r.json().get("routes", [])
        if not routes:
            raise MapsError("No walking route found")
        route = routes[0]
        steps: list[RouteStep] = []
        for leg in route.get("legs", []):
            for st in leg.get("steps", []):
                nav = st.get("navigationInstruction", {})
                steps.append(RouteStep(
                    instruction=_clean_instruction(nav.get("instructions", "Continue")),
                    maneuver=nav.get("maneuver"),
                    distance_m=float(st.get("distanceMeters", 0)),
                    start=LatLng(lat=st["startLocation"]["latLng"]["latitude"], lng=st["startLocation"]["latLng"]["longitude"]),
                    end=LatLng(lat=st["endLocation"]["latLng"]["latitude"], lng=st["endLocation"]["latLng"]["longitude"]),
                ))
        poly = decode_polyline(route.get("polyline", {}).get("encodedPolyline", ""))
        duration = float(str(route.get("duration", "0s")).rstrip("s") or 0)
        return Route(steps, poly, float(route.get("distanceMeters", 0)), duration)

    async def _legacy_route(self, origin: LatLng, dest: LatLng) -> Route:
        params = {"origin": f"{origin.lat},{origin.lng}", "destination": f"{dest.lat},{dest.lng}",
                  "mode": "walking", "key": self.settings.google_maps_api_key}
        r = await self._http.get(LEGACY_DIRECTIONS_URL, params=params)
        data = r.json()
        if data.get("status") != "OK":
            raise MapsError(f"Directions API: {data.get('status')} {data.get('error_message', '')}")
        route = data["routes"][0]
        steps = [RouteStep(
            instruction=_clean_instruction(s["html_instructions"]),
            maneuver=s.get("maneuver"),
            distance_m=float(s["distance"]["value"]),
            start=LatLng(lat=s["start_location"]["lat"], lng=s["start_location"]["lng"]),
            end=LatLng(lat=s["end_location"]["lat"], lng=s["end_location"]["lng"]),
        ) for leg in route["legs"] for s in leg["steps"]]
        leg = route["legs"][0]
        return Route(steps, decode_polyline(route["overview_polyline"]["points"]),
                     float(leg["distance"]["value"]), float(leg["duration"]["value"]))

    async def reverse_geocode(self, pos: LatLng) -> str | None:
        """Street address. Google Geocoding first; Nominatim if Geocoding is unavailable."""
        if self.google_ok("geocoding"):
            try:
                r = await self._http.get(GEOCODE_URL, params={"latlng": f"{pos.lat},{pos.lng}",
                                                              "key": self.settings.google_maps_api_key})
                data = r.json()
                if data.get("status") == "OK" and data.get("results"):
                    return data["results"][0]["formatted_address"]
                if data.get("status") in ("REQUEST_DENIED", "OVER_QUERY_LIMIT"):
                    self._mark_off("geocoding", data.get("error_message", data["status"]))
                elif data.get("status") == "ZERO_RESULTS":
                    return None
            except Exception as e:  # noqa: BLE001
                log.warning("Reverse geocode failed: %s", e)
        if self.osm:
            return await self._nominatim_reverse(pos)
        return None

    # ------------------------------------------------------------ nearby places
    async def nearby(self, pos: LatLng, radius_m: int) -> tuple[list[NearbyPlace], list[str]]:
        """Named places around the user: Google Places (New) and OpenStreetMap, merged.
        Returns (places, sources used)."""
        tasks, names = [], []
        if self.google_ok("places"):
            tasks.append(self._google_nearby(pos, radius_m))
            names.append("google_places")
        if self.osm:
            tasks.append(self._overpass_nearby(pos, radius_m))
            names.append("osm")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        places: list[NearbyPlace] = []
        used: list[str] = []
        for name, res in zip(names, results):
            if isinstance(res, Exception):
                if name == "google_places":
                    self._mark_off("places", str(res))
                else:
                    log.warning("Nearby search via %s failed: %s", name, res)
                continue
            used.append(name)
            places += res
        # De-duplicate the same place reported by both sources (same name within ~40 m).
        seen: dict[str, NearbyPlace] = {}
        for p in places:
            key = f"{p.name.lower()}|{round(p.lat, 3)}|{round(p.lng, 3)}"
            if key not in seen or p.source == "google_places":
                seen[key] = p
        return list(seen.values()), used

    async def _google_nearby(self, pos: LatLng, radius_m: int) -> list[NearbyPlace]:
        body = {"maxResultCount": 20, "rankPreference": "DISTANCE",
                "locationRestriction": {"circle": {"center": {"latitude": pos.lat, "longitude": pos.lng},
                                                   "radius": float(radius_m)}}}
        r = await self._http.post(PLACES_NEARBY_URL, json=body, headers=self._headers(
            "places.displayName,places.primaryType,places.location,places.shortFormattedAddress,"
            "places.accessibilityOptions"))
        if r.status_code != 200:
            raise MapsError(f"Places nearby failed ({r.status_code}): {r.text[:200]}")
        out = []
        for p in r.json().get("places", []):
            acc = p.get("accessibilityOptions") or {}
            detail = p.get("shortFormattedAddress")
            if acc.get("wheelchairAccessibleEntrance"):
                detail = f"{detail or ''}; step-free entrance".strip("; ")
            out.append(NearbyPlace(name=p.get("displayName", {}).get("text", "?"),
                                   kind=(p.get("primaryType") or "place").replace("_", " "),
                                   lat=p["location"]["latitude"], lng=p["location"]["longitude"],
                                   source="google_places", detail=detail))
        return out

    async def _overpass_nearby(self, pos: LatLng, radius_m: int) -> list[NearbyPlace]:
        """Named POIs plus blind-relevant street furniture: crossings, steps, entrances, stops."""
        # Kept deliberately light: the public Overpass servers are slow (~20 s) and time out on heavier queries.
        a = f"around:{radius_m},{pos.lat},{pos.lng}"
        q = f"""[out:json][timeout:35];
(
  node[name][amenity]({a}); node[name][shop]({a}); node[name][tourism]({a});
  nwr[railway~"^(station|subway_entrance|tram_stop)$"]({a}); node[highway=bus_stop]({a});
  node[highway=crossing]({a}); node[highway=traffic_signals]({a});
  node[entrance][name]({a}); node[entrance][wheelchair]({a});
);
out center tags 300;"""
        key = (round(pos.lat, 3), round(pos.lng, 3), radius_m)   # ~100 m cells, shared across sessions
        hit = self._osm_cache.get(key)
        if hit and time.time() - hit[0] < OSM_CACHE_S:
            return hit[1]
        data, last_err = None, None
        for url, timeout in zip(OVERPASS_URLS, (35.0, 20.0)):
            try:
                r = await self._http.post(url, data={"data": q}, headers=OSM_HEADERS, timeout=timeout)
                if r.status_code == 200:
                    data = r.json()
                    break
                last_err = f"{url} HTTP {r.status_code}"
            except httpx.HTTPError as e:
                last_err = f"{url} {type(e).__name__}"
        if data is None:
            raise MapsError(f"Overpass failed: {last_err}")
        out = []
        for el in data.get("elements", []):
            t = el.get("tags", {})
            lat = el.get("lat") or (el.get("center") or {}).get("lat")
            lng = el.get("lon") or (el.get("center") or {}).get("lon")
            if lat is None or lng is None:
                continue
            kind = _osm_kind(t)
            name = t.get("name") or t.get("ref") or kind
            extras = []
            if t.get("highway") == "traffic_signals" or t.get("crossing") == "traffic_signals":
                extras.append("traffic signals")
            elif t.get("crossing"):
                extras.append(f"{t['crossing']} crossing".replace("_", " "))
            if t.get("tactile_paving") == "yes":
                extras.append("tactile paving")
            if t.get("wheelchair") in ("yes", "limited", "no"):
                extras.append(f"wheelchair {t['wheelchair']}")
            if t.get("addr:street"):
                extras.append(f"{t.get('addr:housenumber', '')} {t['addr:street']}".strip())
            out.append(NearbyPlace(name=name, kind=kind, lat=float(lat), lng=float(lng), source="osm",
                                   detail=", ".join(e for e in extras if e) or None))
        if out:
            self._osm_cache[key] = (time.time(), out)
        return out

    # ------------------------------------------------------------ Nominatim
    async def _nominatim(self, path: str, params: dict):
        async with self._nominatim_lock:
            wait = 1.0 - (time.time() - self._nominatim_last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._nominatim_last = time.time()
            r = await self._http.get(f"{NOMINATIM_URL}/{path}", params={**params, "format": "jsonv2"},
                                     headers=OSM_HEADERS)
        if r.status_code != 200:
            raise MapsError(f"Nominatim {path} failed ({r.status_code})")
        return r.json()

    async def _nominatim_search(self, query: str, near: LatLng | None) -> dict | None:
        params: dict = {"q": query, "limit": 1, "addressdetails": 0}
        if near:  # prefer results around the user (~10 km box) without excluding the rest
            d = 0.09
            params["viewbox"] = f"{near.lng - d},{near.lat + d},{near.lng + d},{near.lat - d}"
        try:
            res = await self._nominatim("search", params)
        except (httpx.HTTPError, MapsError) as e:
            raise MapsError(f"Place search failed: {e}") from e
        if not res:
            return None
        p = res[0]
        return {"name": p.get("name") or query.split(",")[0], "address": p.get("display_name"),
                "location": LatLng(lat=float(p["lat"]), lng=float(p["lon"]))}

    async def _nominatim_reverse(self, pos: LatLng) -> str | None:
        try:
            res = await self._nominatim("reverse", {"lat": pos.lat, "lon": pos.lng, "zoom": 18})
        except Exception as e:  # noqa: BLE001
            log.warning("Nominatim reverse failed: %s", e)
            return None
        name = res.get("display_name")
        if not name:
            return None
        # "Bush House, 30, Aldwych, St Clement Danes, ..., WC2B 4BG, United Kingdom" -> first 4 parts
        return ", ".join(name.split(", ")[:4])


def _osm_kind(t: dict) -> str:
    if t.get("railway") == "subway_entrance":
        return "underground entrance"
    if t.get("railway") in ("station", "tram_stop") or t.get("public_transport") == "station":
        return "station"
    if t.get("highway") == "bus_stop" or t.get("public_transport") in ("platform", "stop_position"):
        return "bus stop"
    if t.get("highway") in ("crossing", "traffic_signals"):
        return "pedestrian crossing"
    if t.get("highway") == "steps":
        return "steps"
    if t.get("entrance"):
        return "entrance"
    for k in ("amenity", "shop", "tourism", "leisure", "office"):
        if t.get(k):
            return t[k].replace("_", " ")
    if t.get("building"):
        return "building"
    return "place"
