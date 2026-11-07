"""Navigation Agent — outdoor leg. Resolves the destination with Places API
(New), plans a walking route with the Routes API, then tracks GPS: advances
steps, pre-announces turns, detects arrival at the building and re-routes when
the user drifts off the path."""
from __future__ import annotations

import logging
import time

from ..config import Settings
from ..schemas import LatLng, Phase, Priority, SessionState, Utterance
from ..services.maps import MapsError, MapsService
from ..utils.geo import bearing_deg, clock_from_bearing, distance_to_polyline_m, haversine_m, spoken_distance

log = logging.getLogger("visionguide.nav")


def _lower_first(s: str) -> str:
    return s[:1].lower() + s[1:] if s else s


class NavigationAgent:
    name = "navigation"

    def __init__(self, maps: MapsService, settings: Settings):
        self.maps = maps
        self.s = settings

    async def start_route(self, session: SessionState) -> list[Utterance]:
        dest = session.destination
        assert dest and dest.place_query
        if not session.position:
            return [Utterance(text="I need your location to plan the route. Please allow location access, then ask again.",
                              priority=Priority.GUIDANCE, category="system")]
        try:
            if dest.place_location is None:  # known buildings (our floor plans) already carry their location
                place = await self.maps.search_place(dest.place_query, session.position)
                if not place:
                    return [Utterance(text=f"I couldn't find {dest.place_query}. Try the full name and the city.",
                                      priority=Priority.GUIDANCE, category="system")]
                dest.place_name, dest.place_address, dest.place_location = place["name"], place["address"], place["location"]
            route = await self.maps.walking_route(session.position, dest.place_location)
        except MapsError as e:
            log.warning("Routing failed: %s", e)
            return [Utterance(text="I couldn't get walking directions right now. I can still guide you with the camera.",
                              priority=Priority.GUIDANCE, category="system")]

        session.route, session.route_polyline, session.step_index = route.steps, route.polyline, 0
        session.last_reroute_at = time.time()
        if route.distance_m <= self.s.arrival_radius_m or not route.steps:
            session.phase = Phase.APPROACH
            return [Utterance(text=f"You're already at {dest.place_name}. Hold the phone at chest height and I'll look for the entrance.",
                              priority=Priority.GUIDANCE, category="nav")]
        session.phase = Phase.OUTDOOR
        minutes = max(1, round(route.duration_s / 60))
        first = route.steps[0]
        text = (f"Starting route to {dest.place_name}, {spoken_distance(route.distance_m)}, about {minutes} "
                f"minute{'s' if minutes != 1 else ''} walking. {first.instruction}, for {spoken_distance(first.distance_m)}.")
        return [Utterance(text=text, priority=Priority.GUIDANCE, category="nav", haptic="light")]

    def on_position(self, session: SessionState) -> tuple[list[Utterance], bool]:
        """Returns (utterances, needs_reroute)."""
        if session.phase != Phase.OUTDOOR or not session.route or not session.position:
            return [], False
        pos, dest = session.position, session.destination
        out: list[Utterance] = []

        if dest and dest.place_location:
            d_final = haversine_m(pos.lat, pos.lng, dest.place_location.lat, dest.place_location.lng)
            if d_final <= self.s.arrival_radius_m:
                session.phase = Phase.APPROACH
                clock = clock_from_bearing(bearing_deg(pos.lat, pos.lng, dest.place_location.lat,
                                                       dest.place_location.lng), session.heading)
                where = f" It's at {clock} o'clock." if clock else ""
                out.append(Utterance(
                    text=f"You're near {dest.place_name}.{where} Hold your phone at chest height, camera forward, "
                         f"and slowly scan. I'll find the entrance.",
                    priority=Priority.GUIDANCE, category="nav", haptic="success", key="nav:approach"))
                return out, False

        idx = session.step_index
        step = session.route[idx]
        d_end = haversine_m(pos.lat, pos.lng, step.end.lat, step.end.lng)
        has_next = idx + 1 < len(session.route)

        if d_end <= self.s.step_advance_radius_m and has_next:
            session.step_index = idx + 1
            nxt = session.route[idx + 1]
            out.append(Utterance(text=f"Now, {_lower_first(nxt.instruction)}. Then continue {spoken_distance(nxt.distance_m)}.",
                                 priority=Priority.GUIDANCE, category="nav", haptic="light",
                                 key=f"nav:step:{idx + 1}"))
        elif has_next and not step.announced_pre and d_end <= self.s.preannounce_m:
            step.announced_pre = True
            nxt = session.route[idx + 1]
            out.append(Utterance(text=f"In {spoken_distance(d_end).removeprefix('about ')}, {_lower_first(nxt.instruction)}.",
                                 priority=Priority.GUIDANCE, category="nav", key=f"nav:pre:{idx + 1}"))

        needs_reroute = False
        good_fix = session.position_accuracy_m is None or session.position_accuracy_m <= 30
        if good_fix and session.route_polyline:
            off = distance_to_polyline_m(pos.lat, pos.lng, session.route_polyline)
            # ">=": Windows' clock ticks every ~15 ms, so with a 0 s interval ">" could skip a needed re-route.
            if off > self.s.offroute_threshold_m and time.time() - session.last_reroute_at >= self.s.reroute_min_interval_s:
                needs_reroute = True
        return out, needs_reroute

    async def reroute(self, session: SessionState) -> list[Utterance]:
        utts = await self.start_route(session)
        for u in utts:
            if u.category == "nav":
                u.text = "You've gone off the route. Recalculating. " + u.text.split(". ", 1)[-1]
        return utts

    def progress_text(self, session: SessionState) -> str | None:
        if session.phase != Phase.OUTDOOR or not session.route or not session.position:
            return None
        dest = session.destination
        step = session.route[min(session.step_index, len(session.route) - 1)]
        parts = []
        if dest and dest.place_location:
            d = haversine_m(session.position.lat, session.position.lng, dest.place_location.lat, dest.place_location.lng)
            clock = clock_from_bearing(bearing_deg(session.position.lat, session.position.lng,
                                                   dest.place_location.lat, dest.place_location.lng), session.heading)
            parts.append(f"{dest.place_name} is {spoken_distance(d)} away" + (f", at {clock} o'clock" if clock else "") + ".")
        parts.append(f"Current step: {step.instruction}.")
        return " ".join(parts)

    async def address(self, pos: LatLng | None) -> str | None:
        return await self.maps.reverse_geocode(pos) if pos else None
