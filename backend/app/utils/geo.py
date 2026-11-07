"""Geometry helpers: distances, bearings, polylines and clock-face directions.

Clock-face directions ("door at 2 o'clock") are the standard orientation
vocabulary taught to blind and low-vision travellers, so every spatial
instruction the system speaks goes through `clock_from_*`.
"""
from __future__ import annotations

import math

EARTH_R = 6_371_000.0  # metres


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Initial bearing from point 1 to point 2, degrees clockwise from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lng2 - lng1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def clock_from_relative_angle(rel_deg: float) -> int:
    """Map an angle relative to the user's facing direction (-180..180, + = right)
    to a clock hour (12 = straight ahead, 3 = right, 9 = left, 6 = behind)."""
    hour = round(((rel_deg % 360) / 30.0)) % 12
    return 12 if hour == 0 else hour


def clock_from_bearing(target_bearing: float, heading: float | None) -> int | None:
    if heading is None:
        return None
    rel = ((target_bearing - heading + 540) % 360) - 180
    return clock_from_relative_angle(rel)


def clock_from_image_x(x_center_norm: float, hfov_deg: float = 60.0) -> int:
    """Horizontal position in the camera frame (0 = left edge, 1 = right edge)
    to a clock hour. A phone camera only sees roughly 11 to 1 o'clock."""
    rel = (x_center_norm - 0.5) * hfov_deg
    return clock_from_relative_angle(rel)


def side_from_image_x(x_center_norm: float) -> str:
    if x_center_norm < 0.36:
        return "left"
    if x_center_norm > 0.64:
        return "right"
    return "ahead"


def decode_polyline(encoded: str) -> list[tuple[float, float]]:
    """Decode a Google encoded polyline into (lat, lng) pairs."""
    points: list[tuple[float, float]] = []
    index = lat = lng = 0
    while index < len(encoded):
        for coord in (0, 1):
            shift = result = 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else (result >> 1)
            if coord == 0:
                lat += delta
            else:
                lng += delta
        points.append((lat / 1e5, lng / 1e5))
    return points


def _to_xy(lat: float, lng: float, ref_lat: float) -> tuple[float, float]:
    """Equirectangular projection in metres; accurate enough at street scale."""
    x = math.radians(lng) * EARTH_R * math.cos(math.radians(ref_lat))
    y = math.radians(lat) * EARTH_R
    return x, y


def distance_to_polyline_m(lat: float, lng: float, poly: list[tuple[float, float]]) -> float:
    if not poly:
        return float("inf")
    if len(poly) == 1:
        return haversine_m(lat, lng, *poly[0])
    px, py = _to_xy(lat, lng, lat)
    best = float("inf")
    for (a_lat, a_lng), (b_lat, b_lng) in zip(poly, poly[1:]):
        ax, ay = _to_xy(a_lat, a_lng, lat)
        bx, by = _to_xy(b_lat, b_lng, lat)
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        t = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg2))
        cx, cy = ax + t * dx, ay + t * dy
        best = min(best, math.hypot(px - cx, py - cy))
    return best


def spoken_distance(m: float | None) -> str:
    """Precise, TTS-friendly distance: one decimal under 3 m (where a step matters),
    whole metres under 10 m, then 5 m / 10 m steps. No "about", no "very close":
    the user hears the measured value exactly as the agents computed it."""
    if m is None:
        return ""
    if m < 3:
        return f"{max(0.3, round(m, 1)):.1f} meters"
    if m < 10:
        return f"{round(m)} meters"
    if m < 100:
        return f"{int(round(m / 5) * 5)} meters"
    if m < 1000:
        return f"{int(round(m / 10) * 10)} meters"
    return f"{m / 1000:.1f} kilometers"


def direction_phrase(clock: int | None, side: str | None = None) -> str:
    """'at 1 o'clock' when the clock position is known, else 'on your left' / 'on your right' / 'ahead'."""
    if clock:
        return f"at {clock} o'clock"
    return {"left": "on your left", "right": "on your right"}.get(side or "", "ahead")


def object_phrase(name: str, clock: int | None, side: str | None, distance_m: float | None) -> str:
    """'Chair at 1 o'clock, 2.4 meters' - the one fixed format every agent speaks measured objects in."""
    dist = spoken_distance(distance_m)
    return f"{name[:1].upper()}{name[1:]} {direction_phrase(clock, side)}" + (f", {dist}" if dist else "")
