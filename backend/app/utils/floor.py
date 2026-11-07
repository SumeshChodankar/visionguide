"""Floor reasoning from room codes, floor indicators and directional signs.

Conventions handled (covers most universities, offices and hospitals):
  443B   -> floor 4          (3-digit: first digit is the floor)
  1205   -> floor 12         (4-digit: first two digits)
  B105   -> floor 1, wing B  (letter prefix is a wing/building, not a floor)
  G12 / G.05 -> ground floor (UK style)
  LG / B1 / basement -> below ground
  "Level 3", "Floor 2", "L2", "4F", "3rd floor" -> explicit indicators
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .text import canonical_room

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
            "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "ground": 0}


@dataclass
class FloorInference:
    floor: int | None
    wing: str | None
    confidence: float
    explanation: str


def infer_floor_from_room(room: str, ground_floor_is_one: bool = True) -> FloorInference:
    code = canonical_room(room)
    m = re.fullmatch(r"([A-Z]{0,2}?)(\d{2,4})([A-Z]?)", code)
    if not m:
        return FloorInference(None, None, 0.0, f"I couldn't read a floor from {room}.")
    prefix, digits, _suffix = m.groups()
    wing = prefix or None

    if prefix in ("G", "LG"):
        floor = 0 if prefix == "G" else -1
        label = "the ground floor" if floor == 0 else "the lower ground floor"
        return FloorInference(floor, None, 0.8, f"Room {code} starts with {prefix}, so it is on {label}.")

    if len(digits) == 2:
        return FloorInference(0 if not ground_floor_is_one else 1, wing, 0.4,
                              f"Room {code} has a two-digit number, so it is probably on the ground floor.")
    if len(digits) == 3:
        floor = int(digits[0])
    else:  # 4 digits
        floor = int(digits[:2])
    if floor == 0:
        return FloorInference(0, wing, 0.7, f"Room {code} starts with 0, so it is likely on the ground floor.")

    wing_txt = f" in wing {wing}" if wing else ""
    return FloorInference(floor, wing, 0.75,
                          f"Room {code} is likely on floor {floor}{wing_txt}, because the number starts with {floor}.")


_FLOOR_PATTERNS = [
    (re.compile(r"\b(?:floor|level|lvl|etage)\s*[-:]?\s*(-?\d{1,2})\b", re.I), 1),
    (re.compile(r"\b(\d{1,2})\s*(?:st|nd|rd|th)?\s*(?:floor|fl\.?)\b", re.I), 1),
    (re.compile(r"^\s*L\s?(-?\d{1,2})\s*$", re.I), 1),
    (re.compile(r"^\s*(\d{1,2})\s*F\s*$", re.I), 1),
    (re.compile(r"^\s*B\s?(\d)\s*$", re.I), -1),  # B1, B2 basements
]


def parse_floor_indicator(text: str) -> int | None:
    """Return the floor a sign/elevator display says we're on, else None."""
    t = text.strip()
    low = t.lower()
    if re.fullmatch(r"\s*(g|gf|ground|ground floor|lobby)\s*", low):
        return 0
    if re.fullmatch(r"\s*(lg|lower ground)\s*", low):
        return -1
    if "basement" in low:
        return -1
    for word, n in ORDINALS.items():
        if re.search(rf"\b{word}\s+floor\b", low):
            return n
    for pat, sign in _FLOOR_PATTERNS:
        m = pat.search(t)
        if m:
            return sign * int(m.group(1))
    return None


RANGE_RE = re.compile(r"(?P<a>[A-Z]?\d{2,4})\s*(?:-|–|—|to)\s*(?P<b>[A-Z]?\d{2,4})", re.I)


def parse_room_range(text: str) -> tuple[int, int] | None:
    """'Rooms 440-460' -> (440, 460)."""
    m = RANGE_RE.search(text)
    if not m:
        return None
    a = int(re.sub(r"\D", "", m.group("a")))
    b = int(re.sub(r"\D", "", m.group("b")))
    return (min(a, b), max(a, b))


def floor_phrase(floor: int | None) -> str:
    if floor is None:
        return "an unknown floor"
    if floor == 0:
        return "the ground floor"
    if floor < 0:
        return "the basement"
    return f"floor {floor}"
