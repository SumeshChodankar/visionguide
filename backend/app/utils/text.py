"""Text normalisation and fuzzy matching tuned for OCR output on signage."""
from __future__ import annotations

import re
from difflib import SequenceMatcher

# Characters OCR commonly confuses on signage. Applied only to the numeric
# part of a room code, where letters are almost never correct.
_DIGIT_FIXES = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1", "|": "1",
                              "Z": "2", "S": "5", "s": "5", "B": "8", "G": "6", "T": "7"})

# A room-like token: optional wing letters, 2-4 digits (at least the first is a
# real digit), optional letter suffix. "443B", "B-105", "G12", "Room 1205".
ROOM_RE = re.compile(
    r"(?<![A-Z0-9])(?:(?:room|rm|office|suite|lab)\.?\s*#?\s*)?"
    r"(?P<code>[A-Z]{0,2}[\s-]?\d[0-9OI]{1,3}[\s-]?[A-Z]?)(?![A-Z0-9])",
    re.IGNORECASE,
)


def normalize(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def canonical_room(code: str) -> str:
    """'Room 443-b' -> '443B', 'rm B 105' -> 'B105'. Fixes OCR digit confusions."""
    s = code.upper().strip()
    s = re.sub(r"^(ROOM|RM\.?|OFFICE|SUITE|LAB)\s*#?\s*", "", s)
    s = re.sub(r"[\s\-_.#]", "", s)
    # Lazy number group so a trailing letter is kept as the suffix ("443B").
    m = re.fullmatch(r"([A-Z]{0,2}?)(\d[0-9OISZBDQGT]{1,3}?)([A-Z]?)", s)
    if not m:
        return s
    prefix, num, suffix = m.groups()
    # A trailing 8 that was really a B ("4438" for "443B") is ambiguous; keep digits.
    return f"{prefix}{num.translate(_DIGIT_FIXES)}{suffix}"


def room_number_value(code: str) -> int | None:
    m = re.search(r"\d{2,4}", canonical_room(code))
    return int(m.group()) if m else None


def room_match_score(seen: str, target: str) -> float:
    """1.0 = exact room match. Handles OCR noise like '4438' vs '443B'."""
    a, b = canonical_room(seen), canonical_room(target)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # '443B' read as '4438' (B->8) or '443' (suffix dropped)
    if b[-1:].isalpha() and a == b[:-1] + "8":
        return 0.9
    if b[-1:].isalpha() and a == b[:-1]:
        return 0.6
    return SequenceMatcher(None, a, b).ratio() * 0.85


def fuzzy_contains(haystack: str, needle: str) -> float:
    """How well does the OCR'd text contain a place name? 0..1."""
    h, n = normalize(haystack), normalize(needle)
    if not h or not n:
        return 0.0
    if n in h:
        return 1.0
    best = 0.0
    for i in range(0, max(1, len(h) - len(n) + 1)):
        best = max(best, SequenceMatcher(None, h[i:i + len(n)], n).ratio())
    return best
