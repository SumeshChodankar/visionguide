"""Voice Agent — speech in, intent out.

Whisper transcribes; a deterministic rule parser handles the core commands in
<1 ms (no LLM round-trip for "stop" or "find elevator"); anything it can't
classify goes to the conversational assistant (Reasoning Agent)."""
from __future__ import annotations

import re

from ..db import Database
from ..schemas import Destination, Intent
from ..services.speech import WhisperService
from ..utils.floor import infer_floor_from_room
from ..utils.text import canonical_room

TARGETS: dict[str, list[str]] = {
    "elevator": ["elevator", "lift", "elevators", "lifts"],
    "stairs": ["stairs", "staircase", "stairway", "stairwell"],
    "escalator": ["escalator"],
    "reception": ["reception", "front desk", "information desk", "info desk", "help desk", "concierge", "check-in", "check in"],
    "exit": ["exit", "way out"],
    "entrance": ["entrance", "main entrance", "way in", "front door"],
    "restroom": ["restroom", "toilet", "toilets", "bathroom", "washroom", "loo", "wc", "lavatory"],
    "door": ["door"],
}
_TARGET_LOOKUP = {syn: k for k, syns in TARGETS.items() for syn in syns}
_TARGET_RE = "|".join(sorted((re.escape(s) for s in _TARGET_LOOKUP), key=len, reverse=True))

NAV_RE = re.compile(r"^(?:please\s+)?(?:can you\s+)?(?:take me|guide me|navigate|bring me|get me|walk me|lead me|"
                    r"directions|i (?:want|need) to go|i'?m going|go|head)\s+(?:to|towards?)\s+(?P<dest>.+)$", re.I)
FIND_RE = re.compile(rf"\b(?:find|where(?:'s| is| are)|locate|look for|search for|guide me to|take me to|show me)\s+"
                     rf"(?:the\s+|a\s+|an\s+)?(?:nearest\s+|closest\s+)?(?P<t>{_TARGET_RE})\b", re.I)
ROOM_IN_TEXT = re.compile(r"\b(?:room|rm|office|suite|lab)\s*(?:number\s*)?#?\s*(?P<code>[A-Z]{0,2}\s?-?\d{2,4}\s?-?[A-Z]?)\b", re.I)
BARE_CODE = re.compile(r"(?<![A-Z0-9])(?P<code>[A-Z]?\d{2,4}[A-Z]?)(?![A-Z0-9])", re.I)
GATE_RE = re.compile(r"\b(gate|platform|terminal|building|block|wing|desk|counter)\s+(?P<id>[A-Z0-9]{1,4})\b", re.I)


def _clean(t: str) -> str:
    t = t.strip().strip(".!?")
    return re.sub(r"\s+", " ", t)


def _strip_connectors(t: str) -> str:
    t = t.strip(" ,")
    for _ in range(3):
        t = re.sub(r"^(?:in|at|of|on|inside|within|the|near)\s+|\s+(?:in|at|of|on|the)$", "", t.strip(" ,"), flags=re.I)
    return t.strip(" ,")


def parse_destination(dest_text: str, db: Database | None = None) -> Destination:
    raw = _clean(dest_text)
    text = re.sub(r"^(the|a|an)\s+", "", raw, flags=re.I)
    d = Destination(raw=raw)

    if db and (saved := db.get_saved_place(text.lower())):
        d.place_query, d.room = saved["query"], saved["room"]
    else:
        m = ROOM_IN_TEXT.search(text) or (BARE_CODE.search(text) if re.search(r"\d", text) else None)
        rest = text
        if m and not GATE_RE.search(text[: m.end()]):
            d.room = canonical_room(m.group("code"))
            rest = (text[: m.start()] + " " + text[m.end():]).strip()
        g = GATE_RE.search(rest)
        if g:  # "Gate 4", "Building C": text to look for on signage
            d.sign_text = f"{g.group(1).title()} {g.group('id').upper()}"
            rest = (rest[: g.start()] + " " + rest[g.end():]).strip()
        t = re.search(rf"\b({_TARGET_RE})\b", rest, re.I)
        if t and not d.room and not d.sign_text:
            d.target_kind = _TARGET_LOOKUP[t.group(1).lower()]
            rest = (rest[: t.start()] + " " + rest[t.end():]).strip()
        rest = _strip_connectors(rest)
        if len(rest) > 2:
            d.place_query = rest
            if g and g.group(1).lower() in ("building", "block", "wing"):
                d.place_query = f"{d.sign_text}, {rest}"   # "Building C, Imperial College"

    if d.room:
        inf = infer_floor_from_room(d.room)
        d.floor_hint, d.floor_explanation = inf.floor, inf.explanation
    return d


def parse_intent(text: str, db: Database | None = None) -> Intent:
    t = _clean(text)
    low = t.lower()

    if re.fullmatch(r"(stop|cancel|end|quit|exit navigation|stop navigation|cancel navigation|never ?mind)( please)?", low):
        return Intent(name="stop", text=t)
    if re.search(r"\b(repeat|say (that|it) again|what did you say|pardon)\b", low):
        return Intent(name="repeat", text=t)
    if re.fullmatch(r"(help|what can you do\??|commands|what can i say\??)", low):
        return Intent(name="help", text=t)
    if re.search(r"\bwhere am i\b|\bmy (current )?location\b|\bwhere are we\b", low):
        return Intent(name="where_am_i", text=t)
    if re.search(r"\b(read|what does)\b.*\b(sign|text|say|says|label|this|that|it)\b", low):
        return Intent(name="read_text", text=t)
    if re.search(r"\bwhat'?s? (is )?(in front|ahead|around|near)|\bdescribe\b|\bwhat (do|can) you see\b|\blook around\b|\bwhat'?s here\b", low):
        return Intent(name="describe", text=t)

    m = NAV_RE.match(t)
    dest_text = m.group("dest") if m else None
    f = FIND_RE.search(t)
    if f and (not dest_text or _clean(re.sub(r"^(?:(?:the|a|an|nearest|closest)\s+)+", "", dest_text, flags=re.I)).lower() in _TARGET_LOOKUP):
        return Intent(name="find", target_kind=_TARGET_LOOKUP[f.group("t").lower()], text=t)
    if dest_text:
        return Intent(name="navigate", destination=parse_destination(dest_text, db), text=t)
    if fm := re.match(r"^(?:find|locate|where is|where's|look for)\s+(?P<d>(?:room|office|suite|lab|gate)\b.+)$", t, re.I):
        return Intent(name="navigate", destination=parse_destination(fm.group("d"), db), text=t)
    # "Room 443B" on its own is a navigation request
    if ROOM_IN_TEXT.fullmatch(t) or re.fullmatch(r"(?:to\s+)?room\s+\S+", low):
        return Intent(name="navigate", destination=parse_destination(t, db), text=t)
    return Intent(name="chat", text=t)


HELP_TEXT = ("You can say: take me to Room 443B in a building name. Find the elevator. Guide me to reception. "
             "Where am I? What is in front of me? Read the sign. Repeat. Or stop. "
             "You can also ask me any question about your surroundings.")


class VoiceAgent:
    name = "voice"

    def __init__(self, whisper: WhisperService, db: Database):
        self.whisper = whisper
        self.db = db

    async def transcribe(self, audio: bytes, filename: str) -> str:
        return await self.whisper.transcribe(audio, filename)

    def understand(self, text: str) -> Intent:
        return parse_intent(text, self.db)
