"""OCR Agent — reads room numbers, office labels, floor indicators, gate
numbers, building identifiers and directional signs, then classifies each
string so the Reasoning Agent can act on it.

Sources: EasyOCR (precise boxes → clock direction) merged with Gemini's sign
transcriptions (better on stylised or angled text, no boxes)."""
from __future__ import annotations

import re

from PIL import Image

from ..schemas import OCRText
from ..services.local_models import OcrEngine
from ..utils.floor import parse_floor_indicator, parse_room_range
from ..utils.geo import clock_from_image_x, side_from_image_x
from ..utils.text import ROOM_RE, canonical_room, normalize

GATE_RE = re.compile(r"\bgate\s*#?\s*([A-Z]?\d{1,3}[A-Z]?)\b", re.I)
BUILDING_RE = re.compile(r"\b(building|bldg|block|wing|hall|house|tower|centre|center|annex|pavilion)\b", re.I)
BUILDING_ID_RE = re.compile(r"\b(?:building|bldg\.?|block|wing)\s+([A-Z0-9]{1,3})\b", re.I)
DIRECTION_RE = re.compile(r"[→←↑↓➔➜⇨⇦]|\b(left|right|straight ahead|this way|ahead)\b", re.I)


def direction_words(text: str) -> str | None:
    t = text.lower()
    if "→" in text or "➔" in text or "➜" in text or "⇨" in text or re.search(r"\bright\b", t):
        return "to the right"
    if "←" in text or "⇦" in text or re.search(r"\bleft\b", t):
        return "to the left"
    if "↑" in text or "straight" in t or re.search(r"\bahead\b", t):
        return "straight ahead"
    if "↓" in text:
        return "straight ahead"  # down-arrows on overhead signs usually mean "here / ahead"
    return None


def classify(text: str) -> tuple[str, str | None, list[int] | None]:
    """Return (kind, value, room_range) for a piece of sign text."""
    rng = parse_room_range(text)
    if rng and (DIRECTION_RE.search(text) or re.search(r"\brooms?\b", text, re.I)):
        return "direction", direction_words(text), list(rng)
    floor = parse_floor_indicator(text)
    if floor is not None:
        return "floor", str(floor), None
    if m := GATE_RE.search(text):
        return "gate", m.group(1).upper(), None
    if m := BUILDING_ID_RE.search(text):
        return "building", m.group(1).upper(), None
    if m := ROOM_RE.search(text):
        code = canonical_room(m.group("code"))
        if re.search(r"\d", code) and len(normalize(code)) <= 6:
            return "room", code, None
    if BUILDING_RE.search(text):
        return "building", text.strip(), None
    if DIRECTION_RE.search(text):
        return "direction", direction_words(text), None
    return "other", None, None


class OCRAgent:
    name = "ocr"

    def __init__(self, engine: OcrEngine, hfov_deg: float = 60.0):
        self.engine = engine
        self.hfov = hfov_deg

    def read(self, img: Image.Image) -> list[OCRText]:
        out: list[OCRText] = []
        for bbox, text, conf in self.engine.read(img):
            cx = (bbox[0] + bbox[2]) / 2
            kind, value, rng = classify(text)
            out.append(OCRText(text=text, confidence=round(conf, 3), bbox=[round(v, 4) for v in bbox],
                               clock=clock_from_image_x(cx, self.hfov), side=side_from_image_x(cx),
                               kind=kind, value=value, room_range=rng))
        return out

    @staticmethod
    def merge_gemini_signs(ocr: list[OCRText], sign_texts: list[str]) -> list[OCRText]:
        """Add Gemini transcriptions that EasyOCR missed (no bbox → assume ahead)."""
        seen = {normalize(o.text) for o in ocr}
        merged = list(ocr)
        for t in sign_texts:
            n = normalize(t)
            if not n or any(n in s or s in n for s in seen if s):
                continue
            kind, value, rng = classify(t)
            merged.append(OCRText(text=t, confidence=0.7, bbox=[], clock=12, side="ahead",
                                  kind=kind, value=value, room_range=rng))
            seen.add(n)
        return merged
