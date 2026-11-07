"""Building knowledge from your own floor plans.

    backend/buildings/<id>/building.json   name, aliases, GPS location + radius, one entry per level:
                                           plan image and the plan's numbered key (exact names)
    backend/buildings/<id>/images/*.jpg    the floor plans (and later: entrance / corridor photos)
    backend/buildings/<id>/notes.json      generated once by Gemini from the plans: where each space
                                           sits on its level (atrium, cores, neighbours). Regenerated
                                           automatically when a plan image changes.

At run time nothing is re-read: lookups ("where is the library?") use building.json + notes.json
instantly; only consult_floor_plan sends a plan image to Gemini, for questions the notes can't answer.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from ..schemas import LatLng
from ..utils.geo import haversine_m

log = logging.getLogger("visionguide.buildings")

PLAN_SYSTEM = """You read architectural floor plans for a navigation assistant that guides blind people.
Return only JSON. Describe positions relative to things a person meets on the way: the central atrium or void,
the stair and lift cores (hatched stairs and small lift shafts), main corridors, and neighbouring spaces.
Use plan directions (top/bottom/left/right of the page) only as a secondary hint and say so; do not claim compass
directions. Never invent spaces or room numbers that are not on the plan."""

PLAN_PROMPT = """This is {building}, {level_name} (level {level}). The plan's numbered key is:
{key}
Return JSON:
{{
  "layout": "2-3 sentences: overall shape of this level, where the atrium/void is, how many stair/lift cores and roughly where",
  "spaces": {{"<key number>": "where this space is: next to which core / along which side of the atrium / near which other numbered spaces"}},
  "circulation": "how you move around: main corridors or walkways around the atrium, bridges between wings",
  "notes": "anything else useful for finding your way on this level (large open areas, terraces, balconies), or empty"
}}"""

SYNONYMS = {
    "cafe": ["café", "coffee", "refectory"],
    "reception": ["front desk", "information", "welcome desk"],
    "library": ["books", "study"],
    "lecture theatre": ["lecture hall", "auditorium", "lecture theater"],
    "student union": ["students union", "su"],
    "computer lab": ["computer room", "computers", "pc lab"],
    "terrace": ["outside", "balcony", "roof"],
    "study": ["group study", "shared study", "quiet study"],
}

FILLER = {"the", "a", "an", "to", "in", "at", "of", "on", "please", "me", "take", "go", "find", "where", "is", "guide",
          "nearest", "closest", "room", "building", "ucl", "east"}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]+", " ", text.replace("&", " and ").replace("/", " "))


def _words(text: str) -> set[str]:
    return {w for w in _norm(text).split() if w not in FILLER}


@dataclass
class Floor:
    level: int
    name: str
    plan: Path
    key: dict[str, str]


@dataclass
class Building:
    id: str
    name: str
    aliases: list[str]
    address: str | None
    location: LatLng
    radius_m: float
    floors: list[Floor]
    dir: Path
    general_notes: list[str] = field(default_factory=list)
    plan_notes: dict[int, dict] = field(default_factory=dict)   # level -> parsed notes

    def floor(self, level: int) -> Floor | None:
        return next((f for f in self.floors if f.level == level), None)

    def level_label(self, level: int) -> str:
        f = self.floor(level)
        return f.name if f else f"Level {level:02d}"


class BuildingService:
    def __init__(self, settings, gemini=None):
        self.settings = settings
        self.gemini = gemini
        self.root = Path(settings.buildings_dir)
        self.buildings: dict[str, Building] = {}
        self._notes_task: asyncio.Task | None = None
        self.load()

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        self.buildings = {}
        if not self.root.is_dir():
            return
        for cfg in sorted(self.root.glob("*/building.json")):
            try:
                b = self._parse(cfg)
                self.buildings[b.id] = b
            except Exception as e:  # noqa: BLE001 - one bad building must not stop the others
                log.warning("Skipping %s: %s", cfg, e)
        if self.buildings:
            log.info("Buildings loaded: %s", ", ".join(f"{b.name} ({len(b.floors)} levels, "
                                                       f"{len(b.plan_notes)} with notes)" for b in self.buildings.values()))

    def _parse(self, cfg: Path) -> Building:
        d = json.loads(cfg.read_text(encoding="utf-8"))
        base = cfg.parent
        floors = [Floor(level=int(f["level"]), name=f.get("name") or f"Level {int(f['level']):02d}",
                        plan=base / f["plan"], key={str(k): v for k, v in f.get("key", {}).items()})
                  for f in d.get("floors", [])]
        b = Building(id=d["id"], name=d["name"], aliases=d.get("aliases", []), address=d.get("address"),
                     location=LatLng(lat=d["location"]["lat"], lng=d["location"]["lng"]),
                     radius_m=float(d.get("radius_m", 100)), floors=sorted(floors, key=lambda f: f.level),
                     dir=base, general_notes=d.get("notes", []))
        notes_file = base / "notes.json"
        if notes_file.exists():
            saved = json.loads(notes_file.read_text(encoding="utf-8"))
            for f in b.floors:
                entry = saved.get(str(f.level))
                if entry and f.plan.exists() and entry.get("plan_sha1") == _sha1(f.plan):
                    b.plan_notes[f.level] = entry["notes"]
        return b

    # ------------------------------------------------------------------ one-time plan reading
    def missing_notes(self) -> list[tuple[Building, Floor]]:
        return [(b, f) for b in self.buildings.values() for f in b.floors
                if f.level not in b.plan_notes and f.plan.exists()]

    def prepare_notes_background(self) -> None:
        """Called at startup: reads any plan without notes, in the background."""
        if self.missing_notes() and getattr(self.gemini, "enabled", False) and hasattr(self.gemini, "image_json"):
            self._notes_task = asyncio.create_task(self.prepare_notes())

    async def prepare_notes(self, concurrency: int = 3) -> int:
        todo = self.missing_notes()
        if not todo:
            return 0
        log.info("Reading %d floor plan(s) with Gemini (one time)...", len(todo))
        sem = asyncio.Semaphore(concurrency)

        async def one(b: Building, f: Floor) -> bool:
            async with sem:
                key = "\n".join(f"{k}. {v}" for k, v in f.key.items()) or "(no key)"
                prompt = PLAN_PROMPT.format(building=b.name, level_name=f.name, level=f.level, key=key)
                notes = await self.gemini.image_json(f.plan.read_bytes(), prompt, PLAN_SYSTEM, timeout=60.0)
                if not notes:
                    log.warning("No notes for %s %s", b.name, f.name)
                    return False
                b.plan_notes[f.level] = notes
                self._save_notes(b)
                return True

        done = sum(await asyncio.gather(*(one(b, f) for b, f in todo)))
        log.info("Floor plans read: %d/%d", done, len(todo))
        return done

    def _save_notes(self, b: Building) -> None:
        out = {str(lv): {"plan_sha1": _sha1(b.floor(lv).plan), "notes": n}
               for lv, n in sorted(b.plan_notes.items()) if b.floor(lv)}
        (b.dir / "notes.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------ queries
    def near(self, pos: LatLng | None, extra_m: float = 0.0) -> Building | None:
        if not pos:
            return None
        best = None
        for b in self.buildings.values():
            d = haversine_m(pos.lat, pos.lng, b.location.lat, b.location.lng)
            if d <= b.radius_m + extra_m and (best is None or d < best[0]):
                best = (d, b)
        return best[1] if best else None

    def by_name(self, text: str | None) -> Building | None:
        """A building named in free text ("the library in Marshgate" -> Marshgate)."""
        if not text:
            return None
        t = f" {_norm(text)} "
        hits = [(len(a), b) for b in self.buildings.values() for a in [b.name, *b.aliases] if f" {_norm(a)} " in t]
        return max(hits, key=lambda h: h[0])[1] if hits else None

    def strip_building(self, text: str, b: Building) -> str:
        """'the library in Marshgate' -> 'the library'."""
        t = _norm(text)
        for a in sorted([b.name, *b.aliases], key=len, reverse=True):
            t = re.sub(rf"\b(in|at|inside)?\s*(the\s+)?{re.escape(_norm(a))}\b", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    def find_space(self, query: str, building: Building | None = None, limit: int = 6) -> list[dict]:
        """Named spaces matching the query, best first: [{building, level, level_name, number, name, where}]."""
        q = _words(query)
        for canon, alts in SYNONYMS.items():          # "coffee" -> also search "cafe"
            if any(_words(a) <= q for a in [canon, *alts] if _words(a)):
                q |= _words(canon)
        if not q:
            return []
        scored = []
        for b in ([building] if building else self.buildings.values()):
            for f in b.floors:
                for num, name in f.key.items():
                    nw = _words(name)
                    for canon, alts in SYNONYMS.items():
                        if _words(canon) <= nw:
                            for a in alts:
                                nw |= _words(a)
                    overlap = len(q & nw)
                    if not overlap:
                        continue
                    score = overlap / len(q) + (0.5 if _words(name) <= q else 0)   # exact name wins
                    where = (b.plan_notes.get(f.level, {}).get("spaces") or {}).get(num)
                    scored.append((score, -f.level, {"building": b.name, "building_id": b.id, "level": f.level,
                                                     "level_name": f.name, "number": num, "name": name,
                                                     "where": where}))
        scored.sort(key=lambda s: (-s[0], -s[1]))
        top = scored[0][0] if scored else 0
        return [s[2] for s in scored if s[0] >= max(0.5, top - 0.01)][:limit]

    def summary(self, b: Building, levels: list[int] | None = None) -> dict:
        """Compact building context for the brain: every level's spaces, plus detailed notes for `levels`."""
        out = {
            "name": b.name,
            "level_numbering": "Level 00 is the ground floor; Level N is floor N.",
            "levels": {f.name: [v for v in f.key.values()] for f in b.floors},
            "about": b.general_notes,
        }
        detail = {}
        for lv in levels or []:
            f, n = b.floor(lv), b.plan_notes.get(lv)
            if f and n:
                detail[f.name] = {"layout": n.get("layout"), "circulation": n.get("circulation"),
                                  "spaces": {f.key.get(k, k): v for k, v in (n.get("spaces") or {}).items()},
                                  "notes": n.get("notes") or None}
        if detail:
            out["level_details"] = detail
        return out

    async def consult(self, b: Building, level: int, question: str, context: str = "") -> str | None:
        """Look at the actual plan image to answer a wayfinding question about one level."""
        f = b.floor(level)
        if not f or not f.plan.exists() or not hasattr(self.gemini, "image_text"):
            return None
        key = "; ".join(f"{k}. {v}" for k, v in f.key.items())
        notes = json.dumps(b.plan_notes.get(level) or {}, ensure_ascii=False)
        prompt = (f"{b.name}, {f.name}. Plan key: {key}. Earlier notes on this plan: {notes}. {context}\n"
                  f"Question from a blind visitor: {question}\n"
                  "Answer in at most 3 short spoken sentences, naming landmarks they will pass (cores, atrium, "
                  "neighbouring spaces). Say clearly if the plan cannot answer it.")
        return await self.gemini.image_text(f.plan.read_bytes(), prompt, PLAN_SYSTEM.replace("Return only JSON. ", ""),
                                            timeout=30.0)

    def info(self) -> list[dict]:
        return [{"id": b.id, "name": b.name, "address": b.address, "location": b.location.model_dump(),
                 "radius_m": b.radius_m, "levels": [{"level": f.level, "name": f.name, "plan": f"/buildings/{b.id}/"
                                                     f"{f.plan.relative_to(b.dir).as_posix()}", "spaces": f.key,
                                                     "has_notes": f.level in b.plan_notes} for f in b.floors]}
                for b in self.buildings.values()]


def _sha1(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()
