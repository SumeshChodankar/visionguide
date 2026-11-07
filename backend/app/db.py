"""SQLite persistence. Hot session state lives in memory; every change is
snapshotted here so a backend restart mid-demo doesn't lose the user's goal,
and the events table gives judges a replayable log of what the agents said."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any

from .schemas import SessionState

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    phase         TEXT NOT NULL,
    destination   TEXT,            -- JSON Destination
    state_json    TEXT NOT NULL    -- full SessionState snapshot
);
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    ts          REAL NOT NULL,
    kind        TEXT NOT NULL,     -- command | utterance | route | hazard | arrival | frame | error
    payload     TEXT NOT NULL      -- JSON
);
CREATE INDEX IF NOT EXISTS idx_events_session_ts ON events(session_id, ts);
CREATE TABLE IF NOT EXISTS memories (
    key         TEXT PRIMARY KEY,      -- "office", "preferred_route", "mobility_aid"
    value       TEXT NOT NULL,         -- "Room 443B in Bush House", "prefers stairs over elevators"
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS saved_places (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT UNIQUE NOT NULL,  -- "work", "home", "physio"
    query       TEXT NOT NULL,         -- what to send to Places, e.g. "Bush House London"
    room        TEXT,                  -- optional indoor target, e.g. "443B"
    created_at  REAL NOT NULL
);
"""


class Database:
    def __init__(self, path: str):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.commit()

    def save_session(self, s: SessionState) -> None:
        dest = s.destination.model_dump_json() if s.destination else None
        with self._lock:
            self._conn.execute(
                """INSERT INTO sessions(id, created_at, updated_at, phase, destination, state_json)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at, phase=excluded.phase,
                   destination=excluded.destination, state_json=excluded.state_json""",
                (s.id, s.created_at, time.time(), s.phase.value, dest, s.model_dump_json()),
            )
            self._conn.commit()

    def load_session(self, session_id: str) -> SessionState | None:
        with self._lock:
            row = self._conn.execute("SELECT state_json FROM sessions WHERE id=?", (session_id,)).fetchone()
        return SessionState.model_validate_json(row["state_json"]) if row else None

    def log(self, session_id: str, kind: str, payload: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO events(session_id, ts, kind, payload) VALUES(?,?,?,?)",
                (session_id, time.time(), kind, json.dumps(payload, default=str)),
            )
            self._conn.commit()

    def events(self, session_id: str, limit: int = 200) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, kind, payload FROM events WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [{"ts": r["ts"], "kind": r["kind"], "payload": json.loads(r["payload"])} for r in rows]

    def save_memory(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO memories(key, value, updated_at) VALUES(?,?,?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
                (key.strip().lower()[:60], value.strip()[:500], time.time()),
            )
            self._conn.commit()

    def memories(self, limit: int = 30) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute("SELECT key, value FROM memories ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return {r["key"]: r["value"] for r in rows}

    def get_saved_place(self, label: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM saved_places WHERE label=?", (label.lower(),)).fetchone()
        return dict(row) if row else None

    def upsert_saved_place(self, label: str, query: str, room: str | None) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO saved_places(label, query, room, created_at) VALUES(?,?,?,?)
                   ON CONFLICT(label) DO UPDATE SET query=excluded.query, room=excluded.room""",
                (label.lower(), query, room, time.time()),
            )
            self._conn.commit()


class SessionStore:
    """In-memory cache in front of SQLite."""

    def __init__(self, db: Database):
        self.db = db
        self._cache: dict[str, SessionState] = {}

    def create(self) -> SessionState:
        s = SessionState()
        self._cache[s.id] = s
        self.db.save_session(s)
        return s

    def get(self, session_id: str) -> SessionState | None:
        if session_id in self._cache:
            return self._cache[session_id]
        s = self.db.load_session(session_id)
        if s:
            self._cache[session_id] = s
        return s

    def save(self, s: SessionState) -> None:
        self._cache[s.id] = s
        self.db.save_session(s)
