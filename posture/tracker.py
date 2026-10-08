"""Presence and desk-time tracking with SQLite persistence.

Pure state machine plus a tiny SQLite store. The detector feeds
``update(present, now)``; the Flask layer reads ``to_dict()``.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import date


class Tracker:
    def __init__(self, db_path: str = "posture.db", away_after_seconds: float = 30.0) -> None:
        self.db_path = db_path
        self.away_after = away_after_seconds
        self._lock = threading.Lock()
        self.state = "unknown"  # present | maybe-away | away | unknown
        self.streak_start: float | None = None
        self.last_seen: float | None = None
        self.last_get_up: float | None = None
        self.last_reminder: float | None = None
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()
        self._load_meta()

    # -- schema ---------------------------------------------------------
    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS days("
            "day TEXT PRIMARY KEY, desk_seconds REAL NOT NULL DEFAULT 0)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS events("
            "ts REAL NOT NULL, kind TEXT NOT NULL, note TEXT DEFAULT '')"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS meta("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.commit()

    def _load_meta(self) -> None:
        cur = self._conn.cursor()
        row = cur.execute(
            "SELECT value FROM meta WHERE key='last_get_up'"
        ).fetchone()
        if row:
            self.last_get_up = float(row[0])
        row = cur.execute(
            "SELECT value FROM meta WHERE key='last_reminder'"
        ).fetchone()
        if row:
            self.last_reminder = float(row[0])

    def _save_meta(self, key: str, value: float) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )
        self._conn.commit()

    def _log(self, ts: float, kind: str, note: str = "") -> None:
        self._conn.execute(
            "INSERT INTO events(ts, kind, note) VALUES(?, ?, ?)", (ts, kind, note)
        )
        self._conn.commit()

    def _add_desk(self, day: str, seconds: float) -> None:
        self._conn.execute(
            "INSERT INTO days(day, desk_seconds) VALUES(?, ?) "
            "ON CONFLICT(day) DO UPDATE "
            "SET desk_seconds = days.desk_seconds + excluded.desk_seconds",
            (day, seconds),
        )
        self._conn.commit()

    # -- state machine --------------------------------------------------
    def update(self, present: bool, now: float | None = None) -> str:
        """Feed one detection sample. Returns the presence state."""
        now = time.time() if now is None else now
        with self._lock:
            if present:
                self.last_seen = now
                if self.state != "present":
                    if self.state == "away" or self.state == "unknown":
                        self._log(now, "sit_down")
                    self.state = "present"
                    if self.streak_start is None:
                        self.streak_start = now
                return self.state
            # No person in frame.
            if self.state == "present":
                self.state = "maybe-away"
                return self.state
            if self.state == "maybe-away":
                assert self.last_seen is not None
                if now - self.last_seen >= self.away_after:
                    self.state = "away"
                    self.last_get_up = now
                    self._save_meta("last_get_up", now)
                    if self.streak_start is not None:
                        self._add_desk(
                            date.fromtimestamp(now).isoformat(),
                            max(0.0, self.last_seen - self.streak_start),
                        )
                        self.streak_start = None
                    self._log(now, "get_up")
                return self.state
            if self.state == "unknown":
                self.state = "away"
                return self.state
            return self.state  # already away

    def continuous_desk_seconds(self, now: float | None = None) -> float:
        now = time.time() if now is None else now
        with self._lock:
            if self.state == "present" and self.streak_start is not None:
                return max(0.0, now - self.streak_start)
            return 0.0

    def desk_today(self, today: str | None = None) -> float:
        today = date.today().isoformat() if today is None else today
        cur = self._conn.cursor()
        row = cur.execute(
            "SELECT desk_seconds FROM days WHERE day=?", (today,)
        ).fetchone()
        return float(row[0]) if row else 0.0

    def reminder_due(self, desk_minutes: float, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            if self.state != "present" or self.streak_start is None:
                return False
            if now - self.streak_start < desk_minutes * 60:
                return False
            if self.last_reminder is not None and self.last_reminder >= self.streak_start:
                return False
            return True

    def mark_reminded(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self.last_reminder = now
            self._save_meta("last_reminder", now)
            self._log(now, "reminder")

    def to_dict(self, desk_minutes: float, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        return {
            "presence": self.state,
            "continuous_desk_seconds": self.continuous_desk_seconds(now),
            "desk_today_seconds": self.desk_today(date.fromtimestamp(now).isoformat())
            + self.continuous_desk_seconds(now),
            "last_get_up": self.last_get_up,
            "reminder_due": self.reminder_due(desk_minutes, now),
        }
