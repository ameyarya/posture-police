"""Presence and desk-time tracking with SQLite persistence.

Pure state machine plus a tiny SQLite store. The detector feeds
``update(present, now)``; the Flask layer reads ``to_dict()``.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import date, datetime, timedelta


class Tracker:
    def __init__(self, db_path: str = "posture.db", away_after_seconds: float = 30.0) -> None:
        self.db_path = db_path
        self.away_after = away_after_seconds
        self._lock = threading.Lock()
        self.state = "unknown"  # present | maybe-away | away | unknown
        self.streak_start: float | None = None
        # Desk time of the current streak up to here is already in `days`.
        self.banked_until: float | None = None
        # Start of the current break (last sighting before going away).
        self.break_start: float | None = None
        # Posture seconds not yet flushed to SQLite: {(day, status): secs}.
        self._posture_buf: dict[tuple[str, str], float] = {}
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
        # Finished desk stretches and breaks, for the history card.
        cur.execute(
            "CREATE TABLE IF NOT EXISTS sessions("
            "kind TEXT NOT NULL, start REAL NOT NULL, end REAL NOT NULL)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS posture("
            "day TEXT NOT NULL, status TEXT NOT NULL, seconds REAL NOT NULL, "
            "PRIMARY KEY(day, status))"
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
                    if self.state == "away" and self.break_start is not None:
                        self._session("break", self.break_start, now)
                    self.break_start = None
                    self.state = "present"
                    if self.streak_start is None:
                        self.streak_start = now
                        self.banked_until = now
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
                        self._bank(self.last_seen)
                        self._session("desk", self.streak_start, self.last_seen)
                        self.streak_start = None
                        self.banked_until = None
                    # The break began when we last saw you, not when the
                    # away timer ran out.
                    self.break_start = self.last_seen
                    self._log(now, "get_up")
                return self.state
            if self.state == "unknown":
                self.state = "away"
                return self.state
            return self.state  # already away

    def _bank(self, upto: float) -> None:
        """Store streak time not yet in `days`. Caller holds the lock."""
        if self.banked_until is None or upto <= self.banked_until:
            return
        self._add_desk(
            date.fromtimestamp(upto).isoformat(), upto - self.banked_until
        )
        self.banked_until = upto

    def _session(self, kind: str, start: float, end: float) -> None:
        if end > start:
            self._conn.execute(
                "INSERT INTO sessions(kind, start, end) VALUES(?, ?, ?)",
                (kind, start, end),
            )
            self._conn.commit()

    def add_posture(self, status: str, seconds: float, now: float | None = None) -> None:
        """Accumulate time spent in a posture; flushed by checkpoint()."""
        now = time.time() if now is None else now
        key = (date.fromtimestamp(now).isoformat(), status)
        with self._lock:
            self._posture_buf[key] = self._posture_buf.get(key, 0.0) + seconds

    def _flush_posture(self) -> None:
        for (day, status), secs in self._posture_buf.items():
            self._conn.execute(
                "INSERT INTO posture(day, status, seconds) VALUES(?, ?, ?) "
                "ON CONFLICT(day, status) DO UPDATE "
                "SET seconds = posture.seconds + excluded.seconds",
                (day, status, secs),
            )
        self._conn.commit()
        self._posture_buf.clear()

    def checkpoint(self, now: float | None = None, final: bool = False) -> None:
        """Persist the running streak so a restart or crash loses little.

        Call periodically and with final=True on shutdown, which also closes
        the open desk stretch or break. Time after the last sighting is left
        unbanked until presence is confirmed again.
        """
        now = time.time() if now is None else now
        with self._lock:
            self._flush_posture()
            if self.streak_start is not None:
                end = now if self.state == "present" else self.last_seen
                self._bank(end)
                if final:
                    self._session("desk", self.streak_start, end)
                    self.streak_start = None
                    self.banked_until = None
            if final and self.break_start is not None:
                self._session("break", self.break_start, now)
                self.break_start = None

    def unbanked_seconds(self, now: float | None = None) -> float:
        now = time.time() if now is None else now
        with self._lock:
            if self.state == "present" and self.banked_until is not None:
                return max(0.0, now - self.banked_until)
            return 0.0

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

    def history(self, days: int = 7, now: float | None = None) -> list[dict]:
        """Per-day totals, oldest first, including the stretch in progress."""
        now = time.time() if now is None else now
        today = date.fromtimestamp(now)
        with self._lock:
            live_desk = (
                (self.streak_start, now)
                if self.state == "present" and self.streak_start is not None
                else None
            )
            live_break = (
                (self.break_start, now)
                if self.state == "away" and self.break_start is not None
                else None
            )
            buf = dict(self._posture_buf)
        cur = self._conn.cursor()
        out = []
        for back in range(days - 1, -1, -1):
            day = today - timedelta(days=back)
            iso = day.isoformat()
            lo = datetime.combine(day, datetime.min.time()).timestamp()
            hi = lo + 86400
            spans = {"desk": [], "break": []}
            for kind, start, end in cur.execute(
                "SELECT kind, start, end FROM sessions WHERE start >= ? AND start < ?",
                (lo, hi),
            ):
                spans.setdefault(kind, []).append(end - start)
            if day == today:
                if live_desk:
                    spans["desk"].append(live_desk[1] - live_desk[0])
                if live_break:
                    spans["break"].append(live_break[1] - live_break[0])
            posture = {
                status: secs
                for status, secs in cur.execute(
                    "SELECT status, seconds FROM posture WHERE day = ?", (iso,)
                )
            }
            for (bday, status), secs in buf.items():
                if bday == iso:
                    posture[status] = posture.get(status, 0.0) + secs
            scored = sum(posture.get(s, 0.0) for s in ("good", "slouch", "lean"))
            desk = self.desk_today(iso)
            if day == today:
                desk += self.unbanked_seconds(now)
            out.append({
                "day": iso,
                "desk_seconds": desk,
                "break_seconds": sum(spans["break"]),
                "breaks": len(spans["break"]),
                "longest_stretch_seconds": max(spans["desk"], default=0.0),
                "posture_seconds": posture,
                "good_ratio": posture.get("good", 0.0) / scored if scored else None,
            })
        return out

    def to_dict(self, desk_minutes: float, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        return {
            "presence": self.state,
            "continuous_desk_seconds": self.continuous_desk_seconds(now),
            "desk_today_seconds": self.desk_today(date.fromtimestamp(now).isoformat())
            + self.unbanked_seconds(now),
            "last_get_up": self.last_get_up,
            "reminder_due": self.reminder_due(desk_minutes, now),
        }
