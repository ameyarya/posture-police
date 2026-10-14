"""Bad-posture alert timing.

Pure state machine, safe to unit test. The vision loop feeds it the
reported posture once per tick; it says when to nudge the user.
"""

from __future__ import annotations

BAD_POSTURES = ("slouch", "lean")


class PostureAlert:
    """Fires once posture has stayed bad for ``after_seconds``.

    While posture stays bad it fires again every ``repeat_seconds``.
    Sitting upright or leaving the desk resets the stretch; brief
    unclear readings in between do not.
    """

    def __init__(self, after_seconds: float = 60.0, repeat_seconds: float = 300.0) -> None:
        self.after = after_seconds
        self.repeat = repeat_seconds
        self.bad_since: float | None = None
        self.last_alert: float | None = None
        self.posture = "starting"

    def update(self, posture: str, present: bool, now: float) -> bool:
        """Feed one sample. Returns True when an alert should go out now."""
        self.posture = posture
        if posture == "good" or not present:
            self.bad_since = None
            self.last_alert = None
            return False
        if posture not in BAD_POSTURES:
            return False  # unclear blip: keep the stretch, never fire on it
        if self.bad_since is None:
            self.bad_since = now
        if self.after <= 0 or now - self.bad_since < self.after:
            return False
        if self.last_alert is not None and now - self.last_alert < self.repeat:
            return False
        self.last_alert = now
        return True

    def bad_seconds(self, now: float) -> float:
        return 0.0 if self.bad_since is None else max(0.0, now - self.bad_since)

    def active(self, now: float) -> bool:
        """Bad right now and past the threshold (drives the dashboard banner)."""
        return (
            self.after > 0
            and self.posture in BAD_POSTURES
            and self.bad_seconds(now) >= self.after
        )
