"""Unit tests for posture scoring and presence tracking.

Run from the project root with the project environment:

    source .venv/bin/activate
    python -m unittest discover -s tests -v
"""

import os
import tempfile
import unittest

import numpy as np

from posture.detector import score_pose
from posture.tracker import Tracker


def upright_kpts() -> np.ndarray:
    k = np.zeros((17, 2))
    k[0] = [320, 180]  # nose
    k[5] = [280, 300]  # left shoulder
    k[6] = [360, 300]  # right shoulder
    k[11] = [285, 430]
    k[12] = [355, 430]
    return k


class ScorePoseTest(unittest.TestCase):
    def test_upright_is_good(self):
        self.assertEqual(score_pose(upright_kpts()).status, "good")

    def test_dropped_head_is_slouch(self):
        k = upright_kpts()
        k[0] = [320, 290]  # nose nearly level with shoulders
        self.assertEqual(score_pose(k).status, "slouch")

    def test_sideways_offset_is_lean(self):
        k = upright_kpts()
        k[0] = [250, 180]  # head far left of shoulder centre
        self.assertEqual(score_pose(k).status, "lean")

    def test_missing_keypoints_is_unclear(self):
        self.assertEqual(score_pose(np.zeros((17, 2))).status, "unclear")


class TrackerTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        self.db = tmp.name
        self.addCleanup(os.unlink, self.db)
        self.tr = Tracker(self.db, away_after_seconds=30.0)

    def test_sit_down_and_stay_present(self):
        self.assertEqual(self.tr.update(True, now=1000.0), "present")
        self.assertEqual(self.tr.update(True, now=1010.0), "present")
        self.assertAlmostEqual(
            self.tr.continuous_desk_seconds(now=1010.0), 10.0
        )

    def test_brief_absence_does_not_end_streak(self):
        self.tr.update(True, now=1000.0)
        self.assertEqual(self.tr.update(False, now=1010.0), "maybe-away")
        self.assertEqual(self.tr.update(True, now=1020.0), "present")
        self.assertIsNone(self.tr.last_get_up)

    def test_long_absence_logs_get_up(self):
        self.tr.update(True, now=1000.0)
        self.tr.update(False, now=1010.0)
        self.assertEqual(self.tr.update(False, now=1045.0), "away")
        self.assertEqual(self.tr.last_get_up, 1045.0)

    def test_reminder_after_threshold(self):
        self.tr.update(True, now=1000.0)
        self.assertFalse(self.tr.reminder_due(30, now=1000.0 + 29 * 60))
        self.assertTrue(self.tr.reminder_due(30, now=1000.0 + 31 * 60))
        self.tr.mark_reminded(now=1000.0 + 31 * 60)
        self.assertFalse(self.tr.reminder_due(30, now=1000.0 + 32 * 60))

    def test_state_survives_reload(self):
        self.tr.update(True, now=1000.0)
        self.tr.update(False, now=1010.0)
        self.tr.update(False, now=1050.0)
        reloaded = Tracker(self.db, away_after_seconds=30.0)
        self.assertEqual(reloaded.last_get_up, 1050.0)

    def _day(self, ts):
        from datetime import date
        return date.fromtimestamp(ts).isoformat()

    def test_checkpoint_survives_restart(self):
        self.tr.update(True, now=1000.0)
        self.tr.update(True, now=1100.0)
        self.tr.checkpoint(now=1100.0)
        reloaded = Tracker(self.db, away_after_seconds=30.0)
        self.assertAlmostEqual(reloaded.desk_today(self._day(1100.0)), 100.0)

    def test_checkpoint_then_get_up_counts_once(self):
        self.tr.update(True, now=1000.0)
        self.tr.checkpoint(now=1060.0)
        self.tr.update(True, now=1090.0)
        self.tr.update(False, now=1100.0)
        self.tr.update(False, now=1130.0)  # away; last seen 1090
        self.assertAlmostEqual(self.tr.desk_today(self._day(1130.0)), 90.0)

    def test_today_total_not_double_counted(self):
        self.tr.update(True, now=1000.0)
        self.tr.checkpoint(now=1060.0)
        d = self.tr.to_dict(30, now=1080.0)
        self.assertAlmostEqual(d["desk_today_seconds"], 80.0)
        self.assertAlmostEqual(d["continuous_desk_seconds"], 80.0)

    def test_checkpoint_while_maybe_away_stops_at_last_seen(self):
        self.tr.update(True, now=1000.0)
        self.tr.update(True, now=1050.0)
        self.tr.update(False, now=1055.0)
        self.tr.checkpoint(now=1070.0)
        self.assertAlmostEqual(self.tr.desk_today(self._day(1070.0)), 50.0)


class HistoryTest(unittest.TestCase):
    """Uses a fixed local noon so every sample lands on one calendar day."""

    def setUp(self):
        from datetime import datetime
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        self.db = tmp.name
        self.addCleanup(os.unlink, self.db)
        self.tr = Tracker(self.db, away_after_seconds=60.0)
        self.t0 = datetime(2026, 10, 14, 12, 0).timestamp()

    def today(self, now):
        return self.tr.history(days=1, now=now)[-1]

    def test_break_runs_from_last_sighting_to_return(self):
        t = self.t0
        self.tr.update(True, now=t)
        self.tr.update(True, now=t + 600)       # 10 min at desk
        self.tr.update(False, now=t + 610)
        self.tr.update(False, now=t + 670)      # away; break began at t+600
        self.tr.update(True, now=t + 900)       # back after 5 min
        day = self.today(t + 900)
        self.assertEqual(day["breaks"], 1)
        self.assertAlmostEqual(day["break_seconds"], 300.0)
        self.assertAlmostEqual(day["longest_stretch_seconds"], 600.0)

    def test_live_stretch_and_live_break_count(self):
        t = self.t0
        self.tr.update(True, now=t)
        self.tr.update(True, now=t + 1200)
        self.assertAlmostEqual(self.today(t + 1200)["longest_stretch_seconds"], 1200.0)
        self.tr.update(False, now=t + 1201)
        self.tr.update(False, now=t + 1300)     # away; break began at t+1200
        day = self.today(t + 1500)
        self.assertEqual(day["breaks"], 1)
        self.assertAlmostEqual(day["break_seconds"], 300.0)

    def test_short_absence_is_not_a_break(self):
        t = self.t0
        self.tr.update(True, now=t)
        self.tr.update(False, now=t + 10)
        self.tr.update(True, now=t + 40)
        self.assertEqual(self.today(t + 40)["breaks"], 0)

    def test_good_ratio_ignores_unclear(self):
        t = self.t0
        self.tr.add_posture("good", 30, now=t)
        self.tr.add_posture("slouch", 10, now=t)
        self.tr.add_posture("unclear", 100, now=t)
        self.assertAlmostEqual(self.today(t)["good_ratio"], 0.75)
        self.tr.checkpoint(now=t)                # flushed to SQLite, same answer
        self.assertAlmostEqual(self.today(t)["good_ratio"], 0.75)

    def test_final_checkpoint_closes_open_stretch(self):
        t = self.t0
        self.tr.update(True, now=t)
        self.tr.checkpoint(now=t + 300, final=True)
        reloaded = Tracker(self.db, away_after_seconds=60.0)
        day = reloaded.history(days=1, now=t + 400)[-1]
        self.assertAlmostEqual(day["longest_stretch_seconds"], 300.0)
        self.assertAlmostEqual(day["desk_seconds"], 300.0)

    def test_history_is_oldest_first(self):
        days = self.tr.history(days=7, now=self.t0)
        self.assertEqual(len(days), 7)
        self.assertEqual(days[-1]["day"], "2026-10-14")
        self.assertEqual(days[0]["day"], "2026-10-08")


if __name__ == "__main__":
    unittest.main()
