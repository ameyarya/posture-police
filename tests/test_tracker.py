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

    def test_missing_keypoints_is_away(self):
        self.assertEqual(score_pose(np.zeros((17, 2))).status, "away")


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


if __name__ == "__main__":
    unittest.main()
