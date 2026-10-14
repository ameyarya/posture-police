"""Unit tests for bad-posture alert timing.

Run from the project root with the project environment:

    source .venv/bin/activate
    python -m unittest discover -s tests -v
"""

import unittest

from posture.alerts import PostureAlert


class PostureAlertTest(unittest.TestCase):
    def test_fires_after_threshold_once(self):
        a = PostureAlert(after_seconds=60, repeat_seconds=300)
        self.assertFalse(a.update("slouch", True, 0))
        self.assertFalse(a.update("slouch", True, 59))
        self.assertTrue(a.update("slouch", True, 60))
        self.assertTrue(a.active(60))
        self.assertFalse(a.update("lean", True, 61))

    def test_repeats_while_still_bad(self):
        a = PostureAlert(after_seconds=60, repeat_seconds=300)
        a.update("slouch", True, 0)
        self.assertTrue(a.update("slouch", True, 60))
        self.assertFalse(a.update("slouch", True, 359))
        self.assertTrue(a.update("slouch", True, 360))

    def test_good_posture_resets(self):
        a = PostureAlert(after_seconds=60)
        a.update("slouch", True, 0)
        a.update("good", True, 50)
        self.assertFalse(a.update("slouch", True, 70))
        self.assertEqual(a.bad_seconds(70), 0.0)

    def test_leaving_desk_resets(self):
        a = PostureAlert(after_seconds=60)
        a.update("slouch", True, 0)
        a.update("absent", False, 30)
        self.assertFalse(a.update("slouch", True, 65))

    def test_unclear_blip_keeps_stretch_but_never_fires(self):
        a = PostureAlert(after_seconds=60)
        a.update("slouch", True, 0)
        self.assertFalse(a.update("unclear", True, 61))  # still seated
        self.assertFalse(a.active(61))
        self.assertTrue(a.update("slouch", True, 62))

    def test_zero_disables(self):
        a = PostureAlert(after_seconds=0)
        a.update("slouch", True, 0)
        self.assertFalse(a.update("slouch", True, 1000))
        self.assertFalse(a.active(1000))


if __name__ == "__main__":
    unittest.main()
