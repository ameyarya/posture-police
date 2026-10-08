"""Unit tests for detection gating and digital zoom (no hardware needed)."""

import unittest

import numpy as np

from posture.detector import crop_center, keypoints_usable


def conf_row(nose: float, shoulder: float) -> np.ndarray:
    row = np.full(17, 0.9)
    row[0] = nose
    row[5] = shoulder
    row[6] = shoulder
    return row


class UsableTest(unittest.TestCase):
    def test_clear_person_is_usable(self):
        self.assertTrue(keypoints_usable(conf_row(0.8, 0.7), 0.3))

    def test_weak_nose_is_not_usable(self):
        self.assertFalse(keypoints_usable(conf_row(0.1, 0.8), 0.3))

    def test_weak_shoulder_is_not_usable(self):
        self.assertFalse(keypoints_usable(conf_row(0.8, 0.2), 0.3))

    def test_half_visible_person_is_not_usable(self):
        # Mirrors the live NexiGo frame: mean conf ~0.12-0.27, junk geometry.
        row = np.full(17, 0.12)
        self.assertFalse(keypoints_usable(row, 0.3))


class CropTest(unittest.TestCase):
    def test_zoom_one_is_identity(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = crop_center(frame, 1.0)
        self.assertEqual(out.shape, (480, 640, 3))

    def test_zoom_two_halves_each_side(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = crop_center(frame, 2.0)
        self.assertEqual(out.shape, (240, 320, 3))


if __name__ == "__main__":
    unittest.main()
