"""Unit tests for detection gating and digital zoom (no hardware needed)."""

import unittest

import numpy as np

from posture.detector import (
    crop_center,
    crop_window,
    draw_pose,
    keypoints_usable,
    pose_metrics,
    score_pose,
)


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


class MetricsTest(unittest.TestCase):
    def test_upright_numbers(self):
        m = pose_metrics(upright_kpts())
        self.assertAlmostEqual(m["head_height"], 1.5)
        self.assertAlmostEqual(m["head_forward"], 0.0)
        self.assertAlmostEqual(m["tilt"], 0.0)

    def test_custom_threshold_respected(self):
        k = upright_kpts()
        k[0] = [320, 250]  # head_height = 0.625
        self.assertEqual(score_pose(k).status, "good")
        self.assertEqual(score_pose(k, head_drop=0.7).status, "slouch")

    def test_overlay_draws_skeleton(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = draw_pose(frame.copy(), upright_kpts())
        self.assertEqual(out.shape, (480, 640, 3))
        self.assertGreater(int((out != 0).sum()), 0)


def upright_kpts() -> np.ndarray:
    k = np.zeros((17, 2))
    k[0] = [320, 180]
    k[5] = [280, 300]
    k[6] = [360, 300]
    k[11] = [285, 430]
    k[12] = [355, 430]
    return k


class CropTest(unittest.TestCase):
    def test_zoom_one_is_identity(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = crop_center(frame, 1.0)
        self.assertEqual(out.shape, (480, 640, 3))

    def test_zoom_two_halves_each_side(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = crop_center(frame, 2.0)
        self.assertEqual(out.shape, (240, 320, 3))

    def test_pan_right_keeps_right_edge(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, 90:] = 255  # bright strip at the right edge
        out = crop_window(frame, 2.0, pan_x=1.0)
        self.assertEqual(out.shape, (50, 50, 3))
        self.assertTrue(np.all(out[:, -10:] == 255))
        self.assertTrue(np.all(out[:, :10] == 0))

    def test_pan_clamps_inside_frame(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out = crop_window(frame, 2.0, pan_x=5.0, pan_y=-5.0)
        self.assertEqual(out.shape, (50, 50, 3))


if __name__ == "__main__":
    unittest.main()
