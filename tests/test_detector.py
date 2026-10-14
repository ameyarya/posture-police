"""Unit tests for detection gating and digital zoom (no hardware needed)."""

import unittest

import numpy as np

from posture.detector import (
    crop_center,
    crop_window,
    draw_pose,
    brighten,
    face_visible,
    key_confidence,
    mean_luma,
    follow_window,
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


class OverlayConfidenceTest(unittest.TestCase):
    def test_low_confidence_points_are_not_drawn(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        kpts = np.zeros((17, 2))
        kpts[:] = [50, 50]
        conf = np.zeros(17)  # nothing trusted
        draw_pose(frame, kpts, None, conf, min_conf=0.4)
        self.assertEqual(int(frame.sum()), 0)
        conf[0] = 0.9
        draw_pose(frame, kpts, None, conf, min_conf=0.4)
        self.assertGreater(int(frame.sum()), 0)


class LowLightTest(unittest.TestCase):
    def test_brighten_lifts_dark_frames(self):
        rng = np.random.default_rng(0)
        dark = rng.integers(0, 40, (120, 160, 3), dtype=np.uint8)
        self.assertGreater(mean_luma(brighten(dark)), mean_luma(dark) + 10)

    def test_mean_luma_uses_box_region(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[40:60, 40:60] = 200
        self.assertAlmostEqual(mean_luma(frame, np.array([50, 50, 20, 20])), 200, delta=1)
        self.assertLess(mean_luma(frame), 20)

    def test_key_confidence_ignores_legs(self):
        row = np.zeros(17)
        row[[0, 5, 6]] = 0.9  # nose and shoulders seen, legs never are
        self.assertAlmostEqual(key_confidence(row), 0.9)
        self.assertEqual(key_confidence(None), 0.0)


class FaceVisibleTest(unittest.TestCase):
    def test_any_confident_face_point_counts(self):
        row = np.zeros(17)
        row[3] = 0.6  # left ear only, e.g. head turned
        self.assertTrue(face_visible(row))

    def test_body_without_face_is_not_present(self):
        row = np.full(17, 0.9)
        row[:5] = 0.1
        self.assertFalse(face_visible(row))

    def test_no_confidences(self):
        self.assertFalse(face_visible(None))


class FollowWindowTest(unittest.TestCase):
    def test_centres_on_subject(self):
        self.assertEqual(follow_window(1280, 720, 640), (160, 1120))
        self.assertEqual(follow_window(1280, 720, 700), (220, 1180))

    def test_clamps_at_edges(self):
        self.assertEqual(follow_window(1280, 720, 1250), (320, 1280))
        self.assertEqual(follow_window(1280, 720, 10), (0, 960))

    def test_narrow_frame_is_whole(self):
        self.assertEqual(follow_window(640, 480, 500), (0, 640))


if __name__ == "__main__":
    unittest.main()
