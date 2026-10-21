"""Unit tests for the MediaPipe pose backend (no model or camera needed)."""

import os
import tempfile
import unittest
from collections import namedtuple
from unittest import mock

import numpy as np

from posture.mp_pose import ensure_model, landmark_confidence, map_landmarks, pose_box

LM = namedtuple("LM", "x y visibility presence")


def landmarks(overrides):
    lms = [LM(0.0, 0.0, 0.0, 0.0) for _ in range(33)]
    for idx, vals in overrides.items():
        lms[idx] = LM(*vals)
    return lms


class MapTest(unittest.TestCase):
    def test_face_and_shoulders_reach_coco_indices(self):
        lms = landmarks({
            0: (0.5, 0.4, 1.0, 1.0),     # nose -> 0
            2: (0.52, 0.38, 1.0, 1.0),   # left eye -> 1
            5: (0.48, 0.38, 1.0, 1.0),   # right eye -> 2
            7: (0.55, 0.38, 1.0, 1.0),   # left ear -> 3
            8: (0.45, 0.38, 1.0, 1.0),   # right ear -> 4
            11: (0.6, 0.7, 0.9, 0.8),    # left shoulder -> 5
            12: (0.4, 0.7, 0.9, 0.8),    # right shoulder -> 6
            23: (0.6, 1.2, 0.1, 0.01),   # left hip (out of frame) -> 11
        })
        kpts, conf = map_landmarks(lms, 640, 480)
        self.assertEqual(kpts.shape, (17, 2))
        self.assertEqual(conf.shape, (17,))
        np.testing.assert_allclose(kpts[0], [320, 192])
        np.testing.assert_allclose(kpts[5], [384, 336])
        self.assertAlmostEqual(conf[0], 1.0)
        self.assertAlmostEqual(conf[5], 0.72)  # presence * visibility
        self.assertLess(conf[11], 0.01)  # hips out of frame score ~0
        self.assertEqual(conf[9], 0.0)  # unmapped wrist stays zero

    def test_short_or_broken_landmarks_are_safe(self):
        kpts, conf = map_landmarks([LM(0.5, 0.5, 1.0, 1.0)] * 2, 640, 480)
        self.assertEqual(conf.sum(), 1.0)  # only nose (index 0) mapped
        kpts, conf = map_landmarks(None, 640, 480)
        self.assertEqual(conf.sum(), 0.0)
        kpts, conf = map_landmarks([object()] * 33, 640, 480)
        self.assertEqual(conf.sum(), 0.0)
        self.assertEqual(landmark_confidence(object()), 0.0)


class BoxTest(unittest.TestCase):
    def test_box_covers_confident_points(self):
        kpts = np.zeros((17, 2))
        kpts[0] = [320, 190]
        kpts[5] = [280, 300]
        kpts[6] = [360, 300]
        conf = np.zeros(17)
        conf[[0, 5, 6]] = 0.9
        box = pose_box(kpts, conf)
        self.assertIsNotNone(box)
        xc, yc, w, h = box
        self.assertAlmostEqual(xc, 320)
        self.assertGreater(w, 80)  # spans the shoulders plus padding

    def test_too_few_points_is_no_box(self):
        kpts = np.zeros((17, 2))
        kpts[0] = [320, 190]
        conf = np.zeros(17)
        conf[0] = 0.9
        self.assertIsNone(pose_box(kpts, conf))


class EnsureModelTest(unittest.TestCase):
    def test_existing_file_needs_no_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pose.task")
            with open(path, "wb") as fh:
                fh.write(b"fake")
            with mock.patch("urllib.request.urlopen") as urlopen:
                self.assertEqual(ensure_model(path), path)
                urlopen.assert_not_called()

    def test_failed_download_raises_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pose.task")
            with mock.patch(
                "urllib.request.urlopen", side_effect=OSError("offline")
            ):
                with self.assertRaises(RuntimeError):
                    ensure_model(path)
            self.assertFalse(os.path.exists(path))
            self.assertFalse(os.path.exists(path + ".tmp"))


if __name__ == "__main__":
    unittest.main()
