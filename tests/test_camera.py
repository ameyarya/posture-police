"""Unit tests for camera discovery and selection (no hardware needed)."""

import unittest

from posture.camera import choose_camera, parse_avfoundation_devices

FFMPEG_SAMPLE = """\
[AVFoundation indev @ 0x787f01c140] AVFoundation video devices:
[AVFoundation indev @ 0x787f01c140] [0] NexiGo N60 FHD Webcam
[AVFoundation indev @ 0x787f01c140] [1] MacBook Air Camera
[AVFoundation indev @ 0x787f01c140] [2] MacBook Air Desk View Camera
[AVFoundation indev @ 0x787f01c140] AVFoundation audio devices:
[AVFoundation indev @ 0x787f01c140] [0] MacBook Air Microphone
"""


def opener_for(*working: int):
    return lambda idx: idx in working


class ParseTest(unittest.TestCase):
    def test_video_only(self):
        self.assertEqual(
            parse_avfoundation_devices(FFMPEG_SAMPLE),
            [
                (0, "NexiGo N60 FHD Webcam"),
                (1, "MacBook Air Camera"),
                (2, "MacBook Air Desk View Camera"),
            ],
        )

    def test_empty(self):
        self.assertEqual(parse_avfoundation_devices(""), [])


class ChooseTest(unittest.TestCase):
    DEVICES = [(0, "NexiGo N60 FHD Webcam"), (1, "MacBook Air Camera")]

    def test_preferred_match_wins(self):
        idx, label, _ = choose_camera(
            self.DEVICES, "nexigo", 1, opener_for(0, 1)
        )
        self.assertEqual((idx, label), (0, "NexiGo N60 FHD Webcam"))

    def test_preferred_shut_falls_back_to_configured(self):
        idx, _, _ = choose_camera(self.DEVICES, "nexigo", 1, opener_for(1))
        self.assertEqual(idx, 1)

    def test_no_preference_uses_configured(self):
        idx, label, _ = choose_camera(self.DEVICES, "", 1, opener_for(0, 1))
        self.assertEqual((idx, label), (1, "MacBook Air Camera"))

    def test_nothing_opens_keeps_configured(self):
        idx, _, reason = choose_camera(self.DEVICES, "nexigo", 0, opener_for())
        self.assertEqual(idx, 0)
        self.assertIn("unavailable", reason)


if __name__ == "__main__":
    unittest.main()
