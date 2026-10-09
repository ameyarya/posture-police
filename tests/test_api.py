"""API-level tests for settings, reminders, and status shape.

These drive the real Flask routes with the real Tracker. Run from the
project root with the project environment:

    source .venv/bin/activate
    python -m unittest discover -s tests -v
"""

import json
import time
import unittest

import app as appmod


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.client = appmod.app.test_client()
        with open("config.yaml", "rb") as fh:
            self._cfg_bytes = fh.read()
        self.addCleanup(self._restore)

    def _restore(self):
        with open("config.yaml", "wb") as fh:
            fh.write(self._cfg_bytes)
        fresh = appmod.load_config()
        appmod.cfg.clear()
        appmod.cfg.update(fresh)
        t = appmod.tracker
        t.state = "unknown"
        t.streak_start = None
        t.last_seen = None
        t.last_reminder = None
        d = appmod.detector
        d.th_drop = 0.6
        d.th_offset = 0.35
        d.th_tilt = 0.25
        d.min_conf = 0.3

    def test_status_shape(self):
        r = self.client.get("/api/status")
        self.assertEqual(r.status_code, 200)
        body = json.loads(r.data)
        for key in (
            "presence", "posture", "continuous_desk_seconds",
            "desk_today_seconds", "last_get_up", "reminder_due",
        ):
            self.assertIn(key, body)

    def test_settings_roundtrip(self):
        r = self.client.post("/api/settings", json={"desk_minutes": 12})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(appmod.cfg["reminders"]["desk_minutes"], 12)
        with open("config.yaml") as fh:
            self.assertIn("12", fh.read())

    def test_video_placeholder_is_jpeg(self):
        jpeg = appmod.placeholder_jpeg()
        self.assertTrue(jpeg.startswith(b"\xff\xd8"))

    def test_missing_config_points_to_project_root(self):
        with self.assertRaises(SystemExit) as ctx:
            appmod.load_config("/nonexistent/config.yaml")
        self.assertIn("project root", str(ctx.exception))

    def test_pick_port_skips_busy(self):
        import socket

        busy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        busy.bind(("127.0.0.1", 0))
        held = busy.getsockname()[1]
        try:
            self.assertNotEqual(
                appmod.pick_port("127.0.0.1", held, tries=3), held
            )
        finally:
            busy.close()

    def test_camera_override_needs_restart(self):
        r = self.client.post("/api/settings", json={"camera_index": 2})
        body = json.loads(r.data)
        self.assertTrue(body["restart_needed"])
        self.assertEqual(appmod.cfg["camera"]["index"], 2)
        self.assertEqual(appmod.cfg["camera"]["prefer_name"], "")

    def test_calibration_applies_live(self):
        r = self.client.post(
            "/api/settings",
            json={"head_drop_ratio": 0.75, "lean_offset_ratio": 0.5},
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(appmod.detector.th_drop, 0.75)
        self.assertEqual(appmod.detector.th_offset, 0.5)
        self.assertEqual(appmod.cfg["inference"]["head_drop_ratio"], 0.75)
        body = json.loads(self.client.get("/api/settings").data)
        self.assertEqual(body["head_drop_ratio"], 0.75)

    def test_cameras_endpoint_lists_devices(self):
        r = self.client.get("/api/cameras")
        self.assertEqual(r.status_code, 200)
        body = json.loads(r.data)
        self.assertIsInstance(body, list)
        self.assertGreater(len(body), 0)
        for cam in body:
            self.assertIn("index", cam)
            self.assertIn("label", cam)
            self.assertIn("working", cam)

    def test_cameras_result_is_cached(self):
        first = json.loads(self.client.get("/api/cameras").data)
        second = json.loads(self.client.get("/api/cameras").data)
        self.assertEqual(first, second)
        self.assertGreater(appmod._cam_cache["at"], 0)

    def test_restart_unavailable_off_dev_server(self):
        r = self.client.post("/api/restart")
        body = json.loads(r.data)
        self.assertIn("ok", body)
        self.assertFalse(body["ok"])

    def test_reminder_flow(self):
        self.client.post("/api/settings", json={"desk_minutes": 1})
        # Pretend the user sat down 61 seconds ago.
        appmod.tracker.update(True, now=time.time() - 61)
        body = json.loads(self.client.get("/api/status").data)
        self.assertTrue(body["reminder_due"])
        self.assertGreater(body["continuous_desk_seconds"], 60)

        r = self.client.post("/api/remind/ack")
        self.assertEqual(r.status_code, 200)
        body = json.loads(self.client.get("/api/status").data)
        self.assertFalse(body["reminder_due"])


if __name__ == "__main__":
    unittest.main()
