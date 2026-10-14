"""API-level tests for settings, reminders, and status shape.

These drive the real Flask routes with the real Tracker. Run from the
project root with the project environment:

    source .venv/bin/activate
    python -m unittest discover -s tests -v
"""

import json
import os
import tempfile
import time
import unittest

import app as appmod


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.client = appmod.app.test_client()
        with open("config.yaml", "rb") as fh:
            self._cfg_bytes = fh.read()
        # Point dashboard overrides at a scratch file; never touch the real one.
        tmpdir = tempfile.mkdtemp()
        self._real_settings_path = appmod.SETTINGS_PATH
        self._real_settings = dict(appmod.settings)
        appmod.SETTINGS_PATH = os.path.join(tmpdir, "settings.local.yaml")
        appmod.settings.clear()
        self.addCleanup(self._restore)

    def _restore(self):
        with open("config.yaml", "rb") as fh:
            changed = fh.read() != self._cfg_bytes
        if changed:
            with open("config.yaml", "wb") as fh:
                fh.write(self._cfg_bytes)
        if os.path.exists(appmod.SETTINGS_PATH):
            os.unlink(appmod.SETTINGS_PATH)
        appmod.SETTINGS_PATH = self._real_settings_path
        appmod.settings.clear()
        appmod.settings.update(self._real_settings)
        fresh = appmod.load_config()
        appmod.cfg.clear()
        appmod.cfg.update(fresh)
        t = appmod.tracker
        t.state = "unknown"
        t.streak_start = None
        t.banked_until = None
        t.last_seen = None
        t.last_reminder = None
        d = appmod.detector
        d.th_drop = 0.6
        d.th_offset = 0.35
        d.th_tilt = 0.25
        d.min_conf = 0.3
        appmod.posture_alert.after = float(
            fresh["reminders"].get("posture_alert_seconds", 60)
        )

    def test_status_shape(self):
        r = self.client.get("/api/status")
        self.assertEqual(r.status_code, 200)
        body = json.loads(r.data)
        for key in (
            "presence", "posture", "continuous_desk_seconds",
            "desk_today_seconds", "last_get_up", "reminder_due",
        ):
            self.assertIn(key, body)

    def test_posture_alert_setting_applies_live(self):
        r = self.client.post("/api/settings", json={"posture_alert_seconds": 90})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(appmod.posture_alert.after, 90)
        body = json.loads(self.client.get("/api/status").data)
        self.assertIn("posture_alert", body)
        self.assertIn("posture_bad_seconds", body)

    def test_refuses_other_machines(self):
        r = self.client.get("/api/status", environ_base={"REMOTE_ADDR": "192.168.1.20"})
        self.assertEqual(r.status_code, 403)

    def test_accepts_ipv4_mapped_loopback(self):
        r = self.client.get("/api/status", environ_base={"REMOTE_ADDR": "::ffff:127.0.0.1"})
        self.assertEqual(r.status_code, 200)

    def test_refuses_unknown_host_header(self):
        r = self.client.get("/api/status", headers={"Host": "evil.example:8080"})
        self.assertEqual(r.status_code, 403)

    def test_accepts_configured_name(self):
        appmod.cfg["server"]["name"] = "posture-cam.local"
        r = self.client.get("/api/status", headers={"Host": "posture-cam.local"})
        self.assertEqual(r.status_code, 200)
        r = self.client.get("/api/status", headers={"Host": "[::1]:80"})
        self.assertEqual(r.status_code, 200)

    def test_settings_roundtrip(self):
        r = self.client.post("/api/settings", json={"desk_minutes": 12})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(appmod.cfg["reminders"]["desk_minutes"], 12)
        with open(appmod.SETTINGS_PATH) as fh:
            self.assertIn("desk_minutes: 12", fh.read())
        with open("config.yaml", "rb") as fh:
            self.assertEqual(fh.read(), self._cfg_bytes)  # defaults untouched
        self.assertEqual(appmod.load_config()["reminders"]["desk_minutes"], 12)

    def test_overlay_toggle_applies_live(self):
        self.client.post("/api/settings", json={"preview_overlay": False})
        self.assertFalse(appmod.detector.preview_overlay)
        body = json.loads(self.client.get("/api/settings").data)
        self.assertFalse(body["preview_overlay"])
        self.client.post("/api/settings", json={"preview_overlay": True})
        self.assertTrue(appmod.detector.preview_overlay)

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
