"""Posture Police POC entry point.

Starts the local webcam detector on a background thread and serves the
localhost dashboard. Run from the project root:

    source .venv/bin/activate
    python app.py

Then open http://127.0.0.1:8080 in a browser.
"""

from __future__ import annotations

import ipaddress
import os
import signal
import subprocess
import threading
import time

# Ultralytics phones home with usage analytics (no images) unless offline.
# Must be set before ultralytics is imported. YOLO_OFFLINE=0 re-enables
# network access, e.g. to download missing model weights.
os.environ.setdefault("YOLO_OFFLINE", "1")

import yaml
from flask import Flask, Response, abort, jsonify, render_template, request

from posture import vlm
from posture.alerts import PostureAlert
from posture import camera as camselect
from posture.detector import Detector
from posture.tracker import Tracker

CONFIG_PATH = "config.yaml"
# Dashboard edits land here, never in config.yaml, so saving settings
# cannot clobber defaults or keys the running server does not know about.
SETTINGS_PATH = "settings.local.yaml"


def load_settings() -> dict:
    if not os.path.exists(SETTINGS_PATH):
        return {}
    with open(SETTINGS_PATH) as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: str = CONFIG_PATH) -> dict:
    """config.yaml defaults with dashboard overrides layered on top."""
    if not os.path.exists(path):
        raise SystemExit(
            f"error: {path} not found - "
            "run from the project root (the folder containing app.py), "
            "e.g. `./run.sh`"
        )
    with open(path) as fh:
        merged = yaml.safe_load(fh)
    for section, values in load_settings().items():
        if isinstance(values, dict):
            merged.setdefault(section, {}).update(values)
    return merged


def set_setting(section: str, key: str, value) -> None:
    """Apply a dashboard change to the live config and the overrides."""
    cfg.setdefault(section, {})[key] = value
    settings.setdefault(section, {})[key] = value


def save_settings() -> None:
    tmp = SETTINGS_PATH + ".tmp"
    with open(tmp, "w") as fh:
        yaml.safe_dump(settings, fh, sort_keys=False)
    os.replace(tmp, SETTINGS_PATH)


settings = load_settings()
cfg = load_config()

app = Flask(__name__)

cam_index, cam_label, cam_reason = camselect.resolve_camera(
    prefer_name=cfg["camera"].get("prefer_name", ""),
    configured_index=cfg["camera"]["index"],
)
print(f"camera: {cam_label} (index {cam_index}) - {cam_reason}", flush=True)

detector = Detector(
    model_name=cfg["inference"]["model"],
    fallback_model=cfg["inference"]["fallback_model"],
    conf=cfg["inference"]["conf"],
    inference_fps=cfg["inference"]["fps"],
    camera_index=cam_index,
    camera_label=cam_label,
    width=cfg["camera"]["width"],
    height=cfg["camera"]["height"],
    min_conf=cfg["inference"]["min_conf"],
    zoom=cfg["camera"]["zoom"],
    pan_x=cfg["camera"].get("pan_x", 0.0),
    pan_y=cfg["camera"].get("pan_y", 0.0),
    head_drop=cfg["inference"].get("head_drop_ratio", 0.6),
    lean_offset=cfg["inference"].get("lean_offset_ratio", 0.35),
    tilt_max=cfg["inference"].get("tilt_ratio", 0.25),
    preview_fps=cfg["camera"].get("preview_fps", 10),
    preview_follow=cfg["camera"].get("preview_follow", True),
    preview_mirror=cfg["camera"].get("preview_mirror", True),
    preview_overlay=cfg["camera"].get("preview_overlay", True),
    preview_enhance=cfg["camera"].get("preview_enhance", True),
)
tracker = Tracker(
    db_path=cfg["storage"]["db_path"],
    away_after_seconds=cfg["presence"]["away_after_seconds"],
)
posture_alert = PostureAlert(
    after_seconds=float(cfg["reminders"].get("posture_alert_seconds", 60)),
    repeat_seconds=float(cfg["reminders"].get("posture_repeat_minutes", 5)) * 60,
)
vlm_state = {"note": None, "ts": None}
mac_notified_streak: float | None = None


def mac_notify(title: str, message: str) -> None:
    """Best-effort macOS notification. Never raises."""
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{message}" with title "{title}"'],
            check=False,
            timeout=5,
        )
    except Exception:
        pass


CHECKPOINT_SECONDS = 60.0


def shutdown(signum, _frame) -> None:
    """Save the running desk stretch before a restart or Ctrl+C."""
    tracker.checkpoint(final=True)
    raise SystemExit(0)


def vision_loop() -> None:
    global mac_notified_streak
    vlm_cfg = cfg.get("vlm", {})
    last_vlm = 0.0
    last_checkpoint = time.time()
    last_tick = last_checkpoint
    while True:
        res = detector.result
        tracker.update(detector.face_visible)
        now = time.time()
        if tracker.state == "present":
            tracker.add_posture(res.status, min(now - last_tick, 5.0), now)
        last_tick = now
        if now - last_checkpoint >= CHECKPOINT_SECONDS:
            last_checkpoint = now
            tracker.checkpoint(now)

        if posture_alert.update(res.status, tracker.state != "away", now):
            what = "slouching" if res.status == "slouch" else "leaning"
            mins = posture_alert.bad_seconds(now) / 60
            mac_notify(
                "Posture Police",
                f"You've been {what} for {mins:.0f} min - sit up straight!",
            )

        desk_minutes = float(cfg["reminders"]["desk_minutes"])
        if tracker.reminder_due(desk_minutes, now):
            streak = tracker.streak_start
            if mac_notified_streak != streak:
                mac_notified_streak = streak
                mac_notify(
                    "Posture Police",
                    f"{desk_minutes:g} min at your desk - time to get up!",
                )

        if vlm_cfg.get("enabled") and res.status in ("good", "slouch", "lean"):
            every = float(vlm_cfg.get("every_minutes", 10)) * 60
            jpeg = detector.latest_jpeg()
            if jpeg and now - last_vlm > every:
                last_vlm = now
                note = vlm.ask_vlm(jpeg, model=vlm_cfg.get("model", "qwen3-vl:4b"))
                if note:
                    vlm_state.update(note=note, ts=now)
        time.sleep(0.5)


LOCAL_HOSTNAMES = {"127.0.0.1", "localhost", "::1"}


def allowed_hosts() -> set[str]:
    name = cfg["server"].get("name")
    return LOCAL_HOSTNAMES | ({name.lower()} if name else set())


def request_hostname(host_header: str) -> str:
    """Host header without its port: 'a.local:80' -> 'a.local', '[::1]:80' -> '::1'."""
    host = host_header.strip().lower()
    if host.startswith("["):
        return host[1:].split("]", 1)[0]
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


@app.before_request
def local_only():
    """Serve this Mac only, under a name we expect.

    The server may bind every interface (port 80 needs that without
    root), so refuse other machines. The Host check stops DNS-rebinding
    pages in your browser from reading the camera feed.
    """
    try:
        ip = ipaddress.ip_address(request.remote_addr or "")
    except ValueError:
        abort(403)
    if getattr(ip, "ipv4_mapped", None):
        ip = ip.ipv4_mapped
    if not ip.is_loopback:
        abort(403)
    if request_hostname(request.host) not in allowed_hosts():
        abort(403)


@app.route("/")
def index():
    return render_template(
        "index.html",
        desk_minutes=cfg["reminders"]["desk_minutes"],
        away_after=cfg["presence"]["away_after_seconds"],
        posture_alert=posture_alert.after,
    )


def placeholder_jpeg() -> bytes | None:
    """First-frame stand-in so the stream responds before the camera does."""
    import cv2
    import numpy as np

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        img, "waiting for camera...", (140, 240),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
    )
    ok, buf = cv2.imencode(".jpg", img)
    return bytes(buf) if ok else None


@app.route("/video_feed")
def video_feed():
    def frames():
        first = placeholder_jpeg()
        if first:
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + first + b"\r\n"
            )
        last_seq = -1
        period = 1.0 / max(detector.preview_fps, 1.0)
        while True:
            seq, jpeg = detector.latest_preview()
            if jpeg and seq != last_seq:
                last_seq = seq
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
            time.sleep(period / 2)

    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    res = detector.result
    out = tracker.to_dict(float(cfg["reminders"]["desk_minutes"]))
    out.update(
        posture=res.status,
        posture_detail=res.detail,
        posture_conf=res.conf,
        model=detector.model_label,
        camera={"index": detector.camera_index, "label": detector.camera_label},
        vlm_note=vlm_state["note"],
        desk_minutes=float(cfg["reminders"]["desk_minutes"]),
        away_after_seconds=float(cfg["presence"]["away_after_seconds"]),
        posture_bad_seconds=posture_alert.bad_seconds(time.time()),
        posture_alert=posture_alert.active(time.time()),
        metrics=detector.metrics,
        thresholds={
            "head_drop_ratio": detector.th_drop,
            "lean_offset_ratio": detector.th_offset,
            "tilt_ratio": detector.th_tilt,
            "min_conf": detector.min_conf,
        },
    )
    return jsonify(out)


@app.route("/api/history")
def api_history():
    days = max(1, min(31, request.args.get("days", 7, type=int)))
    return jsonify(tracker.history(days))


@app.route("/api/settings")
def api_settings_get():
    return jsonify(
        {
            "desk_minutes": cfg["reminders"]["desk_minutes"],
            "away_after_seconds": cfg["presence"]["away_after_seconds"],
            "posture_alert_seconds": posture_alert.after,
            "preview_overlay": detector.preview_overlay,
            "camera_index": cfg["camera"]["index"],
            "head_drop_ratio": detector.th_drop,
            "lean_offset_ratio": detector.th_offset,
            "tilt_ratio": detector.th_tilt,
            "min_conf": detector.min_conf,
        }
    )


_cam_cache: dict = {"at": 0.0, "data": None}
CAM_CACHE_TTL = 120.0


@app.route("/api/cameras")
def api_cameras():
    now = time.time()
    if _cam_cache["data"] is not None and now - _cam_cache["at"] < CAM_CACHE_TTL:
        return jsonify(_cam_cache["data"])
    devices = camselect.list_system_cameras()
    if not devices:
        devices = [(i, f"camera {i}") for i in range(4)]
    out = []
    for idx, name in devices:
        if idx == detector.camera_index:
            working = True
        else:
            try:
                working = camselect.can_open(idx)
            except Exception:
                working = False
        out.append({"index": idx, "label": name, "working": working})
    _cam_cache.update(at=now, data=out)
    return jsonify(out)


@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.get_json(force=True)
    if "desk_minutes" in data:
        set_setting("reminders", "desk_minutes", max(1, float(data["desk_minutes"])))
    if "away_after_seconds" in data:
        val = max(5, float(data["away_after_seconds"]))
        set_setting("presence", "away_after_seconds", val)
        tracker.away_after = val
    if "posture_alert_seconds" in data:
        val = max(0.0, float(data["posture_alert_seconds"]))  # 0 turns it off
        set_setting("reminders", "posture_alert_seconds", val)
        posture_alert.after = val
    if "preview_overlay" in data:
        val = bool(data["preview_overlay"])
        set_setting("camera", "preview_overlay", val)
        detector.preview_overlay = val
    if "camera_index" in data:
        # Saved for the next start; the running detector keeps its camera.
        set_setting("camera", "index", int(data["camera_index"]))
        set_setting("camera", "prefer_name", "")
    for key, lo, hi in (
        ("head_drop_ratio", 0.2, 1.2),
        ("lean_offset_ratio", 0.1, 0.8),
        ("tilt_ratio", 0.05, 0.6),
        ("min_conf", 0.05, 0.9),
    ):
        if key in data:
            set_setting("inference", key, min(hi, max(lo, float(data[key]))))
    # Calibration applies live; no restart needed.
    detector.th_drop = cfg["inference"].get("head_drop_ratio", 0.6)
    detector.th_offset = cfg["inference"].get("lean_offset_ratio", 0.35)
    detector.th_tilt = cfg["inference"].get("tilt_ratio", 0.25)
    detector.min_conf = cfg["inference"].get("min_conf", 0.3)
    save_settings()
    return jsonify({"ok": True, "restart_needed": "camera_index" in data})


@app.route("/api/remind/ack", methods=["POST"])
def api_remind_ack():
    tracker.mark_reminded()
    return jsonify({"ok": True})


ALLOW_RESTART = False


@app.route("/api/restart", methods=["POST"])
def api_restart():
    """Terminate the server; run.sh respawns it. Refuses off-server."""
    if not ALLOW_RESTART:
        return jsonify({"ok": False, "reason": "restart only works on a started server"})

    def _die():
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_die, daemon=True).start()
    return jsonify({"ok": True})


def pick_port(host: str, base: int, tries: int = 10) -> int:
    """First free TCP port from base upward. Never guesses in the dark."""
    import socket

    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    for port in range(base, base + tries):
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            # Match werkzeug: lingering TIME_WAIT sockets from a restarted
            # server must not push us onto the next port.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise OSError(f"no free port in {base}..{base + tries - 1}")


if __name__ == "__main__":
    ALLOW_RESTART = True
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    detector.start()
    threading.Thread(target=vision_loop, daemon=True).start()
    port = pick_port(cfg["server"]["host"], int(cfg["server"]["port"]))
    name = cfg["server"].get("name") or "127.0.0.1"
    url = f"http://{name}/" if port == 80 else f"http://{name}:{port}/"
    print(f"Posture Police running - open {url} in a browser", flush=True)
    app.run(host=cfg["server"]["host"], port=port)
