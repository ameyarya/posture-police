"""Posture Police POC entry point.

Starts the local webcam detector on a background thread and serves the
localhost dashboard. Run from the project root:

    source .venv/bin/activate
    python app.py

Then open http://127.0.0.1:5000 in a browser.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time

import yaml
from flask import Flask, Response, jsonify, render_template, request

from posture import vlm
from posture import camera as camselect
from posture.detector import Detector
from posture.tracker import Tracker

CONFIG_PATH = "config.yaml"


def load_config(path: str = CONFIG_PATH) -> dict:
    if not os.path.exists(path):
        raise SystemExit(
            f"error: {path} not found - "
            "run from the project root (the folder containing app.py), "
            "e.g. `./run.sh`"
        )
    with open(path) as fh:
        return yaml.safe_load(fh)


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)


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
)
tracker = Tracker(
    db_path=cfg["storage"]["db_path"],
    away_after_seconds=cfg["presence"]["away_after_seconds"],
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


def vision_loop() -> None:
    global mac_notified_streak
    vlm_cfg = cfg.get("vlm", {})
    last_vlm = 0.0
    while True:
        res = detector.result
        present = res.status in ("good", "slouch", "lean")
        tracker.update(present)
        now = time.time()

        desk_minutes = float(cfg["reminders"]["desk_minutes"])
        if tracker.reminder_due(desk_minutes, now):
            streak = tracker.streak_start
            if mac_notified_streak != streak:
                mac_notified_streak = streak
                mac_notify(
                    "Posture Police",
                    f"{desk_minutes:g} min at your desk - time to get up!",
                )

        if vlm_cfg.get("enabled") and res.status != "away":
            every = float(vlm_cfg.get("every_minutes", 10)) * 60
            jpeg = detector.latest_jpeg()
            if jpeg and now - last_vlm > every:
                last_vlm = now
                note = vlm.ask_vlm(jpeg, model=vlm_cfg.get("model", "qwen3-vl:4b"))
                if note:
                    vlm_state.update(note=note, ts=now)
        time.sleep(0.5)


@app.route("/")
def index():
    return render_template(
        "index.html",
        desk_minutes=cfg["reminders"]["desk_minutes"],
        away_after=cfg["presence"]["away_after_seconds"],
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
        while True:
            jpeg = detector.latest_jpeg()
            if jpeg:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
            time.sleep(0.2)

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
        metrics=detector.metrics,
        thresholds={
            "head_drop_ratio": detector.th_drop,
            "lean_offset_ratio": detector.th_offset,
            "tilt_ratio": detector.th_tilt,
            "min_conf": detector.min_conf,
        },
    )
    return jsonify(out)


@app.route("/api/settings")
def api_settings_get():
    return jsonify(
        {
            "desk_minutes": cfg["reminders"]["desk_minutes"],
            "away_after_seconds": cfg["presence"]["away_after_seconds"],
            "camera_index": cfg["camera"]["index"],
            "head_drop_ratio": detector.th_drop,
            "lean_offset_ratio": detector.th_offset,
            "tilt_ratio": detector.th_tilt,
            "min_conf": detector.min_conf,
        }
    )


@app.route("/api/cameras")
def api_cameras():
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
    return jsonify(out)


@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.get_json(force=True)
    if "desk_minutes" in data:
        cfg["reminders"]["desk_minutes"] = max(1, float(data["desk_minutes"]))
    if "away_after_seconds" in data:
        val = max(5, float(data["away_after_seconds"]))
        cfg["presence"]["away_after_seconds"] = val
        tracker.away_after = val
    if "camera_index" in data:
        # Saved for the next start; the running detector keeps its camera.
        cfg["camera"]["index"] = int(data["camera_index"])
        cfg["camera"]["prefer_name"] = ""
    for key, lo, hi in (
        ("head_drop_ratio", 0.2, 1.2),
        ("lean_offset_ratio", 0.1, 0.8),
        ("tilt_ratio", 0.05, 0.6),
        ("min_conf", 0.05, 0.9),
    ):
        if key in data:
            cfg["inference"][key] = min(hi, max(lo, float(data[key])))
    # Calibration applies live; no restart needed.
    detector.th_drop = cfg["inference"].get("head_drop_ratio", 0.6)
    detector.th_offset = cfg["inference"].get("lean_offset_ratio", 0.35)
    detector.th_tilt = cfg["inference"].get("tilt_ratio", 0.25)
    detector.min_conf = cfg["inference"].get("min_conf", 0.3)
    save_config(cfg)
    return jsonify({"ok": True, "restart_needed": "camera_index" in data})


@app.route("/api/remind/ack", methods=["POST"])
def api_remind_ack():
    tracker.mark_reminded()
    return jsonify({"ok": True})


def pick_port(host: str, base: int, tries: int = 10) -> int:
    """First free TCP port from base upward. Never guesses in the dark."""
    import socket

    for port in range(base, base + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise OSError(f"no free port in {base}..{base + tries - 1}")


if __name__ == "__main__":
    detector.start()
    threading.Thread(target=vision_loop, daemon=True).start()
    port = pick_port(cfg["server"]["host"], int(cfg["server"]["port"]))
    print(
        f"Posture Police running - open "
        f"http://{cfg['server']['host']}:{port} in a browser",
        flush=True,
    )
    app.run(host=cfg["server"]["host"], port=port)
