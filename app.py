"""Posture Police POC entry point.

Starts the local webcam detector on a background thread and serves the
localhost dashboard. Run from the project root:

    source .venv/bin/activate
    python app.py

Then open http://127.0.0.1:5000 in a browser.
"""

from __future__ import annotations

import subprocess
import threading
import time

import yaml
from flask import Flask, Response, jsonify, render_template, request

from posture import vlm
from posture.detector import Detector
from posture.tracker import Tracker

CONFIG_PATH = "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)


cfg = load_config()

app = Flask(__name__)

detector = Detector(
    model_name=cfg["inference"]["model"],
    fallback_model=cfg["inference"]["fallback_model"],
    conf=cfg["inference"]["conf"],
    inference_fps=cfg["inference"]["fps"],
    camera_index=cfg["camera"]["index"],
    width=cfg["camera"]["width"],
    height=cfg["camera"]["height"],
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


@app.route("/video_feed")
def video_feed():
    def frames():
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
        vlm_note=vlm_state["note"],
    )
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
    save_config(cfg)
    return jsonify({"ok": True})


@app.route("/api/remind/ack", methods=["POST"])
def api_remind_ack():
    tracker.mark_reminded()
    return jsonify({"ok": True})


if __name__ == "__main__":
    detector.start()
    threading.Thread(target=vision_loop, daemon=True).start()
    app.run(host=cfg["server"]["host"], port=cfg["server"]["port"])
