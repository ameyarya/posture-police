# Posture Police
Posture Police helps you keep an eye on yourself while you work.

A local-only Mac app: your webcam checks your sitting posture on-device, tracks desk time and when you last got up, nudges you after a minute of slouching, and reminds you to stand every 30 minutes. Video never leaves your Mac.

See [PLAN.md](PLAN.md) for the original scope and design decisions.

## Requirements

- macOS (Apple Silicon recommended) with a webcam
- [uv](https://docs.astral.sh/uv/) — `brew install uv`. It installs the right Python (3.12, pinned in `.python-version`) for you.

## Quick start

    git clone https://github.com/ameyarya/posture-police.git
    cd posture-police
    uv venv
    uv pip install -r requirements.txt
    ./run.sh

Open http://127.0.0.1:8080 and allow camera access when macOS asks. `run.sh` restarts the server if it stops; press Ctrl+C twice to quit.

The first run downloads the ~7.5 MB YOLO pose model (`yolo26n-pose.pt`); after that everything works offline.

Without uv: create a Python 3.12 virtualenv yourself (`python3.12 -m venv .venv`), then `.venv/bin/pip install -r requirements.txt` and `./run.sh`.

## Setting it up for you

- **Camera.** The app looks for a camera whose name contains `camera.prefer_name` in `config.yaml` (the author's NexiGo) and otherwise falls back to `camera.index`. Pick your camera in the dashboard under Settings and restart.
- **Camera permission.** If the camera won't open, allow your terminal app in System Settings → Privacy & Security → Camera, then restart.
- **Seating.** Sit facing the camera, about an arm's length away, with your head and both shoulders in view.
- **Calibration.** The default thresholds are tuned for the author's desk. Open *Calibrate posture* while sitting upright and move each slider so the live values pass (green).
- **Notifications.** Click *Enable break notifications* on the dashboard. macOS notifications come from Script Editor; allow them in System Settings → Notifications if they don't appear.

Settings you change on the dashboard are saved to `settings.local.yaml` (not committed), layered on top of the defaults in `config.yaml`.

## Privacy

- Frames stay in memory; nothing is written to disk or uploaded.
- The dashboard only listens on `127.0.0.1`, so other devices can't reach it.
- Ultralytics runs with `YOLO_OFFLINE=1`, which turns off its anonymous usage analytics.
- The optional Qwen3-VL note (`vlm.enabled` in `config.yaml`, off by default) talks only to a local Ollama.

## Tests

    .venv/bin/python -m unittest discover -s tests -v

No camera needed.

## Layout

| Path | What it does |
| --- | --- |
| `app.py` | Flask server, settings, reminder and alert loop |
| `posture/detector.py` | Camera capture, YOLO pose inference, posture scoring, preview overlay |
| `posture/tracker.py` | Presence state machine, desk time in SQLite (`posture.db`) |
| `posture/alerts.py` | Bad-posture alert timing |
| `posture/camera.py` | Camera discovery and selection |
| `posture/vlm.py` | Optional local Ollama hook |
| `templates/index.html` | Dashboard |
