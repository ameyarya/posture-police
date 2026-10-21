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

The first run downloads the ~6 MB MediaPipe pose model (`pose_landmarker_lite.task`); after that everything works offline.

Without uv: create a Python 3.12 virtualenv yourself (`python3.12 -m venv .venv`), then `.venv/bin/pip install -r requirements.txt` and `./run.sh`.

## Setting it up for you

- **Camera.** The app looks for a camera whose name contains `camera.prefer_name` in `config.yaml` (the author's NexiGo) and otherwise falls back to `camera.index`. Pick your camera in the dashboard under Settings and restart.
- **Camera permission.** If the camera won't open, allow your terminal app in System Settings → Privacy & Security → Camera, then restart.
- **Seating.** Sit facing the camera, about an arm's length away, with your head and both shoulders in view.
- **Calibration.** The default thresholds are tuned for the author's desk. Open *Calibrate posture* while sitting upright and move each slider so the live values pass (green).
- **Notifications.** Click *Enable break notifications* on the dashboard. macOS notifications come from Script Editor; allow them in System Settings → Notifications if they don't appear.

Settings you change on the dashboard are saved to `settings.local.yaml` (not committed), layered on top of the defaults in `config.yaml`.

### Optional: a friendly URL like http://posture-cam.local/

1. Point the name at your Mac (local only, needs your password once):

       sudo sh -c 'printf "127.0.0.1 posture-cam.local\n::1 posture-cam.local\n" >> /etc/hosts'
       sudo killall -HUP mDNSResponder

2. Add to `settings.local.yaml`:

       server:
         host: "::"
         port: 80
         name: posture-cam.local

3. Restart. If macOS asks whether Python may accept incoming connections, choose **Deny**: that only blocks other devices.

Port 80 can only be opened without root on all interfaces, so the server refuses any request that isn't from this Mac or that uses a hostname other than the configured one.

## Privacy

- Frames stay in memory; nothing is written to disk or uploaded.
- The dashboard only answers requests from this Mac (other devices get 403), and only under expected hostnames, which blocks DNS-rebinding pages from reading the camera feed.
- MediaPipe runs fully offline with no usage analytics; the pose model is fetched once and then never touches the network.
- The optional Qwen3-VL note (`vlm.enabled` in `config.yaml`, off by default) talks only to a local Ollama.

## Tests

    .venv/bin/python -m unittest discover -s tests -v

No camera needed.

## Layout

| Path | What it does |
| --- | --- |
| `app.py` | Flask server, settings, reminder and alert loop |
| `posture/detector.py` | Camera capture, pose inference, posture scoring, preview overlay |
| `posture/mp_pose.py` | MediaPipe backend: model fetch, landmark mapping |
| `posture/tracker.py` | Presence state machine, desk time in SQLite (`posture.db`) |
| `posture/alerts.py` | Bad-posture alert timing |
| `posture/camera.py` | Camera discovery and selection |
| `posture/vlm.py` | Optional local Ollama hook |
| `templates/index.html` | Dashboard |
