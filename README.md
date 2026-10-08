# Posture Police
Posture Police helps you keep an eye on yourself while you work.

Local-only Mac POC: your webcam checks sitting posture on-device, tracks desk time and last get-up, and reminds you to stand about every 30 minutes. No images are uploaded to any cloud.

See [PLAN.md](/Users/am3yarya/Documents/Github/posture-police/PLAN.md) for the approved POC scope: YOLO pose scoring, Flask localhost dashboard, SQLite desk-time tracking, and an optional local Qwen3-VL note via Ollama.

Status: POC implemented, awaiting live webcam trial.

## Run the POC

From the project root:

    uv venv
    uv pip install -r requirements.txt
    source .venv/bin/activate
    python app.py

Then open http://127.0.0.1:5000 in a browser. Allow camera access when macOS asks.

The app prefers the NexiGo webcam automatically (`camera.prefer_name` in `config.yaml`, resolved against the system camera list). If macOS reports the camera as unauthorized, grant Camera access to your terminal in System Settings → Privacy & Security → Camera, then restart the app. You can override the camera index on the dashboard or with `camera.index` in `config.yaml`.

First launch downloads a ~6 MB nano pose model; afterwards everything runs offline. To check the logic without a camera:

    python -m unittest discover -s tests -v

Tune the reminder interval and away threshold on the dashboard or in `config.yaml`.
