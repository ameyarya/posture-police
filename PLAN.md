## Goal
Build a local-only Mac POC for Posture Police: webcam posture checks, desk-time tracking, and 30-minute get-up reminders, served as a localhost dashboard. No images leave the Mac.

## Success Criteria
- `python` app opens Mac webcam locally and shows live preview on localhost.
- Sitting vs. away vs. slouching is inferred on-device at ~2–5 fps without cloud calls.
- Dashboard shows posture status, last-get-up time, and total desk time today.
- Continuous 30 min at desk triggers a visible plus audible reminder; settings allow changing the interval.
- No outbound network calls during normal operation; frames stay in memory unless the user saves an event snapshot.

## Approach
Reuse your mini-t pattern: fast Python detector in the hot loop, heavyweight local VLM only on occasional snapshots, localhost dashboard for control. Primary detection is a nano pose model, not per-frame Qwen. Qwen3-VL via Ollama is an optional second opinion, off by default in the POC.

Key decisions:
- Pose estimator over local Qwen Vision in the loop: YOLO pose nano runs at webcam rate on Apple Silicon; per-frame VLM at 2–4B+ is too slow and memory-heavy for continuous scoring.
- YOLO pose over MediaPipe-only: matches your mini-t YOLO-World tooling and gives person box plus shoulder, neck, and hip keypoints in one pass. MediaPipe Pose Landmarker stays a documented fallback.
- Flask localhost dashboard over headless script: you chose dashboard; it gives live view, stats, and reminder settings in one surface.
- Away equals no person for 30 s continuous; desk time equals present-time accumulation stored in SQLite. Simple, testable, matches your 30-min standard choice.

## Steps
1. Scaffold POC: `.venv`, `requirements.txt` with `opencv-python`, `ultralytics`, `flask`, `config.yaml` for camera index, sample rate, intervals, and ports.
2. Capture loop: OpenCV `VideoCapture(0)` reader thread, downscaled frames, FPS guard, macOS camera-permission error message.
3. Pose loop: Ultralytics `yolo26n-pose` or `yolov8n-pose` inference at 2–5 fps, keep largest person, extract nose, shoulders, and hips, compute forward-head plus slouch angles with hysteresis to avoid flicker.
4. Presence state machine: present, maybe-away, and away states; update last-get-up on present-to-away transition; accumulate desk seconds only while present.
5. Storage plus API: SQLite tables for sessions and events; Flask routes for MJPEG preview, `/api/status`, `/api/stats`, and `/api/settings`.
6. Dashboard plus reminders: single page with live view, status pill, last-up and desk-time cards, 30-minute countdown, test-reminder button, sound plus banner plus optional macOS `osascript` notification.
7. Optional VLM hook: snapshot endpoint that sends one JPEG to Ollama `qwen3-vl:4b` every N minutes for a textual posture note; failures never block pose scoring.
8. Docs plus ignore rules: README run steps, privacy note, and `.gitignore` for `.venv`, weights, and event snapshots.

## Validation Plan
- `python3 -m py_compile` on new modules plus `pip install -r requirements.txt` in a fresh venv.
- Manual: open dashboard, sit, slouch, leave frame 35 s; expect status flips, last-get-up updates, and 30-minute test reminder fires with sound and banner.
- Privacy check: run with network disabled after model download; expect full scoring and reminders to keep working.
- Highest-risk check is webcam plus pose-model throughput on your Mac; measure fps and lower input resolution or inference rate first if it lags.

## Risks / Open Questions
- macOS camera permission denial; mitigation is a clear settings prompt.
- Python 3.14 plus Ultralytics plus OpenCV wheel compatibility on ARM; pin versions after first install.
- Single-person assumption; multi-person frames use the largest box in the POC.
- Qwen hook is heavier at ~3.3 GB for the 4B tag; keep it optional.
- Open questions: None; intervals and thresholds live in settings for tuning.

## Sources
- https://docs.ultralytics.com/tasks/pose/
- https://docs.opencv.org/4.13.0/dd/d43/tutorial_py_video_display.html
- https://flask.palletsprojects.com/
- https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
- https://ollama.com/library/qwen3-vl
- https://github.com/ameyarya/ai-driven-mini-t
- https://raw.githubusercontent.com/ameyarya/ai-driven-mini-t/main/README.md
