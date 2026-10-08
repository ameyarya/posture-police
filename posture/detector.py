"""Webcam capture plus YOLO pose inference plus posture scoring.

Everything runs locally in this process. Frames are kept in memory and are
never uploaded anywhere; the only network use in the whole app is the
optional Ollama hook in posture/vlm.py.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

# COCO keypoint indices used by Ultralytics pose models.
NOSE = 0
L_SHOULDER = 5
R_SHOULDER = 6
L_HIP = 11
R_HIP = 12

# Heuristic thresholds for a frontal desk webcam. These are starting points
# meant to be tuned from the dashboard settings, not medical measurements.
HEAD_FORWARD_RATIO = 0.35  # nose offset from shoulder centre / shoulder width
HEAD_DROP_RATIO = 0.60  # nose-above-shoulders distance / shoulder width
SHOULDER_TILT_RATIO = 0.25  # shoulder height difference / shoulder width
HYSTERESIS_FRAMES = 3  # consecutive frames required to change posture state


@dataclass
class PostureResult:
    status: str  # good | slouch | lean | away | starting | error
    detail: str = ""
    conf: float = 0.0


def score_pose(kpts: np.ndarray) -> PostureResult:
    """Score one person's keypoints. Pure function, safe to unit test.

    kpts is a (17, 2) or (17, 3) array of COCO keypoints in pixels.
    """
    try:
        pts = np.asarray(kpts, dtype=float)
        nose = pts[NOSE, :2]
        lsh = pts[L_SHOULDER, :2]
        rsh = pts[R_SHOULDER, :2]
    except (IndexError, ValueError, TypeError):
        return PostureResult("away", "no usable keypoints")

    shoulder_width = float(np.linalg.norm(rsh - lsh))
    if shoulder_width < 1e-6:
        return PostureResult("away", "shoulders not visible")

    mid_shoulder = (lsh + rsh) / 2.0
    head_forward = abs(float(nose[0] - mid_shoulder[0])) / shoulder_width
    head_height = float(mid_shoulder[1] - nose[1]) / shoulder_width
    tilt = abs(float(lsh[1] - rsh[1])) / shoulder_width

    if head_height < HEAD_DROP_RATIO:
        return PostureResult(
            "slouch",
            f"head dropped toward desk (ratio {head_height:.2f})",
        )
    if head_forward > HEAD_FORWARD_RATIO or tilt > SHOULDER_TILT_RATIO:
        return PostureResult(
            "lean",
            f"leaning (offset {head_forward:.2f}, tilt {tilt:.2f})",
        )
    return PostureResult("good", "upright")


class Detector:
    """Owns the camera and the pose model on a background thread."""

    def __init__(
        self,
        model_name: str = "yolo26n-pose",
        fallback_model: str = "yolov8n-pose",
        conf: float = 0.5,
        inference_fps: float = 3.0,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.conf = conf
        self.inference_fps = inference_fps
        self.camera_index = camera_index
        self.width = width
        self.height = height

        self._lock = threading.Lock()
        self._result = PostureResult("starting", "warming up")
        self._pending_status = "starting"
        self._pending_count = 0
        self._jpeg: bytes | None = None
        self._model = None
        self._model_label = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    # -- public state ---------------------------------------------------
    @property
    def result(self) -> PostureResult:
        with self._lock:
            return self._result

    @property
    def model_label(self) -> str:
        with self._lock:
            return self._model_label

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    # -- internals ------------------------------------------------------
    def _load_model(self) -> None:
        from ultralytics import YOLO

        for name in (self.model_name, self.fallback_model):
            try:
                self._model = YOLO(name if name.endswith(".pt") else f"{name}.pt")
                with self._lock:
                    self._model_label = name
                return
            except Exception:
                continue
        with self._lock:
            self._result = PostureResult(
                "error", "could not load pose model (offline first run?)"
            )

    def _set_status(self, status: str, detail: str, conf: float) -> None:
        """Hysteresis: a new posture must persist before it is reported."""
        with self._lock:
            if status == self._result.status:
                self._pending_count = 0
                self._pending_status = status
                self._result = PostureResult(status, detail, conf)
                return
            if status != self._pending_status:
                self._pending_status = status
                self._pending_count = 1
                return
            self._pending_count += 1
            if self._pending_count >= HYSTERESIS_FRAMES:
                self._result = PostureResult(status, detail, conf)
                self._pending_count = 0

    def _loop(self) -> None:
        self._load_model()
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            with self._lock:
                self._result = PostureResult(
                    "error",
                    "cannot open webcam (grant camera permission and check index)",
                )
            return
        try:
            period = 1.0 / max(self.inference_fps, 0.5)
            next_run = 0.0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    self._set_status("error", "lost webcam frame", 0.0)
                    time.sleep(0.5)
                    continue
                now = time.monotonic()
                if self._model is not None and now >= next_run:
                    next_run = now + period
                    self._infer(frame)
                self._store_jpeg(frame)
        finally:
            cap.release()

    def _infer(self, frame: np.ndarray) -> None:
        try:
            res = self._model.predict(frame, conf=self.conf, verbose=False)[0]
        except Exception as exc:
            self._set_status("error", f"inference failed: {exc}", 0.0)
            return
        if res.keypoints is None or len(res.keypoints) == 0:
            self._set_status("away", "no person in frame", 0.0)
            return
        # Largest person wins the POC.
        boxes = res.boxes.xywh.cpu().numpy() if res.boxes is not None else None
        kpts = res.keypoints.xy.cpu().numpy()
        idx = 0
        if boxes is not None and len(boxes) == len(kpts):
            idx = int(np.argmax(boxes[:, 2] * boxes[:, 3]))
        confs = None
        if res.keypoints.conf is not None:
            confs = res.keypoints.conf.cpu().numpy()
        mean_conf = float(np.mean(confs[idx])) if confs is not None else 0.0
        scored = score_pose(kpts[idx])
        scored.conf = mean_conf
        self._set_status(scored.status, scored.detail, scored.conf)

    def _store_jpeg(self, frame: np.ndarray) -> None:
        small = frame
        if frame.shape[1] > 640:
            scale = 640.0 / frame.shape[1]
            small = cv2.resize(
                frame, (640, int(frame.shape[0] * scale))
            )
        with self._lock:
            status = self._result.status
        cv2.putText(
            small, status, (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
        )
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._lock:
                self._jpeg = bytes(buf)
