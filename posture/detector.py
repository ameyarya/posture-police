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


def pose_metrics(kpts: np.ndarray) -> dict | None:
    """Geometry ratios for one person, or None when unusable.

    Pure function. kpts is a (17, 2) or (17, 3) array of COCO keypoints.
    """
    try:
        pts = np.asarray(kpts, dtype=float)
        nose = pts[NOSE, :2]
        lsh = pts[L_SHOULDER, :2]
        rsh = pts[R_SHOULDER, :2]
    except (IndexError, ValueError, TypeError):
        return None

    shoulder_width = float(np.linalg.norm(rsh - lsh))
    if shoulder_width < 1e-6:
        return None

    mid_shoulder = (lsh + rsh) / 2.0
    return {
        "head_height": float(mid_shoulder[1] - nose[1]) / shoulder_width,
        "head_forward": abs(float(nose[0] - mid_shoulder[0])) / shoulder_width,
        "tilt": abs(float(lsh[1] - rsh[1])) / shoulder_width,
        "shoulder_width": shoulder_width,
    }


def score_pose(
    kpts: np.ndarray,
    head_drop: float = HEAD_DROP_RATIO,
    lean_offset: float = HEAD_FORWARD_RATIO,
    tilt_max: float = SHOULDER_TILT_RATIO,
) -> PostureResult:
    """Score one person's keypoints. Pure function, safe to unit test."""
    m = pose_metrics(kpts)
    if m is None:
        return PostureResult("away", "no usable keypoints")

    if m["head_height"] < head_drop:
        return PostureResult(
            "slouch",
            f"head dropped toward desk (ratio {m['head_height']:.2f})",
        )
    if m["head_forward"] > lean_offset or m["tilt"] > tilt_max:
        return PostureResult(
            "lean",
            f"leaning (offset {m['head_forward']:.2f}, tilt {m['tilt']:.2f})",
        )
    return PostureResult("good", "upright")


# Limb pairs (COCO indices) drawn on the dashboard video overlay.
SKELETON_PAIRS = (
    (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def draw_pose(
    frame: np.ndarray,
    kpts_xy: np.ndarray,
    box_xywh: np.ndarray | None = None,
) -> np.ndarray:
    """Draw keypoints, limbs, and box on a frame. Returns the frame."""
    pts = np.asarray(kpts_xy, dtype=float)
    h, w = frame.shape[:2]

    def inside(x: float, y: float) -> bool:
        return 0 <= x < w and 0 <= y < h

    for a, b in SKELETON_PAIRS:
        xa, ya = pts[a, :2]
        xb, yb = pts[b, :2]
        if inside(xa, ya) and inside(xb, yb):
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 0), 2)
    for x, y in pts[:, :2]:
        if inside(x, y):
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
    if box_xywh is not None:
        xc, yc, bw, bh = (float(v) for v in box_xywh[:4])
        cv2.rectangle(
            frame,
            (int(xc - bw / 2), int(yc - bh / 2)),
            (int(xc + bw / 2), int(yc + bh / 2)),
            (255, 0, 0),
            2,
        )
    return frame


# Keypoints that must be confidently visible before geometry is trusted.
KEY_CONF_IDS = (NOSE, L_SHOULDER, R_SHOULDER)


def keypoints_usable(conf_row: np.ndarray, min_conf: float) -> bool:
    """True when nose and both shoulders clear the confidence floor."""
    try:
        return bool(np.all(np.asarray(conf_row, dtype=float)[list(KEY_CONF_IDS)] >= min_conf))
    except (IndexError, ValueError, TypeError):
        return False


def crop_center(frame: np.ndarray, zoom: float) -> np.ndarray:
    """Center crop for wide-angle cameras. zoom=1.0 returns the frame."""
    return crop_window(frame, zoom)


def crop_window(
    frame: np.ndarray, zoom: float, pan_x: float = 0.0, pan_y: float = 0.0
) -> np.ndarray:
    """Crop a zoomed window, optionally panned toward a side-placed subject.

    zoom=1.0 and pan 0 returns the frame. pan_x/pan_y in [-1, 1] shift the
    window by that fraction of the available margin; the window is clamped
    inside the frame.
    """
    if zoom <= 1.0 and pan_x == 0 and pan_y == 0:
        return frame
    zoom = max(zoom, 1.0)
    pan_x = max(-1.0, min(1.0, pan_x))
    pan_y = max(-1.0, min(1.0, pan_y))
    h, w = frame.shape[:2]
    cw, ch = int(w / zoom), int(h / zoom)
    margin_x, margin_y = (w - cw) // 2, (h - ch) // 2
    x = max(0, min(w - cw, margin_x + int(pan_x * margin_x)))
    y = max(0, min(h - ch, margin_y + int(pan_y * margin_y)))
    return frame[y : y + ch, x : x + cw]


class Detector:
    """Owns the camera and the pose model on a background thread."""

    def __init__(
        self,
        model_name: str = "yolo26n-pose",
        fallback_model: str = "yolov8n-pose",
        conf: float = 0.5,
        inference_fps: float = 3.0,
        camera_index: int = 0,
        camera_label: str = "",
        width: int = 640,
        height: int = 480,
        min_conf: float = 0.3,
        zoom: float = 1.0,
        pan_x: float = 0.0,
        pan_y: float = 0.0,
        head_drop: float = HEAD_DROP_RATIO,
        lean_offset: float = HEAD_FORWARD_RATIO,
        tilt_max: float = SHOULDER_TILT_RATIO,
    ) -> None:
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.conf = conf
        self.inference_fps = inference_fps
        self.camera_index = camera_index
        self.camera_label = camera_label or f"camera {camera_index}"
        self.width = width
        self.height = height
        self.min_conf = min_conf
        self.zoom = zoom
        self.pan_x = pan_x
        self.pan_y = pan_y
        self.th_drop = head_drop
        self.th_offset = lean_offset
        self.th_tilt = tilt_max
        self._metrics: dict = {}
        self._ov_kpts: np.ndarray | None = None
        self._ov_box: np.ndarray | None = None

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

    @property
    def metrics(self) -> dict:
        with self._lock:
            return dict(self._metrics)

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
                    f"cannot open {self.camera_label} "
                    "(grant camera permission and check index)",
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
                frame = crop_window(frame, self.zoom, self.pan_x, self.pan_y)
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
            with self._lock:
                self._ov_kpts = None
                self._ov_box = None
                self._metrics = {}
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
        with self._lock:
            self._ov_kpts = kpts[idx].copy()
            self._ov_box = boxes[idx].copy() if boxes is not None else None
            self._metrics = pose_metrics(kpts[idx]) or {}
        if confs is None or not keypoints_usable(confs[idx], self.min_conf):
            self._set_status(
                "away",
                "person unclear - sit centered in frame",
                mean_conf,
            )
            return
        scored = score_pose(kpts[idx], self.th_drop, self.th_offset, self.th_tilt)
        scored.conf = mean_conf
        self._set_status(scored.status, scored.detail, scored.conf)

    def _store_jpeg(self, frame: np.ndarray) -> None:
        with self._lock:
            overlay_kpts = None if self._ov_kpts is None else self._ov_kpts.copy()
            overlay_box = None if self._ov_box is None else self._ov_box.copy()
        if overlay_kpts is not None:
            draw_pose(frame, overlay_kpts, overlay_box)
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
