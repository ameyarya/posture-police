"""On-device pose backend: MediaPipe Pose Landmarker.

Runs fully offline after a one-time ~6 MB model download. There is no
analytics or other network use at runtime. Landmarks are mapped onto the
COCO-17 layout the posture scoring in detector.py expects.
"""

from __future__ import annotations

import os
import urllib.request

import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# MediaPipe landmark index -> COCO keypoint index.
MP_TO_COCO = {
    0: 0,    # nose
    2: 1,    # left eye
    5: 2,    # right eye
    7: 3,    # left ear
    8: 4,    # right ear
    11: 5,   # left shoulder
    12: 6,   # right shoulder
    13: 7,   # left elbow
    14: 8,   # right elbow
    15: 9,   # left wrist
    16: 10,  # right wrist
    23: 11,  # left hip
    24: 12,  # right hip
    25: 13,  # left knee
    26: 14,  # right knee
    27: 15,  # left ankle
    28: 16,  # right ankle
}


def landmark_confidence(lm) -> float:
    """Trust for one landmark: in-frame (presence) and unoccluded (visibility).

    Duck-typed; anything missing or unreadable scores 0.
    """
    try:
        return max(0.0, min(1.0, float(lm.presence) * float(lm.visibility)))
    except (AttributeError, TypeError, ValueError):
        return 0.0


def map_landmarks(landmarks, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Map 33 pose landmarks onto COCO-17 pixel keypoints plus confidences.

    Pure function; landmarks only need x/y/presence/visibility attributes.
    Unmapped or unreadable points come back as zeros.
    """
    kpts = np.zeros((17, 2))
    conf = np.zeros(17)
    try:
        items = list(landmarks)
    except TypeError:
        return kpts, conf
    for mp_i, coco_i in MP_TO_COCO.items():
        if mp_i >= len(items):
            continue
        lm = items[mp_i]
        try:
            x, y = float(lm.x), float(lm.y)
        except (AttributeError, TypeError, ValueError):
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        kpts[coco_i] = [x * width, y * height]
        conf[coco_i] = landmark_confidence(lm)
    return kpts, conf


def pose_box(
    kpts: np.ndarray, conf: np.ndarray, min_conf: float = 0.2
) -> np.ndarray | None:
    """xywh box over the confident keypoints, or None when too few qualify."""
    pts = np.asarray(kpts, dtype=float)[np.asarray(conf, dtype=float) >= min_conf]
    if len(pts) < 3:
        return None
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.15 * max(float(hi[0] - lo[0]), float(hi[1] - lo[1]))
    return np.array(
        [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2,
         (hi[0] - lo[0]) + 2 * pad, (hi[1] - lo[1]) + 2 * pad]
    )


def ensure_model(path: str) -> str:
    """Fetch the .task file on first run. Returns path; raises offline."""
    if os.path.exists(path):
        return path
    tmp = path + ".tmp"
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=120) as resp, \
                open(tmp, "wb") as fh:
            fh.write(resp.read())
        os.replace(tmp, path)
    except Exception as exc:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise RuntimeError(
            f"could not download pose model to {path} "
            f"(connect once, then it works offline): {exc}"
        )
    return path


class PoseBackend:
    """Thin PoseLandmarker wrapper. mediapipe is imported lazily so that
    importing this module (and app) never pays the import cost."""

    def __init__(self, model_path: str, detection_conf: float = 0.5) -> None:
        self.model_path = model_path
        self.detection_conf = detection_conf
        self._landmarker = None

    def load(self) -> None:
        import mediapipe as mp

        path = ensure_model(self.model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=self.detection_conf,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> tuple | None:
        """(kpts, conf, box) for the most prominent person, or None."""
        import cv2
        import mediapipe as mp

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect(image)
        if not res.pose_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        kpts, conf = map_landmarks(res.pose_landmarks[0], w, h)
        return kpts, conf, pose_box(kpts, conf)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
