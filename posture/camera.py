"""Camera discovery and selection.

Prefers a named camera (the NexiGo) using the system AVFoundation device
list, verifies the choice with OpenCV, and falls back gracefully. Matching
logic is pure and unit tested; only list_system_cameras() and can_open()
touch hardware or subprocesses.
"""

from __future__ import annotations

import re
import shutil
import subprocess

DEVICE_RE = re.compile(r"\[(\d+)\]\s+(.+?)\s*$")


def parse_avfoundation_devices(text: str) -> list[tuple[int, str]]:
    """Parse `ffmpeg -f avfoundation -list_devices` output.

    Only the video section is returned; the audio section is ignored.
    """
    devices: list[tuple[int, str]] = []
    in_audio = False
    for line in text.splitlines():
        if "AVFoundation audio devices" in line:
            in_audio = True
            continue
        if in_audio:
            continue
        match = DEVICE_RE.search(line)
        if match:
            devices.append((int(match.group(1)), match.group(2).strip()))
    return devices


def list_system_cameras() -> list[tuple[int, str]]:
    """List (index, name) video devices. Returns [] when unavailable."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return []
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-f", "avfoundation",
             "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return parse_avfoundation_devices(proc.stderr + proc.stdout)
    except Exception:
        return []


def can_open(index: int) -> bool:
    """True when OpenCV can open the camera (also requests permission)."""
    import cv2

    cap = cv2.VideoCapture(index)
    ok = cap.isOpened()
    cap.release()
    return bool(ok)


def choose_camera(
    devices: list[tuple[int, str]],
    prefer_name: str = "",
    configured_index: int = 0,
    opener=can_open,
) -> tuple[int, str, str]:
    """Pick a camera index. Returns (index, label, reason). Never raises."""
    names = dict(devices)
    if prefer_name:
        want = prefer_name.lower()
        for idx, name in devices:
            if want in name.lower():
                if opener(idx):
                    return idx, name, "preferred camera matched and opened"
                break
    if opener(configured_index):
        return (
            configured_index,
            names.get(configured_index, f"camera {configured_index}"),
            "configured camera opened",
        )
    for idx, name in devices:
        if opener(idx):
            return idx, name, "fallback: first working camera"
    return (
        configured_index,
        names.get(configured_index, f"camera {configured_index}"),
        "unavailable (grant camera permission and restart?)",
    )


def resolve_camera(
    prefer_name: str = "", configured_index: int = 0
) -> tuple[int, str, str]:
    """Discover devices and choose one, probing indexes when unlisted."""
    devices = list_system_cameras()
    if not devices:
        devices = [(i, f"camera {i}") for i in range(4)]
    return choose_camera(devices, prefer_name, configured_index)
