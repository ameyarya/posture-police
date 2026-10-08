"""Optional local VLM second opinion through Ollama. Disabled by default.

When enabled, the app sends one JPEG snapshot at a time to a local
Qwen3-VL model over localhost. This never blocks pose scoring: any failure
returns None and the dashboard keeps working on the pose model alone.
"""

from __future__ import annotations

import base64
import json
import urllib.request

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

PROMPT = (
    "You are a posture coach. Look at this webcam photo of a person "
    "working at a desk and reply in one short sentence: is their sitting "
    "posture good, or are they slouching or leaning? No preamble."
)


def ask_vlm(
    jpeg: bytes,
    model: str = "qwen3-vl:4b",
    timeout: float = 90.0,
) -> str | None:
    """Send one snapshot to the local VLM. Returns its note or None."""
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
                "images": [base64.b64encode(jpeg).decode()],
            }
        ],
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return str(data["message"]["content"]).strip()
    except Exception:
        return None
