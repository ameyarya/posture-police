#!/bin/sh
# Start Posture Police from the project root. Respawns the server after a
# crash or an /api/restart request; Ctrl+C twice to stop for good.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "missing .venv - run: uv venv && uv pip install -r requirements.txt"
  exit 1
fi
while true; do
  .venv/bin/python app.py
  echo "server stopped - restarting in 2s (Ctrl+C now to stay stopped)"
  sleep 2
done
