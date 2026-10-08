#!/bin/sh
# Start Posture Police from the project root.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "missing .venv - run: uv venv && uv pip install -r requirements.txt"
  exit 1
fi
exec .venv/bin/python app.py
