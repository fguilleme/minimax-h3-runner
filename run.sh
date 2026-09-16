#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/home/francois/comfy/ComfyUI/.venv/bin/python
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m h3runner.run \
  --config "$ROOT/config.json" \
  --work-dir "$ROOT/runs/fox-56f" \
  --output "$ROOT/output/minimax-h3-fox-56f-10fps.mp4" \
  "$@"
