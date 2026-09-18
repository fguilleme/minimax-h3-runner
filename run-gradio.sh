#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/home/francois/comfy/ComfyUI/.venv/bin/python
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m h3runner.gradio_app \
  --host "${H3_GRADIO_HOST:-127.0.0.1}" \
  --port "${H3_GRADIO_PORT:-7860}" \
  --comfy-root /home/francois/comfy/ComfyUI \
  --runs-dir "$ROOT/runs/gradio" \
  --log-dir "$ROOT/log/gradio" \
  --pid "$ROOT/runs/gradio/server.pid" \
  "$@"
