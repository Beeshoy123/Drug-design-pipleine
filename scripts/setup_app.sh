#!/bin/sh
# Installs the web app into the existing REINVENT4 environment (idempotent).
# The app reuses .venv-reinvent because it shares RDKit.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
uv pip install --python "$ROOT/.venv-reinvent/bin/python" -r "$ROOT/app/requirements.txt"
echo "App deps installed."
