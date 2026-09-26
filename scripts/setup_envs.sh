#!/bin/sh
# Recreates the two Python environments for the pipeline tools.
# Usage: sh ./scripts/setup_envs.sh
#
# Requires: uv (https://docs.astral.sh/uv/), and on Debian/Ubuntu:
#   sudo apt-get install -y libxrender1 libxext6 libsm6   # RDKit drawing libs
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> REINVENT4 environment (.venv-reinvent, Python 3.12, CPU torch)"
uv venv "$ROOT/.venv-reinvent" --python 3.12
uv pip install --python "$ROOT/.venv-reinvent/bin/python" \
    torch==2.12.0 torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$ROOT/.venv-reinvent/bin/python" scipy
uv pip install --python "$ROOT/.venv-reinvent/bin/python" "$ROOT/tools/REINVENT4"
"$ROOT/.venv-reinvent/bin/reinvent" --version

echo "==> AiZynthFinder environment (.venv-aizynth, Python 3.12)"
uv venv "$ROOT/.venv-aizynth" --python 3.12
uv pip install --python "$ROOT/.venv-aizynth/bin/python" "$ROOT/tools/aizynthfinder"
"$ROOT/.venv-aizynth/bin/aizynthcli" --help >/dev/null && echo "aizynthcli OK"

echo "==> Web app deps (into .venv-reinvent, shares RDKit)"
uv pip install --python "$ROOT/.venv-reinvent/bin/python" -r "$ROOT/app/requirements.txt"

echo "==> Done. Next: download models (see tools/README.md)"
