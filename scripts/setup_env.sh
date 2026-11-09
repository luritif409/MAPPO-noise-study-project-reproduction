#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uv venv --python 3.10 "$PROJECT_DIR/.venv"
uv pip install --python "$PROJECT_DIR/.venv/bin/python" torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$PROJECT_DIR/.venv/bin/python" -r "$PROJECT_DIR/requirements-minimal.txt"
uv pip install --python "$PROJECT_DIR/.venv/bin/python" -e "$PROJECT_DIR/vendor/on-policy"
"$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/tools/system_info.py"

