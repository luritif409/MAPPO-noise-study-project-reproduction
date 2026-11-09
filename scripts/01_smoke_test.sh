#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/tools/run_training.py" --stage smoke --seed 1 --threads 1

