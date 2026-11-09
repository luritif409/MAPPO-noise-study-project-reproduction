#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for SEED in 1 2 3; do
  "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/tools/run_training.py" --stage baseline --seed "$SEED" --threads 8
done

