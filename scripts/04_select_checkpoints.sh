#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for SEED in 1 2 3; do
  "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/tools/select_checkpoint.py" \
    --run-metadata "$PROJECT_DIR/outputs/runs/baseline_seed${SEED}.json" \
    --train-seed "$SEED" --validation-episodes 20
done

