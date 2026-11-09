#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/tools/evaluate_noise.py" \
  --train-seeds 1 2 3 \
  --noise-levels 0 0.05 0.10 0.20 0.30 \
  --episodes 100 --test-seed-start 2001

