#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/tools/plot_training.py" \
  --run-metadata \
  "$PROJECT_DIR/outputs/runs/baseline_seed1.json" \
  "$PROJECT_DIR/outputs/runs/baseline_seed2.json" \
  "$PROJECT_DIR/outputs/runs/baseline_seed3.json" \
  --output "$PROJECT_DIR/outputs/figures/baseline_learning_curves.png"
