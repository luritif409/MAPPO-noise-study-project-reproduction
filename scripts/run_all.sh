#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/setup_env.sh"
bash "$SCRIPT_DIR/01_smoke_test.sh"
bash "$SCRIPT_DIR/02_pilot.sh"
bash "$SCRIPT_DIR/03_train_baselines.sh"
bash "$SCRIPT_DIR/04_select_checkpoints.sh"
bash "$SCRIPT_DIR/05_evaluate_noise.sh"
bash "$SCRIPT_DIR/06_plot_training.sh"

