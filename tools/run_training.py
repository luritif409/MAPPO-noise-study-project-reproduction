#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor" / "on-policy"
TRAIN_SCRIPT = UPSTREAM / "onpolicy" / "scripts" / "train" / "train_mpe.py"
RESULTS_ROOT = UPSTREAM / "onpolicy" / "scripts" / "results" / "MPE" / "simple_spread" / "rmappo"


def latest_run(experiment):
    folder = RESULTS_ROOT / experiment
    runs = [p for p in folder.glob("run*") if p.is_dir()]
    if not runs:
        raise RuntimeError(f"No run directory found under {folder}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Run a pinned limited-reproduction MAPPO training stage.")
    parser.add_argument("--stage", choices=["smoke", "pilot", "baseline"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    default_steps = {"smoke": 10_000, "pilot": 100_000, "baseline": 2_000_000}
    steps = args.steps or default_steps[args.stage]
    experiment = f"{args.stage}_seed{args.seed}"
    command = [
        sys.executable, str(TRAIN_SCRIPT),
        "--env_name", "MPE",
        "--algorithm_name", "rmappo",
        "--experiment_name", experiment,
        "--scenario_name", "simple_spread",
        "--num_agents", "3",
        "--num_landmarks", "3",
        "--seed", str(args.seed),
        "--n_training_threads", "1",
        "--n_rollout_threads", str(args.threads),
        "--n_eval_rollout_threads", "8",
        "--num_mini_batch", "1",
        "--episode_length", "25",
        "--num_env_steps", str(steps),
        "--ppo_epoch", "10",
        "--gain", "0.01",
        "--lr", "7e-4",
        "--critic_lr", "7e-4",
        "--clip_param", "0.2",
        "--gamma", "0.99",
        "--gae_lambda", "0.95",
        "--save_interval", "100",
        "--log_interval", "10",
        "--eval_interval", "100",
        "--use_eval",
        "--use_wandb"
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(UPSTREAM) + os.pathsep + env.get("PYTHONPATH", "")
    log_dir = ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{experiment}.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=TRAIN_SCRIPT.parent.parent, env=env, stdout=log, stderr=subprocess.STDOUT)
    duration = time.perf_counter() - started
    if completed.returncode != 0:
        raise SystemExit(f"Training failed; inspect {log_path}")
    run_dir = latest_run(experiment)
    checkpoint_dir = ROOT / "outputs" / "checkpoints" / experiment
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for file in (run_dir / "models").glob("*.pt"):
        shutil.copy2(file, checkpoint_dir / file.name)
    metadata = {
        "stage": args.stage,
        "seed": args.seed,
        "steps": steps,
        "rollout_threads": args.threads,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration,
        "command": command,
        "upstream_run_dir": str(run_dir.resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "log": str(log_path.resolve()),
        "return_code": completed.returncode,
    }
    metadata_path = ROOT / "outputs" / "runs" / f"{experiment}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metadata_path.resolve())


if __name__ == "__main__":
    main()

