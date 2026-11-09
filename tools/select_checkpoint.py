#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import torch

from eval_core import ROOT, evaluate_episode


def episode_number(path):
    return int(path.stem.split("ep")[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-metadata", required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--validation-episodes", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    metadata = json.loads(Path(args.run_metadata).read_text(encoding="utf-8"))
    checkpoint_dir = Path(metadata["checkpoint_dir"])
    candidates = sorted(checkpoint_dir.glob("actor_ep*.pt"), key=episode_number)
    if len(candidates) > args.max_candidates:
        indexes = [round(i * (len(candidates) - 1) / (args.max_candidates - 1)) for i in range(args.max_candidates)]
        candidates = [candidates[i] for i in sorted(set(indexes))]
    if not candidates:
        candidates = [checkpoint_dir / "actor.pt"]
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    rows = []
    validation_seeds = range(1001, 1001 + args.validation_episodes)
    for checkpoint in candidates:
        for episode_seed in validation_seeds:
            result = evaluate_episode(checkpoint, episode_seed, 0.0, device)
            result.update({"checkpoint": str(checkpoint), "train_seed": args.train_seed})
            rows.append(result)
    frame = pd.DataFrame(rows)
    summary = frame.groupby("checkpoint", as_index=False).agg(
        mean_reward=("episode_reward", "mean"),
        std_reward=("episode_reward", "std"),
        success_rate=("success", "mean"),
    ).sort_values(["mean_reward", "success_rate"], ascending=False)
    best_checkpoint = Path(summary.iloc[0]["checkpoint"])
    best_dir = ROOT / "outputs" / "best" / f"seed_{args.train_seed}"
    best_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_checkpoint, best_dir / "actor.pt")
    critic_candidate = best_checkpoint.with_name(best_checkpoint.name.replace("actor_", "critic_"))
    if critic_candidate.exists():
        shutil.copy2(critic_candidate, best_dir / "critic.pt")
    selection_dir = ROOT / "outputs" / "validation"
    selection_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(selection_dir / f"seed_{args.train_seed}_episodes.csv", index=False)
    summary.to_csv(selection_dir / f"seed_{args.train_seed}_summary.csv", index=False)
    (best_dir / "selection.json").write_text(json.dumps({
        "train_seed": args.train_seed,
        "selected_checkpoint": str(best_checkpoint),
        "validation_episodes": args.validation_episodes,
        "mean_reward": float(summary.iloc[0]["mean_reward"]),
        "success_rate": float(summary.iloc[0]["success_rate"]),
    }, indent=2), encoding="utf-8")
    print(best_dir.resolve())


if __name__ == "__main__":
    main()

