#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from eval_core import ROOT, evaluate_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--test-seed-start", type=int, default=2001)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    rows = []
    for train_seed in args.train_seeds:
        checkpoint = ROOT / "outputs" / "best" / f"seed_{train_seed}" / "actor.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Best checkpoint missing: {checkpoint}")
        for noise in args.noise_levels:
            for episode_seed in range(args.test_seed_start, args.test_seed_start + args.episodes):
                result = evaluate_episode(checkpoint, episode_seed, noise, device)
                result["train_seed"] = train_seed
                rows.append(result)

    output_dir = ROOT / "outputs" / "evaluation"
    figure_dir = ROOT / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "noise_episode_results.csv", index=False)
    per_seed = raw.groupby(["train_seed", "noise_std"], as_index=False).agg(
        mean_reward=("episode_reward", "mean"),
        success_rate=("success", "mean"),
        mean_collisions=("collision_pairs", "mean"),
        mean_final_distance=("final_min_dists", "mean"),
    )
    per_seed.to_csv(output_dir / "noise_results_per_training_seed.csv", index=False)
    summary = per_seed.groupby("noise_std", as_index=False).agg(
        reward_mean=("mean_reward", "mean"),
        reward_std=("mean_reward", "std"),
        success_mean=("success_rate", "mean"),
        success_std=("success_rate", "std"),
        collisions_mean=("mean_collisions", "mean"),
        collisions_std=("mean_collisions", "std"),
    )
    baseline_reward = float(summary.loc[summary.noise_std == 0, "reward_mean"].iloc[0])
    summary["reward_degradation_pct"] = (baseline_reward - summary.reward_mean) / abs(baseline_reward) * 100
    summary.to_csv(output_dir / "noise_summary.csv", index=False)

    for metric, ylabel, filename in [
        ("reward_mean", "Average episode reward", "noise_vs_reward.png"),
        ("success_mean", "Success rate", "noise_vs_success.png"),
        ("collisions_mean", "Collision pairs per episode", "noise_vs_collisions.png"),
    ]:
        std_col = metric.replace("_mean", "_std")
        plt.figure(figsize=(7, 4.5))
        plt.errorbar(summary.noise_std, summary[metric], yerr=summary[std_col], marker="o", capsize=4)
        plt.xlabel("Observation noise σ")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(figure_dir / filename, dpi=180)
        plt.close()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

