#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-metadata", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/figures/baseline_learning_curves.png")
    args = parser.parse_args()
    plt.figure(figsize=(8, 5))
    plotted = 0
    for metadata_path in args.run_metadata:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        summary_path = Path(metadata["upstream_run_dir"]) / "logs" / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        # matching = [value for key, value in summary.items()
        #             if key.endswith("average_episode_rewards/average_episode_rewards")
        #             and "eval_average" not in key]
        matching = [
            value
            for key, value in summary.items()
            if key.replace("\\", "/").endswith(
                "average_episode_rewards/average_episode_rewards"
            )
               and "eval_average" not in key.replace("\\", "/")
        ]
        if not matching:
            continue
        values = matching[0]
        plt.plot([row[1] for row in values], [row[2] for row in values],
                 label=f"seed {metadata['seed']}")
        plotted += 1
    if not plotted:
        raise RuntimeError("No average_episode_rewards series found in supplied runs")
    plt.xlabel("Environment steps")
    plt.ylabel("Average episode reward")
    plt.grid(alpha=0.25)
    plt.legend()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    print(output.resolve())


if __name__ == "__main__":
    main()

