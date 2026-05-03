#!/usr/bin/env python3
"""
Plot distributions of model scores and per-task solve rates for LiveBench
LCB_generation and coding_completion benchmarks.

Usage:
    python livebench_plots.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

DATA_DIR = Path(__file__).parent

TASKS = {
    "LCB_generation": {
        "leaderboard": DATA_DIR / "livebench_LCB_generation_leaderboard.json",
        "ordered": DATA_DIR / "livebench_LCB_generation_ordered.yaml",
        "title": "LiveBench LCB Generation",
    },
    "coding_completion": {
        "leaderboard": DATA_DIR / "livebench_coding_completion_leaderboard.json",
        "ordered": DATA_DIR / "livebench_coding_completion_ordered.yaml",
        "title": "LiveBench Coding Completion",
    },
}


def load_model_scores(leaderboard_path: Path) -> list[float]:
    with open(leaderboard_path) as f:
        data = json.load(f)
    return [entry["score"] for entry in data["results"]]


def load_solve_rates(ordered_path: Path) -> list[float]:
    with open(ordered_path) as f:
        data = yaml.safe_load(f)
    return [task["metrics"]["solve_rate"] for task in data["tasks"]]


def plot_distributions():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, (task_key, cfg) in enumerate(TASKS.items()):
        scores = load_model_scores(cfg["leaderboard"])
        rates = load_solve_rates(cfg["ordered"])

        # Top row: model score distribution
        ax = axes[0, col]
        ax.hist(scores, bins=25, edgecolor="black", alpha=0.7, color="#4C72B0")
        ax.axvline(np.median(scores), color="red", linestyle="--", label=f"median={np.median(scores):.3f}")
        ax.set_xlabel("Model Pass Rate (pass@1)")
        ax.set_ylabel("Count")
        ax.set_title(f"{cfg['title']}\nModel Score Distribution (n={len(scores)})")
        ax.legend()

        # Bottom row: per-task solve rate distribution
        ax = axes[1, col]
        ax.hist(rates, bins=20, edgecolor="black", alpha=0.7, color="#55A868")
        ax.axvline(np.median(rates), color="red", linestyle="--", label=f"median={np.median(rates):.3f}")
        ax.set_xlabel("Per-Task Solve Rate")
        ax.set_ylabel("Count")
        ax.set_title(f"{cfg['title']}\nPer-Task Solve Rate Distribution (n={len(rates)})")
        ax.legend()

        # Annotate saturation
        saturated = sum(1 for r in rates if r >= 0.95)
        impossible = sum(1 for r in rates if r <= 0.05)
        ax.annotate(
            f"Saturated (>=95%): {saturated}\nNear-impossible (<=5%): {impossible}",
            xy=(0.97, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
        )

    fig.tight_layout(pad=2.0)
    out_path = DATA_DIR / "livebench_distributions.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_distributions()
