#!/usr/bin/env python3
"""
One chart: mean uplift in actor count (mean p50 − baseline) vs baseline # actors.

Data source (only): this folder's results/baseline_{3,5,10,20,50}/runs/*/detailed_estimates.csv,
rows with step_name == ScenarioLevelMetric_NumActors. One mean uplift per baseline (not 6).

Writes only: results/numactors_uplift_only.png

Usage:
    python analyze_results.py
"""

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit("matplotlib is required: pip install matplotlib") from e

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
OUTPUT_CHART = RESULTS_DIR / "numactors_uplift_only.png"

BASELINE_ACTORS = [3, 5, 10, 20, 50]
NUM_ACTORS_STEP = "ScenarioLevelMetric_NumActors"


def load_p50_for_baseline(baseline: int) -> List[float]:
    baseline_dir = RESULTS_DIR / f"baseline_{baseline}"
    if not baseline_dir.exists():
        return []

    runs_root = baseline_dir / "runs"
    if not runs_root.is_dir():
        return []

    values: List[float] = []
    for run_dir in sorted(d for d in runs_root.iterdir() if d.is_dir()):
        csv_path = run_dir / "detailed_estimates.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("has_error", "").strip().lower() == "true":
                    continue
                if row.get("step_name", "").strip() != NUM_ACTORS_STEP:
                    continue
                try:
                    raw = row.get("percentile_50th") or row.get("estimate")
                    if raw is None or not str(raw).strip():
                        continue
                    values.append(float(str(raw).strip()))
                except (ValueError, TypeError):
                    continue
    return values


def main():
    xs: List[int] = []
    ys: List[float] = []
    yerrs: List[float] = []

    for b in BASELINE_ACTORS:
        p50s = load_p50_for_baseline(b)
        if not p50s:
            continue
        arr = np.array(p50s, dtype=float)
        xs.append(b)
        ys.append(float(np.mean(arr) - b))
        yerrs.append(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0)

    if not xs:
        raise SystemExit("No valid NumActors rows under results/baseline_*/runs/*/detailed_estimates.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(xs, ys, yerr=yerrs, fmt="o-", capsize=5, markersize=8, linewidth=2, color="#2e86ab")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    # Exactly five conditions (3,5,10,20,50); avoid default xticks like 0,10,…,50 (looks like 6 points)
    ax.set_xticks(BASELINE_ACTORS)
    ax.set_xticklabels([str(b) for b in BASELINE_ACTORS])
    ax.set_xlim(0, 55)
    ax.set_xlabel("Baseline number of actors", fontsize=12)
    ax.set_ylabel("Uplift (additional actors)", fontsize=12)
    ax.set_title("Initial access bundle: uplifted # actors vs. baseline", fontsize=14, weight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    OUTPUT_CHART.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_CHART, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUTPUT_CHART}")


if __name__ == "__main__":
    main()
