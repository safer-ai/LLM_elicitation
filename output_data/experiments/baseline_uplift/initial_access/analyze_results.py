#!/usr/bin/env python3
"""
Build one chart: mean uplift (p50 − baseline probability) vs baseline %.
Writes: results/baseline_uplift_only.png

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
OUTPUT_CHART = RESULTS_DIR / "baseline_uplift_only.png"

BASELINE_VALUES = [10, 20, 30, 40, 50, 60, 70, 80, 90]


def load_estimates_for_baseline(baseline: int) -> List[Dict]:
    baseline_dir = RESULTS_DIR / f"baseline_{baseline}"
    if not baseline_dir.exists():
        return []

    runs_root = baseline_dir / "runs"
    if runs_root.is_dir():
        run_dirs = sorted(d for d in runs_root.iterdir() if d.is_dir())
    else:
        run_dirs = sorted(
            d for d in baseline_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        )

    estimates = []
    for run_dir in run_dirs:
        csv_path = run_dir / "detailed_estimates.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                if row.get("has_error", "").strip().lower() == "true":
                    continue
                try:
                    raw_ml = row.get("most_likely_estimate") or row.get("estimate")
                    if raw_ml is None or not str(raw_ml).strip():
                        continue
                    estimates.append({
                        "p50": float(row["percentile_50th"].strip()),
                    })
                except (ValueError, KeyError):
                    continue
    return estimates


def compute_statistics(estimates: List[Dict]):
    if not estimates:
        return None
    p50_values = [e["p50"] for e in estimates]
    return {
        "n": len(p50_values),
        "p50_mean": float(np.mean(p50_values)),
        "p50_std": float(np.std(p50_values, ddof=1)) if len(p50_values) > 1 else 0.0,
    }


def main():
    all_stats = {}
    for baseline in BASELINE_VALUES:
        est = load_estimates_for_baseline(baseline)
        all_stats[baseline] = compute_statistics(est)

    xs, ys, yerrs = [], [], []
    for baseline in BASELINE_VALUES:
        stats = all_stats.get(baseline)
        if not stats or stats["n"] <= 0:
            continue
        xs.append(baseline)
        ys.append((stats["p50_mean"] - baseline / 100.0) * 100.0)
        yerrs.append(stats["p50_std"] * 100.0)

    if not xs:
        raise SystemExit("No valid estimates found under results/baseline_*/runs/")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(xs, ys, yerr=yerrs, fmt="o-", capsize=5, markersize=8, linewidth=2, color="#2e86ab")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Baseline probability (%)", fontsize=12)
    ax.set_ylabel("Uplift (percentage points)", fontsize=12)
    ax.set_title("Initial access: uplift vs. stated baseline", fontsize=14, weight="bold")
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    OUTPUT_CHART.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_CHART, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUTPUT_CHART}")


if __name__ == "__main__":
    main()
