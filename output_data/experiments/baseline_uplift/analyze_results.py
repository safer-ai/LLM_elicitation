#!/usr/bin/env python3
"""
Analyze results from the Baseline Uplift Curve Experiment.

This script:
1. Loads estimates from each baseline condition
2. Computes statistics (mean, std, CV)
3. Generates the baseline uplift curve plot
4. Outputs summary statistics

Usage:
    python analyze_results.py
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Optional matplotlib (for plotting)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plots will be skipped")

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_DIR = EXPERIMENT_DIR / "results"

BASELINE_VALUES = [10, 20, 30, 40, 50, 60, 70, 80, 90]


def load_estimates_for_baseline(baseline: int) -> List[Dict]:
    """Load all estimates for a given baseline value."""
    baseline_dir = RESULTS_DIR / f"baseline_{baseline}"

    if not baseline_dir.exists():
        print(f"Warning: No results directory for baseline {baseline}%")
        return []

    estimates = []

    # Find all run directories
    run_dirs = sorted([d for d in baseline_dir.iterdir()
                       if d.is_dir() and d.name.startswith("run_")])

    for run_dir in run_dirs:
        csv_path = run_dir / "detailed_estimates.csv"

        if not csv_path.exists():
            continue

        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                # Skip error rows
                if row.get("has_error", "").strip().lower() == "true":
                    continue

                try:
                    estimate = {
                        "baseline": baseline,
                        "run": run_dir.name,
                        "most_likely_estimate": float(row["most_likely_estimate"].strip()),
                        "p25": float(row["percentile_25th"].strip()),
                        "p50": float(row["percentile_50th"].strip()),
                        "p75": float(row["percentile_75th"].strip()),
                    }
                    estimates.append(estimate)
                except (ValueError, KeyError) as e:
                    continue

    return estimates


def compute_statistics(estimates: List[Dict]) -> Dict:
    """Compute statistics for a set of estimates."""
    if not estimates:
        return None

    p50_values = [e["p50"] for e in estimates]
    most_likely_values = [e["most_likely_estimate"] for e in estimates]

    return {
        "n": len(estimates),
        "p50_mean": np.mean(p50_values),
        "p50_std": np.std(p50_values, ddof=1) if len(p50_values) > 1 else 0,
        "p50_min": np.min(p50_values),
        "p50_max": np.max(p50_values),
        "p50_cv": np.std(p50_values, ddof=1) / np.mean(p50_values) * 100 if np.mean(p50_values) > 0 else 0,
        "most_likely_mean": np.mean(most_likely_values),
        "most_likely_std": np.std(most_likely_values, ddof=1) if len(most_likely_values) > 1 else 0,
    }


def print_summary_table(all_stats: Dict[int, Dict]):
    """Print a summary table of all statistics."""
    print("\n" + "=" * 90)
    print("BASELINE UPLIFT CURVE - SUMMARY STATISTICS")
    print("=" * 90)
    print()
    print(f"{'Baseline':>10} | {'N':>5} | {'P50 Mean':>10} | {'P50 Std':>10} | {'P50 CV%':>10} | {'Uplift':>10}")
    print("-" * 90)

    prev_mean = None
    for baseline in BASELINE_VALUES:
        stats = all_stats.get(baseline)

        if stats is None:
            print(f"{baseline:>9}% | {'N/A':>5} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10}")
            continue

        # Compute uplift from baseline
        uplift = (stats['p50_mean'] - baseline/100) * 100  # in percentage points

        print(f"{baseline:>9}% | {stats['n']:>5} | {stats['p50_mean']:>10.4f} | "
              f"{stats['p50_std']:>10.4f} | {stats['p50_cv']:>9.1f}% | {uplift:>+9.1f}pp")

        prev_mean = stats['p50_mean']

    print("-" * 90)


def compute_curve_metrics(all_stats: Dict[int, Dict]) -> Dict:
    """Compute metrics about the baseline-output relationship."""
    baselines = []
    outputs = []

    for baseline in BASELINE_VALUES:
        stats = all_stats.get(baseline)
        if stats and stats['n'] > 0:
            baselines.append(baseline / 100)  # Convert to 0-1 scale
            outputs.append(stats['p50_mean'])

    if len(baselines) < 2:
        return None

    baselines = np.array(baselines)
    outputs = np.array(outputs)

    # Linear regression
    slope, intercept = np.polyfit(baselines, outputs, 1)

    # Compute correlation
    correlation = np.corrcoef(baselines, outputs)[0, 1]

    # Compute average uplift
    uplifts = outputs - baselines
    avg_uplift = np.mean(uplifts)

    # Check for ceiling effect (output flattening at high baselines)
    if len(outputs) >= 3:
        high_baseline_slope = (outputs[-1] - outputs[-3]) / (baselines[-1] - baselines[-3])
        low_baseline_slope = (outputs[2] - outputs[0]) / (baselines[2] - baselines[0])
        ceiling_effect = low_baseline_slope - high_baseline_slope
    else:
        ceiling_effect = None

    return {
        "slope": slope,
        "intercept": intercept,
        "correlation": correlation,
        "avg_uplift": avg_uplift,
        "ceiling_effect": ceiling_effect,
    }


def plot_baseline_uplift_curve(all_stats: Dict[int, Dict], output_path: Path):
    """Generate the baseline uplift curve plot."""
    if not HAS_MATPLOTLIB:
        print("Skipping plot generation (matplotlib not available)")
        return

    baselines = []
    outputs = []
    stds = []
    uplifts = []

    for baseline in BASELINE_VALUES:
        stats = all_stats.get(baseline)
        if stats and stats['n'] > 0:
            baselines.append(baseline)
            outputs.append(stats['p50_mean'] * 100)  # Convert to percentage
            stds.append(stats['p50_std'] * 100)
            uplifts.append((stats['p50_mean'] - baseline/100) * 100)

    if not baselines:
        print("No data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Baseline vs Output
    ax1 = axes[0]
    ax1.errorbar(baselines, outputs, yerr=stds, fmt='o-', capsize=5,
                 markersize=8, linewidth=2, color='C0', label='LLM Output (p50)')
    ax1.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Identity (no change)')
    ax1.fill_between([0, 100], [0, 100], [100, 100], alpha=0.1, color='green',
                     label='Uplift region')

    ax1.set_xlabel("Input Baseline Probability (%)", fontsize=11, weight='bold')
    ax1.set_ylabel("LLM Output Probability (p50, %)", fontsize=11, weight='bold')
    ax1.set_title("A. Baseline Uplift Curve", fontsize=12, weight='bold')
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel B: Uplift by Baseline
    ax2 = axes[1]
    ax2.bar(baselines, uplifts, width=8, color='C1', alpha=0.8, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=np.mean(uplifts), color='red', linestyle='--',
                label=f'Mean uplift: {np.mean(uplifts):.1f}pp')

    ax2.set_xlabel("Input Baseline Probability (%)", fontsize=11, weight='bold')
    ax2.set_ylabel("Uplift (percentage points)", fontsize=11, weight='bold')
    ax2.set_title("B. LLM Uplift by Baseline", fontsize=12, weight='bold')
    ax2.set_xlim(0, 100)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle("Baseline Uplift Curve Experiment\n"
                 "How does LLM output probability change with input baseline?",
                 fontsize=14, weight='bold', y=1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved plot: {output_path}")

    # Also save PDF
    pdf_path = output_path.with_suffix('.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved plot: {pdf_path}")

    plt.close(fig)


def save_results_csv(all_stats: Dict[int, Dict], output_path: Path):
    """Save results to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['baseline_pct', 'n', 'p50_mean', 'p50_std', 'p50_cv',
                        'uplift_pp', 'most_likely_mean', 'most_likely_std'])

        for baseline in BASELINE_VALUES:
            stats = all_stats.get(baseline)
            if stats:
                uplift = (stats['p50_mean'] - baseline/100) * 100
                writer.writerow([
                    baseline,
                    stats['n'],
                    f"{stats['p50_mean']:.6f}",
                    f"{stats['p50_std']:.6f}",
                    f"{stats['p50_cv']:.2f}",
                    f"{uplift:.2f}",
                    f"{stats['most_likely_mean']:.6f}",
                    f"{stats['most_likely_std']:.6f}",
                ])
            else:
                writer.writerow([baseline, 0, '', '', '', '', '', ''])

    print(f"Saved results: {output_path}")


def main():
    print("=" * 70)
    print("BASELINE UPLIFT CURVE - ANALYSIS")
    print("=" * 70)

    # Load all estimates
    all_estimates = {}
    all_stats = {}

    print("\nLoading estimates...")
    for baseline in BASELINE_VALUES:
        estimates = load_estimates_for_baseline(baseline)
        all_estimates[baseline] = estimates

        stats = compute_statistics(estimates)
        all_stats[baseline] = stats

        n = len(estimates)
        print(f"  Baseline {baseline:2d}%: {n:3d} estimates")

    # Print summary
    print_summary_table(all_stats)

    # Compute curve metrics
    metrics = compute_curve_metrics(all_stats)
    if metrics:
        print("\n" + "=" * 70)
        print("CURVE METRICS")
        print("=" * 70)
        print(f"  Linear fit: output = {metrics['slope']:.4f} * baseline + {metrics['intercept']:.4f}")
        print(f"  Correlation (r): {metrics['correlation']:.4f}")
        print(f"  Average uplift: {metrics['avg_uplift']*100:.2f} percentage points")
        if metrics['ceiling_effect'] is not None:
            print(f"  Ceiling effect indicator: {metrics['ceiling_effect']:.4f}")
            if metrics['ceiling_effect'] > 0.1:
                print("    -> Suggests ceiling effect (slope decreases at high baselines)")
            else:
                print("    -> No strong ceiling effect detected")

    # Generate outputs
    print("\n" + "=" * 70)
    print("GENERATING OUTPUTS")
    print("=" * 70)

    # Save CSV
    csv_path = EXPERIMENT_DIR / "baseline_uplift_results.csv"
    save_results_csv(all_stats, csv_path)

    # Generate plot
    plot_path = EXPERIMENT_DIR / "baseline_uplift_curve.png"
    plot_baseline_uplift_curve(all_stats, plot_path)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
