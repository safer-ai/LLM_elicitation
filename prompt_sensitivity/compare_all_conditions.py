#!/usr/bin/env python3
"""
Complete analysis of prompt sensitivity across ALL 7 conditions.

Conditions:
1. control - Full prompt with baseline + CI + reasoning structure
2. no_baseline - No baseline estimate, has CI + reasoning
3. no_ci - Has baseline, no CI + reasoning
4. no_baseline_no_ci - Minimal: no baseline, no CI + reasoning
5. skip_analysis - Minimized analysis step + reasoning
6. trim_reasoning - No reasoning structure scaffolding
7. trim_all - Bare minimum prompt

Uses prompt characteristics to classify runs since prompts_dir not saved in metadata.
"""

import json
import csv
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, List, Dict
import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import least_squares, minimize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configuration
OUTPUT_DIR = Path(__file__).parent / "output"
RUNS_DIR = OUTPUT_DIR / "runs"

# Lower threshold to include trim_all runs
MIN_DURATION = 30.0

# Quantile grid
QUANTILE_GRID_SIZE = 201
Q_GRID = np.linspace(0.001, 0.999, QUANTILE_GRID_SIZE)

N_PERMUTATIONS = 5_000

# Run Classification
def classify_run(analysis_len: int, estimation_len: int, has_reasoning: bool, scenario: str) -> str:
    """Classify run based on prompt characteristics."""
    if scenario == 'scenario_no_baseline.yaml':
        return 'no_baseline'
    elif scenario == 'scenario_no_ci.yaml':
        return 'no_ci'
    elif scenario == 'scenario_no_baseline_no_ci.yaml':
        return 'no_baseline_no_ci'
    elif scenario == 'scenario_control.yaml':
        # Distinguish between control, skip_analysis, trim_reasoning, trim_all
        if analysis_len < 2500:
            return 'trim_all'
        elif analysis_len < 4000:
            return 'skip_analysis'
        elif analysis_len > 5000:
            if not has_reasoning:
                return 'trim_reasoning'
            else:
                return 'control'
    return 'unknown'


# Beta Fitting
def fit_beta_from_percentiles(
    p25: float, p50: float, p75: float
) -> Optional[Tuple[float, float, float]]:
    """Fit Beta(alpha, beta) using least-squares on all three percentiles."""
    if not (0 < p25 < p50 < p75 < 1):
        return None

    p25 = max(p25, 0.005)
    p75 = min(p75, 0.995)
    p50 = max(min(p50, 0.995), 0.005)

    if not (p25 < p50 < p75):
        return None

    def residuals(params):
        a, b = params
        if a <= 0.1 or b <= 0.1:
            return [1e10, 1e10, 1e10]
        r1 = sp_stats.beta.cdf(p25, a, b) - 0.25
        r2 = sp_stats.beta.cdf(p50, a, b) - 0.50
        r3 = sp_stats.beta.cdf(p75, a, b) - 0.75
        return [r1, r2, r3]

    best_solution = None
    best_cost = float("inf")

    iqr = p75 - p25
    loc = p50
    est_concentration = max(2, min(1000, 0.1 / (iqr ** 2))) if iqr > 0 else 50
    est_alpha = est_concentration * loc
    est_beta = est_concentration * (1 - loc)

    init_guesses = [
        (est_alpha, est_beta),
        (est_alpha * 0.5, est_beta * 0.5),
        (est_alpha * 2, est_beta * 2),
        (2.0, 2.0), (5.0, 5.0), (10.0, 10.0), (20.0, 20.0),
        (50.0, 50.0), (100.0, 100.0), (200.0, 200.0),
        (3.0, 8.0), (8.0, 3.0), (5.0, 15.0), (15.0, 5.0),
        (20.0, 80.0), (80.0, 20.0), (50.0, 200.0), (200.0, 50.0),
        (100.0, 20.0), (20.0, 100.0), (200.0, 40.0), (40.0, 200.0),
        (0.5, 0.5), (1.0, 1.0), (1.0, 2.0), (2.0, 1.0),
    ]

    for a0, b0 in init_guesses:
        try:
            result = least_squares(
                residuals,
                [a0, b0],
                bounds=([0.1, 0.1], [2000, 2000]),
                ftol=1e-12,
                xtol=1e-12
            )

            if result.success and result.cost < best_cost:
                best_solution = result.x
                best_cost = result.cost

        except (ValueError, RuntimeError):
            continue

    if best_solution is None:
        return None

    a_fit, b_fit = best_solution
    p25_err = abs(sp_stats.beta.cdf(p25, a_fit, b_fit) - 0.25)
    p50_err = abs(sp_stats.beta.cdf(p50, a_fit, b_fit) - 0.50)
    p75_err = abs(sp_stats.beta.cdf(p75, a_fit, b_fit) - 0.75)
    max_err = max(p25_err, p50_err, p75_err)

    # Try minimax refinement if needed
    if max_err > 0.04:
        def max_error_obj(params):
            a, b = params
            if a <= 0.1 or b <= 0.1 or a > 2000 or b > 2000:
                return 1.0
            e25 = abs(sp_stats.beta.cdf(p25, a, b) - 0.25)
            e50 = abs(sp_stats.beta.cdf(p50, a, b) - 0.50)
            e75 = abs(sp_stats.beta.cdf(p75, a, b) - 0.75)
            return max(e25, e50, e75)

        try:
            result = minimize(
                max_error_obj, [a_fit, b_fit],
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if result.success:
                a_new, b_new = result.x
                if 0.1 < a_new < 2000 and 0.1 < b_new < 2000:
                    new_max_err = max_error_obj([a_new, b_new])
                    if new_max_err < max_err:
                        a_fit, b_fit = a_new, b_new
                        max_err = new_max_err
        except (ValueError, RuntimeError):
            pass

    if max_err > 0.05:
        return None

    return (float(a_fit), float(b_fit), float(max_err))


# Quantile Functions and Wasserstein
def quantile_function(alpha: float, beta: float) -> np.ndarray:
    """Compute quantile function of Beta(alpha, beta) on Q_GRID."""
    return sp_stats.beta.ppf(Q_GRID, alpha, beta)


def w1_distance(qf1: np.ndarray, qf2: np.ndarray) -> float:
    """L1-Wasserstein distance between two distributions."""
    return np.trapz(np.abs(qf1 - qf2), Q_GRID)


def w2_squared(qf1: np.ndarray, qf2: np.ndarray) -> float:
    """L2-Wasserstein squared distance between two distributions."""
    return np.trapz((qf1 - qf2) ** 2, Q_GRID)


# Data Loading
def load_all_runs():
    """Load all runs and classify them by prompt characteristics."""
    condition_runs = defaultdict(list)

    for run_dir in sorted(RUNS_DIR.glob('202603*')):
        if not run_dir.is_dir():
            continue

        json_file = run_dir / 'full_results.json'
        if not json_file.exists():
            continue

        with open(json_file) as f:
            data = json.load(f)

        # Check duration
        duration = data['run_metadata']['duration_seconds']
        if duration < MIN_DURATION:
            continue

        scenario = data['run_metadata']['config_used']['scenario_file'].split('/')[-1]

        # Get prompt characteristics
        response = data['results_per_step'][0]['results_per_task'][0]['rounds_data'][0]['responses'][0]

        analysis_len = len(response.get('analysis_user_prompt', ''))
        estimation_len = len(response.get('estimation_user_prompt', ''))
        has_reasoning = '<reasoning_structure>' in response.get('estimation_user_prompt', '')

        # Classify run
        condition = classify_run(analysis_len, estimation_len, has_reasoning, scenario)

        if condition != 'unknown':
            condition_runs[condition].append(run_dir.name)

    return condition_runs


def load_estimates_for_run(run_id):
    """Load estimates from a run's CSV file."""
    csv_path = RUNS_DIR / run_id / "detailed_estimates.csv"

    if not csv_path.exists():
        return []

    estimates = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                estimates.append({
                    "p25": float(row["percentile_25th"]),
                    "p50": float(row["percentile_50th"]),
                    "p75": float(row["percentile_75th"]),
                })
            except (ValueError, KeyError):
                continue

    return estimates


# Main Analysis
def main():
    print("=" * 80)
    print("PROMPT SENSITIVITY: Complete Analysis of All 7 Conditions")
    print("=" * 80)
    print()

    # Load data
    print("Loading and classifying runs by prompt characteristics...")
    condition_runs = load_all_runs()

    print(f"\nRuns per condition (MIN_DURATION = {MIN_DURATION}s):")
    print("-" * 80)

    condition_order = [
        'control',
        'no_baseline',
        'no_ci',
        'no_baseline_no_ci',
        'skip_analysis',
        'trim_reasoning',
        'trim_all'
    ]

    for condition_name in condition_order:
        count = len(condition_runs.get(condition_name, []))
        print(f"  {condition_name:20s}: {count:2d} runs")

    # Load and fit distributions
    print("\nFitting Beta distributions...")
    print("-" * 80)

    condition_qfs = {}
    condition_params = {}

    for condition_name in condition_order:
        run_ids = condition_runs.get(condition_name, [])
        qfs = []
        params = []
        fit_failures = 0

        for run_id in run_ids:
            estimates = load_estimates_for_run(run_id)

            for est in estimates:
                result = fit_beta_from_percentiles(est["p25"], est["p50"], est["p75"])
                if result is not None:
                    alpha, beta, error = result
                    qf = quantile_function(alpha, beta)
                    qfs.append(qf)
                    params.append((alpha, beta))
                else:
                    fit_failures += 1

        condition_qfs[condition_name] = qfs
        condition_params[condition_name] = params
        print(f"  {condition_name:20s}: {len(qfs):3d} distributions, {fit_failures:2d} failed")

    # Compute Wasserstein distances to control
    print("\n" + "=" * 80)
    print("WASSERSTEIN DISTANCES TO CONTROL")
    print("=" * 80)

    control_qfs = condition_qfs['control']

    results = []

    for condition_name in condition_order:
        if condition_name == 'control':
            continue

        cond_qfs = condition_qfs[condition_name]

        if len(cond_qfs) == 0:
            print(f"\n{condition_name}: NO DATA")
            continue

        # Compute pairwise W1 and W2 distances
        w1_distances = []
        w2_distances = []

        for qf_control in control_qfs:
            for qf_cond in cond_qfs:
                w1_distances.append(w1_distance(qf_control, qf_cond))
                w2_distances.append(np.sqrt(w2_squared(qf_control, qf_cond)))

        mean_w1 = np.mean(w1_distances)
        std_w1 = np.std(w1_distances)
        mean_w2 = np.mean(w2_distances)
        std_w2 = np.std(w2_distances)

        results.append({
            'condition': condition_name,
            'n': len(cond_qfs),
            'mean_w1': mean_w1,
            'std_w1': std_w1,
            'mean_w2': mean_w2,
            'std_w2': std_w2
        })

        print(f"\n{condition_name:20s} (n={len(cond_qfs):2d}):")
        print(f"  W1 distance: {mean_w1:.6f} ± {std_w1:.6f}")
        print(f"  W2 distance: {mean_w2:.6f} ± {std_w2:.6f}")

    # Within-control variance
    print(f"\nWITHIN-CONTROL variance:")
    within_w1 = []
    within_w2 = []
    for i, qf1 in enumerate(control_qfs):
        for j, qf2 in enumerate(control_qfs):
            if i < j:
                within_w1.append(w1_distance(qf1, qf2))
                within_w2.append(np.sqrt(w2_squared(qf1, qf2)))

    print(f"  W1 distance: {np.mean(within_w1):.6f} ± {np.std(within_w1):.6f}")
    print(f"  W2 distance: {np.mean(within_w2):.6f} ± {np.std(within_w2):.6f}")

    # Save results to file
    output_file = OUTPUT_DIR / "wasserstein_distances_all_conditions.txt"
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("WASSERSTEIN DISTANCES TO CONTROL CONDITION\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'Condition':<20} {'N':<5} {'Mean W1':<12} {'Std W1':<12} {'Mean W2':<12} {'Std W2':<12}\n")
        f.write("-" * 80 + "\n")

        for r in results:
            f.write(f"{r['condition']:<20} {r['n']:<5} {r['mean_w1']:<12.6f} {r['std_w1']:<12.6f} {r['mean_w2']:<12.6f} {r['std_w2']:<12.6f}\n")

        f.write("\nWithin-control variance:\n")
        f.write(f"  W1: {np.mean(within_w1):.6f} ± {np.std(within_w1):.6f}\n")
        f.write(f"  W2: {np.mean(within_w2):.6f} ± {np.std(within_w2):.6f}\n")

    print(f"\nResults saved to: {output_file}")

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Plot 1: W1 distances with error bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    conditions = [r['condition'] for r in results]
    w1_means = [r['mean_w1'] for r in results]
    w1_stds = [r['std_w1'] for r in results]
    w2_means = [r['mean_w2'] for r in results]
    w2_stds = [r['std_w2'] for r in results]

    # W1 plot
    x_pos = np.arange(len(conditions))
    ax1.bar(x_pos, w1_means, yerr=w1_stds, capsize=5, alpha=0.7,
            color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'])
    ax1.axhline(y=np.mean(within_w1), color='blue', linestyle='--', linewidth=2,
                label=f'Within-control: {np.mean(within_w1):.6f}')
    ax1.set_xlabel('Condition', fontsize=12)
    ax1.set_ylabel('W1 Wasserstein Distance', fontsize=12)
    ax1.set_title('W1 Distance from Control Condition', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([c.replace('_', '\n') for c in conditions], rotation=0, ha='center')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # W2 plot
    ax2.bar(x_pos, w2_means, yerr=w2_stds, capsize=5, alpha=0.7,
            color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'])
    ax2.axhline(y=np.mean(within_w2), color='blue', linestyle='--', linewidth=2,
                label=f'Within-control: {np.mean(within_w2):.6f}')
    ax2.set_xlabel('Condition', fontsize=12)
    ax2.set_ylabel('W2 Wasserstein Distance', fontsize=12)
    ax2.set_title('W2 Distance from Control Condition', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([c.replace('_', '\n') for c in conditions], rotation=0, ha='center')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_chart = OUTPUT_DIR / "wasserstein_distances_all_conditions.png"
    plt.savefig(output_chart, dpi=150, bbox_inches='tight')
    print(f"Chart saved to: {output_chart}")

    plt.close()

    # Plot 2: Combined plot with both W1 and W2
    fig, ax = plt.subplots(figsize=(14, 8))

    x_pos = np.arange(len(conditions))
    width = 0.35

    bars1 = ax.bar(x_pos - width/2, w1_means, width, yerr=w1_stds, capsize=5,
                   alpha=0.8, label='W1 Distance', color='steelblue')
    bars2 = ax.bar(x_pos + width/2, w2_means, width, yerr=w2_stds, capsize=5,
                   alpha=0.8, label='W2 Distance', color='coral')

    ax.axhline(y=np.mean(within_w1), color='blue', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'W1 within-control: {np.mean(within_w1):.6f}')
    ax.axhline(y=np.mean(within_w2), color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'W2 within-control: {np.mean(within_w2):.6f}')

    ax.set_xlabel('Condition', fontsize=13, fontweight='bold')
    ax.set_ylabel('Wasserstein Distance', fontsize=13, fontweight='bold')
    ax.set_title('Prompt Sensitivity: Wasserstein Distances from Control Condition',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([c.replace('_', '\n') for c in conditions], rotation=0, ha='center', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_chart2 = OUTPUT_DIR / "wasserstein_combined_all_conditions.png"
    plt.savefig(output_chart2, dpi=150, bbox_inches='tight')
    print(f"Combined chart saved to: {output_chart2}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
