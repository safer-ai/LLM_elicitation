#!/usr/bin/env python3
"""Compare prompt sensitivity across 4 experimental conditions.

Conditions:
1. Control (baseline + confidence intervals)
2. No baseline (no baseline + confidence intervals)
3. No confidence (baseline + no confidence intervals)
4. No baseline no CI (no baseline + no confidence intervals)

Analysis:
- Fit Beta distributions from percentiles
- Compute Wasserstein distances between conditions
- Run Fréchet ANOVA to test for differences
- Compute ICC (Intraclass Correlation Coefficient)
"""

import json
import csv
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, List, Dict
import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import least_squares
from itertools import combinations


# Configuration
OUTPUT_DIR = Path(__file__).parent / "output"
RUNS_DIR = OUTPUT_DIR / "runs"

CONDITIONS = {
    "control": "scenario_control.yaml",
    "no_baseline": "scenario_no_baseline.yaml",
    "no_confidence": "scenario_no_ci.yaml",
    "no_baseline_no_ci": "scenario_no_baseline_no_ci.yaml",
}

MIN_DURATION = 50.0  # Filter out test runs


# Beta Distribution Fitting
def fit_beta_from_percentiles(
    p25: float, p50: float, p75: float
) -> Optional[Tuple[float, float, float]]:
    """Fit Beta(alpha, beta) on [0,1] using least-squares on all three percentiles.

    Returns (alpha, beta, max_error) where max_error = max(|CDF(p_i) - target_i|)
    Returns None if fitting fails or max_error > 0.05.
    """
    # Validation
    if not (0 < p25 < p50 < p75 < 1):
        return None

    # Clamp percentiles away from 0/1
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

    # Estimate concentration based on IQR and location
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

    # Check errors
    a_fit, b_fit = best_solution
    p25_err = abs(sp_stats.beta.cdf(p25, a_fit, b_fit) - 0.25)
    p50_err = abs(sp_stats.beta.cdf(p50, a_fit, b_fit) - 0.50)
    p75_err = abs(sp_stats.beta.cdf(p75, a_fit, b_fit) - 0.75)
    max_error = max(p25_err, p50_err, p75_err)

    if max_error > 0.05:
        return None

    return (float(a_fit), float(b_fit), float(max_error))


# Wasserstein Distance
def wasserstein_distance_beta(alpha1: float, beta1: float,
                              alpha2: float, beta2: float,
                              n_samples: int = 10000) -> float:
    """Compute L2 Wasserstein distance between two Beta distributions."""
    # Using quantile-based computation
    q_grid = np.linspace(0.001, 0.999, 201)

    q1 = sp_stats.beta.ppf(q_grid, alpha1, beta1)
    q2 = sp_stats.beta.ppf(q_grid, alpha2, beta2)

    # L2 Wasserstein distance via quantile functions
    w2_squared = np.trapz((q1 - q2) ** 2, q_grid)
    return np.sqrt(w2_squared)


# Load Data
def load_registry():
    """Load the run registry."""
    with open(OUTPUT_DIR / "run_registry.json", "r") as f:
        return json.load(f)


def filter_valid_runs(registry):
    """Filter valid runs by duration and group by condition."""
    condition_runs = defaultdict(list)

    for run in registry["runs"]:
        duration = run["metadata"]["duration_seconds"]
        if duration < MIN_DURATION:
            continue

        scenario_file = run["metadata"]["scenario_file"].split("/")[-1]

        for condition_name, scenario_filename in CONDITIONS.items():
            if scenario_file == scenario_filename:
                condition_runs[condition_name].append(run["run_id"])
                break

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
                    "estimate": float(row["estimate"]),
                    "p25": float(row["percentile_25th"]),
                    "p50": float(row["percentile_50th"]),
                    "p75": float(row["percentile_75th"]),
                    "task_name": row["task_name"],
                    "step_name": row["step_name"],
                })
            except (ValueError, KeyError):
                continue

    return estimates


# Main Analysis
def main():
    print("=" * 80)
    print("PROMPT SENSITIVITY ANALYSIS: 4-Condition Comparison")
    print("=" * 80)
    print()

    # Load data
    print("Loading data...")
    registry = load_registry()
    condition_runs = filter_valid_runs(registry)

    # Print run counts
    print("\nRuns per condition:")
    for condition_name in ["control", "no_baseline", "no_confidence", "no_baseline_no_ci"]:
        count = len(condition_runs[condition_name])
        print(f"  {condition_name:20s}: {count:2d} runs")

    # Load all estimates
    print("\nLoading estimates...")
    condition_data = {}

    for condition_name, run_ids in condition_runs.items():
        all_estimates = []
        for run_id in run_ids:
            estimates = load_estimates_for_run(run_id)
            all_estimates.extend(estimates)

        condition_data[condition_name] = all_estimates
        print(f"  {condition_name:20s}: {len(all_estimates):3d} estimates")

    # Fit Beta distributions
    print("\nFitting Beta distributions...")
    condition_betas = {}

    for condition_name, estimates in condition_data.items():
        betas = []
        failed = 0

        for est in estimates:
            result = fit_beta_from_percentiles(est["p25"], est["p50"], est["p75"])
            if result is not None:
                alpha, beta, error = result
                betas.append({
                    "alpha": alpha,
                    "beta": beta,
                    "error": error,
                    "p50": est["p50"],
                    "estimate": est["estimate"],
                })
            else:
                failed += 1

        condition_betas[condition_name] = betas
        print(f"  {condition_name:20s}: {len(betas):3d} fitted, {failed:2d} failed")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for condition_name in ["control", "no_baseline", "no_confidence", "no_baseline_no_ci"]:
        betas = condition_betas[condition_name]
        p50s = [b["p50"] for b in betas]
        alphas = [b["alpha"] for b in betas]
        betas_param = [b["beta"] for b in betas]

        print(f"\n{condition_name.upper()}:")
        print(f"  Median estimate (p50):")
        print(f"    Mean: {np.mean(p50s):.4f}")
        print(f"    Std:  {np.std(p50s):.4f}")
        print(f"    Min:  {np.min(p50s):.4f}")
        print(f"    Max:  {np.max(p50s):.4f}")
        print(f"  Beta parameters:")
        print(f"    Alpha - Mean: {np.mean(alphas):.2f}, Std: {np.std(alphas):.2f}")
        print(f"    Beta  - Mean: {np.mean(betas_param):.2f}, Std: {np.std(betas_param):.2f}")

    # Pairwise Wasserstein distances
    print("\n" + "=" * 80)
    print("PAIRWISE WASSERSTEIN DISTANCES (Mean ± Std)")
    print("=" * 80)

    condition_names = ["control", "no_baseline", "no_confidence", "no_baseline_no_ci"]

    for cond1, cond2 in combinations(condition_names, 2):
        betas1 = condition_betas[cond1]
        betas2 = condition_betas[cond2]

        # Compute all pairwise distances
        distances = []
        for b1 in betas1:
            for b2 in betas2:
                dist = wasserstein_distance_beta(
                    b1["alpha"], b1["beta"],
                    b2["alpha"], b2["beta"]
                )
                distances.append(dist)

        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        print(f"{cond1:20s} vs {cond2:20s}: {mean_dist:.6f} ± {std_dist:.6f}")

    # Within-condition Wasserstein distances (for ICC-like measure)
    print("\n" + "=" * 80)
    print("WITHIN-CONDITION VARIATION (Wasserstein distances)")
    print("=" * 80)

    for condition_name in condition_names:
        betas = condition_betas[condition_name]

        if len(betas) < 2:
            print(f"{condition_name:20s}: N/A (insufficient data)")
            continue

        # Compute all pairwise within-condition distances
        distances = []
        for i, b1 in enumerate(betas):
            for b2 in betas[i+1:]:
                dist = wasserstein_distance_beta(
                    b1["alpha"], b1["beta"],
                    b2["alpha"], b2["beta"]
                )
                distances.append(dist)

        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        print(f"{condition_name:20s}: {mean_dist:.6f} ± {std_dist:.6f}")

    # Simple ANOVA on medians
    print("\n" + "=" * 80)
    print("ONE-WAY ANOVA ON MEDIAN ESTIMATES (p50)")
    print("=" * 80)

    groups = [
        [b["p50"] for b in condition_betas["control"]],
        [b["p50"] for b in condition_betas["no_baseline"]],
        [b["p50"] for b in condition_betas["no_confidence"]],
        [b["p50"] for b in condition_betas["no_baseline_no_ci"]],
    ]

    f_stat, p_value = sp_stats.f_oneway(*groups)
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value:     {p_value:.6f}")

    if p_value < 0.001:
        print("Result: HIGHLY SIGNIFICANT difference between conditions (p < 0.001)")
    elif p_value < 0.01:
        print("Result: Significant difference between conditions (p < 0.01)")
    elif p_value < 0.05:
        print("Result: Significant difference between conditions (p < 0.05)")
    else:
        print("Result: NO significant difference between conditions")

    # Post-hoc pairwise t-tests
    print("\n" + "=" * 80)
    print("POST-HOC PAIRWISE T-TESTS (Welch's t-test)")
    print("=" * 80)

    for cond1, cond2 in combinations(condition_names, 2):
        group1 = [b["p50"] for b in condition_betas[cond1]]
        group2 = [b["p50"] for b in condition_betas[cond2]]

        t_stat, p_val = sp_stats.ttest_ind(group1, group2, equal_var=False)

        print(f"{cond1:20s} vs {cond2:20s}:")
        print(f"  t = {t_stat:7.4f}, p = {p_val:.6f}", end="")

        if p_val < 0.001:
            print(" ***")
        elif p_val < 0.01:
            print(" **")
        elif p_val < 0.05:
            print(" *")
        else:
            print()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
