#!/usr/bin/env python3
"""Proper Fréchet ANOVA analysis for prompt sensitivity across 4 conditions.

Uses distributional approach:
1. Fit Beta(α,β) from (p25, p50, p75) for each run
2. Compute quantile functions for each fitted Beta
3. Run Fréchet ANOVA to test if conditions differ distributionally
4. Compute ICC_F to quantify between-condition vs within-condition variance
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


# Configuration
OUTPUT_DIR = Path(__file__).parent / "output"
RUNS_DIR = OUTPUT_DIR / "runs"

CONDITIONS = {
    "control": "scenario_control.yaml",
    "no_baseline": "scenario_no_baseline.yaml",
    "no_confidence": "scenario_no_ci.yaml",
    "no_baseline_no_ci": "scenario_no_baseline_no_ci.yaml",
}

MIN_DURATION = 50.0
QUANTILE_GRID_SIZE = 201
N_PERMUTATIONS = 5_000

# Quantile grid - avoid exact 0 and 1
Q_GRID = np.linspace(0.001, 0.999, QUANTILE_GRID_SIZE)


# Beta Fitting (from existing code)
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


def w2_squared(qf1: np.ndarray, qf2: np.ndarray) -> float:
    """L2-Wasserstein squared distance between two distributions."""
    return np.trapz((qf1 - qf2) ** 2, Q_GRID)


# Fréchet ANOVA
def _w2sq_matrix_to_center(qf_matrix: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Vectorized W2-squared from each row of qf_matrix to center."""
    diffs_sq = (qf_matrix - center[np.newaxis, :]) ** 2
    return np.trapz(diffs_sq, Q_GRID, axis=1)


def _frechet_anova_from_matrix(
    qf_matrix: np.ndarray, labels: np.ndarray, group_ids: List, n_total: int
) -> Tuple[float, float, float, float, float]:
    """Compute Fréchet ANOVA statistic from (n, grid) matrix and label array."""
    k = len(group_ids)

    # Pooled Fréchet mean and variance
    mu_pooled = qf_matrix.mean(axis=0)
    d2_pooled = _w2sq_matrix_to_center(qf_matrix, mu_pooled)
    v_pooled = d2_pooled.mean()

    # Per-group
    group_vars = np.zeros(k)
    lambdas = np.zeros(k)
    for j, gid in enumerate(group_ids):
        mask = labels == j
        n_g = mask.sum()
        lambdas[j] = n_g / n_total
        mu_g = qf_matrix[mask].mean(axis=0)
        d2_g = _w2sq_matrix_to_center(qf_matrix[mask], mu_g)
        group_vars[j] = d2_g.mean()

    f_n = v_pooled - np.sum(lambdas * group_vars)

    u_n = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            u_n += lambdas[i] * lambdas[j] * (group_vars[i] - group_vars[j]) ** 2

    t_n = n_total * f_n + n_total * u_n
    icc_f = max(f_n / v_pooled, 0.0) if v_pooled > 0 else 0.0

    return t_n, f_n, u_n, v_pooled, icc_f


def frechet_anova_statistic(
    groups: Dict[str, List[np.ndarray]]
) -> Tuple[float, float, float, float, float, np.ndarray]:
    """Compute Fréchet ANOVA test statistic.

    Returns (T_n, F_n, U_n, V_pooled, ICC_F, group_variances).
    """
    group_names = sorted(groups.keys())
    all_qfs = []
    labels = []
    for j, g in enumerate(group_names):
        for qf in groups[g]:
            all_qfs.append(qf)
            labels.append(j)

    qf_matrix = np.array(all_qfs)
    labels_arr = np.array(labels)
    n_total = len(all_qfs)

    t_n, f_n, u_n, v_pooled, icc_f = _frechet_anova_from_matrix(
        qf_matrix, labels_arr, group_names, n_total
    )

    # Also compute individual group variances
    group_vars = np.zeros(len(group_names))
    for j, gname in enumerate(group_names):
        mask = labels_arr == j
        mu_g = qf_matrix[mask].mean(axis=0)
        d2_g = _w2sq_matrix_to_center(qf_matrix[mask], mu_g)
        group_vars[j] = d2_g.mean()

    return t_n, f_n, u_n, v_pooled, icc_f, group_vars


def permutation_test(
    groups: Dict[str, List[np.ndarray]],
    observed_t: float,
    n_perm: int = N_PERMUTATIONS,
    seed: int = 42,
) -> float:
    """Vectorized permutation test for Fréchet ANOVA."""
    rng = np.random.RandomState(seed)
    group_names = sorted(groups.keys())

    all_qfs = []
    labels = []
    for j, g in enumerate(group_names):
        for qf in groups[g]:
            all_qfs.append(qf)
            labels.append(j)

    qf_matrix = np.array(all_qfs)
    labels_arr = np.array(labels)
    n_total = len(all_qfs)

    count_ge = 0
    for _ in range(n_perm):
        perm_labels = rng.permutation(labels_arr)
        t_perm, _, _, _, _ = _frechet_anova_from_matrix(
            qf_matrix, perm_labels, group_names, n_total
        )
        if t_perm >= observed_t:
            count_ge += 1

    return (count_ge + 1) / (n_perm + 1)


def asymptotic_pvalue(t_n: float, k: int) -> float:
    """Asymptotic p-value: T_n ~ chi-squared(k-1) under H0."""
    return 1.0 - sp_stats.chi2.cdf(t_n, df=k - 1)


# Data Loading
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
    print("FRÉCHET ANOVA: Prompt Sensitivity Analysis (4 Conditions)")
    print("=" * 80)
    print()

    # Load data
    print("Loading data...")
    registry = load_registry()
    condition_runs = filter_valid_runs(registry)

    print("\nRuns per condition:")
    for condition_name in ["control", "no_baseline", "no_confidence", "no_baseline_no_ci"]:
        count = len(condition_runs[condition_name])
        print(f"  {condition_name:20s}: {count:2d} runs")

    # Load and fit distributions
    print("\nFitting Beta distributions...")
    condition_qfs = {}

    for condition_name, run_ids in condition_runs.items():
        qfs = []
        fit_failures = 0

        for run_id in run_ids:
            estimates = load_estimates_for_run(run_id)

            for est in estimates:
                result = fit_beta_from_percentiles(est["p25"], est["p50"], est["p75"])
                if result is not None:
                    alpha, beta, error = result
                    qf = quantile_function(alpha, beta)
                    qfs.append(qf)
                else:
                    fit_failures += 1

        condition_qfs[condition_name] = qfs
        print(f"  {condition_name:20s}: {len(qfs):3d} distributions, {fit_failures:2d} failed")

    # Fréchet ANOVA
    print("\n" + "=" * 80)
    print("FRÉCHET ANOVA RESULTS")
    print("=" * 80)

    groups = {cond: qfs for cond, qfs in condition_qfs.items()}
    t_n, f_n, u_n, v_pooled, icc_f, group_vars = frechet_anova_statistic(groups)

    k = len(groups)
    n_total = sum(len(qfs) for qfs in groups.values())

    print(f"\nSample sizes:")
    for i, cond in enumerate(sorted(groups.keys())):
        print(f"  {cond:20s}: n = {len(groups[cond])}")

    print(f"\nTest Statistics:")
    print(f"  T_n (test statistic):     {t_n:.4f}")
    print(f"  F_n (between-group var):  {f_n:.6f}")
    print(f"  U_n (var of group vars):  {f_n:.6f}")
    print(f"  V_pooled (total var):     {v_pooled:.6f}")
    print(f"  ICC_F (between/total):    {icc_f:.4f}")

    print(f"\nInterpretation:")
    print(f"  ICC_F = {icc_f:.1%} of variance is BETWEEN conditions")
    print(f"        = {1-icc_f:.1%} of variance is WITHIN conditions (prompt sensitivity)")

    # P-values
    p_asymptotic = asymptotic_pvalue(t_n, k)
    print(f"\nAsymptotic p-value (χ²({k-1})): {p_asymptotic:.6f}")

    print(f"\nRunning permutation test ({N_PERMUTATIONS:,} permutations)...")
    p_perm = permutation_test(groups, t_n, n_perm=N_PERMUTATIONS)
    print(f"Permutation p-value:        {p_perm:.6f}")

    if p_perm < 0.001:
        print("\n*** HIGHLY SIGNIFICANT difference between conditions (p < 0.001)")
    elif p_perm < 0.01:
        print("\n** Significant difference between conditions (p < 0.01)")
    elif p_perm < 0.05:
        print("\n* Significant difference between conditions (p < 0.05)")
    else:
        print("\nNO significant difference between conditions")

    # Group-specific variances
    print("\n" + "=" * 80)
    print("WITHIN-CONDITION VARIANCES (W² Wasserstein)")
    print("=" * 80)

    for i, cond in enumerate(sorted(groups.keys())):
        print(f"{cond:20s}: {group_vars[i]:.6f}")

    relative_vars = group_vars / group_vars[0]  # Relative to control
    print(f"\nRelative to control:")
    for i, cond in enumerate(sorted(groups.keys())):
        print(f"{cond:20s}: {relative_vars[i]:.2f}x")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n1. MAIN EFFECT: Conditions differ significantly (p = {p_perm:.6f})")
    print(f"2. EFFECT SIZE: {icc_f:.1%} of variance between conditions")
    print(f"3. PROMPT SENSITIVITY: {1-icc_f:.1%} of variance within conditions")
    print(f"\n4. CONDITION-SPECIFIC PROMPT SENSITIVITY:")
    for i, cond in enumerate(sorted(groups.keys())):
        print(f"   {cond:20s}: {relative_vars[i]:.2f}x baseline variability")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
