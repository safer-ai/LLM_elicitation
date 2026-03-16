#!/usr/bin/env python3
"""Fréchet ANOVA on fitted Beta distributions using Wasserstein distance (PERCENTILE VERSION).

Replaces scalar ANOVA on point estimates with a distributional test:
  1. Fit Beta(α, β) on [0,1] from each observation's (p25, p50, p75)
     using least-squares minimization on all three percentiles.
  2. Compute L2-Wasserstein distances via quantile functions
  3. Run Fréchet ANOVA (Dubey & Müller, 2019) with permutation test

Beta fitting: Use all three percentiles to minimize squared error:
  - minimize Σ(CDF(p_i) - target_i)² for i ∈ {25, 50, 75}
  - Reject fits where max|CDF(p_i) - target_i| > 0.05

Usage:
    python3 frechet_anova_percentile.py
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, List, Dict

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import least_squares, minimize

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPERIMENTS_DIR = Path(__file__).parent.parent

DATA_DIRS = {
    ("Claude Sonnet 4.5", "TA0002 (50%)"): EXPERIMENTS_DIR / "percentile_claude_TA0002_50pct",
    ("Claude Sonnet 4.5", "TA0007 (85%)"): EXPERIMENTS_DIR / "percentile_claude_TA0007_85pct",
    ("Claude Sonnet 4.5", "T1657 (30%)"):  EXPERIMENTS_DIR / "percentile_claude_T1657_30pct",
    ("GPT-4o", "TA0002 (50%)"): EXPERIMENTS_DIR / "percentile_gpt4o_TA0002_50pct",
    ("GPT-4o", "TA0007 (85%)"): EXPERIMENTS_DIR / "percentile_gpt4o_TA0007_85pct",
    ("GPT-4o", "T1657 (30%)"):  EXPERIMENTS_DIR / "percentile_gpt4o_T1657_30pct",
    ("Gemini 2.5 Pro", "TA0002 (50%)"): EXPERIMENTS_DIR / "percentile_gemini_TA0002_50pct",
    ("Gemini 2.5 Pro", "TA0007 (85%)"): EXPERIMENTS_DIR / "percentile_gemini_TA0007_85pct",
    ("Gemini 2.5 Pro", "T1657 (30%)"):  EXPERIMENTS_DIR / "percentile_gemini_T1657_30pct",
}

STEPS = ["TA0002 (50%)", "TA0007 (85%)", "T1657 (30%)"]
MODELS = ["Claude Sonnet 4.5", "GPT-4o", "Gemini 2.5 Pro"]

QUANTILE_GRID_SIZE = 201
N_PERMUTATIONS = 5_000

# Quantile grid — avoid exact 0 and 1 where Beta quantile function diverges
Q_GRID = np.linspace(0.001, 0.999, QUANTILE_GRID_SIZE)


# ---------------------------------------------------------------------------
# Step 1: Beta Distribution Fitting from Percentiles
# ---------------------------------------------------------------------------

def fit_beta_from_percentiles(
    p25: float, p50: float, p75: float
) -> Optional[Tuple[float, float, float]]:
    """Fit Beta(alpha, beta) on [0,1] using least-squares on all three percentiles.

    Minimizes squared error across:
      - CDF(p25) = 0.25
      - CDF(p50) = 0.50
      - CDF(p75) = 0.75

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
    # Narrower IQR requires higher α+β; location determines α/(α+β)
    iqr = p75 - p25
    loc = p50  # approximate location

    # For Beta, IQR ≈ 1.35 * σ where σ ≈ sqrt(αβ/((α+β)²(α+β+1)))
    # Rough heuristic: concentration ≈ 0.5 / IQR² for narrow distributions
    est_concentration = max(2, min(1000, 0.1 / (iqr ** 2))) if iqr > 0 else 50
    est_alpha = est_concentration * loc
    est_beta = est_concentration * (1 - loc)

    init_guesses = [
        # Data-driven initial guess
        (est_alpha, est_beta),
        (est_alpha * 0.5, est_beta * 0.5),
        (est_alpha * 2, est_beta * 2),
        # Standard guesses
        (2.0, 2.0), (5.0, 5.0), (10.0, 10.0), (20.0, 20.0),
        (50.0, 50.0), (100.0, 100.0), (200.0, 200.0),
        # Skewed distributions
        (3.0, 8.0), (8.0, 3.0), (5.0, 15.0), (15.0, 5.0),
        (20.0, 80.0), (80.0, 20.0), (50.0, 200.0), (200.0, 50.0),
        # High concentration for narrow distributions
        (100.0, 20.0), (20.0, 100.0), (200.0, 40.0), (40.0, 200.0),
        (300.0, 60.0), (60.0, 300.0), (500.0, 100.0), (100.0, 500.0),
        # Low concentration
        (0.5, 0.5), (1.0, 1.0), (1.0, 2.0), (2.0, 1.0),
    ]

    for a0, b0 in init_guesses:
        try:
            result = least_squares(
                residuals,
                [a0, b0],
                bounds=([0.1, 0.1], [2000, 2000]),  # Increased upper bounds
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

    # Check all errors from least_squares solution
    a_fit, b_fit = best_solution
    p25_err = abs(sp_stats.beta.cdf(p25, a_fit, b_fit) - 0.25)
    p50_err = abs(sp_stats.beta.cdf(p50, a_fit, b_fit) - 0.50)
    p75_err = abs(sp_stats.beta.cdf(p75, a_fit, b_fit) - 0.75)
    max_err = max(p25_err, p50_err, p75_err)

    # If max_err is close to threshold, try minimax refinement
    # least_squares minimizes sum of squares, not max error
    if max_err > 0.04:
        def max_error_obj(params):
            a, b = params
            if a <= 0.1 or b <= 0.1 or a > 2000 or b > 2000:
                return 1.0
            e25 = abs(sp_stats.beta.cdf(p25, a, b) - 0.25)
            e50 = abs(sp_stats.beta.cdf(p50, a, b) - 0.50)
            e75 = abs(sp_stats.beta.cdf(p75, a, b) - 0.75)
            return max(e25, e50, e75)

        # Try Nelder-Mead refinement from least_squares solution
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

    return (a_fit, b_fit, max_err)


# ---------------------------------------------------------------------------
# Step 2: Wasserstein Distance
# ---------------------------------------------------------------------------

def quantile_function(alpha: float, beta: float, grid: np.ndarray = Q_GRID) -> np.ndarray:
    """Compute the quantile function of Beta(alpha, beta) on the grid."""
    return sp_stats.beta.ppf(grid, alpha, beta)


def w2_squared(qf1: np.ndarray, qf2: np.ndarray) -> float:
    """L2-Wasserstein squared distance between two distributions."""
    return np.trapz((qf1 - qf2) ** 2, Q_GRID)


def w1_distance(qf1: np.ndarray, qf2: np.ndarray) -> float:
    """L1-Wasserstein distance between two distributions."""
    return np.trapz(np.abs(qf1 - qf2), Q_GRID)


# ---------------------------------------------------------------------------
# Step 3 & 4: Vectorized Fréchet ANOVA
# ---------------------------------------------------------------------------

def _w2sq_matrix_to_center(qf_matrix: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Vectorized W2-squared from each row of qf_matrix to center.
    qf_matrix: (n, grid_size), center: (grid_size,)
    Returns: (n,) array of squared distances.
    """
    diffs_sq = (qf_matrix - center[np.newaxis, :]) ** 2
    return np.trapz(diffs_sq, Q_GRID, axis=1)


def _frechet_anova_from_matrix(
    qf_matrix: np.ndarray, labels: np.ndarray, group_ids: List, n_total: int
) -> Tuple[float, float, float, float, float]:
    """Compute Fréchet ANOVA statistic from a (n, grid) matrix and label array."""
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
) -> Tuple[float, float, float, float, float]:
    """Compute Fréchet ANOVA test statistic T_n.
    Returns (T_n, F_n, U_n, V_pooled, ICC_F).
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

    return _frechet_anova_from_matrix(qf_matrix, labels_arr, group_names, n_total)


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


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_and_fit(data_dir: Path) -> Dict[str, List[Tuple[np.ndarray, dict]]]:
    """Load CSV data, fit Beta distributions from percentiles, return {expert: [(qf, meta), ...]}.

    Each entry is a quantile function array + metadata dict.
    """
    expert_dists = defaultdict(list)
    fit_failures = 0
    validation_warnings = 0
    total = 0

    run_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    for run_dir in run_dirs:
        csv_path = run_dir / "detailed_estimates.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                # Skip rows with errors
                if row.get("has_error", "").strip().lower() == "true":
                    continue

                total += 1
                try:
                    p25_val = float(row.get("percentile_25th", "").strip())
                    p50_val = float(row.get("percentile_50th", "").strip())
                    p75_val = float(row.get("percentile_75th", "").strip())
                except (ValueError, AttributeError):
                    fit_failures += 1
                    continue

                result = fit_beta_from_percentiles(p25_val, p50_val, p75_val)
                if result is None:
                    fit_failures += 1
                    continue

                alpha, beta_param, max_error = result
                if max_error > 0.02:
                    validation_warnings += 1

                qf = quantile_function(alpha, beta_param)
                expert = row["expert_name"]
                expert_dists[expert].append((qf, {
                    "alpha": alpha, "beta": beta_param,
                    "p25": p25_val, "p50": p50_val, "p75": p75_val,
                    "max_error": max_error,
                    "model": row.get("model", ""),
                }))

    if fit_failures > 0:
        pct = 100 * fit_failures / total if total > 0 else 0
        print(f"  Warning: {fit_failures}/{total} rows ({pct:.1f}%) failed Beta fitting", flush=True)
    if validation_warnings > 0:
        pct = 100 * validation_warnings / (total - fit_failures) if (total - fit_failures) > 0 else 0
        print(f"  Warning: {validation_warnings} rows ({pct:.1f}%) have max fitting error > 0.02", flush=True)

    return dict(expert_dists)


# ---------------------------------------------------------------------------
# Experiment Runners
# ---------------------------------------------------------------------------

def run_persona_frechet_anova(model: str, step: str) -> dict:
    """Run Fréchet ANOVA with persona as grouping factor."""
    data_dir = DATA_DIRS[(model, step)]
    expert_dists = load_and_fit(data_dir)

    if len(expert_dists) < 2:
        return {"error": f"Not enough groups: {len(expert_dists)}"}

    groups = {expert: [qf for qf, _ in entries] for expert, entries in expert_dists.items()}

    t_n, f_n, u_n, v_pooled, icc_f = frechet_anova_statistic(groups)
    k = len(groups)
    p_asymptotic = asymptotic_pvalue(t_n, k)

    print(f"  T_n = {t_n:.4f}, ICC_F = {icc_f:.4f}, p_asymptotic = {p_asymptotic:.4f}", flush=True)
    print(f"  Running permutation test ({N_PERMUTATIONS} permutations)...", flush=True)

    p_perm = permutation_test(groups, t_n)

    # Average pairwise W1 — sample-based for speed
    group_names = sorted(groups.keys())
    within_w1_samples = []
    between_w1_samples = []
    rng_w1 = np.random.RandomState(123)
    max_pairs = 500
    for g in group_names:
        gqfs = groups[g]
        for i in range(len(gqfs)):
            for j in range(i + 1, len(gqfs)):
                within_w1_samples.append(w1_distance(gqfs[i], gqfs[j]))
    for i, g1 in enumerate(group_names):
        for g2 in group_names[i + 1:]:
            pairs = [(a, b) for a in groups[g1] for b in groups[g2]]
            if len(pairs) > max_pairs:
                idx = rng_w1.choice(len(pairs), max_pairs, replace=False)
                pairs = [pairs[ii] for ii in idx]
            for qf1, qf2 in pairs:
                between_w1_samples.append(w1_distance(qf1, qf2))

    mean_w1_within = np.mean(within_w1_samples) if within_w1_samples else 0.0
    mean_w1_between = np.mean(between_w1_samples) if between_w1_samples else 0.0

    n_total = sum(len(v) for v in groups.values())

    return {
        "model": model,
        "step": step,
        "n": n_total,
        "k_groups": k,
        "T_n": t_n,
        "F_n": f_n,
        "U_n": u_n,
        "V_pooled": v_pooled,
        "ICC_F": icc_f,
        "p_asymptotic": p_asymptotic,
        "p_permutation": p_perm,
        "mean_W1_within": mean_w1_within,
        "mean_W1_between": mean_w1_between,
    }


def run_cross_model_frechet_anova(step: str) -> dict:
    """Run Fréchet ANOVA with model as grouping factor."""
    groups = {}

    for model in MODELS:
        data_dir = DATA_DIRS[(model, step)]
        expert_dists = load_and_fit(data_dir)
        # Pool all experts for this model
        all_qfs = []
        for entries in expert_dists.values():
            all_qfs.extend([qf for qf, _ in entries])
        groups[model] = all_qfs

    if len(groups) < 2:
        return {"error": f"Not enough groups: {len(groups)}"}

    t_n, f_n, u_n, v_pooled, icc_f = frechet_anova_statistic(groups)
    k = len(groups)
    p_asymptotic = asymptotic_pvalue(t_n, k)

    print(f"  T_n = {t_n:.4f}, ICC_F = {icc_f:.4f}, p_asymptotic = {p_asymptotic:.4f}", flush=True)
    print(f"  Running permutation test ({N_PERMUTATIONS} permutations)...", flush=True)

    p_perm = permutation_test(groups, t_n)

    # Between-group W1
    group_names = sorted(groups.keys())
    between_w1_samples = []
    rng_w1 = np.random.RandomState(123)
    max_pairs = 500
    for i, g1 in enumerate(group_names):
        for g2 in group_names[i + 1:]:
            pairs = [(a, b) for a in groups[g1] for b in groups[g2]]
            if len(pairs) > max_pairs:
                idx = rng_w1.choice(len(pairs), max_pairs, replace=False)
                pairs = [pairs[ii] for ii in idx]
            for qf1, qf2 in pairs:
                between_w1_samples.append(w1_distance(qf1, qf2))

    mean_w1_between = np.mean(between_w1_samples) if between_w1_samples else 0.0
    n_total = sum(len(v) for v in groups.values())

    return {
        "step": step,
        "n": n_total,
        "k_groups": k,
        "T_n": t_n,
        "F_n": f_n,
        "U_n": u_n,
        "V_pooled": v_pooled,
        "ICC_F": icc_f,
        "p_asymptotic": p_asymptotic,
        "p_permutation": p_perm,
        "mean_W1_between": mean_w1_between,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_file = Path(__file__).parent / "frechet_anova_results_percentile.txt"

    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("FRÉCHET ANOVA RESULTS (PERCENTILE-BASED ELICITATION)\n")
        f.write("=" * 80 + "\n\n")

        # Experiment 1: Persona ANOVA (within-model)
        f.write("-" * 80 + "\n")
        f.write("EXPERIMENT 1: PERSONA VARIANCE (WITHIN-MODEL)\n")
        f.write("-" * 80 + "\n\n")
        f.write("H₀: All 10 expert personas produce the same distribution.\n\n")

        persona_results = []
        for model in MODELS:
            for step in STEPS:
                print(f"\n>>> Persona ANOVA: {model}, {step}", flush=True)
                try:
                    result = run_persona_frechet_anova(model, step)
                    if "error" not in result:
                        persona_results.append(result)
                except Exception as e:
                    print(f"  ERROR: {e}", flush=True)
                    continue

        # Table format
        f.write(f"{'Model':<20} {'Step':<15} {'N':>5} {'T_n':>8} {'p (perm)':>10} {'ICC_F':>8} {'W1_within':>10} {'W1_between':>11}\n")
        f.write("-" * 100 + "\n")
        for r in persona_results:
            f.write(f"{r['model']:<20} {r['step']:<15} {r['n']:>5} {r['T_n']:>8.4f} {r['p_permutation']:>10.4f} {r['ICC_F']:>8.3f} {r['mean_W1_within']:>10.4f} {r['mean_W1_between']:>11.4f}\n")

        # Experiment 2: Cross-model ANOVA
        f.write("\n\n")
        f.write("-" * 80 + "\n")
        f.write("EXPERIMENT 2: MODEL VARIANCE (ACROSS MODELS)\n")
        f.write("-" * 80 + "\n\n")
        f.write("H₀: All 3 models (Claude, GPT-4o, Gemini) produce the same distribution.\n\n")

        model_results = []
        for step in STEPS:
            print(f"\n>>> Cross-Model ANOVA: {step}", flush=True)
            try:
                result = run_cross_model_frechet_anova(step)
                if "error" not in result:
                    model_results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                continue

        f.write(f"{'Step':<15} {'N':>5} {'T_n':>8} {'p (perm)':>10} {'ICC_F':>8} {'W1_between':>11}\n")
        f.write("-" * 70 + "\n")
        for r in model_results:
            f.write(f"{r['step']:<15} {r['n']:>5} {r['T_n']:>8.4f} {r['p_permutation']:>10.4f} {r['ICC_F']:>8.3f} {r['mean_W1_between']:>11.4f}\n")

        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("END OF RESULTS\n")
        f.write("=" * 80 + "\n")

    print(f"\n✓ Results written to: {output_file}\n", flush=True)


if __name__ == "__main__":
    main()
