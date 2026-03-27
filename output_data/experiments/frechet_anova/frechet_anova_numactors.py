#!/usr/bin/env python3
"""Fréchet ANOVA on fitted PERT distributions for num_actors using Wasserstein distance.

Adapts the percentile Beta fitting approach for integer count data:
  1. Fit PERT(a, m, b) from each observation's (p25, p50, p75)
     PERT is a scaled Beta on [a, b] with mode m
  2. Compute L2-Wasserstein distances via quantile functions
  3. Run Fréchet ANOVA (Dubey & Müller, 2019) with permutation test

PERT fitting: Solve for (a, m, b) such that:
  - CDF_PERT(p25) = 0.25
  - CDF_PERT(p50) = 0.50
  - CDF_PERT(p75) = 0.75
  - Subject to: 0 ≤ a ≤ m ≤ b

Usage:
    python3 frechet_anova_numactors.py
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from scipy.optimize import minimize, least_squares

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPERIMENTS_DIR = Path(__file__).parent.parent

DATA_DIRS = {
    "GPT-4o": EXPERIMENTS_DIR / "numactors_gpt4o",
    "Gemini 2.5 Pro": EXPERIMENTS_DIR / "numactors_gemini",
    "Claude Sonnet 4.5": EXPERIMENTS_DIR / "numactors_claude",
}

MODELS = ["GPT-4o", "Gemini 2.5 Pro", "Claude Sonnet 4.5"]

PROB_DIRS = {
    ("Claude Sonnet 4.5", "TA0002 (50%)"): EXPERIMENTS_DIR / "anova_probability",
    ("Claude Sonnet 4.5", "TA0007 (85%)"): EXPERIMENTS_DIR / "pilot_experiments" / "cross_step_TA0007_85pct",
    ("Claude Sonnet 4.5", "T1657 (30%)"): EXPERIMENTS_DIR / "pilot_experiments" / "cross_step_T1657_30pct",
    ("GPT-4o", "TA0002 (50%)"): EXPERIMENTS_DIR / "pilot_experiments" / "cross_model_gpt4o",
    ("GPT-4o", "TA0007 (85%)"): EXPERIMENTS_DIR / "pilot_experiments" / "cross_model_gpt4o_TA0007_85pct",
    ("GPT-4o", "T1657 (30%)"): EXPERIMENTS_DIR / "pilot_experiments" / "cross_model_gpt4o_T1657_30pct",
    ("Gemini 2.5 Pro", "TA0002 (50%)"): EXPERIMENTS_DIR / "pilot_experiments" / "cross_model_gemini_TA0002_50pct",
    ("Gemini 2.5 Pro", "TA0007 (85%)"): EXPERIMENTS_DIR / "pilot_experiments" / "cross_model_gemini_TA0007_85pct",
    ("Gemini 2.5 Pro", "T1657 (30%)"): EXPERIMENTS_DIR / "pilot_experiments" / "cross_model_gemini_T1657_30pct",
}

PROB_STEPS = ["TA0002 (50%)", "TA0007 (85%)", "T1657 (30%)"]

QUANTILE_GRID_SIZE = 201
N_PERMUTATIONS = 5_000

# Quantile grid
Q_GRID = np.linspace(0.001, 0.999, QUANTILE_GRID_SIZE)


# ---------------------------------------------------------------------------
# Step 1: PERT Distribution Fitting from Percentiles
# ---------------------------------------------------------------------------

def pert_cdf(x: float, a: float, m: float, b: float) -> float:
    """CDF of PERT distribution at x.

    PERT is a scaled Beta on [a, b] with mode m.
    Shape parameters: alpha = 1 + 4*(m-a)/(b-a), beta = 1 + 4*(b-m)/(b-a)
    Mean constraint: mu = (a + 4*m + b) / 6
    """
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0

    # Standardize to [0, 1]
    x_std = (x - a) / (b - a)
    m_std = (m - a) / (b - a)

    # PERT shape parameters
    alpha = 1 + 4 * m_std
    beta_param = 1 + 4 * (1 - m_std)

    return sp_stats.beta.cdf(x_std, alpha, beta_param)


def pert_ppf(q: float, a: float, m: float, b: float) -> float:
    """Quantile function (inverse CDF) of PERT distribution."""
    if q <= 0:
        return a
    if q >= 1:
        return b

    m_std = (m - a) / (b - a)
    alpha = 1 + 4 * m_std
    beta_param = 1 + 4 * (1 - m_std)

    x_std = sp_stats.beta.ppf(q, alpha, beta_param)
    return a + x_std * (b - a)


def fit_pert_from_percentiles(
    p25: float, p50: float, p75: float
) -> Optional[Tuple[float, float, float, float]]:
    """Fit PERT(a, m, b) from three percentiles using numerical optimization.

    Solves for (a, m, b) such that:
      - CDF_PERT(p25) = 0.25
      - CDF_PERT(p50) = 0.50
      - CDF_PERT(p75) = 0.75
      - Subject to: 0 ≤ a ≤ m ≤ b

    Returns (a, m, b, max_error) or None if fitting fails.
    """
    # Validation: must be ordered
    if not (0 <= p25 < p50 < p75):
        return None

    # Ensure non-negative (integer counts)
    if p25 < 0:
        return None

    def residuals(params):
        """Sum of squared errors for CDF matching."""
        a, m, b = params

        # Enforce ordering constraints
        if not (0 <= a <= m <= b):
            return [1e10, 1e10, 1e10]

        # Prevent degenerate distributions
        if (b - a) < 0.1:
            return [1e10, 1e10, 1e10]

        try:
            r1 = pert_cdf(p25, a, m, b) - 0.25
            r2 = pert_cdf(p50, a, m, b) - 0.50
            r3 = pert_cdf(p75, a, m, b) - 0.75
            return [r1, r2, r3]
        except (ValueError, RuntimeError, ZeroDivisionError):
            return [1e10, 1e10, 1e10]

    # Initial guesses
    # Heuristic: a ≈ p25 - IQR, m ≈ p50, b ≈ p75 + IQR
    iqr = p75 - p25

    init_guesses = [
        # Data-driven
        (max(0, p25 - iqr), p50, p75 + iqr),
        (max(0, p25 - 0.5*iqr), p50, p75 + 0.5*iqr),
        (max(0, p25 - 2*iqr), p50, p75 + 2*iqr),
        # Conservative
        (0, p50, p75 + iqr),
        (p25*0.5, p50, p75*1.5),
        (p25*0.3, p50, p75*2),
        # Symmetric around p50
        (p50 - 1.5*iqr, p50, p50 + 1.5*iqr),
        (p50 - 2*iqr, p50, p50 + 2*iqr),
        # Wide bounds
        (0, p50, p75 + 3*iqr),
        (0, p50, p75*3),
    ]

    best_solution = None
    best_cost = float("inf")

    for a0, m0, b0 in init_guesses:
        # Ensure valid initial guess
        if not (0 <= a0 <= m0 <= b0) or (b0 - a0) < 0.1:
            continue

        try:
            result = least_squares(
                residuals,
                [a0, m0, b0],
                bounds=([0, 0, 0], [1e6, 1e6, 1e6]),
                ftol=1e-12,
                xtol=1e-12,
                max_nfev=2000
            )

            if result.success and result.cost < best_cost:
                a_fit, m_fit, b_fit = result.x
                # Verify ordering
                if 0 <= a_fit <= m_fit <= b_fit and (b_fit - a_fit) >= 0.1:
                    best_solution = result.x
                    best_cost = result.cost

        except (ValueError, RuntimeError):
            continue

    if best_solution is None:
        return None

    a_fit, m_fit, b_fit = best_solution

    # Check fit quality
    try:
        p25_err = abs(pert_cdf(p25, a_fit, m_fit, b_fit) - 0.25)
        p50_err = abs(pert_cdf(p50, a_fit, m_fit, b_fit) - 0.50)
        p75_err = abs(pert_cdf(p75, a_fit, m_fit, b_fit) - 0.75)
        max_err = max(p25_err, p50_err, p75_err)
    except (ValueError, RuntimeError, ZeroDivisionError):
        return None

    # Reject poor fits
    if max_err > 0.05:
        return None

    return (a_fit, m_fit, b_fit, max_err)


# ---------------------------------------------------------------------------
# Step 2: Wasserstein Distance
# ---------------------------------------------------------------------------

def quantile_function_pert(a: float, m: float, b: float, grid: np.ndarray = Q_GRID) -> np.ndarray:
    """Compute the quantile function of PERT(a, m, b) on the grid."""
    return np.array([pert_ppf(q, a, m, b) for q in grid])


def w2_squared(qf1: np.ndarray, qf2: np.ndarray) -> float:
    """L2-Wasserstein squared distance between two distributions."""
    return np.trapz((qf1 - qf2) ** 2, Q_GRID)


def w1_distance(qf1: np.ndarray, qf2: np.ndarray) -> float:
    """L1-Wasserstein distance between two distributions."""
    return np.trapz(np.abs(qf1 - qf2), Q_GRID)


# ---------------------------------------------------------------------------
# Step 3 & 4: Vectorized Fréchet ANOVA (same as probability version)
# ---------------------------------------------------------------------------

def _w2sq_matrix_to_center(qf_matrix: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Vectorized W2-squared from each row of qf_matrix to center."""
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
    """Compute Fréchet ANOVA test statistic T_n."""
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
    """Load CSV data, fit PERT distributions from percentiles.

    Returns {expert: [(qf, meta), ...]}
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

                result = fit_pert_from_percentiles(p25_val, p50_val, p75_val)
                if result is None:
                    fit_failures += 1
                    continue

                a, m, b, max_error = result
                if max_error > 0.02:
                    validation_warnings += 1

                qf = quantile_function_pert(a, m, b)
                expert = row["expert_name"]
                expert_dists[expert].append((qf, {
                    "a": a, "m": m, "b": b,
                    "p25": p25_val, "p50": p50_val, "p75": p75_val,
                    "max_error": max_error,
                    "model": row.get("model", ""),
                }))

    if fit_failures > 0:
        pct = 100 * fit_failures / total if total > 0 else 0
        print(f"  Warning: {fit_failures}/{total} rows ({pct:.1f}%) failed PERT fitting", flush=True)
    if validation_warnings > 0:
        pct = 100 * validation_warnings / (total - fit_failures) if (total - fit_failures) > 0 else 0
        print(f"  Warning: {validation_warnings} rows ({pct:.1f}%) have max fitting error > 0.02", flush=True)

    return dict(expert_dists)


# ---------------------------------------------------------------------------
# Experiment Runners
# ---------------------------------------------------------------------------

def run_persona_frechet_anova(model: str) -> dict:
    """Run Fréchet ANOVA with persona as grouping factor."""
    data_dir = DATA_DIRS[model]
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

    # Average pairwise W1
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


def run_cross_model_frechet_anova() -> dict:
    """Run Fréchet ANOVA with model as grouping factor."""
    groups = {}

    for model in MODELS:
        data_dir = DATA_DIRS[model]
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
# CV Dominance Analysis
# ---------------------------------------------------------------------------

def _load_raw_quantity_estimates(data_dir: Path) -> pd.DataFrame:
    """Load point estimates from numactors CSVs.

    Returns DataFrame with columns: task_name, expert_name, estimate.
    The 'estimate' column is the LLM's direct mode response (number of actors).
    """
    rows = []
    run_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    for run_dir in run_dirs:
        csv_path = run_dir / "detailed_estimates.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                if row.get("has_error", "").strip().lower() == "true":
                    continue
                try:
                    est = float(row["estimate"].strip())
                except (ValueError, KeyError):
                    continue
                rows.append({
                    "task_name": row.get("task_name", ""),
                    "expert_name": row.get("expert_name", ""),
                    "estimate": est,
                })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["task_name", "expert_name", "estimate"])


def _load_raw_probability_estimates(data_dir: Path) -> pd.DataFrame:
    """Load point estimates from probability CSVs.

    Returns DataFrame with columns: task_name, expert_name, estimate.
    The 'estimate' column is the LLM's most_likely_estimate (probability).
    """
    rows = []
    run_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    for run_dir in run_dirs:
        csv_path = run_dir / "detailed_estimates.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                if row.get("has_error", "").strip().lower() == "true":
                    continue
                try:
                    est = float(row["most_likely_estimate"].strip())
                except (ValueError, KeyError):
                    continue
                rows.append({
                    "task_name": row.get("task_name", ""),
                    "expert_name": row.get("expert_name", ""),
                    "estimate": est,
                })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["task_name", "expert_name", "estimate"])


def _compute_cv(values: np.ndarray) -> Optional[float]:
    """Compute coefficient of variation = std / mean.

    Returns None when there are fewer than 2 values or mean is zero.
    """
    if len(values) < 2:
        return None
    mean = np.mean(values)
    if mean == 0.0:
        return None
    return float(np.std(values, ddof=1) / mean)


def cv_dominance_analysis() -> dict:
    """Check whether CV(numactors) > CV(probability) at every (task, model, step) cell.

    This is the dominance approach: instead of comparing average CVs across steps with
    different baselines — which conflates ceiling effects at 50%/85%/30% — we verify
    the inequality cell by cell.  A single dominant majority makes the argument without
    requiring any cross-baseline averaging.

    Returns a dict with:
        comparisons: list of per-cell records
        n_dominate:  cells where CV(quantity) > CV(probability)
        n_tied:      cells where they are equal (rare in practice)
        n_not_dominate: cells where CV(quantity) <= CV(probability)
    """
    comparisons = []
    skipped = []
    n_dominate = 0
    n_tied = 0
    n_not_dominate = 0

    for model in MODELS:
        quant_dir = DATA_DIRS[model]
        if not quant_dir.exists():
            continue
        quant_df = _load_raw_quantity_estimates(quant_dir)
        if quant_df.empty:
            continue

        for step in PROB_STEPS:
            prob_dir = PROB_DIRS.get((model, step))
            if prob_dir is None or not prob_dir.exists():
                skipped.append(f"{model} / {step}  (directory not found)")
                continue
            prob_df = _load_raw_probability_estimates(prob_dir)
            if prob_df.empty:
                skipped.append(f"{model} / {step}  (no data loaded)")
                continue

            common_tasks = sorted(
                set(quant_df["task_name"].unique()) & set(prob_df["task_name"].unique())
            )
            if not common_tasks:
                qty_tasks = quant_df["task_name"].unique().tolist()
                prob_tasks = prob_df["task_name"].unique().tolist()
                skipped.append(
                    f"{model} / {step}  (no task overlap: "
                    f"qty={qty_tasks}, prob={prob_tasks})"
                )
                continue
            for task in common_tasks:
                q_vals = quant_df.loc[quant_df["task_name"] == task, "estimate"].values
                p_vals = prob_df.loc[prob_df["task_name"] == task, "estimate"].values

                cv_q = _compute_cv(q_vals)
                cv_p = _compute_cv(p_vals)

                if cv_q is None or cv_p is None:
                    continue

                if cv_q > cv_p:
                    n_dominate += 1
                    dominance = "quantity > prob"
                elif cv_q == cv_p:
                    n_tied += 1
                    dominance = "tied"
                else:
                    n_not_dominate += 1
                    dominance = "prob >= quantity"

                comparisons.append({
                    "model": model,
                    "step": step,
                    "task": task,
                    "cv_quantity": cv_q,
                    "cv_probability": cv_p,
                    "dominance": dominance,
                    "n_quantity": len(q_vals),
                    "n_probability": len(p_vals),
                })

    return {
        "comparisons": comparisons,
        "skipped": skipped,
        "n_dominate": n_dominate,
        "n_tied": n_tied,
        "n_not_dominate": n_not_dominate,
    }


def format_cv_dominance_section(result: dict) -> str:
    """Format the CV dominance analysis results as a report section."""
    comparisons = result["comparisons"]
    skipped = result.get("skipped", [])
    n_dom = result["n_dominate"]
    n_not = result["n_not_dominate"]
    n_tied = result["n_tied"]
    n_total = n_dom + n_not + n_tied

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("CV DOMINANCE ANALYSIS: QUANTITY vs PROBABILITY NODES")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Goal: test whether CV(NumActors) > CV(probability_step) holds at every")
    lines.append("(task, model, step) cell — without averaging across steps/baselines.")
    lines.append("")
    lines.append("Why dominance instead of averages: averaging CV across probability steps")
    lines.append("with different baselines (30%/50%/85%) conflates ceiling effects. The")
    lines.append("dominance check sidesteps this by comparing within each cell.")
    lines.append("")
    lines.append("CV = std(point estimates across all runs×experts) / mean  (ddof=1)")
    lines.append("")

    if not comparisons:
        lines.append("  No comparable (task, model, step) cells found.")
        if skipped:
            lines.append("")
            lines.append("  Skipped (no task overlap between quantity and probability dirs):")
            for s in skipped:
                lines.append(f"    {s}")
        return "\n".join(lines)

    lines.append(f"{'Model':<22} {'Step':<16} {'Task':<22} {'CV(qty)':>8} {'CV(prob)':>9} {'Result':<18}")
    lines.append("-" * 100)
    for c in sorted(comparisons, key=lambda x: (x["model"], x["step"], x["task"])):
        lines.append(
            f"{c['model']:<22} {c['step']:<16} {c['task']:<22} "
            f"{c['cv_quantity']:>7.1%} {c['cv_probability']:>8.1%}  {c['dominance']}"
        )

    lines.append("")
    lines.append("-" * 80)
    lines.append("SUMMARY")
    lines.append("-" * 80)
    lines.append(f"  Total cells compared:          {n_total}")
    lines.append(f"  Quantity CV > Probability CV:  {n_dom}  ({100*n_dom/n_total:.1f}%)")
    if n_tied:
        lines.append(f"  Tied:                          {n_tied}  ({100*n_tied/n_total:.1f}%)")
    lines.append(f"  Probability CV >= Quantity CV: {n_not}  ({100*n_not/n_total:.1f}%)")
    lines.append("")

    if n_not > 0:
        lines.append("  Exceptions (cells where dominance fails):")
        for c in comparisons:
            if c["dominance"] != "quantity > prob":
                lines.append(
                    f"    {c['model']} / {c['step']} / {c['task']}: "
                    f"CV(qty)={c['cv_quantity']:.1%}  CV(prob)={c['cv_probability']:.1%}"
                )
        lines.append("")

    if skipped:
        lines.append(f"  Skipped ({len(skipped)} (model, step) pairs — no task overlap):")
        for s in skipped:
            lines.append(f"    {s}")
        lines.append("")

    if n_dom == n_total:
        lines.append("  ✓ Strict dominance holds: CV(NumActors) > CV(probability) in every comparable cell.")
    elif n_total > 0 and n_dom / n_total >= 0.9:
        lines.append(f"  ✓ Near-strict dominance: holds in {n_dom}/{n_total} cells ({100*n_dom/n_total:.0f}%).")
        lines.append("    Exceptions are isolated cases, not a systematic pattern.")
    else:
        lines.append(f"  ⚠ Dominance holds in {n_dom}/{n_total} cells; check exceptions above.")

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


CV_HEATMAP_ROW_LABELS = [
    ("# of actors", "quantity"),
    ("TA0002 (50%)", "TA0002 (50%)"),
    ("TA0007 (85%)", "TA0007 (85%)"),
    ("T1657 (30%)", "T1657 (30%)"),
]


def build_cv_matrix_for_model(model: str) -> Optional[dict]:
    """Build mean / CV / n matrices for Delphi-style table (rows=nodes, cols=tasks)."""
    quant_dir = DATA_DIRS.get(model)
    if quant_dir is None or not quant_dir.exists():
        return None
    quant_df = _load_raw_quantity_estimates(quant_dir)
    if quant_df.empty:
        return None

    tasks: set = set(quant_df["task_name"].unique())
    for step in PROB_STEPS:
        prob_dir = PROB_DIRS.get((model, step))
        if prob_dir is not None and prob_dir.exists():
            p = _load_raw_probability_estimates(prob_dir)
            if not p.empty:
                tasks |= set(p["task_name"].unique())
    col_labels = sorted(tasks)
    if not col_labels:
        return None

    n_rows = len(CV_HEATMAP_ROW_LABELS)
    n_cols = len(col_labels)
    mean_arr = np.full((n_rows, n_cols), np.nan)
    cv_arr = np.full((n_rows, n_cols), np.nan)
    n_arr = np.full((n_rows, n_cols), np.nan)
    is_prob_row = np.zeros(n_rows, dtype=bool)

    for ri, (row_name, kind) in enumerate(CV_HEATMAP_ROW_LABELS):
        if kind == "quantity":
            for ci, task in enumerate(col_labels):
                vals = quant_df.loc[quant_df["task_name"] == task, "estimate"].values
                if len(vals) < 2:
                    continue
                mean_arr[ri, ci] = np.mean(vals)
                cv = _compute_cv(vals)
                if cv is not None:
                    cv_arr[ri, ci] = cv
                n_arr[ri, ci] = len(vals)
        else:
            is_prob_row[ri] = True
            prob_dir = PROB_DIRS.get((model, kind))
            if prob_dir is None or not prob_dir.exists():
                continue
            prob_df = _load_raw_probability_estimates(prob_dir)
            if prob_df.empty:
                continue
            for ci, task in enumerate(col_labels):
                vals = prob_df.loc[prob_df["task_name"] == task, "estimate"].values
                if len(vals) < 2:
                    continue
                mean_arr[ri, ci] = np.mean(vals)
                cv = _compute_cv(vals)
                if cv is not None:
                    cv_arr[ri, ci] = cv
                n_arr[ri, ci] = len(vals)

    row_labels = [r[0] for r in CV_HEATMAP_ROW_LABELS]
    return {
        "row_labels": row_labels,
        "col_labels": col_labels,
        "mean": mean_arr,
        "cv": cv_arr,
        "n": n_arr,
        "is_prob_row": is_prob_row,
    }


def plot_cv_heatmap_delphi_style(
    model: str,
    out_path: Path,
    title_suffix: str = "LLM Delphi (runs × personas pooled)",
) -> bool:
    """Save a human-Delphi-style heatmap: each cell shows mean, CV%, and n."""
    data = build_cv_matrix_for_model(model)
    if data is None:
        return False

    mean_arr = data["mean"]
    cv_arr = data["cv"]
    n_arr = data["n"]
    row_labels = data["row_labels"]
    col_labels = data["col_labels"]
    is_prob_row = data["is_prob_row"]

    n_rows, n_cols = cv_arr.shape
    fig_w = max(8.0, 1.2 * n_cols + 4)
    fig_h = max(4.0, 0.65 * n_rows + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    valid_cv = cv_arr[~np.isnan(cv_arr)]
    vmax = float(np.max(valid_cv) * 1.15) if valid_cv.size else 0.5
    vmax = max(vmax, 0.05)

    display = np.ma.masked_invalid(cv_arr)
    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(display, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_title(f"{model}\n{title_suffix}", fontsize=11, pad=12)

    for i in range(n_rows):
        for j in range(n_cols):
            if np.isnan(cv_arr[i, j]) and np.isnan(mean_arr[i, j]):
                txt = "—"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="0.4")
                continue
            m = mean_arr[i, j]
            cv_pct = 100.0 * cv_arr[i, j]
            n_i = int(n_arr[i, j]) if not np.isnan(n_arr[i, j]) else 0
            if is_prob_row[i]:
                mean_str = f"{m * 100:.1f}%"
            else:
                mean_str = f"{m:.2f}"
            cell = f"{mean_str}\nCV {cv_pct:.0f}%\nn={n_i}"
            lum = cv_arr[i, j] / vmax if vmax > 0 else 0.5
            ax.text(
                j, i, cell, ha="center", va="center", fontsize=8,
                color="black" if lum < 0.55 else "white",
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Coefficient of variation", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def save_cv_heatmap_figures(output_dir: Optional[Path] = None) -> List[Path]:
    """Write one PNG per model (Delphi-style CV table). Returns paths written."""
    base = output_dir if output_dir is not None else Path(__file__).parent
    written: List[Path] = []
    for model in MODELS:
        safe = model.replace(" ", "_").replace(".", "")
        path = base / f"cv_heatmap_delphi_style_{safe}.png"
        if plot_cv_heatmap_delphi_style(model, path):
            written.append(path)
    if written:
        combined_path = base / "cv_heatmap_delphi_style_all_models.png"
        _plot_combined_cv_heatmaps(combined_path)
        written.append(combined_path)
    return written


def _plot_combined_cv_heatmaps(out_path: Path) -> None:
    """Single figure with 3 subplots (one per model)."""
    n_models = len(MODELS)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    global_vmax = 0.05
    all_data = []
    for model in MODELS:
        d = build_cv_matrix_for_model(model)
        all_data.append(d)
        if d is not None:
            v = d["cv"][~np.isnan(d["cv"])]
            if v.size:
                global_vmax = max(global_vmax, float(np.max(v) * 1.1))

    for ax, model, data in zip(axes, MODELS, all_data):
        if data is None:
            ax.set_visible(False)
            continue
        cv_arr = data["cv"]
        mean_arr = data["mean"]
        n_arr = data["n"]
        row_labels = data["row_labels"]
        col_labels = data["col_labels"]
        is_prob_row = data["is_prob_row"]
        n_rows, n_cols = cv_arr.shape

        display = np.ma.masked_invalid(cv_arr)
        im = ax.imshow(display, aspect="auto", cmap=plt.cm.RdYlGn_r, vmin=0.0, vmax=global_vmax)
        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_rows))
        ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=7)
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_title(model, fontsize=9)

        for i in range(n_rows):
            for j in range(n_cols):
                if np.isnan(cv_arr[i, j]) and np.isnan(mean_arr[i, j]):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="0.4")
                    continue
                m = mean_arr[i, j]
                cv_pct = 100.0 * cv_arr[i, j]
                n_i = int(n_arr[i, j]) if not np.isnan(n_arr[i, j]) else 0
                mean_str = f"{m * 100:.1f}%" if is_prob_row[i] else f"{m:.2f}"
                cell = f"{mean_str}\n{cv_pct:.0f}%\nn={n_i}"
                lum = cv_arr[i, j] / global_vmax if global_vmax > 0 else 0.5
                ax.text(
                    j, i, cell, ha="center", va="center", fontsize=6,
                    color="black" if lum < 0.55 else "white",
                )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("LLM elicitation: CV by node × task (pooled runs × personas)", fontsize=11, y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_file = Path(__file__).parent / "frechet_anova_results_numactors.txt"

    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("FRÉCHET ANOVA RESULTS (NUM_ACTORS - PERT FITTING)\n")
        f.write("=" * 80 + "\n\n")
        f.write("Distribution: PERT (scaled Beta on [a,b] with mode m)\n")
        f.write("Data: GPT-4o, Gemini 2.5 Pro, and Claude Sonnet 4.5\n\n")

        # Experiment 1: Persona ANOVA (within-model)
        f.write("-" * 80 + "\n")
        f.write("EXPERIMENT 1: PERSONA VARIANCE (WITHIN-MODEL)\n")
        f.write("-" * 80 + "\n\n")
        f.write("H₀: All 10 expert personas produce the same distribution.\n\n")

        persona_results = []
        for model in MODELS:
            print(f"\n>>> Persona ANOVA: {model}", flush=True)
            try:
                result = run_persona_frechet_anova(model)
                if "error" not in result:
                    persona_results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                continue

        # Table format
        f.write(f"{'Model':<20} {'N':>5} {'T_n':>8} {'p (perm)':>10} {'ICC_F':>8} {'W1_within':>10} {'W1_between':>11}\n")
        f.write("-" * 80 + "\n")
        for r in persona_results:
            f.write(f"{r['model']:<20} {r['n']:>5} {r['T_n']:>8.4f} {r['p_permutation']:>10.4f} {r['ICC_F']:>8.3f} {r['mean_W1_within']:>10.4f} {r['mean_W1_between']:>11.4f}\n")

        # Experiment 2: Cross-model ANOVA
        f.write("\n\n")
        f.write("-" * 80 + "\n")
        f.write("EXPERIMENT 2: MODEL VARIANCE (ACROSS MODELS)\n")
        f.write("-" * 80 + "\n\n")
        f.write("H₀: All three models (GPT-4o, Gemini, Claude) produce the same distribution.\n\n")

        print(f"\n>>> Cross-Model ANOVA", flush=True)
        try:
            result = run_cross_model_frechet_anova()
            if "error" not in result:
                f.write(f"{'N':>5} {'T_n':>8} {'p (perm)':>10} {'ICC_F':>8} {'W1_between':>11}\n")
                f.write("-" * 50 + "\n")
                f.write(f"{result['n']:>5} {result['T_n']:>8.4f} {result['p_permutation']:>10.4f} {result['ICC_F']:>8.3f} {result['mean_W1_between']:>11.4f}\n")
            else:
                f.write(f"ERROR: {result['error']}\n")
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            f.write(f"ERROR: {e}\n")

        # Experiment 3: CV dominance analysis
        print(f"\n>>> CV Dominance Analysis (quantity vs probability nodes)", flush=True)
        try:
            cv_result = cv_dominance_analysis()
            cv_section = format_cv_dominance_section(cv_result)
            f.write(cv_section + "\n")
            n_dom = cv_result["n_dominate"]
            n_total = cv_result["n_dominate"] + cv_result["n_not_dominate"] + cv_result["n_tied"]
            print(f"  Dominance holds in {n_dom}/{n_total} cells", flush=True)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            f.write(f"\nCV dominance analysis error: {e}\n")

        print(f"\n>>> CV heatmaps (human-Delphi-style tables)", flush=True)
        try:
            heatmap_paths = save_cv_heatmap_figures(Path(__file__).parent)
            for p in heatmap_paths:
                f.write(f"\nCV heatmap written: {p}\n")
                print(f"  Saved: {p}", flush=True)
        except Exception as e:
            print(f"  ERROR (heatmaps): {e}", flush=True)
            f.write(f"\nCV heatmap error: {e}\n")

        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("END OF RESULTS\n")
        f.write("=" * 80 + "\n")

    print(f"\n✓ Results written to: {output_file}\n", flush=True)


if __name__ == "__main__":
    main()
