#!/usr/bin/env python3
"""Fréchet ANOVA on fitted Beta distributions using Wasserstein distance.

Replaces scalar ANOVA on point estimates with a distributional test:
  1. Fit Beta(α, β) on [0,1] from each observation's (min, max, confidence)
     using symmetric confidence intervals (equal tail probabilities).
  2. Compute L2-Wasserstein distances via quantile functions
  3. Run Fréchet ANOVA (Dubey & Müller, 2019) with permutation test

Beta fitting: treat (min, max) as a symmetric confidence interval with
coverage = confidence. That is, CDF(lo) = (1-conf)/2, CDF(hi) = (1+conf)/2.
The mode/point estimate is NOT used as a fitting constraint.

Usage:
    python3 frechet_anova.py
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, List, Dict

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import fsolve

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPERIMENTS_DIR = Path(__file__).parent.parent

DATA_DIRS = {
    ("Claude Sonnet 4.5", "TA0002 (50%)"): EXPERIMENTS_DIR / "anova_probability",
    ("Claude Sonnet 4.5", "TA0007 (85%)"): EXPERIMENTS_DIR / "pilot_experiments" / "cross_step_TA0007_85pct",
    ("Claude Sonnet 4.5", "T1657 (30%)"):  EXPERIMENTS_DIR / "pilot_experiments" / "cross_step_T1657_30pct",
    ("GPT-4o", "TA0002 (50%)"): EXPERIMENTS_DIR / "pilot_experiments" / "cross_model_gpt4o",
    ("GPT-4o", "TA0007 (85%)"): EXPERIMENTS_DIR / "pilot_experiments" / "cross_model_gpt4o_TA0007_85pct",
    ("GPT-4o", "T1657 (30%)"):  EXPERIMENTS_DIR / "pilot_experiments" / "cross_model_gpt4o_T1657_30pct",
    ("Gemini 2.5 Pro", "TA0002 (50%)"): EXPERIMENTS_DIR / "pilot_experiments" / "cross_model_gemini_TA0002_50pct",
    ("Gemini 2.5 Pro", "TA0007 (85%)"): EXPERIMENTS_DIR / "pilot_experiments" / "cross_model_gemini_TA0007_85pct",
    ("Gemini 2.5 Pro", "T1657 (30%)"):  EXPERIMENTS_DIR / "pilot_experiments" / "cross_model_gemini_T1657_30pct",
}

STEPS = ["TA0002 (50%)", "TA0007 (85%)", "T1657 (30%)"]
MODELS = ["Claude Sonnet 4.5", "GPT-4o", "Gemini 2.5 Pro"]

DEFAULT_CONFIDENCE = 0.85
QUANTILE_GRID_SIZE = 201
N_PERMUTATIONS = 5_000

# Quantile grid — avoid exact 0 and 1 where Beta quantile function diverges
Q_GRID = np.linspace(0.001, 0.999, QUANTILE_GRID_SIZE)


# ---------------------------------------------------------------------------
# Step 1: Beta Distribution Fitting
# ---------------------------------------------------------------------------

def fit_beta_from_elicitation(
    lo: float, hi: float, confidence: float
) -> Optional[Tuple[float, float]]:
    """Fit Beta(alpha, beta) on [0,1] using symmetric confidence interval.

    Treats (lo, hi) as a symmetric confidence interval with equal tail
    probabilities:
      - CDF(lo) = (1 - confidence) / 2
      - CDF(hi) = (1 + confidence) / 2

    Returns (alpha, beta) or None if fitting fails.
    """
    if not (0 <= lo < hi <= 1 and 0 < confidence < 1):
        return None

    # Clamp bounds away from 0/1: CDF(0)=0 and CDF(1)=1 for all Beta
    # distributions, so exact boundary values make the symmetric CI
    # constraints unsatisfiable.
    lo = max(lo, 0.005)
    hi = min(hi, 0.995)
    if lo >= hi:
        return None

    target_lo_cdf = (1.0 - confidence) / 2.0
    target_hi_cdf = (1.0 + confidence) / 2.0

    def equations(params):
        a, b = params
        if a <= 0.01 or b <= 0.01:
            return [1e10, 1e10]
        eq1 = sp_stats.beta.cdf(lo, a, b) - target_lo_cdf
        eq2 = sp_stats.beta.cdf(hi, a, b) - target_hi_cdf
        return [eq1, eq2]

    best_solution = None
    best_residual = float("inf")

    init_guesses = [
        (2.0, 2.0), (5.0, 5.0), (1.5, 1.5), (10.0, 10.0),
        (3.0, 8.0), (8.0, 3.0), (1.2, 0.8), (0.8, 1.2),
        (1.5, 3.0), (3.0, 1.5), (15.0, 5.0), (5.0, 15.0),
        (0.5, 0.5), (1.0, 2.0), (2.0, 1.0), (20.0, 20.0),
    ]

    for a0, b0 in init_guesses:
        try:
            sol, info, ier, msg = fsolve(equations, [a0, b0], full_output=True)
            a_sol, b_sol = sol
            if ier == 1 and a_sol > 0.01 and b_sol > 0.01:
                residual = sum(x**2 for x in info["fvec"])
                if residual < best_residual:
                    best_residual = residual
                    best_solution = (a_sol, b_sol)
                    if residual < 1e-20:
                        break
        except (ValueError, RuntimeError):
            continue

    if best_solution is None or best_residual > 1e-6:
        return None

    return best_solution


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
    """Load CSV data, fit Beta distributions, return {expert: [(qf, meta), ...]}.

    Each entry is a quantile function array + metadata dict.
    """
    expert_dists = defaultdict(list)
    fit_failures = 0
    total = 0

    run_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    for run_dir in run_dirs:
        csv_path = run_dir / "detailed_estimates.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                total += 1
                try:
                    lo_val = float(row.get("minimum_estimate", "").strip())
                    hi_val = float(row.get("maximum_estimate", "").strip())
                except (ValueError, AttributeError):
                    fit_failures += 1
                    continue

                conf_str = row.get("confidence_in_range", "").strip()
                try:
                    conf_val = float(conf_str) if conf_str else DEFAULT_CONFIDENCE
                except ValueError:
                    conf_val = DEFAULT_CONFIDENCE

                params = fit_beta_from_elicitation(lo_val, hi_val, conf_val)
                if params is None:
                    fit_failures += 1
                    continue

                alpha, beta_param = params
                qf = quantile_function(alpha, beta_param)
                expert = row["expert_name"]
                expert_dists[expert].append((qf, {
                    "alpha": alpha, "beta": beta_param,
                    "lo": lo_val, "hi": hi_val,
                    "confidence": conf_val,
                    "model": row.get("model", ""),
                }))

    if fit_failures > 0:
        pct = 100 * fit_failures / total if total > 0 else 0
        print(f"  Warning: {fit_failures}/{total} rows ({pct:.1f}%) failed Beta fitting", flush=True)

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

    return {
        "model": model, "step": step,
        "k": k, "n_total": sum(len(v) for v in groups.values()),
        "t_n": t_n, "f_n": f_n, "u_n": u_n,
        "v_pooled": v_pooled, "icc_f": icc_f,
        "p_asymptotic": p_asymptotic, "p_perm": p_perm,
        "mean_within_w1": np.mean(within_w1_samples) if within_w1_samples else 0,
        "mean_between_w1": np.mean(between_w1_samples) if between_w1_samples else 0,
        "group_sizes": {g: len(groups[g]) for g in group_names},
    }


def run_cross_model_frechet_anova(step: str) -> dict:
    """Run Fréchet ANOVA with model as grouping factor."""
    groups = {}
    for model in MODELS:
        data_dir = DATA_DIRS[(model, step)]
        expert_dists = load_and_fit(data_dir)
        model_qfs = [qf for entries in expert_dists.values() for qf, _ in entries]
        if model_qfs:
            groups[model] = model_qfs

    if len(groups) < 2:
        return {"error": f"Not enough model groups: {len(groups)}"}

    t_n, f_n, u_n, v_pooled, icc_f = frechet_anova_statistic(groups)
    k = len(groups)
    p_asymptotic = asymptotic_pvalue(t_n, k)

    print(f"  T_n = {t_n:.4f}, ICC_F = {icc_f:.4f}, p_asymptotic = {p_asymptotic:.4f}", flush=True)
    print(f"  Running permutation test ({N_PERMUTATIONS} permutations)...", flush=True)

    p_perm = permutation_test(groups, t_n)

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

    return {
        "step": step,
        "k": k, "n_total": sum(len(v) for v in groups.values()),
        "t_n": t_n, "f_n": f_n, "u_n": u_n,
        "v_pooled": v_pooled, "icc_f": icc_f,
        "p_asymptotic": p_asymptotic, "p_perm": p_perm,
        "mean_between_w1": np.mean(between_w1_samples) if between_w1_samples else 0,
        "group_sizes": {g: len(groups[g]) for g in group_names},
    }


# ---------------------------------------------------------------------------
# Output Formatting
# ---------------------------------------------------------------------------

def format_results(persona_results: list, model_results: list) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("FRÉCHET ANOVA RESULTS — Beta Distributions with Wasserstein Distance")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Method: Fit Beta(α,β) on [0,1] using symmetric confidence interval.")
    lines.append("        (min, max) treated as CI with equal tail probs: CDF(lo)=(1-c)/2, CDF(hi)=(1+c)/2.")
    lines.append("        Fréchet ANOVA (Dubey & Müller, 2019) with L2-Wasserstein distance.")
    lines.append(f"        Permutation test: {N_PERMUTATIONS} permutations.")
    lines.append(f"        Quantile grid: {QUANTILE_GRID_SIZE} points.")
    lines.append(f"        Default confidence for missing values: {DEFAULT_CONFIDENCE}")
    lines.append("")
    lines.append("Note: Asymptotic p-values (T_n ~ χ²(k-1)) are shown for reference but are")
    lines.append("      unreliable at these sample sizes. The permutation p-value is the primary test.")
    lines.append("")

    lines.append("-" * 80)
    lines.append("EXPERIMENT 1: PERSONA FRÉCHET ANOVA")
    lines.append("-" * 80)
    lines.append("")

    for r in persona_results:
        if "error" in r:
            lines.append(f"  {r.get('model', '?')} / {r.get('step', '?')}: {r['error']}")
            continue
        sig = "NOT significant" if r["p_perm"] > 0.05 else "SIGNIFICANT"
        lines.append(f"  {r['model']} — {r['step']}")
        lines.append(f"    N = {r['n_total']} distributions, k = {r['k']} personas")
        lines.append(f"    T_n = {r['t_n']:.4f}")
        lines.append(f"    p (permutation) = {r['p_perm']:.4f}  → {sig} at α=0.05")
        lines.append(f"    p (asymptotic)  = {r['p_asymptotic']:.4f}")
        lines.append(f"    Fréchet ICC     = {r['icc_f']:.4f}  ({r['icc_f']*100:.1f}% of distributional variance from persona)")
        lines.append(f"    Mean within-persona W1  = {r['mean_within_w1']:.4f}")
        lines.append(f"    Mean between-persona W1 = {r['mean_between_w1']:.4f}")
        lines.append("")

    lines.append("-" * 80)
    lines.append("EXPERIMENT 2: CROSS-MODEL FRÉCHET ANOVA")
    lines.append("-" * 80)
    lines.append("")

    for r in model_results:
        if "error" in r:
            lines.append(f"  {r.get('step', '?')}: {r['error']}")
            continue
        sig = "NOT significant" if r["p_perm"] > 0.05 else "SIGNIFICANT"
        lines.append(f"  {r['step']}")
        lines.append(f"    N = {r['n_total']} distributions, k = {r['k']} models")
        sizes = ", ".join(f"{m}: {n}" for m, n in sorted(r["group_sizes"].items()))
        lines.append(f"    Group sizes: {sizes}")
        lines.append(f"    T_n = {r['t_n']:.4f}")
        lines.append(f"    p (permutation) = {r['p_perm']:.4f}  → {sig} at α=0.05")
        lines.append(f"    p (asymptotic)  = {r['p_asymptotic']:.4f}")
        lines.append(f"    Fréchet ICC     = {r['icc_f']:.4f}  ({r['icc_f']*100:.1f}% of distributional variance from model)")
        lines.append(f"    Mean between-model W1 = {r['mean_between_w1']:.4f}")
        lines.append("")

    lines.append("-" * 80)
    lines.append("COMPARISON: SCALAR ANOVA vs FRÉCHET ANOVA")
    lines.append("-" * 80)
    lines.append("")
    lines.append("  Persona experiments:")
    lines.append(f"  {'Model':<20} {'Step':<16} {'Scalar p':>10} {'Scalar ICC':>11} {'Fréchet p':>10} {'Fréchet ICC':>12}")
    lines.append("  " + "-" * 79)

    scalar_persona = {
        ("Claude Sonnet 4.5", "TA0002 (50%)"): (0.12, 0.058),
        ("GPT-4o", "TA0002 (50%)"): (0.65, 0.000),
        ("Claude Sonnet 4.5", "TA0007 (85%)"): (0.01, 0.144),
        ("GPT-4o", "TA0007 (85%)"): (0.15, 0.051),
        ("Claude Sonnet 4.5", "T1657 (30%)"): (0.87, 0.000),
        ("GPT-4o", "T1657 (30%)"): (0.36, 0.011),
    }

    for r in persona_results:
        if "error" in r:
            continue
        key = (r["model"], r["step"])
        sp, si = scalar_persona.get(key, ("—", "—"))
        sp_str = f"{sp:.2f}" if isinstance(sp, float) else sp
        si_str = f"{si:.3f}" if isinstance(si, float) else si
        lines.append(f"  {r['model']:<20} {r['step']:<16} {sp_str:>10} {si_str:>11} {r['p_perm']:>10.4f} {r['icc_f']:>12.4f}")

    lines.append("")
    lines.append("  Cross-model experiments:")
    lines.append(f"  {'Step':<16} {'Scalar p':>10} {'Scalar ICC':>11} {'Fréchet p':>10} {'Fréchet ICC':>12}")
    lines.append("  " + "-" * 59)

    scalar_model = {
        "TA0002 (50%)": ("<0.0001", 0.651),
        "TA0007 (85%)": ("<0.0001", 0.279),
        "T1657 (30%)": ("<0.0001", 0.607),
    }

    for r in model_results:
        if "error" in r:
            continue
        sp, si = scalar_model.get(r["step"], ("—", "—"))
        si_str = f"{si:.3f}" if isinstance(si, float) else si
        lines.append(f"  {r['step']:<16} {sp:>10} {si_str:>11} {r['p_perm']:>10.4f} {r['icc_f']:>12.4f}")

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    print("=" * 60, flush=True)
    print("Fréchet ANOVA — Beta Distributions + Wasserstein Distance", flush=True)
    print("=" * 60, flush=True)

    persona_results = []
    for model in ["Claude Sonnet 4.5", "GPT-4o"]:
        for step in STEPS:
            t0 = time.time()
            print(f"\nPersona ANOVA: {model} / {step}", flush=True)
            result = run_persona_frechet_anova(model, step)
            persona_results.append(result)
            print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    model_results = []
    for step in STEPS:
        t0 = time.time()
        print(f"\nCross-Model ANOVA: {step}", flush=True)
        result = run_cross_model_frechet_anova(step)
        model_results.append(result)
        print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    output = format_results(persona_results, model_results)
    print("\n" + output, flush=True)

    output_path = Path(__file__).parent / "frechet_anova_results.txt"
    with open(output_path, "w") as f:
        f.write(output)
    print(f"\nSaved to: {output_path}", flush=True)
