#!/usr/bin/env python3
"""Fréchet ANOVA on fitted Beta distributions — TEMPERATURE-VARIANCE VERSION.

Mirror of frechet_anova_percentile.py, but with temperature as the grouping
factor instead of expert_name. Persona is held fixed (Academic Security
Researcher); each elicitation produces one Beta distribution.

For each (model, step), groups are temperatures (5 levels). Per group, n=10
samples are repeats at the same T. Test: is between-temperature variance
significantly larger than within-temperature variance?

Math is identical to the percentile version (Dubey & Müller 2019, L2-W2).
Only the data-loading and experiment-runner blocks differ.

Usage:
    python3 frechet_anova_temperature.py
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

# Provider short names match run_temperature_sweep.sh / config_<provider>_temperature.yaml
PROVIDERS = ["claude", "gpt4o"]

# Display names for tables
MODEL_DISPLAY = {
    "claude": "Claude Sonnet 4.6",
    "gpt4o": "GPT-4o",
}

# Per-provider temperature sweeps (must match run_temperature_sweep.sh)
TEMPS_BY_PROVIDER = {
    "claude": [0.0, 0.25, 0.5, 0.75, 1.0],
    "gpt4o":  [0.0, 0.5,  1.0,  1.5,  2.0],
}

STEPS = ["TA0002 (50%)", "TA0007 (85%)", "T1657 (30%)"]
STEP_SHORT = {  # display name -> directory short label
    "TA0002 (50%)": "TA0002",
    "TA0007 (85%)": "TA0007",
    "T1657 (30%)":  "T1657",
}


def temp_dir(provider: str, step_display: str, temp: float) -> Path:
    """Path to one temperature group's run dirs (10 runs × 1 elicitation each)."""
    short = STEP_SHORT[step_display]
    return EXPERIMENTS_DIR / f"temperature_{provider}_{short}_t{temp}"

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

def load_temp_group(data_dir: Path) -> List[Tuple[np.ndarray, dict]]:
    """Load all elicitations in one temperature directory.

    Each subdirectory `run_*` contains a CSV with one row per expert (here: 1 row,
    since num_experts=1). Returns a flat list of (quantile_function, meta) tuples,
    one per successful elicitation.
    """
    entries = []
    fit_failures = 0
    validation_warnings = 0
    total = 0

    if not data_dir.exists():
        return entries

    run_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    for run_dir in run_dirs:
        csv_path = run_dir / "detailed_estimates.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
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
                entries.append((qf, {
                    "alpha": alpha, "beta": beta_param,
                    "p25": p25_val, "p50": p50_val, "p75": p75_val,
                    "max_error": max_error,
                    "model": row.get("model", ""),
                    "temperature": row.get("temperature", ""),
                }))

    if fit_failures > 0:
        pct = 100 * fit_failures / total if total > 0 else 0
        print(f"    Warning: {fit_failures}/{total} rows ({pct:.1f}%) failed Beta fitting in {data_dir.name}", flush=True)
    if validation_warnings > 0:
        pct = 100 * validation_warnings / (total - fit_failures) if (total - fit_failures) > 0 else 0
        print(f"    Warning: {validation_warnings} rows ({pct:.1f}%) have max fitting error > 0.02 in {data_dir.name}", flush=True)

    return entries


# ---------------------------------------------------------------------------
# Experiment Runners
# ---------------------------------------------------------------------------

def run_temperature_frechet_anova(provider: str, step: str) -> dict:
    """Run Fréchet ANOVA with TEMPERATURE as grouping factor.

    For one (provider, step), pulls 5 temperature directories; each contributes
    one group of ~10 quantile functions. Tests whether temperature creates
    distinct distributional clusters.
    """
    temps = TEMPS_BY_PROVIDER[provider]
    groups: Dict[str, List[np.ndarray]] = {}
    for t in temps:
        d = temp_dir(provider, step, t)
        entries = load_temp_group(d)
        if entries:
            # Use temperature value as group key (string for sortable display)
            groups[f"T={t}"] = [qf for qf, _ in entries]

    if len(groups) < 2:
        return {"error": f"Not enough temperature groups present: {list(groups.keys())}"}

    t_n, f_n, u_n, v_pooled, icc_f = frechet_anova_statistic(groups)
    k = len(groups)
    p_asymptotic = asymptotic_pvalue(t_n, k)

    print(f"  T_n = {t_n:.4f}, ICC_F = {icc_f:.4f}, p_asymptotic = {p_asymptotic:.4f}", flush=True)
    print(f"  Running permutation test ({N_PERMUTATIONS} permutations)...", flush=True)

    p_perm = permutation_test(groups, t_n)

    # W1 distances: within = repeats at same T; between = across T values.
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
        "model": MODEL_DISPLAY[provider],
        "provider": provider,
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
        "groups_present": group_names,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_file = Path(__file__).parent / "frechet_anova_results_temperature.txt"

    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("FRÉCHET ANOVA RESULTS (TEMPERATURE-VARIANCE)\n")
        f.write("=" * 80 + "\n\n")

        f.write("-" * 80 + "\n")
        f.write("EXPERIMENT: TEMPERATURE VARIANCE (FIXED PERSONA: Academic Security Researcher)\n")
        f.write("-" * 80 + "\n\n")
        f.write("H₀: All temperature levels produce the same belief distribution.\n")
        f.write("Claude temps: 0.0, 0.25, 0.5, 0.75, 1.0  |  GPT-4o temps: 0.0, 0.5, 1.0, 1.5, 2.0\n\n")

        results = []
        for provider in PROVIDERS:
            for step in STEPS:
                model_name = MODEL_DISPLAY[provider]
                print(f"\n>>> Temperature ANOVA: {model_name}, {step}", flush=True)
                try:
                    r = run_temperature_frechet_anova(provider, step)
                    if "error" not in r:
                        results.append(r)
                    else:
                        print(f"  Skipped: {r['error']}", flush=True)
                except Exception as e:
                    print(f"  ERROR: {e}", flush=True)
                    continue

        f.write(f"{'Model':<20} {'Step':<15} {'N':>5} {'k':>3} {'T_n':>8} {'p (perm)':>10} {'ICC_F':>8} {'W1_within':>10} {'W1_between':>11}\n")
        f.write("-" * 110 + "\n")
        for r in results:
            f.write(
                f"{r['model']:<20} {r['step']:<15} {r['n']:>5} {r['k_groups']:>3} "
                f"{r['T_n']:>8.4f} {r['p_permutation']:>10.4f} {r['ICC_F']:>8.3f} "
                f"{r['mean_W1_within']:>10.4f} {r['mean_W1_between']:>11.4f}\n"
            )

        f.write("\n\n")
        f.write("Reference (from prior experiments, percentile elicitation):\n")
        f.write("  Persona ICC_F: 6%–25% (mostly non-significant within Claude/GPT-4o)\n")
        f.write("  Cross-model ICC_F: 48%–69% (highly significant across all steps)\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("END OF RESULTS\n")
        f.write("=" * 80 + "\n")

    print(f"\n✓ Results written to: {output_file}\n", flush=True)


if __name__ == "__main__":
    main()
