#!/usr/bin/env python3
"""
Generate the W1 distribution-shift bar chart for the poster.

Single-panel bar chart: W1 distance from control for 6 non-control conditions,
with 95% bootstrap CI error bars (n=10,000 resamples).

Data loading / Beta-fitting logic ported from
  prompt_sensitivity/compare_all_conditions.py
Output: latex/figures/w1_chart (1)_week5.png
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, List
import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import least_squares, minimize
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROMPT_SENS_DIR = Path(__file__).parent.parent / "prompt_sensitivity"
RUNS_DIR        = PROMPT_SENS_DIR / "output" / "runs"
OUT             = Path(__file__).parent.parent / "latex" / "figures" / "w1_chart (1)_week5.png"

MIN_DURATION       = 30.0
QUANTILE_GRID_SIZE = 201
Q_GRID             = np.linspace(0.001, 0.999, QUANTILE_GRID_SIZE)

# ---------------------------------------------------------------------------
# Run classification  (identical to compare_all_conditions.py)
# ---------------------------------------------------------------------------
def classify_run(analysis_len: int, estimation_len: int, has_reasoning: bool, scenario: str) -> str:
    if scenario == "scenario_no_baseline.yaml":
        return "no_baseline"
    elif scenario == "scenario_no_ci.yaml":
        return "no_ci"
    elif scenario == "scenario_no_baseline_no_ci.yaml":
        return "no_baseline_no_ci"
    elif scenario == "scenario_control.yaml":
        if analysis_len < 2500:
            return "trim_all"
        elif analysis_len < 4000:
            return "skip_analysis"
        elif analysis_len > 5000:
            return "trim_reasoning" if not has_reasoning else "control"
    return "unknown"


# ---------------------------------------------------------------------------
# Beta fitting  (identical to compare_all_conditions.py)
# ---------------------------------------------------------------------------
def fit_beta_from_percentiles(p25: float, p50: float, p75: float) -> Optional[Tuple[float, float, float]]:
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
        return [sp_stats.beta.cdf(p25, a, b) - 0.25,
                sp_stats.beta.cdf(p50, a, b) - 0.50,
                sp_stats.beta.cdf(p75, a, b) - 0.75]

    iqr = p75 - p25
    loc = p50
    est_conc = max(2, min(1000, 0.1 / (iqr ** 2))) if iqr > 0 else 50
    init_guesses = [
        (est_conc * loc, est_conc * (1 - loc)),
        (est_conc * loc * 0.5, est_conc * (1 - loc) * 0.5),
        (est_conc * loc * 2,   est_conc * (1 - loc) * 2),
        (2, 2), (5, 5), (10, 10), (20, 20), (50, 50), (100, 100), (200, 200),
        (3, 8), (8, 3), (5, 15), (15, 5), (20, 80), (80, 20),
        (50, 200), (200, 50), (100, 20), (20, 100),
        (0.5, 0.5), (1, 1), (1, 2), (2, 1),
    ]

    best_sol, best_cost = None, float("inf")
    for a0, b0 in init_guesses:
        try:
            r = least_squares(residuals, [a0, b0], bounds=([0.1, 0.1], [2000, 2000]),
                              ftol=1e-12, xtol=1e-12)
            if r.success and r.cost < best_cost:
                best_sol, best_cost = r.x, r.cost
        except (ValueError, RuntimeError):
            pass

    if best_sol is None:
        return None

    a_fit, b_fit = best_sol
    errs = [abs(sp_stats.beta.cdf(p, a_fit, b_fit) - q)
            for p, q in [(p25, 0.25), (p50, 0.50), (p75, 0.75)]]
    if max(errs) > 0.05:
        return None
    return float(a_fit), float(b_fit), float(max(errs))


def quantile_function(alpha: float, beta: float) -> np.ndarray:
    return sp_stats.beta.ppf(Q_GRID, alpha, beta)


def w1_distance(qf1: np.ndarray, qf2: np.ndarray) -> float:
    return float(np.trapezoid(np.abs(qf1 - qf2), Q_GRID))


# ---------------------------------------------------------------------------
# Data loading  (identical to compare_all_conditions.py)
# ---------------------------------------------------------------------------
def load_all_runs():
    condition_runs = defaultdict(list)
    for run_dir in sorted(RUNS_DIR.glob("202603*")):
        if not run_dir.is_dir():
            continue
        json_file = run_dir / "full_results.json"
        if not json_file.exists():
            continue
        with open(json_file) as f:
            data = json.load(f)
        if data["run_metadata"]["duration_seconds"] < MIN_DURATION:
            continue
        scenario = data["run_metadata"]["config_used"]["scenario_file"].split("/")[-1]
        resp = data["results_per_step"][0]["results_per_task"][0]["rounds_data"][0]["responses"][0]
        analysis_len  = len(resp.get("analysis_user_prompt", ""))
        estimation_len = len(resp.get("estimation_user_prompt", ""))
        has_reasoning  = "<reasoning_structure>" in resp.get("estimation_user_prompt", "")
        condition = classify_run(analysis_len, estimation_len, has_reasoning, scenario)
        if condition != "unknown":
            condition_runs[condition].append(run_dir.name)
    return condition_runs


def load_estimates_for_run(run_id: str) -> List[dict]:
    csv_path = RUNS_DIR / run_id / "detailed_estimates.csv"
    if not csv_path.exists():
        return []
    estimates = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                estimates.append({
                    "p25": float(row["percentile_25th"]),
                    "p50": float(row["percentile_50th"]),
                    "p75": float(row["percentile_75th"]),
                })
            except (ValueError, KeyError):
                pass
    return estimates




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
CONDITION_ORDER = [
    "no_ci",
    "no_baseline",
    "no_baseline_no_ci",
    "skip_analysis",
    "trim_reasoning",
    "trim_all",
]

DISPLAY_NAMES = {
    "no_ci":             "No CI",
    "no_baseline":       "No Baseline",
    "no_baseline_no_ci": "No Baseline\n+ No CI",
    "skip_analysis":     "Skip Analysis",
    "trim_reasoning":    "Trim Reasoning",
    "trim_all":          "Trim All",
}

COLORS = {
    "no_ci":             "#2ca02c",
    "no_baseline":       "#d62728",
    "no_baseline_no_ci": "#ff7f0e",
    "skip_analysis":     "#9467bd",
    "trim_reasoning":    "#8c564b",
    "trim_all":          "#e377c2",
}

print("Loading and classifying runs...")
condition_runs = load_all_runs()

print("Fitting Beta distributions...")
all_conditions = ["control"] + CONDITION_ORDER
condition_qfs: dict = {}
for cname in all_conditions:
    qfs = []
    for run_id in condition_runs.get(cname, []):
        for est in load_estimates_for_run(run_id):
            result = fit_beta_from_percentiles(est["p25"], est["p50"], est["p75"])
            if result is not None:
                qfs.append(quantile_function(result[0], result[1]))
    condition_qfs[cname] = qfs
    print(f"  {cname:20s}: {len(qfs)} distributions")

control_qfs = condition_qfs["control"]

# Within-control: all pairwise W1 distances between control runs
within_w1 = [w1_distance(q1, q2)
             for i, q1 in enumerate(control_qfs)
             for j, q2 in enumerate(control_qfs) if i < j]
mean_within_w1 = float(np.mean(within_w1)) if within_w1 else 0.0

print(f"\nWithin-control mean W1 = {mean_within_w1:.4f}")

# Mean pairwise W1 (every control × every condition pair) + bootstrap CI
def bootstrap_pairwise_ci(pw: np.ndarray, n_boot: int = 10_000,
                          seed: int = 42) -> Tuple[float, float]:
    """95% bootstrap CI on the mean of pairwise W1 distances."""
    rng = np.random.default_rng(seed)
    n = len(pw)
    boot_means = pw[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

records = []
for cname in CONDITION_ORDER:
    cond_qfs = condition_qfs[cname]
    if not cond_qfs:
        print(f"  {cname}: no data — skipped")
        continue
    pw = np.array([w1_distance(qc, qx)
                   for qc in control_qfs
                   for qx in cond_qfs])
    mean_w1 = float(pw.mean())
    lo, hi  = bootstrap_pairwise_ci(pw)
    records.append({
        "cname":   cname,
        "label":   DISPLAY_NAMES[cname],
        "color":   COLORS[cname],
        "mean_w1": mean_w1,
        "lo":      lo,
        "hi":      hi,
    })
    print(f"  {cname:20s}: W1={mean_w1:.4f} [{lo:.4f}, {hi:.4f}]  (n_pairs={len(pw)})")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))

x     = np.arange(len(records))
WIDTH = 0.55
ANNOT_PAD  = 0.004
ANNOT_BBOX = dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85, edgecolor="none")

for i, r in enumerate(records):
    err = [[r["mean_w1"] - r["lo"]], [r["hi"] - r["mean_w1"]]]
    ax.bar(i, r["mean_w1"], WIDTH, color=r["color"], edgecolor="black", linewidth=0.6)
    ax.errorbar(i, r["mean_w1"], yerr=err,
                fmt="none", ecolor="black", elinewidth=1.2, capsize=5)
    ax.text(i, r["hi"] + ANNOT_PAD, f"{r['mean_w1']:.3f}",
            ha="center", va="bottom", fontsize=9, bbox=ANNOT_BBOX)

# Within-control reference line
blend = ax.get_yaxis_transform()
ax.axhline(mean_within_w1, ls="--", c="steelblue", lw=1.2)
ax.text(0.50, mean_within_w1 + 0.002,
        f"Within-control ({mean_within_w1:.3f})",
        color="steelblue", ha="center", va="bottom", fontsize=8.5, transform=blend)

ax.set_xticks(x)
ax.set_xticklabels([r["label"] for r in records], fontsize=10)
ax.set_ylabel("W1 Distance from Control  (lower = better)", fontsize=11)
ax.set_ylim(0, max(r["hi"] for r in records) + ANNOT_PAD + 0.025)
ax.set_title("Prompt Sensitivity: Distribution Shift (W₁) from Control",
             fontsize=12, pad=8)
ax.grid(axis="y", alpha=0.25)

fig.text(0.01, -0.04,
         "Mean W₁ over all control × condition distribution pairs.  "
         "Error bars: 95% bootstrap CI (n = 10,000 resamples).",
         fontsize=7.5, color="#555", ha="left")

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"\nSaved: {OUT}")
