#!/usr/bin/env python3
"""
Poster Figure 1: Prompt sensitivity — Distribution Shift + Output Stability.

Two side-by-side panels with 95% bootstrap CIs:
  Left:  W₂ distance from control (4 key conditions)
  Right: Relative within-condition variability (×)

Output: latex/figures/prompt_sensitivity_chart_week4.png

Data source: prompt_sensitivity/output/runs/  (same as compare_all_conditions.py)
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple
import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import least_squares, minimize
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────────────
RUNS_DIR = Path("/Users/madhav/SaferAI/LLM_elicitation/prompt_sensitivity/output/runs")
OUT      = Path("/Users/madhav/SaferAI/LLM_elicitation/latex/figures/prompt_sensitivity_chart_week4.png")

# ── config ─────────────────────────────────────────────────────────────────────
MIN_DURATION = 30.0
Q_GRID       = np.linspace(0.001, 0.999, 201)
N_BOOT       = 10_000
SEED         = 42

# Display order and styling for the 4 conditions in the poster
PLOT_CONDITIONS = [
    ("control",           "Control\n(baseline+CI)",   "#2ca02c"),
    ("no_ci",             "No CI\n(baseline only)",   "#1f77b4"),
    ("no_baseline",       "No Baseline\n(CI only)",   "#d62728"),
    ("no_baseline_no_ci", "No Baseline\nNo CI",        "#9467bd"),
]


# ── run classification (from compare_all_conditions.py) ────────────────────────
def classify_run(analysis_len: int, has_reasoning: bool, scenario: str) -> str:
    if scenario == "scenario_no_baseline.yaml":
        return "no_baseline"
    if scenario == "scenario_no_ci.yaml":
        return "no_ci"
    if scenario == "scenario_no_baseline_no_ci.yaml":
        return "no_baseline_no_ci"
    if scenario == "scenario_control.yaml":
        if analysis_len < 2500:
            return "trim_all"
        if analysis_len < 4000:
            return "skip_analysis"
        if analysis_len > 5000:
            return "control" if has_reasoning else "trim_reasoning"
    return "unknown"


def load_all_runs() -> dict:
    condition_runs: dict = defaultdict(list)
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
        resp = (data["results_per_step"][0]["results_per_task"][0]
                ["rounds_data"][0]["responses"][0])
        analysis_len = len(resp.get("analysis_user_prompt", ""))
        has_reasoning = "<reasoning_structure>" in resp.get("estimation_user_prompt", "")
        cond = classify_run(analysis_len, has_reasoning, scenario)
        if cond != "unknown":
            condition_runs[cond].append(run_dir.name)
    return condition_runs


def load_estimates_for_run(run_id: str) -> list:
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
                continue
    return estimates


# ── beta fitting (from compare_all_conditions.py) ──────────────────────────────
def fit_beta_from_percentiles(
    p25: float, p50: float, p75: float
) -> Optional[Tuple[float, float, float]]:
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
        return [
            sp_stats.beta.cdf(p25, a, b) - 0.25,
            sp_stats.beta.cdf(p50, a, b) - 0.50,
            sp_stats.beta.cdf(p75, a, b) - 0.75,
        ]

    iqr = p75 - p25
    loc = p50
    c   = max(2, min(1000, 0.1 / iqr**2)) if iqr > 0 else 50
    inits = [
        (c * loc, c * (1 - loc)), (c * loc * .5, c * (1 - loc) * .5),
        (c * loc * 2, c * (1 - loc) * 2),
        (2, 2), (5, 5), (10, 10), (20, 20), (50, 50), (100, 100), (200, 200),
        (3, 8), (8, 3), (5, 15), (15, 5), (20, 80), (80, 20),
        (0.5, 0.5), (1, 1), (1, 2), (2, 1),
    ]
    best, best_cost = None, float("inf")
    for a0, b0 in inits:
        try:
            r = least_squares(residuals, [a0, b0],
                              bounds=([0.1, 0.1], [2000, 2000]),
                              ftol=1e-12, xtol=1e-12)
            if r.success and r.cost < best_cost:
                best, best_cost = r.x, r.cost
        except (ValueError, RuntimeError):
            continue
    if best is None:
        return None
    a, b = best
    max_err = max(
        abs(sp_stats.beta.cdf(p25, a, b) - 0.25),
        abs(sp_stats.beta.cdf(p50, a, b) - 0.50),
        abs(sp_stats.beta.cdf(p75, a, b) - 0.75),
    )
    if max_err > 0.04:
        def obj(params):
            aa, bb = params
            if aa <= 0.1 or bb <= 0.1 or aa > 2000 or bb > 2000:
                return 1.0
            return max(
                abs(sp_stats.beta.cdf(p25, aa, bb) - 0.25),
                abs(sp_stats.beta.cdf(p50, aa, bb) - 0.50),
                abs(sp_stats.beta.cdf(p75, aa, bb) - 0.75),
            )
        try:
            r2 = minimize(obj, [a, b], method="Nelder-Mead",
                          options={"maxiter": 1000, "xatol": 1e-8, "fatol": 1e-8})
            if r2.success:
                an, bn = r2.x
                if 0.1 < an < 2000 and 0.1 < bn < 2000 and obj([an, bn]) < max_err:
                    a, b, max_err = an, bn, obj([an, bn])
        except (ValueError, RuntimeError):
            pass
    if max_err > 0.05:
        return None
    return float(a), float(b), float(max_err)


def quantile_function(alpha: float, beta: float) -> np.ndarray:
    return sp_stats.beta.ppf(Q_GRID, alpha, beta)


def w2(qf1: np.ndarray, qf2: np.ndarray) -> float:
    """L2-Wasserstein distance between two quantile functions."""
    return float(np.sqrt(np.trapezoid((qf1 - qf2) ** 2, Q_GRID)))


# ── bootstrap ──────────────────────────────────────────────────────────────────
def bootstrap_mean_ci(values, n_boot=N_BOOT, seed=SEED):
    """95% bootstrap CI for the mean of a list of values."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n < 2:
        m = float(arr.mean()) if n == 1 else float("nan")
        return m, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = arr[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(arr.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading and classifying runs...")
    condition_runs = load_all_runs()
    for cname, _, _ in PLOT_CONDITIONS:
        print(f"  {cname:20s}: {len(condition_runs.get(cname, []))} runs")

    print("\nFitting Beta distributions...")
    condition_qfs: dict = {}
    for cname, _, _ in PLOT_CONDITIONS:
        qfs = []
        for run_id in condition_runs.get(cname, []):
            for est in load_estimates_for_run(run_id):
                result = fit_beta_from_percentiles(est["p25"], est["p50"], est["p75"])
                if result is not None:
                    alpha, beta, _ = result
                    qfs.append(quantile_function(alpha, beta))
        condition_qfs[cname] = qfs
        print(f"  {cname:20s}: {len(qfs)} distributions")

    control_qfs = condition_qfs["control"]
    if not control_qfs:
        raise RuntimeError("No control distributions found — check RUNS_DIR and MIN_DURATION.")

    print("\nComputing pairwise W₂ distances...")

    # W₂ shift from control: pairwise distances between condition and control
    shift_pairs: dict = {}
    for cname, _, _ in PLOT_CONDITIONS:
        if cname == "control":
            shift_pairs[cname] = []   # trivially 0
            continue
        cqfs = condition_qfs[cname]
        dists = [w2(qfc, qf) for qfc in control_qfs for qf in cqfs]
        shift_pairs[cname] = dists
        print(f"  shift {cname:20s}: {len(dists)} pairs, mean={np.mean(dists):.4f}")

    # Within-condition pairwise W₂ (all pairs i < j)
    within_pairs: dict = {}
    for cname, _, _ in PLOT_CONDITIONS:
        cqfs = condition_qfs[cname]
        dists = [w2(cqfs[i], cqfs[j])
                 for i in range(len(cqfs)) for j in range(i + 1, len(cqfs))]
        within_pairs[cname] = dists
        if dists:
            print(f"  within {cname:20s}: {len(dists)} pairs, mean={np.mean(dists):.4f}")

    # Reference: within-control mean (used to normalise relative variability)
    ctrl_within_mean = float(np.mean(within_pairs["control"])) if within_pairs["control"] else 1.0
    print(f"\nWithin-control W₂ mean: {ctrl_within_mean:.4f}")

    # Bootstrap CIs
    print("\nBootstrapping 95% CIs...")
    shift_stats: dict = {}
    var_stats: dict = {}

    for cname, _, _ in PLOT_CONDITIONS:
        if cname == "control":
            shift_stats[cname] = (0.0, 0.0, 0.0)
        else:
            shift_stats[cname] = bootstrap_mean_ci(shift_pairs[cname])

        wdata = within_pairs[cname]
        if len(wdata) < 2:
            var_stats[cname] = (
                float(np.mean(wdata)) / ctrl_within_mean if wdata else 1.0,
                float("nan"), float("nan"),
            )
        else:
            wmean, wlo, whi = bootstrap_mean_ci(wdata)
            var_stats[cname] = (
                wmean / ctrl_within_mean,
                wlo   / ctrl_within_mean,
                whi   / ctrl_within_mean,
            )

    # Print summary
    print("\n{:22s}  {:>28s}  {:>28s}".format(
        "Condition", "W₂ shift [lo, hi]", "Rel. variability [lo, hi]"))
    for cname, label, _ in PLOT_CONDITIONS:
        sm, slo, shi = shift_stats[cname]
        vm, vlo, vhi = var_stats[cname]
        print(f"  {cname:20s}  {sm:.4f} [{slo:.4f}, {shi:.4f}]"
              f"  {vm:.2f}× [{vlo:.2f}×, {vhi:.2f}×]")

    # ── plot ──────────────────────────────────────────────────────────────────
    ANNOT_BBOX = dict(boxstyle="round,pad=0.15", facecolor="white",
                      alpha=0.85, edgecolor="none")
    ANNOT_PAD  = 0.012   # relative units (for shift panel)

    cnames  = [c for c, _, _ in PLOT_CONDITIONS]
    labels  = [l for _, l, _ in PLOT_CONDITIONS]
    colors  = [col for _, _, col in PLOT_CONDITIONS]
    x       = np.arange(len(cnames))
    width   = 0.55

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.30)

    # ── left panel: W₂ shift ──────────────────────────────────────────────────
    max_shift_hi = max(shi for _, slo, shi in shift_stats.values())
    ylim_shift   = max(max_shift_hi + 0.04, 0.05)
    ann_pad_shift = ylim_shift * 0.04

    for i, cname in enumerate(cnames):
        sm, slo, shi = shift_stats[cname]
        ax1.bar(i, sm, width, color=colors[i], edgecolor="black", linewidth=0.6)
        if cname != "control":
            ax1.errorbar(i, sm,
                         yerr=[[sm - slo], [shi - sm]],
                         fmt="none", ecolor="black", elinewidth=1.2, capsize=5)
        ann_y = shi + ann_pad_shift if cname != "control" else ann_pad_shift
        ax1.text(i, ann_y, f"{sm:.3f}", ha="center", va="bottom",
                 fontsize=9, bbox=ANNOT_BBOX)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("W₂ distance from control", fontsize=11)
    ax1.set_title("Distribution Shift\n"
                  "(How much does removing baseline/CI\nmove estimates?)",
                  fontsize=10)
    ax1.set_ylim(0, ylim_shift)

    # ── right panel: relative variability ─────────────────────────────────────
    # Dynamic annotation offset: 4% of the y-range (computed after bars are drawn)
    all_tops_var = [(vhi if not np.isnan(vhi) else vm) for vm, vlo, vhi in var_stats.values()]
    max_var_hi   = max(all_tops_var)
    ylim_var     = max_var_hi + 1.5
    ann_pad_var  = ylim_var * 0.04

    for i, cname in enumerate(cnames):
        vm, vlo, vhi = var_stats[cname]
        ax2.bar(i, vm, width, color=colors[i], edgecolor="black", linewidth=0.6)
        if not (np.isnan(vlo) or np.isnan(vhi)):
            ax2.errorbar(i, vm,
                         yerr=[[vm - vlo], [vhi - vm]],
                         fmt="none", ecolor="black", elinewidth=1.2, capsize=5)
        ann_y = (vhi if not np.isnan(vhi) else vm) + ann_pad_var
        ax2.text(i, ann_y, f"{vm:.1f}×", ha="center", va="bottom",
                 fontsize=9, bbox=ANNOT_BBOX)

    # reference line at 1.0× — label on the LEFT to avoid bar overlap
    blend2 = ax2.get_yaxis_transform()
    ax2.axhline(1.0, ls="--", c="grey", lw=1)
    ax2.text(0.01, 1.0 + ylim_var * 0.02, "Control (1.0×)",
             color="grey", ha="left", va="bottom", fontsize=9, transform=blend2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Relative output variability (×)", fontsize=11)
    ax2.set_title("Output Stability\n"
                  "(How variable are repeated samples?)",
                  fontsize=10)
    ax2.set_ylim(0, ylim_var)

    plt.tight_layout()
    fig.text(0.02, -0.02,
             "Error bars: 95% bootstrap CI (n = 10,000 resamples).  "
             "Relative variability = mean pairwise within-condition W₂ / mean pairwise within-control W₂.",
             fontsize=8, color="#555", ha="left")
    plt.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
