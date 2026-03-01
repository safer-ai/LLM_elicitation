#!/usr/bin/env python3
"""Analyze 10x10 ANOVA experiment results and produce summary."""

import csv
import sys
from pathlib import Path
from collections import defaultdict
import math

def load_estimates(anova_dir: str) -> tuple[dict[str, list[float]], dict[str, str]]:
    """Load all estimates grouped by expert name.
    Returns ({expert: [estimates]}, {metadata dict with model, task, step})."""
    base = Path(anova_dir)
    expert_estimates = defaultdict(list)
    metadata = {"model": None, "task": None, "step": None}

    run_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not run_dirs:
        print(f"ERROR: No run directories found in {anova_dir}")
        sys.exit(1)

    for run_dir in run_dirs:
        csv_path = run_dir / "detailed_estimates.csv"
        if not csv_path.exists():
            print(f"WARNING: Missing {csv_path}")
            continue
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = row.get("most_likely_estimate", "").strip()
                if not raw:
                    continue
                try:
                    estimate = float(raw)
                except ValueError:
                    continue
                expert = row["expert_name"]
                expert_estimates[expert].append(estimate)
                if metadata["model"] is None:
                    metadata["model"] = row.get("model", "Unknown")
                    metadata["task"] = row.get("task_name", "Unknown")
                    metadata["step"] = row.get("step_name", "Unknown")

    return dict(expert_estimates), metadata


def compute_stats(expert_estimates: dict[str, list[float]]) -> dict:
    experts = sorted(expert_estimates.keys())
    k = len(experts)
    n_per_group = len(next(iter(expert_estimates.values())))

    all_values = []
    group_means = {}
    group_vars = {}

    for expert in experts:
        vals = expert_estimates[expert]
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
        group_means[expert] = mu
        group_vars[expert] = var
        all_values.extend(vals)

    grand_mean = sum(all_values) / len(all_values)
    N = len(all_values)

    # Within-group variance (pooled)
    ss_within = sum(group_vars[e] * (n_per_group - 1) for e in experts)
    df_within = N - k
    ms_within = ss_within / df_within
    within_std = math.sqrt(ms_within)

    # Between-group variance
    ss_between = sum(n_per_group * (group_means[e] - grand_mean) ** 2 for e in experts)
    df_between = k - 1
    ms_between = ss_between / df_between

    # F-statistic
    f_stat = ms_between / ms_within

    # Between-expert std (of means)
    mean_vals = list(group_means.values())
    between_std = math.sqrt(sum((m - grand_mean) ** 2 for m in mean_vals) / (k - 1))

    # ICC(1,1) = (MS_between - MS_within) / (MS_between + (n-1)*MS_within)
    icc = (ms_between - ms_within) / (ms_between + (n_per_group - 1) * ms_within)
    icc = max(icc, 0.0)  # ICC can be negative; floor at 0

    # p-value approximation using F-distribution
    # Using scipy if available, otherwise note it
    p_value_f = None
    p_value_kw = None
    h_stat = None
    try:
        from scipy import stats as sp_stats
        p_value_f = 1 - sp_stats.f.cdf(f_stat, df_between, df_within)

        # Kruskal-Wallis
        groups = [expert_estimates[e] for e in experts]
        h_stat, p_value_kw = sp_stats.kruskal(*groups)
    except ImportError:
        pass

    return {
        "experts": experts,
        "group_means": group_means,
        "group_vars": group_vars,
        "grand_mean": grand_mean,
        "within_std": within_std,
        "between_std": between_std,
        "f_stat": f_stat,
        "df_between": df_between,
        "df_within": df_within,
        "p_value_f": p_value_f,
        "h_stat": h_stat,
        "p_value_kw": p_value_kw,
        "icc": icc,
        "k": k,
        "n_per_group": n_per_group,
        "N": N,
    }


def format_summary(stats: dict, model: str, task: str, step: str, scenario: str,
                   baseline: str, temp: str) -> str:
    lines = []
    lines.append(f"EXPERIMENT: Expert Persona Diversity — {model}")
    lines.append("")
    lines.append("")
    lines.append("SETUP")
    lines.append(f"Model:     {model}")
    lines.append(f"Task:      {task}")
    lines.append(f"Step:      {step} (baseline probability {baseline})")
    lines.append(f"Scenario:  {scenario}")
    lines.append(f"Rounds:    1 (no Delphi deliberation)")
    lines.append(f"Temp:      {temp}")
    lines.append("")
    lines.append("DESIGN")
    lines.append(f"{stats['k']} expert personas × {stats['n_per_group']} independent runs = {stats['N']} probability estimates.")
    lines.append(f"Each run executes all {stats['k']} experts concurrently. With delphi_rounds=1,")
    lines.append("experts never see each other's answers — all estimates are independent.")
    lines.append(f"This is a balanced one-way ANOVA design ({stats['k']} groups, {stats['n_per_group']} obs/group).")
    lines.append("")
    lines.append("RESULTS")
    lines.append("Expert Means (sorted):")

    sorted_experts = sorted(stats["group_means"].items(), key=lambda x: x[1])
    max_name_len = max(len(e) for e in stats["experts"])
    for expert, mean in sorted_experts:
        lines.append(f"  {expert:<{max_name_len}}  {mean:.3f}")

    lines.append("")
    lines.append(f"Within-expert std (pooled):  {stats['within_std']:.3f}  (noise from LLM sampling)")
    lines.append(f"Between-expert std (means):  {stats['between_std']:.3f}  (persona-driven spread)")
    lines.append("")
    lines.append("STATISTICAL TESTS")

    if stats["p_value_f"] is not None:
        lines.append(f"One-way ANOVA:    F({stats['df_between']},{stats['df_within']}) = {stats['f_stat']:.2f},  p = {stats['p_value_f']:.2f}   → {'NOT significant' if stats['p_value_f'] > 0.05 else 'SIGNIFICANT'} at α=0.05")
    else:
        lines.append(f"One-way ANOVA:    F({stats['df_between']},{stats['df_within']}) = {stats['f_stat']:.2f}  (install scipy for p-value)")

    if stats["h_stat"] is not None:
        lines.append(f"Kruskal-Wallis:   H = {stats['h_stat']:.2f},       p = {stats['p_value_kw']:.2f}   → {'NOT significant' if stats['p_value_kw'] > 0.05 else 'SIGNIFICANT'} at α=0.05")

    icc_pct = stats["icc"] * 100
    lines.append(f"ICC(1,1):         {stats['icc']:.3f}  (only {icc_pct:.1f}% of variance from persona identity)")

    lines.append("")
    lines.append("CONCLUSION")
    if stats["p_value_f"] is not None and stats["p_value_f"] > 0.05:
        lines.append(f"Expert personas on {model} do NOT generate meaningful diversity")
        lines.append(f"in probability estimates. The {stats['k']} personas produce estimates that are")
        lines.append("statistically indistinguishable from running the same persona repeatedly.")
    elif stats["p_value_f"] is not None:
        lines.append(f"Expert personas on {model} show STATISTICALLY SIGNIFICANT differences")
        lines.append(f"in probability estimates (p={stats['p_value_f']:.4f}). However, the practical")
        lines.append(f"significance should be evaluated: ICC={stats['icc']:.3f} means persona identity")
        lines.append(f"explains only {icc_pct:.1f}% of variance.")
    else:
        lines.append("(Install scipy for full statistical conclusions.)")

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_anova.py <anova_results_dir> [baseline_pct]")
        print("Example: python analyze_anova.py output_data/experiments/pilot_experiments/cross_model_gpt4o 50%")
        sys.exit(1)

    anova_dir = sys.argv[1]
    baseline = sys.argv[2] if len(sys.argv) > 2 else "unknown"

    estimates, metadata = load_estimates(anova_dir)
    stats = compute_stats(estimates)

    summary = format_summary(
        stats,
        model=metadata["model"],
        task=metadata["task"],
        step=metadata["step"],
        scenario="OC3 Ransomware on large enterprise",
        baseline=baseline,
        temp="1.0",
    )

    print(summary)

    output_path = Path(anova_dir) / "experiment_summary.txt"
    with open(output_path, "w") as f:
        f.write(summary)
    print(f"\nSaved to: {output_path}")
