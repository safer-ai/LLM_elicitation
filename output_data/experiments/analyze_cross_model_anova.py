#!/usr/bin/env python3
"""Cross-model ANOVA: test whether model identity explains variance in estimates.

Usage:
    python3 analyze_cross_model_anova.py <step_label> <baseline_pct> <claude_dir> <gpt4o_dir> <gemini_dir>

Example:
    python3 analyze_cross_model_anova.py "TA0002 - Execution" 50 \
        output_data/experiments/anova_probability \
        output_data/experiments/pilot_experiments/cross_model_gpt4o \
        output_data/experiments/pilot_experiments/cross_model_gemini_TA0002_50pct
"""

import csv
import sys
import math
from pathlib import Path
from collections import defaultdict


def load_all_estimates(model_dir: str) -> list[float]:
    """Load all probability estimates from a model's experiment directory."""
    base = Path(model_dir)
    estimates = []

    run_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not run_dirs:
        print(f"ERROR: No run directories found in {model_dir}")
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
                    val = float(raw)
                except ValueError:
                    continue
                estimates.append(val)

    return estimates


def compute_cross_model_stats(model_estimates: dict[str, list[float]]) -> dict:
    """One-way ANOVA with model as the grouping factor."""
    models = sorted(model_estimates.keys())
    k = len(models)
    group_sizes = {m: len(model_estimates[m]) for m in models}

    all_values = []
    group_means = {}
    group_vars = {}

    for model in models:
        vals = model_estimates[model]
        n = len(vals)
        mu = sum(vals) / n
        var = sum((v - mu) ** 2 for v in vals) / (n - 1)
        group_means[model] = mu
        group_vars[model] = var
        all_values.extend(vals)

    grand_mean = sum(all_values) / len(all_values)
    N = len(all_values)

    ss_within = sum(group_vars[m] * (group_sizes[m] - 1) for m in models)
    df_within = N - k
    ms_within = ss_within / df_within
    within_std = math.sqrt(ms_within)

    ss_between = sum(group_sizes[m] * (group_means[m] - grand_mean) ** 2 for m in models)
    df_between = k - 1
    ms_between = ss_between / df_between

    f_stat = ms_between / ms_within

    mean_vals = list(group_means.values())
    between_std = math.sqrt(sum((m - grand_mean) ** 2 for m in mean_vals) / (k - 1))

    n_harmonic = k / sum(1.0 / group_sizes[m] for m in models)
    icc = (ms_between - ms_within) / (ms_between + (n_harmonic - 1) * ms_within)
    icc = max(icc, 0.0)

    p_value_f = None
    p_value_kw = None
    h_stat = None
    try:
        from scipy import stats as sp_stats
        p_value_f = 1 - sp_stats.f.cdf(f_stat, df_between, df_within)
        groups = [model_estimates[m] for m in models]
        h_stat, p_value_kw = sp_stats.kruskal(*groups)
    except ImportError:
        pass

    return {
        "models": models,
        "group_means": group_means,
        "group_vars": group_vars,
        "group_sizes": group_sizes,
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
        "N": N,
    }


def format_cross_model_summary(stats: dict, step: str, baseline: str) -> str:
    lines = []
    lines.append("EXPERIMENT: Cross-Model Variance (Positive Control)")
    lines.append("")
    lines.append("")
    lines.append("SETUP")
    lines.append(f"Models:    {', '.join(stats['models'])}")
    lines.append(f"Step:      {step} (baseline probability {baseline}%)")
    lines.append("Scenario:  OC3 Ransomware on large enterprise")
    lines.append("Rounds:    1 (no Delphi deliberation)")
    lines.append("Temp:      1.0")
    lines.append("")
    lines.append("DESIGN")
    size_str = " + ".join(f"{stats['group_sizes'][m]}" for m in stats["models"])
    lines.append(f"{stats['k']} models × ~{stats['group_sizes'][stats['models'][0]]} estimates each = {stats['N']} total.")
    lines.append(f"Each model ran 10 independent runs with all 10 expert personas.")
    lines.append(f"Since the persona experiment showed personas ≈ noise, the")
    lines.append(f"10 per-run persona estimates are treated as independent draws.")
    lines.append(f"One-way ANOVA with model as the grouping factor ({stats['k']} groups).")
    lines.append("")
    lines.append("RESULTS")
    lines.append("Model Means:")

    max_name_len = max(len(m) for m in stats["models"])
    sorted_models = sorted(stats["group_means"].items(), key=lambda x: x[1])
    for model, mean in sorted_models:
        n = stats["group_sizes"][model]
        lines.append(f"  {model:<{max_name_len}}  {mean:.3f}  (n={n})")

    mean_range = max(stats["group_means"].values()) - min(stats["group_means"].values())
    lines.append("")
    lines.append(f"Range of model means:    {mean_range:.3f}  ({mean_range*100:.1f} percentage points)")
    lines.append(f"Within-model std:        {stats['within_std']:.3f}  (pooled LLM sampling noise)")
    lines.append(f"Between-model std:       {stats['between_std']:.3f}  (model-driven spread)")
    lines.append("")
    lines.append("STATISTICAL TESTS")

    if stats["p_value_f"] is not None:
        sig = "NOT significant" if stats["p_value_f"] > 0.05 else "SIGNIFICANT"
        lines.append(f"One-way ANOVA:    F({stats['df_between']},{stats['df_within']}) = {stats['f_stat']:.2f},  p = {stats['p_value_f']:.4f}   → {sig} at α=0.05")
    else:
        lines.append(f"One-way ANOVA:    F({stats['df_between']},{stats['df_within']}) = {stats['f_stat']:.2f}  (install scipy for p-value)")

    if stats["h_stat"] is not None:
        sig_kw = "NOT significant" if stats["p_value_kw"] > 0.05 else "SIGNIFICANT"
        lines.append(f"Kruskal-Wallis:   H = {stats['h_stat']:.2f},       p = {stats['p_value_kw']:.4f}   → {sig_kw} at α=0.05")

    icc_pct = stats["icc"] * 100
    lines.append(f"ICC(1,1):         {stats['icc']:.3f}  ({icc_pct:.1f}% of variance from model identity)")

    lines.append("")
    lines.append("CONCLUSION")
    if stats["p_value_f"] is not None and stats["p_value_f"] <= 0.05:
        lines.append(f"Model identity DOES generate significant variance in estimates.")
        lines.append(f"The {mean_range*100:.1f}pp spread across model means far exceeds the")
        lines.append(f"~1–3pp persona spread observed in the persona experiments.")
        lines.append(f"ICC={stats['icc']:.3f} means model choice explains {icc_pct:.1f}% of variance,")
        lines.append(f"compared to 0–6% for persona choice.")
    elif stats["p_value_f"] is not None:
        lines.append(f"Model identity does NOT generate significant variance (p={stats['p_value_f']:.4f}).")
        lines.append(f"Like personas, different models produce similar estimates.")
    else:
        lines.append("(Install scipy for full statistical conclusions.)")

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python3 analyze_cross_model_anova.py <step_label> <baseline_pct> <claude_dir> <gpt4o_dir> <gemini_dir>")
        print()
        print("Example:")
        print('  python3 analyze_cross_model_anova.py "TA0002 - Execution" 50 \\')
        print("      output_data/experiments/anova_probability \\")
        print("      output_data/experiments/pilot_experiments/cross_model_gpt4o \\")
        print("      output_data/experiments/pilot_experiments/cross_model_gemini_TA0002_50pct")
        sys.exit(1)

    step_label = sys.argv[1]
    baseline_pct = sys.argv[2]
    claude_dir = sys.argv[3]
    gpt4o_dir = sys.argv[4]
    gemini_dir = sys.argv[5]

    print(f"Loading estimates for step: {step_label} ({baseline_pct}% baseline)")
    print(f"  Claude dir:  {claude_dir}")
    print(f"  GPT-4o dir:  {gpt4o_dir}")
    print(f"  Gemini dir:  {gemini_dir}")
    print()

    model_estimates = {
        "Claude Sonnet 4.5": load_all_estimates(claude_dir),
        "GPT-4o": load_all_estimates(gpt4o_dir),
        "Gemini 2.5 Pro": load_all_estimates(gemini_dir),
    }

    for model, ests in model_estimates.items():
        print(f"  {model}: {len(ests)} estimates loaded")

    print()

    stats = compute_cross_model_stats(model_estimates)
    summary = format_cross_model_summary(stats, step_label, baseline_pct)

    print(summary)

    output_dir = Path(gemini_dir).parent
    output_path = output_dir / f"cross_model_anova_{step_label.split(' - ')[0].strip()}.txt"
    with open(output_path, "w") as f:
        f.write(summary)
    print(f"\nSaved to: {output_path}")
