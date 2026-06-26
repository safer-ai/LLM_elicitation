#!/usr/bin/env python3
"""
Analyse the output of run_probe.py.

Reads the latest probe_a.csv and probe_b.csv from results/ and produces:
  - results/contamination_report.md
  - Printed summary to stdout

Key metrics:
  Probe A (recall):
    - Brier(recall): treat stated pass_rate as a probability → Brier vs true binary outcome
      Compare to Exp I control Brier = 0.137 and chance Brier = 0.25
    - Spearman rho between stated pass_rate and true outcome (all rows with numeric answer)
    - By-confidence breakdown: precision when model says "high" confidence
    - By forecasted-model breakdown
    - By task-family breakdown
    - Stated-unknown rate (how often model says it doesn't know)

  Probe B (recognition):
    - Recognition rate overall and by family
    - By FST bin (easy tasks may be more recognised)

Usage (from repo root):
    python intra_benchmark_calibration/experiments/0b_contamination_probe/analyse_probe.py \\
        [--results-dir intra_benchmark_calibration/experiments/0b_contamination_probe/results]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

EXP_I_CONTROL_BRIER = 0.1377   # from Experiment I
CHANCE_BRIER = 0.25


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2))


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTROL_GLOB = (
    "intra_benchmark_calibration/experiments/I_prompt_ablation_brier/"
    "results/control/*/*_intra_estimates.csv"
)


def load_latest(results_dir: Path, suffix: str) -> pd.DataFrame:
    csvs = sorted(results_dir.glob(f"*_{suffix}.csv"))
    if not csvs:
        sys.exit(f"No {suffix} CSV found in {results_dir}")
    path = csvs[-1]
    print(f"Loading {path.name} …")
    return pd.read_csv(path)


def load_control_p50() -> Optional[pd.DataFrame]:
    """Load the Experiment I control run's per-(task, model) p50 so we can compute
    the matched Exp I Brier on the EXACT same answered subset."""
    matches = sorted(REPO_ROOT.glob(DEFAULT_CONTROL_GLOB))
    if not matches:
        return None
    df = pd.read_csv(matches[-1])
    df = df.dropna(subset=["p50", "outcome"]).drop_duplicates(
        subset=["target_task_id", "forecasted_model"])
    return df[["target_task_id", "forecasted_model", "p50", "outcome"]].rename(
        columns={"p50": "control_p50"})


def analyse_probe_a(df: pd.DataFrame, control: Optional[pd.DataFrame]) -> list[str]:
    lines = ["## Probe A — Direct numeric recall\n"]
    n_total = len(df)
    lines.append(f"Total rows: {n_total}  |  "
                 f"Forecasted models: {df['forecasted_model'].nunique()}  |  "
                 f"Tasks: {df['task_id'].nunique()}\n")

    # Drop rows where true_outcome is NaN (model not evaluated on this task)
    df = df.dropna(subset=["true_outcome"])
    n_eval = len(df)
    lines.append(f"Rows with known ground-truth outcome: {n_eval}/{n_total}\n")

    # Rows where model gave a numeric answer
    df_num = df.dropna(subset=["pass_rate"])
    n_num = len(df_num)
    n_unknown = n_eval - n_num
    lines.append(f"Rows with numeric pass_rate answer: {n_num}  |  "
                 f"'unknown' / unparseable: {n_unknown} "
                 f"({100*n_unknown/max(n_eval,1):.1f}%)\n")

    if n_num < 5:
        lines.append("Too few numeric answers to compute statistics.\n")
        return lines

    probs = df_num["pass_rate"].to_numpy(dtype=float)
    outcomes = df_num["true_outcome"].to_numpy(dtype=float)

    # Brier score for the recall answers
    br = brier(probs, outcomes)
    lines.append(f"### Headline recall metrics\n")
    lines.append(f"| Metric | Value | Reference |\n|---|---|---|")
    lines.append(f"| Brier(recall) | **{br:.4f}** | chance=0.25, Exp I control=0.137 |")
    rho, p_rho = stats.spearmanr(probs, outcomes)
    lines.append(f"| Spearman ρ(stated, true) | {rho:.3f} | p={p_rho:.2e} |")
    # Fraction correct when treating as binary (threshold 0.5)
    preds_bin = (probs >= 0.5).astype(float)
    acc = float(np.mean(preds_bin == outcomes))
    lines.append(f"| Binary accuracy (threshold 0.5) | {acc:.3f} | — |")
    lines.append(f"| Mean stated pass_rate | {probs.mean():.3f} | "
                 f"mean true outcome = {outcomes.mean():.3f} |")
    lines.append("")

    # Matched comparison: Exp I control Brier on the EXACT same answered (task,model) pairs.
    if control is not None:
        merged = df_num.merge(control, on=["target_task_id", "forecasted_model"], how="inner")
        if len(merged) >= 5:
            recall_b = brier(merged["pass_rate"].to_numpy(float),
                             merged["true_outcome"].to_numpy(float))
            ctrl_b = brier(merged["control_p50"].to_numpy(float),
                           merged["outcome"].to_numpy(float))
            lines.append("### Matched comparison (same answered pairs)\n")
            lines.append(f"On the {len(merged)} pairs where the model gave a numeric recall "
                         f"answer AND Exp I has a p50:")
            lines.append("")
            lines.append("| Source | Brier on these pairs |\n|---|---|")
            lines.append(f"| Recall (Probe A) | **{recall_b:.4f}** |")
            lines.append(f"| Exp I control (reasoning from source tasks) | **{ctrl_b:.4f}** |")
            gap = recall_b - ctrl_b
            lines.append("")
            lines.append(f"Recall is **{gap:+.4f}** Brier vs Exp I reasoning on the same pairs. "
                         f"{'Recall is much worse → genuine reasoning, not memorization.' if gap > 0.05 else 'Recall ≈ reasoning → inspect for contamination.'}\n")

    # Verdict
    if br <= EXP_I_CONTROL_BRIER + 0.02:
        verdict = ("⚠️  Recall Brier is close to Experiment I Brier — possible "
                   "contamination. Inspect by-model and by-family tables.")
    elif br <= 0.20:
        verdict = "🟡  Partial recall (Brier < 0.20) — weak contamination possible."
    else:
        verdict = ("✅  Recall Brier >> Experiment I Brier — model is NOT "
                   "accurately recalling outcomes. Exp I signal is genuine.")
    lines.append(f"**Verdict:** {verdict}\n")

    # By confidence
    lines.append("### By stated confidence\n")
    lines.append("| Confidence | N | Brier | Spearman ρ | % correct (thresh 0.5) |")
    lines.append("|---|---|---|---|---|")
    for conf in ["high", "medium", "low", "none"]:
        sub = df_num[df_num["confidence"].str.lower() == conf]
        if len(sub) < 3:
            continue
        p = sub["pass_rate"].to_numpy(dtype=float)
        o = sub["true_outcome"].to_numpy(dtype=float)
        b = brier(p, o)
        r, _ = stats.spearmanr(p, o) if len(sub) > 2 else (float("nan"), None)
        a = float(np.mean((p >= 0.5) == o))
        lines.append(f"| {conf} | {len(sub)} | {b:.4f} | {r:.3f} | {a:.3f} |")
    lines.append("")

    # By forecasted model
    lines.append("### By forecasted model\n")
    lines.append("| Model | N | Brier | Spearman ρ | % unknown |")
    lines.append("|---|---|---|---|---|")
    for fm in sorted(df["forecasted_model"].unique()):
        sub_all = df[df["forecasted_model"] == fm]
        sub_num = df_num[df_num["forecasted_model"] == fm]
        if len(sub_num) < 2:
            continue
        p = sub_num["pass_rate"].to_numpy(dtype=float)
        o = sub_num["true_outcome"].to_numpy(dtype=float)
        b = brier(p, o)
        r, _ = stats.spearmanr(p, o) if len(sub_num) > 2 else (float("nan"), None)
        pct_unk = 100 * (1 - len(sub_num) / max(len(sub_all), 1))
        lines.append(f"| {fm} | {len(sub_num)} | {b:.4f} | {r:.3f} | {pct_unk:.0f}% |")
    lines.append("")

    # By task family
    lines.append("### By task family\n")
    lines.append("| Family | N | Brier | Spearman ρ |")
    lines.append("|---|---|---|---|")
    for fam in sorted(df_num["task_family"].unique()):
        sub = df_num[df_num["task_family"] == fam]
        if len(sub) < 2:
            continue
        p = sub["pass_rate"].to_numpy(dtype=float)
        o = sub["true_outcome"].to_numpy(dtype=float)
        b = brier(p, o)
        r, _ = stats.spearmanr(p, o) if len(sub) > 2 else (float("nan"), None)
        lines.append(f"| {fam} | {len(sub)} | {b:.4f} | {r:.3f} |")
    lines.append("")

    return lines


def analyse_probe_b(df: pd.DataFrame) -> list[str]:
    lines = ["## Probe B — Task recognition\n"]
    n = len(df)
    lines.append(f"Total tasks probed: {n}\n")
    if n == 0:
        return lines

    def pct(col_val):
        return 100 * (df["recognized"].str.lower() == col_val).sum() / n

    lines.append(f"| Recognized | Count | % |")
    lines.append("|---|---|---|")
    for v in ["yes", "unsure", "no"]:
        cnt = (df["recognized"].str.lower() == v).sum()
        lines.append(f"| {v} | {cnt} | {100*cnt/n:.0f}% |")
    lines.append("")

    # Verdict
    pct_yes = pct("yes")
    pct_unsure_yes = pct("yes") + pct("unsure")
    if pct_yes > 30:
        verdict = ("⚠️  High recognition rate — tasks are known to the model. "
                   "This doesn't prove outcome memorization but is a prerequisite for it.")
    elif pct_unsure_yes > 50:
        verdict = "🟡  Moderate uncertainty about recognition."
    else:
        verdict = "✅  Low recognition rate — tasks are largely unknown to the model."
    lines.append(f"**Verdict:** {verdict}\n")

    # By family
    lines.append("### By task family\n")
    lines.append("| Family | N | Recognized (yes) | Unsure | Not recognized |")
    lines.append("|---|---|---|---|---|")
    for fam in sorted(df["task_family"].unique()):
        sub = df[df["task_family"] == fam]
        n_fam = len(sub)
        y = (sub["recognized"].str.lower() == "yes").sum()
        u = (sub["recognized"].str.lower() == "unsure").sum()
        no = (sub["recognized"].str.lower() == "no").sum()
        lines.append(f"| {fam} | {n_fam} | {y} ({100*y/n_fam:.0f}%) | "
                     f"{u} ({100*u/n_fam:.0f}%) | {no} ({100*no/n_fam:.0f}%) |")
    lines.append("")

    # By FST difficulty quartile
    df["fst_minutes"] = pd.to_numeric(df["fst_minutes"], errors="coerce")
    df_fst = df.dropna(subset=["fst_minutes"])
    if len(df_fst) >= 4:
        df_fst = df_fst.copy()
        df_fst["fst_bin"] = pd.qcut(df_fst["fst_minutes"], q=3,
                                    labels=["easy", "medium", "hard"])
        lines.append("### By difficulty (FST tertile)\n")
        lines.append("| Difficulty | N | Recognized (yes) |")
        lines.append("|---|---|---|")
        for lbl in ["easy", "medium", "hard"]:
            sub = df_fst[df_fst["fst_bin"] == lbl]
            y = (sub["recognized"].str.lower() == "yes").sum()
            lines.append(f"| {lbl} | {len(sub)} | {y} ({100*y/max(len(sub),1):.0f}%) |")
        lines.append("")

    return lines


def main(results_dir: Path):
    df_a = load_latest(results_dir, "probe_a")
    df_b = load_latest(results_dir, "probe_b")

    report_lines = [
        "# Experiment 0b — Contamination Probe Report\n",
        f"Forecaster: {df_a['run_id'].iloc[0] if len(df_a) else '?'}  \n",
        f"Experiment I (control) Brier for reference: **{EXP_I_CONTROL_BRIER}**  \n",
        f"Chance Brier: **{CHANCE_BRIER}**  \n",
        "---\n",
    ]

    control = load_control_p50()
    if control is None:
        print("WARNING: Exp I control CSV not found — skipping matched comparison.")
    report_lines += analyse_probe_a(df_a, control)
    report_lines += ["---\n"]
    report_lines += analyse_probe_b(df_b)

    report_lines += [
        "---\n",
        "## Overall conclusion\n",
        "See per-section verdicts above. The key question is whether Brier(recall) from "
        "Probe A is close to the Experiment I Brier (0.137). If it is, Exp I accuracy is "
        "potentially recall-driven. If Brier(recall) >> 0.137 (closer to chance 0.25), "
        "the Exp I signal is genuine reasoning from the provided source tasks.\n",
    ]

    report = "\n".join(report_lines)
    out = results_dir / "contamination_report.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path(__file__).parent / "results",
    )
    args = parser.parse_args()
    main(args.results_dir)
