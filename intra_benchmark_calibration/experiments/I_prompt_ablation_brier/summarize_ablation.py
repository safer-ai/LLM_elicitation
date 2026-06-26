#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aggregate the Experiment I prompt ablation (Level-0 bracket) into a single
Brier-on-p50 report.

Conditions are auto-discovered under results/ (any subdir that contains a run
dir). `control` is the reference; every other condition is compared against it.
For each condition this:
  1. locates the latest run dir under results/<condition>/,
  2. computes Brier-on-p50 per repeat_index and the condition mean +/- sd,
  3. (optional) reports mean CRPS if a plots/scored_with_crps.csv exists,
  4. for each non-control condition vs control, computes the PAIRED per-cell
     Brier delta with a bootstrap 95% CI and a Wilcoxon signed-rank p-value
     (the real significance test — exploits the matched cell design),
  5. asserts each condition evaluated the SAME cells as control (same set of
     (forecasted_model, target_task_id, target_bin) keys) while its prompt
     actually differs (disjoint prompt_hash sets) — the controlled-variable
     guarantee,
  6. writes summary/ablation_brier.{md,csv} (+ per-repeat csv) and prints it.

Scoring helpers are imported from analyse_results.py so headline Brier matches
the rest of the pipeline (Brier-on-p50, last Delphi round only).

Usage (from anywhere):
    python intra_benchmark_calibration/experiments/I_prompt_ablation_brier/summarize_ablation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from intra_benchmark_calibration.analyse_results import (  # noqa: E402
    brier_on_p50,
    latest_run_dir,
    load_run,
    valid_rows,
)

REFERENCE = "control"
CELL_KEY_COLS = ["forecasted_model", "target_task_id", "target_bin"]
PREFERRED_ORDER = [
    "control",
    # scaffold decomposition (Level 0 + Level 1 scaffold)
    "no_ground_truth_summary",
    "trim_reasoning",
    "skip_analysis",
    "trim_all",
    # evidence ablations (Level 1 evidence)
    "no_bin_rate",
    "no_task_outcomes",
    "no_source_context",
    # floor
    "minimal",
]


def _discover_conditions(results_root: Path) -> List[str]:
    """Any subdir of results/ that contains at least one run dir."""
    found = []
    if not results_root.is_dir():
        return found
    for d in sorted(results_root.iterdir()):
        if not d.is_dir():
            continue
        if any(rd.is_dir() and rd.name[:4].isdigit() for rd in d.iterdir()):
            found.append(d.name)
    # Stable, readable order: preferred names first, then any extras.
    ordered = [c for c in PREFERRED_ORDER if c in found]
    ordered += [c for c in found if c not in PREFERRED_ORDER]
    return ordered


def _load_condition(condition: str, run_dir: Path) -> pd.DataFrame:
    df_full, _ = load_run(run_dir)
    df = valid_rows(df_full, last_round_only=True)
    if df.empty:
        raise ValueError(f"No valid (parsed-p50) rows for condition '{condition}' in {run_dir}")
    df = df.copy()
    df["condition"] = condition
    return df


def _maybe_crps_mean(run_dir: Path) -> Optional[float]:
    scored = run_dir / "plots" / "scored_with_crps.csv"
    if not scored.exists():
        return None
    sdf = pd.read_csv(scored)
    if "crps" not in sdf.columns:
        return None
    vals = sdf["crps"].dropna()
    return float(vals.mean()) if len(vals) else None


def _per_repeat_brier(df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
    rows: Dict[int, Dict[str, float]] = {}
    has_rep = "repeat_index" in df.columns
    reps = sorted(df["repeat_index"].dropna().unique()) if has_rep else [1]
    for rep in reps:
        sub = df[df["repeat_index"] == rep] if has_rep else df
        rows[int(rep)] = {"brier": brier_on_p50(sub), "n": int(len(sub))}
    return rows


def _per_cell_brier(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over repeats of (p50 - outcome)^2 per cell key. Returns a frame
    indexed by the cell key with a `cell_brier` column."""
    d = df.copy()
    d["se"] = (d["p50"].astype(float) - d["outcome"].astype(int)) ** 2
    g = d.groupby(CELL_KEY_COLS, dropna=False)["se"].mean().rename("cell_brier")
    return g.reset_index()


def _bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(diffs, dtype=float)
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _paired_vs_control(
    control_df: pd.DataFrame, cond_df: pd.DataFrame
) -> Dict[str, float]:
    """Paired per-cell Brier comparison: delta = brier_cond - brier_control.
    Positive delta => the condition is WORSE (higher Brier) than control."""
    c = _per_cell_brier(control_df).rename(columns={"cell_brier": "brier_control"})
    x = _per_cell_brier(cond_df).rename(columns={"cell_brier": "brier_cond"})
    merged = c.merge(x, on=CELL_KEY_COLS, how="inner")
    diffs = (merged["brier_cond"] - merged["brier_control"]).to_numpy()
    lo, hi = _bootstrap_ci(diffs)
    out = {
        "n_paired_cells": int(len(merged)),
        "mean_delta": float(np.mean(diffs)) if len(diffs) else float("nan"),
        "ci_lo": lo,
        "ci_hi": hi,
        "frac_cells_worse": float(np.mean(diffs > 0)) if len(diffs) else float("nan"),
    }
    try:
        from scipy.stats import wilcoxon
        nonzero = diffs[diffs != 0]
        if len(nonzero) >= 1:
            out["wilcoxon_p"] = float(wilcoxon(nonzero).pvalue)
    except Exception:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default=str(EXP_DIR / "results"),
                    help="Root containing per-condition subdirs (default: ./results)")
    args = ap.parse_args()
    results_root = Path(args.results_root).resolve()

    conditions = _discover_conditions(results_root)
    if not conditions:
        print(f"No conditions with run dirs found under {results_root}. Run a config first.")
        return 1

    loaded: Dict[str, pd.DataFrame] = {}
    run_dirs: Dict[str, Path] = {}
    crps: Dict[str, Optional[float]] = {}
    for cond in conditions:
        rd = latest_run_dir(results_root / cond)
        loaded[cond] = _load_condition(cond, rd)
        run_dirs[cond] = rd
        crps[cond] = _maybe_crps_mean(rd)
        print(f"[{cond}] run dir: {rd}  ({len(loaded[cond])} scored rows)")

    # ---- per-condition Brier summary -------------------------------------
    summary_records = []
    per_repeat_long = []
    for cond in conditions:
        df = loaded[cond]
        pr = _per_repeat_brier(df)
        briers = np.array([v["brier"] for v in pr.values()], dtype=float)
        rec = {
            "condition": cond,
            "n_repeats": len(pr),
            "n_cells_total": int(sum(v["n"] for v in pr.values())),
            "brier_mean": float(briers.mean()),
            "brier_sd": float(briers.std(ddof=1)) if len(briers) > 1 else 0.0,
            "brier_pooled": brier_on_p50(df),
            "crps_mean": crps[cond] if crps[cond] is not None else float("nan"),
        }
        summary_records.append(rec)
        for rep, v in pr.items():
            per_repeat_long.append({"condition": cond, "repeat_index": rep,
                                    "brier": v["brier"], "n": v["n"]})

    summary_df = pd.DataFrame(summary_records).set_index("condition")
    repeat_df = pd.DataFrame(per_repeat_long)

    # ---- provenance check: recorded label matches the folder -------------
    checks: List[Tuple[str, bool, str]] = []
    for cond in conditions:
        df = loaded[cond]
        if "experiment_label" in df.columns:
            labels = set(df["experiment_label"].dropna().astype(str).unique())
            ok = labels == {cond}
            checks.append((
                f"{cond}: run's recorded experiment_label matches its folder",
                ok,
                f"recorded={sorted(labels) or ['<none>']}, folder='{cond}'",
            ))

    # ---- paired comparisons vs control -----------------------------------
    paired: Dict[str, Dict[str, float]] = {}
    if REFERENCE in loaded:
        ctrl = loaded[REFERENCE]
        ctrl_keys = set(map(tuple, ctrl[CELL_KEY_COLS].itertuples(index=False, name=None)))
        ctrl_hashes = set(ctrl["prompt_hash"]) if "prompt_hash" in ctrl.columns else set()
        for cond in conditions:
            if cond == REFERENCE:
                continue
            x = loaded[cond]
            paired[cond] = _paired_vs_control(ctrl, x)
            x_keys = set(map(tuple, x[CELL_KEY_COLS].itertuples(index=False, name=None)))
            checks.append((f"{cond}: identical cell set vs control", x_keys == ctrl_keys,
                           f"control={len(ctrl_keys)}, {cond}={len(x_keys)}, shared={len(ctrl_keys & x_keys)}"))
            if ctrl_hashes and "prompt_hash" in x.columns:
                overlap = len(ctrl_hashes & set(x["prompt_hash"]))
                checks.append((f"{cond}: prompt differs from control (disjoint prompt_hash)",
                               overlap == 0, f"overlap={overlap} (expected 0)"))
    else:
        print(f"WARNING: reference condition '{REFERENCE}' not found; skipping paired deltas.")

    # ---- write outputs ----------------------------------------------------
    out_dir = EXP_DIR / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "ablation_brier.csv")
    repeat_df.to_csv(out_dir / "ablation_brier_per_repeat.csv", index=False)

    lines = ["# Experiment I — Level-0 prompt ablation: Brier-on-p50", ""]
    for cond in conditions:
        lines.append(f"- `{cond}` run dir: `{run_dirs[cond].name}`")
    lines += ["", "## Brier-on-p50 by condition", "",
              "| Condition | repeats | cells | Brier mean | Brier sd | Brier pooled | CRPS mean |",
              "|---|---|---|---|---|---|---|"]
    for cond, r in summary_df.iterrows():
        crps_str = "-" if pd.isna(r["crps_mean"]) else f"{r['crps_mean']:.4f}"
        lines.append(
            f"| {cond} | {int(r['n_repeats'])} | {int(r['n_cells_total'])} | "
            f"{r['brier_mean']:.4f} | {r['brier_sd']:.4f} | {r['brier_pooled']:.4f} | {crps_str} |"
        )

    if paired:
        lines += ["", "## Paired per-cell delta vs control",
                  "_delta = Brier(condition) − Brier(control), averaged over matched cells. "
                  "Negative = condition better (lower Brier) than control._", "",
                  "| Condition | paired cells | mean delta | 95% CI (bootstrap) | cells worse | Wilcoxon p |",
                  "|---|---|---|---|---|---|"]
        for cond, p in paired.items():
            wp = p.get("wilcoxon_p")
            wp_str = "-" if wp is None else f"{wp:.3g}"
            lines.append(
                f"| {cond} | {p['n_paired_cells']} | {p['mean_delta']:+.4f} | "
                f"[{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}] | {p['frac_cells_worse']*100:.0f}% | {wp_str} |"
            )
        lines += ["",
                  "_Interpretation: if a condition's 95% CI excludes 0, the prompt change has a "
                  "statistically reliable effect on accuracy across the 300 matched cells._"]

    lines += ["", "## Per-repeat Brier", "", "| Condition | repeat | Brier | n |",
              "|---|---|---|---|"]
    for _, r in repeat_df.iterrows():
        lines.append(f"| {r['condition']} | {int(r['repeat_index'])} | {r['brier']:.4f} | {int(r['n'])} |")

    if checks:
        lines += ["", "## Controlled-variable checks", ""]
        for name, ok, detail in checks:
            lines.append(f"- [{'PASS' if ok else 'FAIL'}] {name} — {detail}")

    (out_dir / "ablation_brier.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote: {out_dir / 'ablation_brier.md'}")
    print(f"Wrote: {out_dir / 'ablation_brier.csv'}")

    if checks and not all(ok for _, ok, _ in checks):
        print("\nWARNING: a controlled-variable check FAILED — inspect before trusting the deltas.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
