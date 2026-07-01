#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GEPA Gate B — proxy-ranking validation ($0, offline).

GEPA scores hundreds of candidate prompts; scoring each on the full 300-cell
grand Brier is too expensive. The plan is to score candidates on a cheap
stratified *subsample* of cells (the "proxy"). That only works if the proxy
ranks prompts the SAME way the full metric does. This script tests exactly that,
using the 10 already-scored prompt variants in
`forecasting_results/prompt_variant_comparison/` — no new API calls.

Crucially, every prompt is scored on the SAME subsampled cells within a draw
(paired mini-batch), exactly as GEPA compares a candidate against its parent —
so the shared task-difficulty noise cancels in the A-vs-B comparison.

Method
------
1. Load the 10 scored prompt variants. Per condition, reduce to one loss value
   per cell — either squared error  se = (p50 - outcome)^2  (Brier) or the
   pre-computed CRPS column (averaged over repeats if present).
2. Restrict to the cells present in ALL conditions (paired design).
3. "Truth" = each condition's full-N mean loss on those common cells -> ranking.
4. For each proxy size N in a grid: draw B stratified subsamples (proportional
   across the 5 difficulty bins), and per draw measure
     - the 10-way rank agreement vs the full ranking (Spearman),
     - whether the proxy's best == the true best,
     - |proxy metric - full metric| (absolute closeness), and
     - pairwise sign-agreement for every prompt pair, bucketed by true gap.

Output
------
`artifacts/proxy_check_{metric}.{md,png}`

Usage
-----
    python intra_benchmark_calibration/experiments/III_gepa_optimization/proxy_check.py
    python intra_benchmark_calibration/experiments/III_gepa_optimization/proxy_check.py --metric crps
    # options: --n-grid 30,50,75,100,150,200,250,300  --n-boot 2000
    #          --threshold 0.8  --seed 0  --metric {brier,crps}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "forecasting_results" / "prompt_variant_comparison"
ARTIFACTS = HERE / "artifacts"

CELL_KEYS = ["forecasted_model", "target_task_id"]
BIN_COL = "target_bin"


def load_conditions(metric: str) -> Dict[str, pd.DataFrame]:
    """Return {condition_name: per-cell frame with [cell keys, target_bin, loss]}.

    metric="brier" computes squared error from p50; metric="crps" uses the
    pre-computed crps column already present in the scored CSVs.
    """
    if metric == "brier":
        need_cols = set(CELL_KEYS + [BIN_COL, "p50", "outcome"])
    else:
        need_cols = set(CELL_KEYS + [BIN_COL, "crps"])

    out: Dict[str, pd.DataFrame] = {}
    for csv in sorted(DATA_DIR.glob("*.csv")):
        name = csv.stem.split("__")[-1]
        df = pd.read_csv(csv)
        if not need_cols.issubset(df.columns):
            raise ValueError(f"{csv.name} missing columns: {need_cols - set(df.columns)}")
        if metric == "brier":
            df = df.dropna(subset=["p50", "outcome"]).copy()
            df["loss"] = (df["p50"].astype(float) - df["outcome"].astype(int)) ** 2
        else:
            df = df.dropna(subset=["crps"]).copy()
            df["loss"] = df["crps"].astype(float)
        g = (df.groupby(CELL_KEYS, as_index=False)
                .agg(loss=("loss", "mean"), target_bin=(BIN_COL, "first")))
        out[name] = g
    if not out:
        raise SystemExit(f"No CSVs found in {DATA_DIR}")
    return out


def common_cells(conds: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide frame indexed by cell key: one loss column per condition + target_bin.
    Restricted to cells present in every condition (paired)."""
    merged = None
    for name, g in conds.items():
        s = g.set_index(CELL_KEYS)[["loss"]].rename(columns={"loss": name})
        merged = s if merged is None else merged.join(s, how="inner")
    bins = next(iter(conds.values())).set_index(CELL_KEYS)["target_bin"]
    merged = merged.join(bins, how="left")
    return merged


def stratified_indices(bins: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ~n row indices, allocated proportionally across bins."""
    idx_all = np.arange(len(bins))
    picks: List[np.ndarray] = []
    uniq, counts = np.unique(bins, return_counts=True)
    total = len(bins)
    for b, c in zip(uniq, counts):
        k = int(round(n * c / total))
        k = max(1, min(k, c))
        rows = idx_all[bins == b]
        picks.append(rng.choice(rows, size=k, replace=False))
    return np.concatenate(picks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metric", choices=["brier", "crps"], default="brier",
                    help="loss metric to use for ranking (default: brier)")
    ap.add_argument("--n-grid", default="30,50,75,100,150,200,250,300")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n_grid = [int(x) for x in args.n_grid.split(",")]
    conds = load_conditions(args.metric)
    wide = common_cells(conds)
    names = list(conds.keys())
    bins = wide["target_bin"].to_numpy()
    loss = {nm: wide[nm].to_numpy() for nm in names}
    n_cells = len(wide)

    # full ("true") ranking on the common cells
    full_loss = {nm: float(np.mean(loss[nm])) for nm in names}
    full_order = sorted(names, key=lambda nm: full_loss[nm])
    full_rank = {nm: i for i, nm in enumerate(full_order)}
    full_vec = np.array([full_rank[nm] for nm in names], dtype=float)
    best_full = full_order[0]

    from scipy.stats import spearmanr

    # Pairs and their TRUE loss gap — the GEPA-relevant unit is a pairwise
    # (parent vs child) decision, not a full 10-way ranking.
    pairs = [(a, c) for i, a in enumerate(names) for c in names[i + 1:]]
    gap_thresholds = [0.005, 0.010]
    # Use zero-padded string keys to avoid float formatting ambiguity (0.010 -> "0.010")
    gt_keys = {gt: f"{gt:.3f}" for gt in gap_thresholds}

    rng = np.random.default_rng(args.seed)
    rows = []
    for n in n_grid:
        if n > n_cells:
            continue
        rhos = np.empty(args.n_boot)
        top1 = 0
        pair_true_gap = {p: full_loss[p[1]] - full_loss[p[0]] for p in pairs}
        pair_correct = {p: 0 for p in pairs}
        abs_errs: List[float] = []
        for b in range(args.n_boot):
            sel = stratified_indices(bins, n, rng)
            proxy_loss = {nm: float(np.mean(loss[nm][sel])) for nm in names}
            proxy_order = sorted(names, key=lambda nm: proxy_loss[nm])
            proxy_rank = {nm: i for i, nm in enumerate(proxy_order)}
            proxy_vec = np.array([proxy_rank[nm] for nm in names], dtype=float)
            rhos[b] = spearmanr(full_vec, proxy_vec).correlation
            if proxy_order[0] == best_full:
                top1 += 1
            for p in pairs:
                proxy_gap = proxy_loss[p[1]] - proxy_loss[p[0]]
                if np.sign(proxy_gap) == np.sign(pair_true_gap[p]) and pair_true_gap[p] != 0:
                    pair_correct[p] += 1
            for nm in names:
                abs_errs.append(abs(proxy_loss[nm] - full_loss[nm]))
        abs_errs_arr = np.asarray(abs_errs)
        row = {
            "N_target": n,
            "mean_spearman": float(np.mean(rhos)),
            "p_spearman_ge_thr": float(np.mean(rhos >= args.threshold)),
            "top1_hit_rate": top1 / args.n_boot,
            "metric_abs_err_mean": float(np.mean(abs_errs_arr)),
            "metric_abs_err_p90": float(np.percentile(abs_errs_arr, 90)),
            "p_within_0.01": float(np.mean(abs_errs_arr <= 0.01)),
            "p_within_0.02": float(np.mean(abs_errs_arr <= 0.02)),
        }
        for gt in gap_thresholds:
            sel_pairs = [p for p in pairs if abs(pair_true_gap[p]) >= gt]
            acc = (np.mean([pair_correct[p] / args.n_boot for p in sel_pairs])
                   if sel_pairs else float("nan"))
            row[f"pair_acc_gap_ge_{gt_keys[gt]}"] = acc
            row[f"n_pairs_gap_ge_{gt_keys[gt]}"] = len(sel_pairs)
        rows.append(row)

    res = pd.DataFrame(rows)
    res.attrs["gap_thresholds"] = gap_thresholds
    res.attrs["gt_keys"] = gt_keys
    res.attrs["metric"] = args.metric
    passing = res[res["mean_spearman"] >= args.threshold]
    min_pass = int(passing["N_target"].min()) if len(passing) else None
    pair_col = f"pair_acc_gap_ge_{gt_keys[0.010]}"
    pair_pass = res[res[pair_col] >= 0.90] if pair_col in res.columns else res.iloc[0:0]
    min_pair_pass = int(pair_pass["N_target"].min()) if len(pair_pass) else None

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    stem = f"proxy_check_{args.metric}"
    _plot(res, args.threshold, min_pass, args.metric, ARTIFACTS / f"{stem}.png")
    _write_md(res, conds, full_loss, full_order, n_cells, args, min_pass,
              min_pair_pass, ARTIFACTS / f"{stem}.md")

    print(res.to_string(index=False))
    print(f"\nFull-{args.metric.upper()} ranking (best->worst): {full_order}")
    print(f"Strict 10-way rank: min N for mean Spearman >= {args.threshold} = {min_pass}")
    print(f"Pairwise decisions on real gaps (>=0.01): min N for >=90% correct = {min_pair_pass}")
    print(f"Wrote {ARTIFACTS / stem}.md and {stem}.png")
    return 0


def _plot(res, thr, min_pass, metric: str, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gt_keys = res.attrs.get("gt_keys", {0.005: "0.005", 0.010: "0.010"})
    col_010 = f"pair_acc_gap_ge_{gt_keys.get(0.010, '0.010')}"
    col_005 = f"pair_acc_gap_ge_{gt_keys.get(0.005, '0.005')}"
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axhline(0.9, ls="--", color="gray", alpha=0.7, label="90% decision accuracy")
    if col_010 in res.columns:
        ax.plot(res["N_target"], res[col_010], "-o", color="#2e7d32",
                label=f"pairwise decision acc (true gap >= 0.010) — GEPA-relevant")
    if col_005 in res.columns:
        ax.plot(res["N_target"], res[col_005], "-^", color="#80b46a",
                label=f"pairwise decision acc (true gap >= 0.005)")
    ax.plot(res["N_target"], res["mean_spearman"], "-s", color="#1f5fb4",
            label="mean Spearman, strict 10-way rank (pessimistic)")
    ax.plot(res["N_target"], res["top1_hit_rate"], ":d", color="#d9831f",
            label="P(proxy picks true best of 10, incl. ties)")
    ax.set_xlabel("proxy size N (cells per candidate)")
    ax.set_ylabel(f"agreement with full-N {metric.upper()}")
    ax.set_title(f"GEPA Gate B — proxy reliability vs. cells per candidate ({metric.upper()})")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")


def _write_md(res, conds, full_loss, full_order, n_cells, args, min_pass,
              min_pair_pass, out: Path):
    metric = args.metric
    M = metric.upper()
    gts = res.attrs.get("gap_thresholds", [0.005, 0.010])
    gt_keys = res.attrs.get("gt_keys", {gt: f"{gt:.3f}" for gt in gts})

    L = []
    L.append(f"# GEPA Gate B — proxy-ranking validation ({M})\n")
    L.append(f"**Question:** can a cheap stratified subsample of cells stand in for the full-N "
             f"{M} when comparing prompts? **$0** — reuses the 10 already-scored prompt variants "
             f"in `forecasting_results/prompt_variant_comparison/`. Every prompt is scored on the "
             f"SAME subsampled cells per draw (paired), exactly as GEPA compares a candidate to "
             f"its parent.\n")
    L.append(f"- Conditions tested: **{len(conds)}** prompt variants · common paired cells: "
             f"**{n_cells}** · bootstrap draws per N: {args.n_boot} (stratified by difficulty bin)\n")

    sp = f"N = {min_pass}" if min_pass else "none of the tested N"
    pp = f"N = {min_pair_pass}" if min_pair_pass else "none of the tested N"
    L.append("## TL;DR\n")
    L.append(f"- **Strict 10-way ranking** (incl. tied prompts): needs **{sp}** for mean "
             f"Spearman >= {args.threshold}. Pessimistic — penalizes shuffling of tied prompts.")
    L.append(f"- **Pairwise decision on a real gap (>=0.01 {M}):** >=90% correct already at "
             f"**{pp}** — this is the GEPA-relevant number (parent-vs-child comparisons).\n")

    L.append("## Results by proxy size\n")
    head = f"| Proxy N | mean Spearman (10-way) | P(picks true best) |"
    sep = "|---|---|---|"
    for gt in gts:
        head += f" pairwise acc, gap>={gt_keys[gt]} |"
        sep += "---|"
    L.append(head)
    L.append(sep)
    for _, r in res.iterrows():
        line = (f"| {int(r['N_target'])} | {r['mean_spearman']:.3f} | "
                f"{r['top1_hit_rate']*100:.0f}% |")
        for gt in gts:
            line += f" {r[f'pair_acc_gap_ge_{gt_keys[gt]}']*100:.0f}% |"
        L.append(line)
    L.append("")
    npairs = {gt: int(res.iloc[0][f"n_pairs_gap_ge_{gt_keys[gt]}"]) for gt in gts}
    L.append(f"_\"pairwise acc, gap>=g\" = among prompt pairs whose **true** {M} differs by "
             "at least g, how often the proxy gets the better one right "
             + ", ".join([f"({int(npairs[gt])} pairs at g={gt_keys[gt]})" for gt in gts]) + "._\n")

    L.append(f"## Absolute closeness: is the proxy {M} value near the full-N {M}?\n")
    L.append(f"| Proxy N | avg |error| in {M} | within +/-0.01 | within +/-0.02 |")
    L.append("|---|---|---|---|")
    for _, r in res.iterrows():
        L.append(f"| {int(r['N_target'])} | {r['metric_abs_err_mean']:.4f} | "
                 f"{r['p_within_0.01']*100:.0f}% | {r['p_within_0.02']*100:.0f}% |")
    L.append("")

    L.append("## Recommendation for GEPA\n")
    L.append(f"- **Use the proxy only for coarse filtering** (keep/kill on gaps >= ~0.01 {M}). "
             "Don't trust it to split hairs between near-tied candidates.")
    L.append("- **Re-validate surviving Pareto finalists on the full cell set** before declaring "
             "a winner.")
    L.append("- Prefer a **proxy of ~100–150 cells** over the 50 originally assumed.\n")

    L.append(f"## Reference: full-{M} ranking on the common cells (best -> worst)\n")
    L.append(f"| Rank | Condition | full {M} |")
    L.append("|---|---|---|")
    for i, nm in enumerate(full_order, 1):
        L.append(f"| {i} | `{nm}` | {full_loss[nm]:.4f} |")
    L.append("")
    L.append("> Caveat: validates ranking on 10 hand-built prompts only. GEPA's candidates may "
             "cluster even more tightly — treat the proxy as a coarse filter and always confirm "
             "finalists on the full set.\n")

    out.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
