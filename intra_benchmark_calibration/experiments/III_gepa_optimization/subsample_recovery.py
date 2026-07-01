#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Does a smaller cell count still recover our headline conclusions? ($0, offline)

Implements the original plan: treat the full-N Brier as the reference, then
repeatedly subsample N (< full) cells and ask whether the *qualitative* result
of each completed experiment survives:

  * Model sweep   : is GPT-5.5 still the best forecaster at N = 50, 100, ...?
  * Prompt ablation: is the best prompt still best; is the ranking preserved?

For each experiment and each N it reports, over many stratified subsamples:
  - P(top-1 preserved): the full-N winner is still ranked #1
  - mean Spearman      : rank-agreement of the whole ordering vs full-N
and draws the per-condition Brier *distributions* (the "BS_N histograms") so you
can see at which N they blur together.

Data: the committed scored CSVs under `forecasting_results/` (no API calls).

Usage:
    python intra_benchmark_calibration/experiments/III_gepa_optimization/subsample_recovery.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
FR = REPO_ROOT / "forecasting_results"
ARTIFACTS = HERE / "artifacts"

CELL_KEYS = ["forecasted_model", "target_task_id"]
BIN_COL = "target_bin"
N_BOOT = 2000
SEED = 0

DATASETS = {
    "model_sweep": {
        "dir": FR / "forecaster_model_comparison",
        "label_from": lambda stem: stem.replace("forecaster_", ""),
        "unit": "forecaster",
        "claim": "GPT-5.5 is the best forecaster",
    },
    "prompt_ablation": {
        "dir": FR / "prompt_variant_comparison",
        "label_from": lambda stem: stem.split("__")[-1],
        "unit": "prompt",
        "claim": "the full/least-ablated prompt ranks best",
    },
}


def load_dataset(d) -> Dict[str, pd.DataFrame]:
    out = {}
    for csv in sorted(Path(d["dir"]).glob("*.csv")):
        nm = d["label_from"](csv.stem)
        df = pd.read_csv(csv).dropna(subset=["p50", "outcome"]).copy()
        df["se"] = (df["p50"].astype(float) - df["outcome"].astype(int)) ** 2
        g = (df.groupby(CELL_KEYS, as_index=False)
                .agg(se=("se", "mean"), target_bin=(BIN_COL, "first")))
        out[nm] = g
    return out


def common_frame(conds: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for nm, g in conds.items():
        s = g.set_index(CELL_KEYS)[["se"]].rename(columns={"se": nm})
        merged = s if merged is None else merged.join(s, how="inner")
    bins = next(iter(conds.values())).set_index(CELL_KEYS)["target_bin"]
    return merged.join(bins, how="left")


def stratified_indices(bins: np.ndarray, n: int, rng) -> np.ndarray:
    idx_all = np.arange(len(bins))
    uniq, counts = np.unique(bins, return_counts=True)
    total = len(bins)
    picks = []
    for b, c in zip(uniq, counts):
        k = max(1, min(int(round(n * c / total)), c))
        picks.append(rng.choice(idx_all[bins == b], size=k, replace=False))
    return np.concatenate(picks)


def analyse(name: str, meta: dict):
    from scipy.stats import spearmanr
    conds = load_dataset(meta)
    wide = common_frame(conds)
    names = list(conds.keys())
    bins = wide["target_bin"].to_numpy()
    se = {nm: wide[nm].to_numpy() for nm in names}
    n_cells = len(wide)

    full = {nm: float(np.mean(se[nm])) for nm in names}
    order = sorted(names, key=lambda nm: full[nm])
    best = order[0]
    full_vec = np.array([{nm: i for i, nm in enumerate(order)}[nm] for nm in names], float)

    grid = [n for n in [30, 50, 75, 100, 150, 200, 250] if n <= n_cells]
    rng = np.random.default_rng(SEED)
    rows = []
    dist_at = {}  # N -> {cond: array of subsample Briers} for histograms
    for n in grid:
        top1 = 0
        rhos = np.empty(N_BOOT)
        briers = {nm: np.empty(N_BOOT) for nm in names}
        for b in range(N_BOOT):
            sel = stratified_indices(bins, n, rng)
            pb = {nm: float(np.mean(se[nm][sel])) for nm in names}
            for nm in names:
                briers[nm][b] = pb[nm]
            po = sorted(names, key=lambda nm: pb[nm])
            if po[0] == best:
                top1 += 1
            pv = np.array([{nm: i for i, nm in enumerate(po)}[nm] for nm in names], float)
            rhos[b] = spearmanr(full_vec, pv).correlation
        rows.append({"N": n, "p_top1_preserved": top1 / N_BOOT,
                     "mean_spearman": float(np.mean(rhos))})
        dist_at[n] = briers
    return {"name": name, "meta": meta, "names": names, "order": order, "best": best,
            "full": full, "n_cells": n_cells, "table": pd.DataFrame(rows),
            "dist_at": dist_at}


def plot_hist(res, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = sorted(res["dist_at"].keys())
    Ns = [grid[1] if len(grid) > 1 else grid[0], grid[min(3, len(grid) - 1)]]
    fig, axes = plt.subplots(1, len(Ns), figsize=(12, 4.8), sharex=True)
    if len(Ns) == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab10")
    order = res["order"]
    colors = {nm: cmap(i) for i, nm in enumerate(order)}
    for ax, n in zip(axes, Ns):
        d = res["dist_at"][n]
        for nm in order:
            ax.hist(d[nm], bins=40, alpha=0.5, color=colors[nm],
                    label=f"{nm} (full={res['full'][nm]:.3f})")
            ax.axvline(res["full"][nm], color=colors[nm], lw=1.5, ls="--")
        ax.set_title(f"N = {n} cells")
        ax.set_xlabel("Brier on the subsample")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"frequency over {N_BOOT} subsamples")
    axes[-1].legend(fontsize=7.5, loc="upper right")
    fig.suptitle(f"{res['name']}: per-{res['meta']['unit']} Brier distributions "
                 f"(dashed = full-{res['n_cells']} value)")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    results = {name: analyse(name, meta) for name, meta in DATASETS.items()}
    for name, res in results.items():
        plot_hist(res, ARTIFACTS / f"recovery_{name}_hist.png")

    L = ["# Subsample recovery — do smaller-N Briers preserve our conclusions?\n",
         "Reference = full-N Brier on the committed `forecasting_results/` data. For each N "
         "we draw 2000 stratified subsamples and check whether the experiment's headline "
         "(its best condition + overall ranking) survives. $0 — no new API calls.\n"]
    for name, res in results.items():
        meta = res["meta"]
        L.append(f"## {name} — claim: *{meta['claim']}*\n")
        L.append(f"- {len(res['names'])} {meta['unit']}s · matched cells: **{res['n_cells']}** "
                 f"· full-N winner: **`{res['best']}`** (Brier {res['full'][res['best']]:.4f})")
        L.append(f"- full-N ranking (best->worst): " +
                 ", ".join([f"`{nm}` {res['full'][nm]:.3f}" for nm in res["order"]]) + "\n")
        L.append("| N (cells) | P(full-N winner still #1) | mean Spearman (full ranking) |")
        L.append("|---|---|---|")
        for _, r in res["table"].iterrows():
            L.append(f"| {int(r['N'])} | {r['p_top1_preserved']*100:.0f}% | {r['mean_spearman']:.3f} |")
        L.append("")
        L.append(f"![{name} histograms](recovery_{name}_hist.png)\n")
    L.append("## Reading it\n")
    L.append("- **P(full-N winner still #1)** = how often the subsample agrees on the single "
             "best condition. This is the \"is GPT-5.5 still best?\" number.")
    L.append("- **mean Spearman** = how well the whole ordering is preserved (1.0 = identical).")
    L.append("- In the histograms, a condition is *reliably distinguishable* at a given N only "
             "when its Brier distribution barely overlaps its neighbours'. Conditions whose "
             "true Briers are within ~0.005 (e.g. Opus vs Sonnet) overlap heavily and cannot "
             "be separated at small N — but the clear winner/loser still separate.\n")
    (ARTIFACTS / "subsample_recovery.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    for name, res in results.items():
        print(f"\n=== {name} (winner={res['best']}, {res['n_cells']} cells) ===")
        print(res["table"].to_string(index=False))
    print(f"\nWrote {ARTIFACTS/'subsample_recovery.md'} + histogram PNGs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
