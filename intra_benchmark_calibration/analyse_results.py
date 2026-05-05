#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Post-hoc scoring and plotting for an intra-benchmark calibration run.

Given the JSON+CSV produced by `run_calibration.py`, computes:

  - Headline: Brier on p50 across all valid elicitations.
  - MAE / RMSE between p50 and binary outcome.
  - Mean bias (p50 - outcome).
  - Calibration error (Expected Calibration Error, 10-bin equal-width).
  - Cross-cell breakdowns: Brier per (source_bin, target_bin), Brier per
    forecasted_model.
  - V2 stub: CRPS by fitting a Beta distribution to (p25, p50, p75) per
    elicitation and integrating the squared CDF error against the binary
    outcome.

Plots written to `<run_dir>/plots/`:

  1. calibration_scatter.png        — p50 vs outcome jittered, colour by source_bin
  2. brier_heatmap.png              — source_bin x target_bin heatmap of mean Brier
  3. per_model_brier.png            — bar chart of Brier per forecasted_model
  4. reliability_diagram.png        — 10-bin reliability with empirical pass fraction
  5. metr_style_logfst.png          — empirical pass rate vs forecaster mean p50,
                                      binned by log10(target_fst_minutes), with
                                      bootstrap CIs on the empirical
  6. expert_distributions.png       — violin of (p25, p50, p75) per cell
                                      (only if Delphi rounds > 1, otherwise points)

Usage:
    python intra_benchmark_calibration/analyse_results.py \\
        -r intra_benchmark_calibration/results/<run_id>
    # or auto-pick the most recent run:
    python intra_benchmark_calibration/analyse_results.py --latest
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist
from scipy.stats import kendalltau, spearmanr

logger = logging.getLogger(__name__)


# ============================================================================
# I/O
# ============================================================================

def find_run_files(run_dir: Path) -> Tuple[Path, Path]:
    """Find the (csv_path, json_path) inside a run directory."""
    csvs = sorted(run_dir.glob("*_estimates.csv"))
    jsons = sorted(run_dir.glob("*_results.json"))
    if not csvs or not jsons:
        raise FileNotFoundError(f"Expected one *_estimates.csv and one *_results.json in {run_dir}")
    if len(csvs) > 1 or len(jsons) > 1:
        logger.warning(f"Multiple result files in {run_dir}; using newest of each.")
    return csvs[-1], jsons[-1]


def load_run(run_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    csv_path, json_path = find_run_files(run_dir)
    df = pd.read_csv(csv_path)
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info(f"Loaded {len(df)} CSV rows from {csv_path.name}")
    logger.info(f"Loaded {len(data.get('elicitations', []))} elicitation records from {json_path.name}")
    return df, data


def latest_run_dir(results_root: Path) -> Path:
    candidates = [d for d in results_root.iterdir() if d.is_dir() and d.name[:4].isdigit()]
    if not candidates:
        raise FileNotFoundError(f"No run dirs found under {results_root}")
    return max(candidates, key=lambda p: p.name)


# ============================================================================
# Filtering / shaping
# ============================================================================

def valid_rows(df: pd.DataFrame, *, last_round_only: bool = True) -> pd.DataFrame:
    """
    Keep only rows where p50 was successfully parsed. Optionally restrict to
    each cell's final Delphi round (default — matches headline reporting).

    With multi-forecaster / multi-repeat runs, "each cell" means each
    (condition_id, forecaster_model, repeat_index, expert_id) group: the same
    cell can appear under several forecasters and repeats, and each must keep
    its own last-round row. Falls back to grouping by `condition_id` only if
    those columns are absent (legacy CSVs).
    """
    out = df.dropna(subset=["p50"]).copy()
    if last_round_only:
        group_keys = ["condition_id"]
        for k in ("forecaster_model", "repeat_index", "expert_id"):
            if k in out.columns:
                group_keys.append(k)
        idx = out.groupby(group_keys, dropna=False)["delphi_round"].idxmax()
        out = out.loc[idx].reset_index(drop=True)
    return out


def filter_rows(
    df: pd.DataFrame,
    *,
    forecaster_model: Optional[str] = None,
    repeat_index: Optional[int] = None,
) -> pd.DataFrame:
    """Apply optional `--forecaster-model` / `--repeat-index` CLI filters.

    Returns the unfiltered frame when both filters are None or when the
    relevant columns are missing (legacy CSVs).
    """
    out = df
    if forecaster_model is not None and "forecaster_model" in out.columns:
        before = len(out)
        out = out[out["forecaster_model"] == forecaster_model]
        logger.info(
            f"--forecaster-model filter: {forecaster_model!r} kept {len(out)}/{before} rows"
        )
    if repeat_index is not None and "repeat_index" in out.columns:
        before = len(out)
        out = out[out["repeat_index"].astype("Int64") == int(repeat_index)]
        logger.info(
            f"--repeat-index filter: {repeat_index} kept {len(out)}/{before} rows"
        )
    return out


# ============================================================================
# Scoring
# ============================================================================

def brier_on_p50(df: pd.DataFrame) -> float:
    return float(np.mean((df["p50"].astype(float) - df["outcome"].astype(int)) ** 2))


def basic_stats(df: pd.DataFrame) -> Dict[str, float]:
    p50 = df["p50"].astype(float).values
    o = df["outcome"].astype(int).values
    if len(p50) < 2:
        return {"n": int(len(p50))}
    sp_r, sp_p = spearmanr(p50, o)
    kt_r, kt_p = kendalltau(p50, o)
    return {
        "n": int(len(p50)),
        "brier_p50": float(np.mean((p50 - o) ** 2)),
        "mae": float(np.mean(np.abs(p50 - o))),
        "rmse": float(np.sqrt(np.mean((p50 - o) ** 2))),
        "bias": float(np.mean(p50 - o)),
        "spearman_r": float(sp_r),
        "spearman_p": float(sp_p),
        "kendall_tau": float(kt_r),
        "kendall_p": float(kt_p),
    }


def expected_calibration_error(df: pd.DataFrame, n_bins: int = 10) -> Dict[str, float]:
    """
    Standard ECE: bin elicitations by p50 into n equal-width bins, compute
    |mean_p50 - empirical_pass_rate| per bin, weight by bin population.
    """
    p50 = df["p50"].astype(float).values
    o = df["outcome"].astype(int).values
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p50, edges) - 1, 0, n_bins - 1)
    total = len(p50)
    ece = 0.0
    bin_data: List[Dict[str, float]] = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        mean_p50 = float(np.mean(p50[mask]))
        emp = float(np.mean(o[mask]))
        ece += (n / total) * abs(mean_p50 - emp)
        bin_data.append({"bin": b, "edge_lo": float(edges[b]), "edge_hi": float(edges[b + 1]),
                         "n": n, "mean_p50": mean_p50, "empirical_pass": emp})
    return {"ece": float(ece), "bins": bin_data, "n_bins": n_bins}


# ============================================================================
# CRPS via Beta-fit (V2 STUB)
# ============================================================================

def fit_beta_to_percentiles(p25: float, p50: float, p75: float,
                            init: Tuple[float, float] = (2.0, 2.0)) -> Optional[Tuple[float, float]]:
    """
    Fit a Beta(alpha, beta) by minimising squared error between target
    percentiles (0.25, 0.50, 0.75) and the candidate Beta CDF inverse at the
    same probabilities.

    Returns (alpha, beta) or None if the fit failed / inputs are degenerate.
    """
    targets = np.array([p25, p50, p75], dtype=float)
    if np.any(np.isnan(targets)) or not np.all((0.0 < targets) & (targets < 1.0)):
        return None
    if not (targets[0] <= targets[1] <= targets[2]):
        # tolerate small parsing noise
        targets = np.sort(targets)

    def loss(params: np.ndarray) -> float:
        a, b = params
        if a <= 0 or b <= 0:
            return 1e9
        try:
            preds = beta_dist.ppf([0.25, 0.50, 0.75], a, b)
        except Exception:
            return 1e9
        if np.any(np.isnan(preds)):
            return 1e9
        return float(np.sum((preds - targets) ** 2))

    res = minimize(loss, x0=np.array(init), method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 500})
    if not res.success:
        return None
    a, b = float(res.x[0]), float(res.x[1])
    if a <= 0 or b <= 0:
        return None
    return a, b


def crps_beta(alpha: float, beta_param: float, outcome: int) -> float:
    """
    CRPS for a Beta(alpha, beta) forecast against a binary outcome in {0, 1}:
        outcome=1: CRPS = integral_0^1 F(y)^2 dy
        outcome=0: CRPS = integral_0^1 (1 - F(y))^2 dy
    """
    if outcome == 1:
        integrand = lambda y: beta_dist.cdf(y, alpha, beta_param) ** 2
    else:
        integrand = lambda y: (1.0 - beta_dist.cdf(y, alpha, beta_param)) ** 2
    val, _err = quad(integrand, 0.0, 1.0, limit=100)
    return float(val)


def compute_crps_per_row(df: pd.DataFrame) -> pd.DataFrame:
    """V2 stub: add `beta_alpha`, `beta_beta`, `crps` columns where fittable."""
    out = df.copy()
    alphas: List[Optional[float]] = []
    betas: List[Optional[float]] = []
    crpses: List[Optional[float]] = []
    n_fit_failed = 0
    for _, r in out.iterrows():
        try:
            p25, p50, p75 = float(r["p25"]), float(r["p50"]), float(r["p75"])
        except (TypeError, ValueError):
            alphas.append(None); betas.append(None); crpses.append(None)
            continue
        fit = fit_beta_to_percentiles(p25, p50, p75)
        if fit is None:
            alphas.append(None); betas.append(None); crpses.append(None)
            n_fit_failed += 1
            continue
        a, b = fit
        crps = crps_beta(a, b, int(r["outcome"]))
        alphas.append(a); betas.append(b); crpses.append(crps)

    out["beta_alpha"] = alphas
    out["beta_beta"] = betas
    out["crps"] = crpses
    if n_fit_failed:
        logger.info(f"CRPS: Beta fit failed for {n_fit_failed}/{len(out)} rows (degenerate p25/p50/p75).")
    return out


# ============================================================================
# Plotting
# ============================================================================

def _save(fig: plt.Figure, path: Optional[Path], show: bool) -> None:
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved {path.name}")
    if show:
        plt.show()
    plt.close(fig)


def _has_meaningful_source_bin(df: pd.DataFrame) -> bool:
    """True iff source_bin column has at least one non-NaN value
    (false when the run is entirely all_except_target mode)."""
    return "source_bin" in df.columns and df["source_bin"].notna().any()


def plot_calibration_scatter(df: pd.DataFrame, out: Optional[Path], show: bool) -> None:
    """p50 vs outcome jitter plot. Coloured by source_bin if available, else by target_bin."""
    fig, ax = plt.subplots(figsize=(8, 6))
    rng = np.random.default_rng(0)
    yj = df["outcome"].astype(int).values + rng.uniform(-0.04, 0.04, size=len(df))
    if _has_meaningful_source_bin(df):
        c, label = df["source_bin"].fillna(-1), "source_bin"
    else:
        c, label = df["target_bin"], "target_bin (all_except_target mode)"
    sc = ax.scatter(df["p50"], yj, c=c, cmap="viridis",
                    s=80, alpha=0.75, edgecolor="black", linewidth=0.6)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="perfect (y=p)")
    plt.colorbar(sc, ax=ax, label=label)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.15, 1.15)
    ax.set_xlabel("Forecaster p50 (median predicted P(solve))")
    ax.set_ylabel("True binary outcome (jittered)")
    types = df["source_profile_type"].unique() if "source_profile_type" in df.columns else ["?"]
    ax.set_title(
        f"Calibration scatter — n={len(df)}, Brier-on-p50={brier_on_p50(df):.3f}, "
        f"mode={'/'.join(map(str, types))}"
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="center right")
    _save(fig, out, show)


def plot_brier_heatmap(df: pd.DataFrame, out: Optional[Path], show: bool) -> None:
    """
    Mean Brier per (source_bin, target_bin) cell when source_bin is meaningful;
    otherwise (all_except_target mode), per target_bin only as a 1-D bar chart.
    """
    df = df.assign(se=(df["p50"].astype(float) - df["outcome"].astype(int)) ** 2)

    if not _has_meaningful_source_bin(df):
        # All_except_target mode -> reduce to a 1-D plot over target_bin.
        grouped = df.groupby("target_bin").agg(brier=("se", "mean"), n=("se", "size"))
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(grouped.index.astype(str), grouped["brier"], color="steelblue", edgecolor="black")
        for bar, n in zip(bars, grouped["n"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"n={n}", ha="center", fontsize=9)
        ax.axhline(0.25, color="grey", linestyle="--", alpha=0.6, label="chance Brier (0.25)")
        ax.set_ylim(0, max(0.5, grouped["brier"].max() * 1.15))
        ax.set_xlabel("target_bin (j)")
        ax.set_ylabel("Brier on p50")
        ax.set_title("Brier-on-p50 per target_bin (all_except_target mode — i is collapsed)")
        ax.legend()
        _save(fig, out, show)
        return

    pivot = df.pivot_table(index="source_bin", columns="target_bin",
                            values="se", aggfunc="mean")
    counts = df.pivot_table(index="source_bin", columns="target_bin",
                             values="se", aggfunc="size")
    annot = pivot.copy().astype(object)
    for r in pivot.index:
        for c in pivot.columns:
            v = pivot.loc[r, c]
            n = counts.loc[r, c] if not pd.isna(counts.loc[r, c]) else 0
            annot.loc[r, c] = "" if pd.isna(v) else f"{v:.3f}\n(n={int(n)})"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=annot, fmt="", cmap="RdYlGn_r", vmin=0, vmax=1,
                cbar_kws={"label": "Mean Brier on p50 (lower = better)"},
                ax=ax, linewidths=1, linecolor="gray")
    ax.set_title("Brier-on-p50 per (source_bin, target_bin)")
    ax.set_xlabel("target_bin (j)")
    ax.set_ylabel("source_bin (i)")
    _save(fig, out, show)


def plot_per_model_brier(df: pd.DataFrame, out: Optional[Path], show: bool) -> None:
    """Bar chart of Brier per forecasted_model."""
    df = df.assign(se=(df["p50"].astype(float) - df["outcome"].astype(int)) ** 2)
    grouped = df.groupby("forecasted_model").agg(brier=("se", "mean"), n=("se", "size"))
    grouped = grouped.sort_values("brier")

    fig, ax = plt.subplots(figsize=(max(7, len(grouped) * 0.8), 5))
    bars = ax.bar(grouped.index, grouped["brier"], color="steelblue", edgecolor="black")
    for bar, n in zip(bars, grouped["n"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"n={n}", ha="center", fontsize=9)
    ax.axhline(0.25, color="grey", linestyle="--", alpha=0.6, label="chance Brier (0.25)")
    ax.set_ylim(0, max(0.5, grouped["brier"].max() * 1.15))
    ax.set_ylabel("Brier on p50 (lower = better)")
    ax.set_xlabel("Forecasted model")
    ax.set_title("Per-model calibration (Brier-on-p50)")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    plt.setp(ax.get_xticklabels(), ha="right")
    _save(fig, out, show)


def plot_per_forecaster_model_brier(df: pd.DataFrame, out: Optional[Path], show: bool) -> None:
    """Bar chart of Brier per `forecaster_model` (the LLM doing the forecasting).

    No-op when the column is absent (legacy CSV) or there is only one
    forecaster — the headline `per_model_brier.png` already covers that case.
    """
    if "forecaster_model" not in df.columns:
        return
    forecasters = sorted(df["forecaster_model"].dropna().unique().tolist())
    if len(forecasters) < 2:
        logger.info(
            f"Skipping per-forecaster-model Brier plot: only "
            f"{len(forecasters)} forecaster(s) present."
        )
        return

    df = df.assign(se=(df["p50"].astype(float) - df["outcome"].astype(int)) ** 2)
    grouped = df.groupby("forecaster_model").agg(brier=("se", "mean"), n=("se", "size"))
    grouped = grouped.sort_values("brier")

    fig, ax = plt.subplots(figsize=(max(7, len(grouped) * 0.9), 5))
    bars = ax.bar(grouped.index, grouped["brier"], color="darkorange", edgecolor="black")
    for bar, n in zip(bars, grouped["n"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"n={n}", ha="center", fontsize=9)
    ax.axhline(0.25, color="grey", linestyle="--", alpha=0.6, label="chance Brier (0.25)")
    ax.set_ylim(0, max(0.5, grouped["brier"].max() * 1.15))
    ax.set_ylabel("Brier on p50 (lower = better)")
    ax.set_xlabel("Forecaster model (LLM running the elicitation)")
    ax.set_title("Per-forecaster calibration (Brier-on-p50)")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    plt.setp(ax.get_xticklabels(), ha="right")
    _save(fig, out, show)


def plot_reliability_diagram(df: pd.DataFrame, out: Optional[Path], show: bool,
                              n_bins: int = 10) -> None:
    """Reliability: bin by p50, plot empirical pass fraction per bin."""
    p50 = df["p50"].astype(float).values
    o = df["outcome"].astype(int).values
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p50, edges) - 1, 0, n_bins - 1)

    centres, emp_rates, sample_counts, mean_p50_per_bin = [], [], [], []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        centres.append(0.5 * (edges[b] + edges[b + 1]))
        emp_rates.append(float(np.mean(o[mask])))
        sample_counts.append(int(mask.sum()))
        mean_p50_per_bin.append(float(np.mean(p50[mask])))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect calibration")
    sizes = [200 + 30 * c for c in sample_counts]
    ax.scatter(mean_p50_per_bin, emp_rates, s=sizes, alpha=0.75,
               edgecolor="black", color="darkorange",
               label="reliability (size ∝ n in bin)")
    for x, y, n in zip(mean_p50_per_bin, emp_rates, sample_counts):
        ax.annotate(f"n={n}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean forecaster p50 in bin")
    ax.set_ylabel("Empirical pass fraction in bin")
    ax.set_title(f"Reliability diagram (p50 binned in {n_bins} equal-width bins)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left")
    _save(fig, out, show)


def plot_metr_style_logfst(df: pd.DataFrame, out: Optional[Path], show: bool,
                            n_bins: int = 6, n_bootstrap: int = 1000,
                            seed: int = 7) -> None:
    """
    METR-style: bin target tasks post hoc by log10(FST), plot empirical pass
    rate per FST-bin (with 95% bootstrap CI) vs mean forecaster p50 per FST-bin.
    Bins here are a PLOTTING choice — independent of the experiment's source_bin
    / target_bin partitioning.
    """
    df = df.copy()
    df = df.dropna(subset=["target_fst_minutes", "p50"])
    if len(df) < 4:
        logger.warning(f"Not enough rows ({len(df)}) for METR-style plot; skipping.")
        return

    fst = df["target_fst_minutes"].astype(float).values
    log_fst = np.log10(fst)
    edges = np.quantile(log_fst, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-6; edges[-1] += 1e-6
    idx = np.clip(np.digitize(log_fst, edges) - 1, 0, n_bins - 1)

    rng = np.random.default_rng(seed)
    centres, emp, emp_lo, emp_hi, mean_p50, ns = [], [], [], [], [], []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        o = df["outcome"].values[mask].astype(int)
        p = df["p50"].values[mask].astype(float)
        emp_b = float(np.mean(o))
        boot = rng.choice(o, size=(n_bootstrap, len(o)), replace=True).mean(axis=1)
        lo = float(np.percentile(boot, 2.5))
        hi = float(np.percentile(boot, 97.5))
        centres.append(10 ** float(0.5 * (edges[b] + edges[b + 1])))
        emp.append(emp_b); emp_lo.append(emp_b - lo); emp_hi.append(hi - emp_b)
        mean_p50.append(float(np.mean(p))); ns.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(centres, emp, yerr=[emp_lo, emp_hi],
                fmt="o-", color="black", markersize=8, capsize=4, label="Empirical pass rate (95% bootstrap CI)")
    ax.plot(centres, mean_p50, "s--", color="tab:red", markersize=8,
            label="Mean forecaster p50")
    for x, y, n in zip(centres, emp, ns):
        ax.annotate(f"n={n}", (x, y), xytext=(5, 8), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Target task FST (minutes, log scale; binning is post-hoc)")
    ax.set_ylabel("P(solve)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"METR-style: empirical vs forecaster pass rate by log10(FST), {n_bins}-bin equal-count")
    ax.grid(True, alpha=0.3, linestyle="--", which="both")
    ax.legend(loc="best")
    _save(fig, out, show)


def plot_per_cell_distributions(df: pd.DataFrame, out: Optional[Path], show: bool) -> None:
    """
    For each cell: vertical line from p25 to p75, dot at p50, marker at outcome.
    With Delphi rounds=1 and num_experts=1 each cell has one elicitation; with
    higher num_experts/rounds we'd average across the final round.
    """
    df = df.copy().sort_values(["source_bin", "target_bin", "forecasted_model", "target_task_id"])
    df = df.reset_index(drop=True)
    cells = list(df["condition_id"])
    x = np.arange(len(cells))

    fig, ax = plt.subplots(figsize=(max(8, len(cells) * 0.6), 6))
    for i, row in df.iterrows():
        try:
            p25 = float(row["p25"]); p50 = float(row["p50"]); p75 = float(row["p75"])
        except (TypeError, ValueError):
            continue
        ax.plot([i, i], [p25, p75], color="steelblue", linewidth=2.5, alpha=0.6)
        ax.plot(i, p50, marker="s", color="steelblue", markersize=10, markeredgecolor="black")
    ax.scatter(x, df["outcome"].astype(int).values, marker="X", s=140, color="crimson",
               edgecolor="black", zorder=10, label="True outcome")
    ax.set_xticks(x)
    ax.set_xticklabels([c[:34] for c in cells], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Probability / outcome")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Per-cell forecaster distributions vs outcomes (n={len(df)} cells)")
    ax.legend(handles=[
        Line2D([0], [0], marker="s", color="steelblue", linestyle="None",
               markersize=10, markeredgecolor="black", label="p25–p75 IQR (line) and p50 (square)"),
        Line2D([0], [0], marker="X", color="crimson", linestyle="None",
               markersize=11, markeredgecolor="black", label="True binary outcome"),
    ], loc="best")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    plt.tight_layout()
    _save(fig, out, show)


# ============================================================================
# Main
# ============================================================================

def write_stats_text(stats: Dict, ece: Dict, crps_summary: Optional[Dict], out: Path,
                      source_profile_types: Optional[List[str]] = None) -> None:
    sp_type_str = "/".join(source_profile_types) if source_profile_types else "?"
    lines = [
        "=" * 70,
        "INTRA-BENCHMARK CALIBRATION STATISTICS (v1)",
        "=" * 70,
        f"  Source-profile type(s) in this run: {sp_type_str}",
        f"  N elicitations analysed (final-round, parsed p50): {stats.get('n', 0)}",
        "",
        "--- Headline (Brier on p50) ---",
        f"  Brier-on-p50:                   {stats.get('brier_p50', float('nan')):.4f}",
        f"  (chance baseline = 0.25; perfect = 0)",
        "",
        "--- Error metrics ---",
        f"  MAE:                            {stats.get('mae', float('nan')):.4f}",
        f"  RMSE:                           {stats.get('rmse', float('nan')):.4f}",
        f"  Bias (mean p50 - outcome):      {stats.get('bias', float('nan')):+.4f}",
        "",
        "--- Calibration ---",
        f"  Expected Calibration Error ({ece['n_bins']}-bin equal-width): {ece['ece']:.4f}",
        "",
        "--- Rank correlation between p50 and outcome ---",
        f"  Spearman's rho:                 {stats.get('spearman_r', float('nan')):.4f}  (p={stats.get('spearman_p', float('nan')):.4f})",
        f"  Kendall's tau:                  {stats.get('kendall_tau', float('nan')):.4f}  (p={stats.get('kendall_p', float('nan')):.4f})",
        "",
    ]
    if crps_summary:
        lines += [
            "--- CRPS (v2 stub: Beta fit to (p25, p50, p75)) ---",
            f"  Mean CRPS:                      {crps_summary['mean_crps']:.4f}",
            f"  N rows scored:                  {crps_summary['n_scored']}",
            f"  N Beta-fit failures:            {crps_summary['n_fit_failed']}",
            "",
        ]
    lines += ["=" * 70]
    text = "\n".join(lines)
    print(text)
    out.write_text(text + "\n", encoding="utf-8")
    logger.info(f"Saved {out.name}")


def main(
    run_dir: Path,
    output_subdir: str = "plots",
    show: bool = False,
    forecaster_model: Optional[str] = None,
    repeat_index: Optional[int] = None,
) -> int:
    df_full, json_data = load_run(run_dir)
    df_full = filter_rows(df_full, forecaster_model=forecaster_model, repeat_index=repeat_index)
    df = valid_rows(df_full, last_round_only=True)
    if df.empty:
        logger.error("No valid (parsed-p50) rows to analyse after filtering.")
        return 1

    # If filters were applied, name the output subdir accordingly so a
    # multi-model run can produce side-by-side per-forecaster outputs without
    # clobbering.
    suffix_parts = []
    if forecaster_model is not None:
        suffix_parts.append(f"fm-{forecaster_model.replace('/', '_')}")
    if repeat_index is not None:
        suffix_parts.append(f"rep{repeat_index}")
    if suffix_parts:
        output_subdir = f"{output_subdir}_{'_'.join(suffix_parts)}"
    output_dir = run_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = basic_stats(df)
    ece = expected_calibration_error(df, n_bins=10)

    df_with_crps = compute_crps_per_row(df)
    valid_crps = df_with_crps["crps"].dropna()
    crps_summary = None
    if len(valid_crps) > 0:
        crps_summary = {
            "mean_crps": float(valid_crps.mean()),
            "n_scored": int(len(valid_crps)),
            "n_fit_failed": int(len(df_with_crps) - len(valid_crps)),
        }

    sp_types = (
        sorted(df["source_profile_type"].dropna().unique().tolist())
        if "source_profile_type" in df.columns else None
    )
    write_stats_text(stats, ece, crps_summary, output_dir / "statistics.txt", source_profile_types=sp_types)

    plot_calibration_scatter(df, output_dir / "calibration_scatter.png", show)
    plot_brier_heatmap(df, output_dir / "brier_heatmap.png", show)
    plot_per_model_brier(df, output_dir / "per_model_brier.png", show)
    plot_per_forecaster_model_brier(df, output_dir / "per_forecaster_model_brier.png", show)
    plot_reliability_diagram(df, output_dir / "reliability_diagram.png", show)
    plot_metr_style_logfst(df, output_dir / "metr_style_logfst.png", show)
    plot_per_cell_distributions(df, output_dir / "per_cell_distributions.png", show)

    df_with_crps.to_csv(output_dir / "scored_with_crps.csv", index=False)
    logger.info(f"Wrote per-row scored CSV: {output_dir / 'scored_with_crps.csv'}")
    logger.info(f"All outputs in: {output_dir}")
    return 0


def cli() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("-r", "--run-dir", type=Path, help="Path to a run directory")
    g.add_argument("--latest", action="store_true",
                   help="Auto-pick the most recent run dir under intra_benchmark_calibration/results/")
    p.add_argument("-s", "--show", action="store_true", help="Show plots interactively")
    p.add_argument(
        "--forecaster-model",
        type=str,
        default=None,
        help=(
            "Restrict analysis to rows where forecaster_model equals this "
            "value. Useful for slicing a multi-LLM-forecaster run."
        ),
    )
    p.add_argument(
        "--repeat-index",
        type=int,
        default=None,
        help=(
            "Restrict analysis to one repeat (1-based). Default: include "
            "every repeat in the same plot."
        ),
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.latest:
        repo_root = Path(__file__).resolve().parent
        results_root = repo_root / "results"
        run_dir = latest_run_dir(results_root)
        logger.info(f"--latest: using {run_dir}")
    else:
        run_dir = args.run_dir.resolve()

    if not run_dir.is_dir():
        logger.error(f"Run dir does not exist: {run_dir}")
        return 1
    return main(
        run_dir,
        show=args.show,
        forecaster_model=args.forecaster_model,
        repeat_index=args.repeat_index,
    )


if __name__ == "__main__":
    sys.exit(cli())
