#!/usr/bin/env python3
"""Plot Exp H (bin-1 target-task variance) as two separate figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
REPO = Path(__file__).resolve().parents[3]
DEFAULT_CSV = Path(__file__).parent / "results" / "bin1_estimates_combined.csv"
DEFAULT_OUT = Path(__file__).parent / "results"
FIGURES_FOR_REPORT = REPO / "figures_for_report"

EXP_C_BIN1_BRIER = 0.10  # reference from prior Exp C / pilot notes


def _short_task_id(task_id: str, max_len: int = 28) -> str:
    if len(task_id) <= max_len:
        return task_id
    if "/" in task_id:
        family, name = task_id.split("/", 1)
        budget = max_len - len(family) - 2
        if budget > 8:
            return f"{family}/{name[:budget]}"
    return task_id[: max_len - 1] + "…"


def load_per_task(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    per_task = (
        df.groupby(["target_task_id", "target_fst_minutes"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "brier": float(np.mean((g["p50"] - g["outcome"]) ** 2)),
                    "gt_solve_rate": float(g["outcome"].mean()),
                    "p50_mean": float(g["p50"].mean()),
                    "n_models": len(g),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
        .sort_values("brier")
        .reset_index(drop=True)
    )
    return per_task


def plot_per_task_brier(per_task: pd.DataFrame, out: Path) -> None:
    n = len(per_task)
    fig_h = max(6.0, 0.32 * n + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    colors = [
        "#2ca02c" if b < EXP_C_BIN1_BRIER else "#d62728" if b > 0.25 else "#ff7f0e"
        for b in per_task["brier"]
    ]
    y = np.arange(n)
    ax.barh(y, per_task["brier"], color=colors, edgecolor="white", height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels([_short_task_id(t) for t in per_task["target_task_id"]], fontsize=9)
    ax.invert_yaxis()

    ax.axvline(EXP_C_BIN1_BRIER, color="steelblue", linestyle="--", linewidth=1.5,
               label=f"Exp C reference ({EXP_C_BIN1_BRIER:.2f})")
    ax.axvline(0.25, color="gray", linestyle=":", linewidth=1.5, label="Naive 0.5 (0.25)")
    ax.axvline(per_task["brier"].mean(), color="black", linewidth=1.5,
               label=f"Mean over 20 tasks ({per_task['brier'].mean():.3f})")

    ax.set_xlabel("Per-task Brier score (mean over 12 forecasted models)")
    ax.set_title(
        "Exp H — bin 1 target-task variance\n"
        "Target bin 1; source context from bins 0, 2, 3, 4 (all_except_target)",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_xlim(left=0)
    ax.grid(True, axis="x", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_p50_vs_gt(per_task: pd.DataFrame, out: Path) -> None:
    """Scatter with numbered dots (sorted by Brier desc) + clean index below."""
    per_task = per_task.copy()
    # Number tasks worst-first so outliers get small numbers
    per_task = per_task.sort_values("brier", ascending=False).reset_index(drop=True)
    per_task["num"] = per_task.index + 1

    # Deterministic horizontal jitter within each GT group, sorted by p50 within group
    jitter = np.zeros(len(per_task))
    for _, grp in per_task.groupby("gt_solve_rate"):
        grp_sorted = grp.sort_values("p50_mean")
        n = len(grp_sorted)
        if n > 1:
            spread = np.linspace(-0.028, 0.028, n)
        else:
            spread = np.array([0.0])
        jitter[grp_sorted.index] = spread
    x = per_task["gt_solve_rate"].values + jitter

    fig = plt.figure(figsize=(7.5, 9.5))
    # Top 68% of figure = scatter, bottom 32% = index
    gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.28)
    ax = fig.add_subplot(gs[0])

    vmax = max(0.40, float(per_task["brier"].max()))
    sc = ax.scatter(
        x,
        per_task["p50_mean"].values,
        c=per_task["brier"].values,
        cmap="RdYlGn_r",
        s=130,
        vmin=0,
        vmax=vmax,
        edgecolors="white",
        linewidth=0.8,
        zorder=3,
    )
    # Numbers inside dots
    for i, row in per_task.iterrows():
        ax.text(
            x[i], float(row["p50_mean"]),
            str(int(row["num"])),
            ha="center", va="center",
            fontsize=6.5, fontweight="bold", color="black",
            zorder=4,
        )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.038, pad=0.02)
    cbar.set_label("Brier score", fontsize=9)

    ax.plot([0, 1], [0, 1], color="dimgray", linestyle="--", linewidth=1.1,
            label="Perfect calibration (y = x)", zorder=2)
    ax.set_xlabel("GT solve rate  (fraction of 12 models)", fontsize=11)
    ax.set_ylabel("LLM mean p50  (elicitor: claude-sonnet-4-6)", fontsize=11)
    ax.set_title(
        "Exp H — bin 1 target-task variance\n"
        "Elicited p50 vs empirical solve rate  (20 tasks, colour = Brier)",
        fontsize=12,
    )
    ax.set_xlim(0.32, 1.06)
    ax.set_ylim(0.12, 1.04)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle="--")

    # ---- Index panel below ----
    ax_idx = fig.add_subplot(gs[1])
    ax_idx.axis("off")

    n = len(per_task)
    half = (n + 1) // 2
    col_x = [0.02, 0.52]   # left/right column x positions (axes fraction)
    for col, start in enumerate([0, half]):
        end = half if col == 0 else n
        for row_i, idx in enumerate(range(start, end)):
            row = per_task.iloc[idx]
            num = int(row["num"])
            task = _short_task_id(row["target_task_id"], 30)
            brier = float(row["brier"])
            text = f"{num:2d}.  {task}  ({brier:.3f})"
            y_pos = 1.0 - row_i * (1.0 / (half + 0.5))
            ax_idx.text(
                col_x[col], y_pos, text,
                transform=ax_idx.transAxes,
                fontsize=7.8, va="top", ha="left",
                fontfamily="monospace",
            )

    ax_idx.text(0.5, 1.05, "Task index  (sorted by Brier score, worst first)",
                transform=ax_idx.transAxes, ha="center", fontsize=9,
                fontweight="bold")
    ax_idx.plot([0, 1], [1.02, 1.02], color="lightgray", linewidth=0.8,
               transform=ax_idx.transAxes, clip_on=False)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--copy-to-figures-for-report", action="store_true")
    args = p.parse_args()

    per_task = load_per_task(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    brier_path = args.out_dir / "fig_exp_h_bin1_per_task_brier.png"
    scatter_path = args.out_dir / "fig_exp_h_bin1_p50_vs_gt_solve_rate.png"
    plot_per_task_brier(per_task, brier_path)
    plot_p50_vs_gt(per_task, scatter_path)

    if args.copy_to_figures_for_report:
        FIGURES_FOR_REPORT.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(brier_path, FIGURES_FOR_REPORT / brier_path.name)
        shutil.copy(scatter_path, FIGURES_FOR_REPORT / scatter_path.name)

    print(f"Wrote {brier_path}")
    print(f"Wrote {scatter_path}")
    print(f"Tasks: {len(per_task)}; mean Brier={per_task['brier'].mean():.4f}, std={per_task['brier'].std():.4f}")


if __name__ == "__main__":
    main()
