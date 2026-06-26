#!/usr/bin/env python3
"""
Plot the Experiment I final results as a grouped bar chart.

Reads FINAL_RESULTS.csv and draws ΔBrier and ΔCRPS vs control per condition
(with bootstrap 95% CI error bars), colored by statistical significance:
  - significant (Wilcoxon p < 0.05): solid, saturated colors
  - not significant: faded / hatched

Writes final_results.png (+ .svg) next to the CSV.

Usage:
    python intra_benchmark_calibration/experiments/I_prompt_ablation_brier/summary/plot_final_results.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = Path(__file__).parent
CSV = HERE / "FINAL_RESULTS.csv"

# Colors
BRIER_SIG = "#1f5fb4"     # saturated blue
BRIER_NS = "#a9c4e6"      # faded blue
CRPS_SIG = "#d9831f"      # saturated orange
CRPS_NS = "#f2cfa0"       # faded orange


def main():
    df = pd.read_csv(CSV)
    df = df[df["condition"] != "control"].copy()  # control delta = 0 (reference)
    df = df.sort_values("brier_delta_vs_control").reset_index(drop=True)

    conds = df["condition"].tolist()
    x = np.arange(len(conds))
    w = 0.4

    dB = df["brier_delta_vs_control"].to_numpy()
    dC = df["crps_delta_vs_control"].to_numpy()
    # asymmetric error bars from bootstrap CI
    bErr = np.vstack([dB - df["brier_ci95_low"], df["brier_ci95_high"] - dB])
    cErr = np.vstack([dC - df["crps_ci95_low"], df["crps_ci95_high"] - dC])

    bColors = [BRIER_SIG if s == "yes" else BRIER_NS for s in df["brier_significant"]]
    cColors = [CRPS_SIG if s == "yes" else CRPS_NS for s in df["crps_significant"]]

    fig, ax = plt.subplots(figsize=(12, 6.5))

    ax.bar(x - w / 2, dB, w, color=bColors, edgecolor="black", linewidth=0.6,
           yerr=bErr, capsize=3, ecolor="#444444", error_kw={"linewidth": 1},
           label="_brier")
    ax.bar(x + w / 2, dC, w, color=cColors, edgecolor="black", linewidth=0.6,
           yerr=cErr, capsize=3, ecolor="#444444", error_kw={"linewidth": 1},
           label="_crps")

    # numeric labels above each bar
    for xi, v in zip(x - w / 2, dB):
        ax.text(xi, v + (0.0012 if v >= 0 else -0.0012), f"{v:+.4f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)
    for xi, v in zip(x + w / 2, dC):
        ax.text(xi, v + (0.0012 if v >= 0 else -0.0012), f"{v:+.4f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)

    ax.axhline(0, color="black", linewidth=1.1)
    ax.text(len(conds) - 0.5, 0.0008, "control (reference)", ha="right",
            va="bottom", fontsize=8, color="black", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Δ vs control  (higher = worse than full prompt)", fontsize=10)
    ax.set_title("Experiment I — prompt-ablation impact on forecast accuracy\n"
                 "Brier vs CRPS, paired Δ from control (Sonnet 4.6, 300 matched cells, "
                 "95% bootstrap CI)", fontsize=11)

    legend_elems = [
        Patch(facecolor=BRIER_SIG, edgecolor="black", label="ΔBrier — significant (p<0.05)"),
        Patch(facecolor=BRIER_NS, edgecolor="black", label="ΔBrier — not significant"),
        Patch(facecolor=CRPS_SIG, edgecolor="black", label="ΔCRPS — significant (p<0.05)"),
        Patch(facecolor=CRPS_NS, edgecolor="black", label="ΔCRPS — not significant"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=8.5, framealpha=0.95)

    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()

    for ext in ("png", "svg"):
        out = HERE / f"final_results.{ext}"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
