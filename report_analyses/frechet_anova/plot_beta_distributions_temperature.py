#!/usr/bin/env python3
"""Plot Beta [0,1] distributions for the temperature-variance experiment.

Mirror of plot_beta_distributions_percentile.py for the temperature axis.
Persona is held fixed (Academic Security Researcher); each elicitation
produces one fitted Beta from (p25, p50, p75).

Generates:
  1. Per (model, step): one panel showing 10 individual fitted Betas at each
     temperature (colored by T), so you can eyeball within-T tightness vs.
     between-T separation.
  2. A 2×3 grid (rows = model, cols = step) of mean Beta densities per T.
"""

import csv
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from frechet_anova_temperature import (
    fit_beta_from_percentiles,
    PROVIDERS, MODEL_DISPLAY, TEMPS_BY_PROVIDER, STEPS, temp_dir,
)

X = np.linspace(0.001, 0.999, 500)


def temperature_color(temp: float, all_temps):
    """Colormap based on position of temp in the sweep."""
    cmap = plt.get_cmap("viridis")
    if len(all_temps) <= 1:
        return cmap(0.5)
    norm = (temp - min(all_temps)) / (max(all_temps) - min(all_temps))
    return cmap(norm)


def load_temp_betas(provider: str, step: str, temp: float):
    """Returns list of (alpha, beta) fits from one temperature directory."""
    d = temp_dir(provider, step, temp)
    if not d.exists():
        return []
    params = []
    run_dirs = sorted([r for r in d.iterdir() if r.is_dir() and r.name.startswith("run_")])
    for run_dir in run_dirs:
        csv_path = run_dir / "detailed_estimates.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                if row.get("has_error", "").strip().lower() == "true":
                    continue
                try:
                    p25 = float(row["percentile_25th"])
                    p50 = float(row["percentile_50th"])
                    p75 = float(row["percentile_75th"])
                except (ValueError, KeyError):
                    continue
                result = fit_beta_from_percentiles(p25, p50, p75)
                if result:
                    a, b, _ = result
                    params.append((a, b))
    return params


def mean_beta_pdf(params_list, x):
    if not params_list:
        return np.zeros_like(x)
    pdfs = np.array([sp_stats.beta.pdf(x, a, b) for a, b in params_list])
    return pdfs.mean(axis=0)


def plot_individual_curves_grid():
    """2 rows (model) x 3 cols (step). Each panel: 10 fitted Betas per T overlaid."""
    fig, axes = plt.subplots(len(PROVIDERS), len(STEPS), figsize=(18, 4.5 * len(PROVIDERS)),
                             sharey=False)
    if len(PROVIDERS) == 1:
        axes = np.array([axes])
    fig.suptitle("Temperature-Variance: Individual Fitted Beta Densities (fixed persona = Academic Security Researcher)",
                 fontsize=14, fontweight="bold", y=1.00)

    for row_idx, provider in enumerate(PROVIDERS):
        temps = TEMPS_BY_PROVIDER[provider]
        for col_idx, step in enumerate(STEPS):
            ax = axes[row_idx][col_idx]
            for t in temps:
                params = load_temp_betas(provider, step, t)
                color = temperature_color(t, temps)
                for a, b in params:
                    y = sp_stats.beta.pdf(X, a, b)
                    ax.plot(X, y, color=color, linewidth=0.9, alpha=0.45)
                # Plot mean as a thicker line for legend
                if params:
                    y_mean = mean_beta_pdf(params, X)
                    ax.plot(X, y_mean, color=color, linewidth=2.5,
                            label=f"T={t}" if (row_idx == 0 and col_idx == 0) else None)

            if row_idx == 0:
                ax.set_title(step, fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel("Density", fontsize=10)
                ax.annotate(MODEL_DISPLAY[provider], xy=(0, 0.5), xytext=(-60, 0),
                            xycoords="axes fraction", textcoords="offset points",
                            fontsize=11, fontweight="bold", ha="center", va="center",
                            rotation=90)
            if row_idx == len(PROVIDERS) - 1:
                ax.set_xlabel("Probability Estimate", fontsize=10)
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), fontsize=10,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.tight_layout()
    out = Path(__file__).parent / "beta_temperature_individual_grid.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_means_only_grid():
    """Same grid but showing only the per-T mean Beta PDF (cleaner)."""
    fig, axes = plt.subplots(len(PROVIDERS), len(STEPS), figsize=(18, 4.5 * len(PROVIDERS)),
                             sharey=False)
    if len(PROVIDERS) == 1:
        axes = np.array([axes])
    fig.suptitle("Temperature-Variance: Mean Beta Density per Temperature",
                 fontsize=14, fontweight="bold", y=1.00)

    for row_idx, provider in enumerate(PROVIDERS):
        temps = TEMPS_BY_PROVIDER[provider]
        for col_idx, step in enumerate(STEPS):
            ax = axes[row_idx][col_idx]
            for t in temps:
                params = load_temp_betas(provider, step, t)
                if not params:
                    continue
                y = mean_beta_pdf(params, X)
                color = temperature_color(t, temps)
                ax.plot(X, y, color=color, linewidth=2.2,
                        label=f"T={t}" if (row_idx == 0 and col_idx == 0) else None)
                ax.fill_between(X, y, alpha=0.08, color=color)

            if row_idx == 0:
                ax.set_title(step, fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel("Density", fontsize=10)
                ax.annotate(MODEL_DISPLAY[provider], xy=(0, 0.5), xytext=(-60, 0),
                            xycoords="axes fraction", textcoords="offset points",
                            fontsize=11, fontweight="bold", ha="center", va="center",
                            rotation=90)
            if row_idx == len(PROVIDERS) - 1:
                ax.set_xlabel("Probability Estimate", fontsize=10)
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), fontsize=10,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.tight_layout()
    out = Path(__file__).parent / "beta_temperature_means_grid.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating temperature-variance Beta distribution plots...")
    plot_individual_curves_grid()
    plot_means_only_grid()
    print("Done.")
