#!/usr/bin/env python3
"""Plot Beta [0,1] distributions fitted using symmetric confidence intervals.

Uses the same Beta fitting as frechet_anova.py (symmetric CI, equal tail probs).
Generates:
  1. Persona distributions (Claude) — 10 persona curves per step
  2. Cross-model distributions — 3 model curves per step
  3. Full 3×3 grid
"""

import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from frechet_anova import (
    fit_beta_from_elicitation, STEPS, MODELS,
    DEFAULT_CONFIDENCE, DATA_DIRS,
)
X = np.linspace(0.001, 0.999, 500)

PERSONA_SHORT = {
    "Defensive Security Specialist": "Defensive",
    "Malware Reverse Engineer": "Malware RE",
    "AI/ML Security Researcher": "AI/ML Sec",
    "Threat Intelligence Analyst": "Threat Intel",
    "Security Software Developer": "Sec Dev",
    "Incident Response Specialist": "Incident Resp",
    "Academic Security Researcher": "Academic",
    "Red Team Operator": "Red Team",
    "CTF Competition Expert": "CTF Expert",
    "Security Compliance Officer": "Compliance",
}

PERSONA_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]
MODEL_COLORS = {"Claude Sonnet 4.5": "#4363d8", "GPT-4o": "#e6194b", "Gemini 2.5 Pro": "#3cb44b"}


def load_beta_params(data_dir: Path) -> dict:
    """Load data and fit Beta distributions. Returns {expert: [(alpha, beta), ...]}."""
    expert_params = defaultdict(list)
    run_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    for run_dir in run_dirs:
        csv_path = run_dir / "detailed_estimates.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                try:
                    lo = float(row["minimum_estimate"])
                    hi = float(row["maximum_estimate"])
                except (ValueError, KeyError):
                    continue
                cs = row.get("confidence_in_range", "").strip()
                conf = float(cs) if cs else DEFAULT_CONFIDENCE
                params = fit_beta_from_elicitation(lo, hi, conf)
                if params:
                    expert_params[row["expert_name"]].append(params)
    return dict(expert_params)


def mean_beta_pdf(params_list, x):
    """Average the PDFs from a list of (alpha, beta) tuples."""
    if not params_list:
        return np.zeros_like(x)
    pdfs = np.array([sp_stats.beta.pdf(x, a, b) for a, b in params_list])
    return pdfs.mean(axis=0)


def plot_persona_distributions():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.suptitle("Expert Persona Belief Distributions — Beta [0,1] Symmetric CI Fit (Claude Sonnet 4.5)",
                 fontsize=14, fontweight="bold", y=1.02)

    for col, step in enumerate(STEPS):
        ax = axes[col]
        data = load_beta_params(DATA_DIRS[("Claude Sonnet 4.5", step)])
        personas = sorted(data.keys())

        for i, persona in enumerate(personas):
            y = mean_beta_pdf(data[persona], X)
            ax.plot(X, y, color=PERSONA_COLORS[i % len(PERSONA_COLORS)],
                    linewidth=1.8, label=PERSONA_SHORT.get(persona, persona), alpha=0.85)

        all_params = [p for ps in data.values() for p in ps]
        y_agg = mean_beta_pdf(all_params, X)
        ax.plot(X, y_agg, color="black", linewidth=2.5, linestyle="--", label="Aggregate", alpha=0.9)

        ax.set_title(step, fontsize=12, fontweight="bold")
        ax.set_xlabel("Probability Estimate", fontsize=10)
        if col == 0:
            ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8,
              bbox_to_anchor=(0.5, -0.08), frameon=True)
    fig.tight_layout()
    out = Path(__file__).parent / "beta_persona_distributions.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_cross_model_distributions():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.suptitle("Cross-Model Belief Distributions — Beta [0,1] Symmetric CI Fit",
                 fontsize=14, fontweight="bold", y=1.02)

    for col, step in enumerate(STEPS):
        ax = axes[col]
        for model in MODELS:
            data = load_beta_params(DATA_DIRS[(model, step)])
            all_params = [p for ps in data.values() for p in ps]
            y = mean_beta_pdf(all_params, X)
            ax.plot(X, y, color=MODEL_COLORS[model], linewidth=2.2, label=model, alpha=0.9)
            ax.fill_between(X, y, alpha=0.08, color=MODEL_COLORS[model])

        ax.set_title(step, fontsize=12, fontweight="bold")
        ax.set_xlabel("Probability Estimate", fontsize=10)
        if col == 0:
            ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    out = Path(__file__).parent / "beta_cross_model_distributions.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_full_grid():
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharey=False)
    fig.suptitle("Expert Persona Distributions — Beta [0,1] Symmetric CI Fit (All Models × Steps)",
                 fontsize=14, fontweight="bold", y=1.01)

    for row_idx, model in enumerate(MODELS):
        for col_idx, step in enumerate(STEPS):
            ax = axes[row_idx][col_idx]
            data = load_beta_params(DATA_DIRS[(model, step)])
            personas = sorted(data.keys())

            for i, persona in enumerate(personas):
                y = mean_beta_pdf(data[persona], X)
                label = PERSONA_SHORT.get(persona, persona) if row_idx == 0 and col_idx == 0 else None
                ax.plot(X, y, color=PERSONA_COLORS[i % len(PERSONA_COLORS)],
                        linewidth=1.5, label=label, alpha=0.8)

            all_params = [p for ps in data.values() for p in ps]
            y_agg = mean_beta_pdf(all_params, X)
            label_agg = "Aggregate" if row_idx == 0 and col_idx == 0 else None
            ax.plot(X, y_agg, color="black", linewidth=2.2, linestyle="--",
                    label=label_agg, alpha=0.9)

            if row_idx == 0:
                ax.set_title(step, fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel("Density", fontsize=10)
                ax.annotate(model, xy=(0, 0.5), xytext=(-60, 0),
                           xycoords="axes fraction", textcoords="offset points",
                           fontsize=11, fontweight="bold", ha="center", va="center",
                           rotation=90)
            if row_idx == 2:
                ax.set_xlabel("Probability Estimate", fontsize=10)
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8,
              bbox_to_anchor=(0.5, -0.03), frameon=True)
    fig.tight_layout()
    out = Path(__file__).parent / "beta_full_distribution_grid.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating Beta [0,1] distribution plots...")
    plot_persona_distributions()
    plot_cross_model_distributions()
    plot_full_grid()
    print("Done.")
