#!/usr/bin/env python3
"""Plot individual run distributions using Beta symmetric CI fitting.

Shows faint lines for each individual run and bold line for the mean,
to assess within-persona run-to-run variability.
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
    fit_beta_from_elicitation, DEFAULT_CONFIDENCE, DATA_DIRS,
)

EXPERIMENTS_DIR = Path(__file__).parent
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


def load_individual_beta_params(data_dir: Path) -> dict:
    """Load and fit Beta for each individual observation.
    Returns {expert: [(alpha, beta), ...]}.
    """
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


def plot_individual_runs_beta():
    """Plot individual run distributions with Beta fitting — Claude TA0002 only."""
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    fig.suptitle("Individual Run Distributions per Persona — Beta [0,1] Symmetric CI Fit (Claude, TA0002 50%)",
                 fontsize=13, fontweight="bold", y=1.02)
    
    data_dir = DATA_DIRS[("Claude Sonnet 4.5", "TA0002 (50%)")]
    data = load_individual_beta_params(data_dir)
    personas = sorted(data.keys())
    
    for idx, persona in enumerate(personas):
        ax = axes[idx // 5][idx % 5]
        params_list = data[persona]
        color = PERSONA_COLORS[idx % len(PERSONA_COLORS)]
        
        # Plot each individual run as a faint line
        for alpha, beta in params_list:
            y = sp_stats.beta.pdf(X, alpha, beta)
            ax.plot(X, y, color=color, linewidth=0.6, alpha=0.3)
        
        # Plot the mean PDF as a bold line
        if params_list:
            pdfs = np.array([sp_stats.beta.pdf(X, a, b) for a, b in params_list])
            y_mean = pdfs.mean(axis=0)
            ax.plot(X, y_mean, color=color, linewidth=2.5, alpha=0.95)
        
        ax.set_title(PERSONA_SHORT.get(persona, persona), fontsize=9, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        if idx >= 5:
            ax.set_xlabel("Probability", fontsize=8)
        if idx % 5 == 0:
            ax.set_ylabel("Density", fontsize=8)
    
    fig.tight_layout()
    out = EXPERIMENTS_DIR / "beta_individual_run_distributions.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating individual run distribution plot with Beta fitting...")
    plot_individual_runs_beta()
    print("Done.")
