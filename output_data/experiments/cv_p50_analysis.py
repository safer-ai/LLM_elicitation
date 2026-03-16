"""
Calculate Coefficient of Variation (CV) of expert-mean p50 values.

This matches the paper's likely methodology:
- For each expert: compute mean of their p50 values across runs
- For each risk factor: compute CV across the 10 expert-mean p50s
"""

import json
import yaml
import pandas as pd
from pathlib import Path
import numpy as np
from collections import defaultdict

EXPERIMENTS_DIR = Path(__file__).parent

# Data directories for quantities
QUANT_DATA_DIRS = {
    "GPT-4o": EXPERIMENTS_DIR / "numactors_gpt4o",
    "Gemini 2.5 Pro": EXPERIMENTS_DIR / "numactors_gemini",
    "Claude Sonnet 4.5": EXPERIMENTS_DIR / "numactors_claude",
}

# Data directories for probabilities (one per step)
PROB_DATA_DIRS = {
    "GPT-4o": {
        "T1657": EXPERIMENTS_DIR / "percentile_gpt4o_T1657_30pct",
        "TA0002": EXPERIMENTS_DIR / "percentile_gpt4o_TA0002_50pct",
        "TA0007": EXPERIMENTS_DIR / "percentile_gpt4o_TA0007_85pct",
    },
    "Gemini 2.5 Pro": {
        "T1657": EXPERIMENTS_DIR / "percentile_gemini_T1657_30pct",
        "TA0002": EXPERIMENTS_DIR / "percentile_gemini_TA0002_50pct",
        "TA0007": EXPERIMENTS_DIR / "percentile_gemini_TA0007_85pct",
    },
    "Claude Sonnet 4.5": {
        "T1657": EXPERIMENTS_DIR / "percentile_claude_T1657_30pct",
        "TA0002": EXPERIMENTS_DIR / "percentile_claude_TA0002_50pct",
        "TA0007": EXPERIMENTS_DIR / "percentile_claude_TA0007_85pct",
    },
}

MODELS = ["GPT-4o", "Gemini 2.5 Pro", "Claude Sonnet 4.5"]

# Map short names to full step names in CSV
PROB_STEPS = {
    "T1657": "T1657 - Impact: Financial Theft / Extortion",
    "TA0002": "TA0002 - Execution",
    "TA0007": "TA0007 - Discovery",
}


def load_p50_values_from_csv(data_dir, step_name):
    """
    Load p50 values from CSV files (for probabilities).

    Returns: dict mapping expert_name -> list of p50 values
    """
    expert_p50s = defaultdict(list)

    for run_dir in sorted(data_dir.glob("run_*")):
        csv_file = run_dir / "detailed_estimates.csv"
        if not csv_file.exists():
            continue

        df = pd.read_csv(csv_file)

        # Filter to the specific step
        step_df = df[df['step_name'] == step_name]

        for _, row in step_df.iterrows():
            expert_name = row['expert_name']
            p50 = row['percentile_50th']
            expert_p50s[expert_name].append(p50)

    return expert_p50s


def load_p50_values_from_csv_numactors(data_dir):
    """
    Load p50 values from CSV files (for num_actors quantity).

    Returns: dict mapping expert_name -> list of p50 values
    """
    expert_p50s = defaultdict(list)

    for run_dir in sorted(data_dir.glob("run_*")):
        csv_file = run_dir / "detailed_estimates.csv"
        if not csv_file.exists():
            continue

        df = pd.read_csv(csv_file)

        # Filter to num_actors metric
        metric_df = df[df['step_name'] == 'ScenarioLevelMetric_NumActors']

        for _, row in metric_df.iterrows():
            expert_name = row['expert_name']
            p50 = row['percentile_50th']
            expert_p50s[expert_name].append(p50)

    return expert_p50s


def compute_cv_from_csv(data_dir, step_name):
    """
    Compute CV of expert-mean p50 values from CSV files.
    """
    expert_p50s = load_p50_values_from_csv(data_dir, step_name)

    if not expert_p50s:
        return None

    # Compute mean p50 per expert (ignore NaN values from failed runs)
    expert_means = []
    for expert_name, p50_values in expert_p50s.items():
        if len(p50_values) > 0:
            mean_val = np.nanmean(p50_values)  # Use nanmean to ignore NaN
            if not np.isnan(mean_val):
                expert_means.append(mean_val)

    if len(expert_means) < 2:
        return None

    # Compute CV
    mean = np.mean(expert_means)
    std = np.std(expert_means, ddof=1)  # Sample std
    cv = (std / mean) * 100  # As percentage

    return {
        'cv': cv,
        'mean': mean,
        'std': std,
        'n_experts': len(expert_means),
        'expert_means': expert_means
    }


def compute_cv_from_csv_numactors(data_dir):
    """
    Compute CV of expert-mean p50 values from CSV files (num_actors).
    """
    expert_p50s = load_p50_values_from_csv_numactors(data_dir)

    if not expert_p50s:
        return None

    # Compute mean p50 per expert (ignore NaN values from failed runs)
    expert_means = []
    for expert_name, p50_values in expert_p50s.items():
        if len(p50_values) > 0:
            mean_val = np.nanmean(p50_values)  # Use nanmean to ignore NaN
            if not np.isnan(mean_val):
                expert_means.append(mean_val)

    if len(expert_means) < 2:
        return None

    # Compute CV
    mean = np.mean(expert_means)
    std = np.std(expert_means, ddof=1)  # Sample std
    cv = (std / mean) * 100  # As percentage

    return {
        'cv': cv,
        'mean': mean,
        'std': std,
        'n_experts': len(expert_means),
        'expert_means': expert_means
    }


def main():
    print("=" * 80)
    print("CV of Expert-Mean p50 Values")
    print("=" * 80)

    # Results storage
    prob_cvs = []
    quant_cvs = []

    print("\n" + "=" * 80)
    print("PROBABILITIES (Step-level)")
    print("=" * 80)

    for model in MODELS:
        print(f"\n{model}:")
        print("-" * 40)

        for step_short, step_full in PROB_STEPS.items():
            data_dir = PROB_DATA_DIRS[model][step_short]
            result = compute_cv_from_csv(data_dir, step_full)

            if result:
                prob_cvs.append(result['cv'])
                print(f"  {step_short:10s}: CV = {result['cv']:5.1f}% "
                      f"(mean = {result['mean']:.3f}, n = {result['n_experts']})")

    print("\n" + "=" * 80)
    print("QUANTITIES (Scenario-level)")
    print("=" * 80)

    for model in MODELS:
        print(f"\n{model}:")
        print("-" * 40)

        data_dir = QUANT_DATA_DIRS[model]
        result = compute_cv_from_csv_numactors(data_dir)

        if result:
            quant_cvs.append(result['cv'])
            print(f"  num_actors: CV = {result['cv']:5.1f}% "
                  f"(mean = {result['mean']:.1f}, n = {result['n_experts']})")

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)

    if prob_cvs and quant_cvs:
        print(f"\nProbabilities (n={len(prob_cvs)}):")
        print(f"  Mean CV   = {np.mean(prob_cvs):.1f}%")
        print(f"  Range CV  = {np.min(prob_cvs):.1f}% - {np.max(prob_cvs):.1f}%")
        print(f"  All CVs   = {[f'{cv:.1f}%' for cv in prob_cvs]}")

        print(f"\nQuantities (n={len(quant_cvs)}):")
        print(f"  Mean CV   = {np.mean(quant_cvs):.1f}%")
        print(f"  Range CV  = {np.min(quant_cvs):.1f}% - {np.max(quant_cvs):.1f}%")
        print(f"  All CVs   = {[f'{cv:.1f}%' for cv in quant_cvs]}")

        print(f"\n{'FINDING:':12s}", end="")
        if np.mean(quant_cvs) > np.mean(prob_cvs):
            print(f" Quantities show MORE variance than probabilities")
            print(f"             (CV: {np.mean(quant_cvs):.1f}% > {np.mean(prob_cvs):.1f}%)")
            print(f"             ✓ MATCHES paper's human expert findings")
        else:
            print(f" Quantities show LESS variance than probabilities")
            print(f"             (CV: {np.mean(quant_cvs):.1f}% < {np.mean(prob_cvs):.1f}%)")
            print(f"             ✗ OPPOSITE of paper's human expert findings")


if __name__ == "__main__":
    main()
