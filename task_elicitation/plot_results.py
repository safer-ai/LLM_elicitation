#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot task elicitation results: predicted probabilities vs ground truth.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys


def plot_calibration(results_dir: Path, output_path: Path = None):
    """
    Create calibration plot comparing predicted probabilities to ground truth.

    Args:
        results_dir: Directory containing predictions.csv
        output_path: Where to save the plot (default: results_dir/calibration.png)
    """
    # Load predictions
    csv_path = results_dir / "predictions.csv"
    df = pd.read_csv(csv_path)

    # Filter to valid predictions
    df = df[df['estimate'].notna()].copy()

    print(f"Loaded {len(df)} predictions from {csv_path}")
    print(f"Agents: {df['agent'].unique().tolist()}")

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # === LEFT: Calibration curve ===
    ax = axes[0]

    # Bin predictions
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Calculate observed frequency in each bin
    df['pred_bin'] = pd.cut(df['estimate'], bins=bins, include_lowest=True)
    bin_stats = df.groupby('pred_bin', observed=True).agg({
        'ground_truth': ['mean', 'count', 'std'],
        'estimate': 'mean'
    })

    bin_stats.columns = ['_'.join(col).strip('_') for col in bin_stats.columns.values]
    bin_stats = bin_stats.reset_index()

    # Plot calibration curve
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')

    # Plot per-agent calibration
    colors = plt.cm.tab10(np.linspace(0, 1, len(df['agent'].unique())))

    for i, agent in enumerate(df['agent'].unique()):
        agent_df = df[df['agent'] == agent].copy()
        agent_df['pred_bin'] = pd.cut(agent_df['estimate'], bins=bins, include_lowest=True)
        agent_stats = agent_df.groupby('pred_bin', observed=True).agg({
            'ground_truth': ['mean', 'count'],
            'estimate': 'mean'
        })
        agent_stats.columns = ['_'.join(col).strip('_') for col in agent_stats.columns.values]
        agent_stats = agent_stats.reset_index()

        # Only plot bins with enough samples
        agent_stats = agent_stats[agent_stats['ground_truth_count'] >= 2]

        label = agent.split('/')[-1]  # Shorten name
        ax.scatter(agent_stats['estimate_mean'], agent_stats['ground_truth_mean'],
                  s=agent_stats['ground_truth_count']*10, alpha=0.6,
                  color=colors[i], label=label)

    ax.set_xlabel('Predicted Probability (P50)', fontsize=12)
    ax.set_ylabel('Observed Frequency (Ground Truth)', fontsize=12)
    ax.set_title('Calibration Curve', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_aspect('equal')

    # === RIGHT: Prediction vs Ground Truth Scatter ===
    ax = axes[1]

    # Add jitter to ground truth for visualization
    np.random.seed(42)
    jitter = np.random.normal(0, 0.02, len(df))

    for i, agent in enumerate(df['agent'].unique()):
        agent_df = df[df['agent'] == agent]
        label = agent.split('/')[-1]

        ax.scatter(agent_df['estimate'], agent_df['ground_truth'] + jitter[:len(agent_df)],
                  alpha=0.4, s=30, color=colors[i], label=label)

    # Add horizontal lines for ground truth
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=2, label='Failed (GT=0)')
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.3, linewidth=2, label='Passed (GT=1)')

    ax.set_xlabel('Predicted Probability (P50)', fontsize=12)
    ax.set_ylabel('Ground Truth (0=Fail, 1=Pass)', fontsize=12)
    ax.set_title('Predictions vs Ground Truth', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.15, 1.15)
    ax.grid(alpha=0.3)
    ax.legend(loc='center left', fontsize=9)

    # Add summary stats
    brier = ((df['estimate'] - df['ground_truth']) ** 2).mean()
    accuracy = ((df['estimate'] >= 0.5).astype(int) == df['ground_truth']).mean()

    fig.text(0.5, 0.02,
             f'Brier Score: {brier:.3f}  |  Accuracy @0.5: {accuracy:.1%}  |  N={len(df)} predictions',
             ha='center', fontsize=11, style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    # Save
    if output_path is None:
        output_path = results_dir / "calibration.png"

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output_path}")

    # Also print metrics
    print("\n" + "="*60)
    print("METRICS")
    print("="*60)
    print(f"Brier Score: {brier:.4f}")
    print(f"Accuracy @0.5: {accuracy:.2%}")
    print(f"Mean Predicted: {df['estimate'].mean():.3f}")
    print(f"Mean Actual: {df['ground_truth'].mean():.3f}")

    # Per-agent metrics
    print("\nPer-agent metrics:")
    for agent in df['agent'].unique():
        agent_df = df[df['agent'] == agent]
        brier_agent = ((agent_df['estimate'] - agent_df['ground_truth']) ** 2).mean()
        acc_agent = ((agent_df['estimate'] >= 0.5).astype(int) == agent_df['ground_truth']).mean()
        print(f"  {agent}:")
        print(f"    Brier: {brier_agent:.4f}, Accuracy: {acc_agent:.2%}, N={len(agent_df)}")

    plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot task elicitation results")
    parser.add_argument("results_dir", type=str, nargs='?',
                       help="Results directory (default: latest in output_data/task_elicitation)")
    parser.add_argument("-o", "--output", type=str, help="Output path for plot")

    args = parser.parse_args()

    # Find results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        # Use latest
        base = Path("output_data/task_elicitation")
        if not base.exists():
            print("Error: No results found in output_data/task_elicitation")
            sys.exit(1)

        runs = sorted([d for d in base.iterdir() if d.is_dir()], reverse=True)
        if not runs:
            print("Error: No result directories found")
            sys.exit(1)

        results_dir = runs[0]
        print(f"Using latest results: {results_dir}")

    output_path = Path(args.output) if args.output else None

    plot_calibration(results_dir, output_path)


if __name__ == "__main__":
    main()
