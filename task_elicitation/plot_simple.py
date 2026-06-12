#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple, clear calibration plot.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys


def plot_simple_calibration(results_dir: Path):
    """Create one super clear calibration plot."""

    # Load predictions
    csv_path = results_dir / "predictions.csv"
    df = pd.read_csv(csv_path)
    df = df[df['estimate'].notna()].copy()

    print(f"Loaded {len(df)} predictions")

    # Create bins
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

    df['pred_bin'] = pd.cut(df['estimate'], bins=bins, labels=labels, include_lowest=True)

    # Calculate stats per bin
    stats = df.groupby('pred_bin', observed=True).agg({
        'estimate': 'mean',
        'ground_truth': ['mean', 'count']
    }).reset_index()

    stats.columns = ['bin', 'predicted', 'actual', 'count']

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(stats))
    width = 0.35

    # Bars
    bars1 = ax.bar(x - width/2, stats['predicted'] * 100, width,
                   label='What we predicted', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, stats['actual'] * 100, width,
                   label='What actually happened', color='coral', alpha=0.8)

    # Add count labels on bars
    for i, (p, a, c) in enumerate(zip(stats['predicted'], stats['actual'], stats['count'])):
        ax.text(i - width/2, p*100 + 2, f'{p*100:.0f}%', ha='center', fontsize=9)
        ax.text(i + width/2, a*100 + 2, f'{a*100:.0f}%', ha='center', fontsize=9)
        ax.text(i, -8, f'n={int(c)}', ha='center', fontsize=8, style='italic')

    # Perfect calibration line
    ax.plot([-0.5, len(stats)-0.5], [0, 100], 'k--', alpha=0.3, linewidth=2,
            label='Perfect calibration')

    ax.set_xlabel('Predicted Probability Range', fontsize=13, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=13, fontweight='bold')
    ax.set_title('Model Task Success Prediction vs Reality\n' +
                 'Example: We predicted 40-60% success → Actually 67% passed',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(stats['bin'])
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(-12, 108)
    ax.grid(axis='y', alpha=0.3)

    # Add summary text
    brier = ((df['estimate'] - df['ground_truth']) ** 2).mean()
    acc = ((df['estimate'] >= 0.5).astype(int) == df['ground_truth']).mean()

    ax.text(0.98, 0.02,
            f'Overall: {len(df)} predictions | Brier Score: {brier:.3f} | Accuracy: {acc:.1%}',
            transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

    plt.tight_layout()

    # Save
    output_path = results_dir / "calibration_simple.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_path}")

    # Print table
    print("\n" + "="*70)
    print("CALIBRATION TABLE")
    print("="*70)
    print(f"{'Predicted Range':<20} {'We Said':<15} {'Actually Passed':<20} {'Count':<10}")
    print("-"*70)
    for _, row in stats.iterrows():
        print(f"{row['bin']:<20} {row['predicted']*100:>6.1f}% {row['actual']*100:>12.1f}% {int(row['count']):>15}")
    print("="*70)

    plt.show()


def main():
    # Find latest results
    base = Path("output_data/task_elicitation")
    runs = sorted([d for d in base.iterdir() if d.is_dir()], reverse=True)

    if not runs:
        print("Error: No results found")
        sys.exit(1)

    results_dir = runs[0]
    print(f"Using: {results_dir}\n")

    plot_simple_calibration(results_dir)


if __name__ == "__main__":
    main()
