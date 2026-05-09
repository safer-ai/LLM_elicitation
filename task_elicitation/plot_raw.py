#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Raw scatter plot - no BS.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_raw(results_dir: Path):
    """Just the raw data points."""

    # Load
    df = pd.read_csv(results_dir / "predictions.csv")
    df = df[df['estimate'].notna()].copy()

    print(f"\n{len(df)} predictions:")
    print(f"  Predicted: 0 to 1 (probability)")
    print(f"  Actual: 0 (failed) or 1 (passed)")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Add jitter to y-axis so we can see overlapping points
    np.random.seed(42)
    jitter = np.random.uniform(-0.03, 0.03, len(df))

    # Plot each point
    ax.scatter(df['estimate'], df['ground_truth'] + jitter,
              alpha=0.5, s=80, color='steelblue', edgecolors='black', linewidth=0.5)

    # Reference lines
    ax.axhline(y=0, color='red', linewidth=2, label='FAILED (ground truth = 0)', alpha=0.7)
    ax.axhline(y=1, color='green', linewidth=2, label='PASSED (ground truth = 1)', alpha=0.7)

    ax.set_xlabel('What We Predicted (Probability)', fontsize=14, fontweight='bold')
    ax.set_ylabel('What Actually Happened (0=Fail, 1=Pass)', fontsize=14, fontweight='bold')
    ax.set_title(f'Task Success Predictions vs Reality\n{len(df)} predictions from 2 models on 30 tasks each',
                fontsize=15, fontweight='bold', pad=20)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.15, 1.15)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['FAILED', 'PASSED'])
    ax.grid(alpha=0.3, axis='x')
    ax.legend(fontsize=11, loc='center left')

    plt.tight_layout()

    output_path = results_dir / "calibration_raw.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_path}")

    # Print some examples
    print("\nExample predictions:")
    print("-" * 80)
    for i, row in df.head(10).iterrows():
        actual = "PASSED" if row['ground_truth'] == 1 else "FAILED"
        print(f"  Predicted: {row['estimate']:.2f} → Actually: {actual} | {row['target_task_id'][:40]}")

    plt.show()


def main():
    base = Path("output_data/task_elicitation")
    runs = sorted([d for d in base.iterdir() if d.is_dir()], reverse=True)
    results_dir = runs[0]
    print(f"Using: {results_dir}")
    plot_raw(results_dir)


if __name__ == "__main__":
    main()
