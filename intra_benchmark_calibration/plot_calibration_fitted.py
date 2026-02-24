#!/usr/bin/env python
"""
Calibration plotting for intra-benchmark experiments using Monte Carlo Beta aggregation.

This script generates 4 types of plots from the consensus CSV 
(detailed_estimates_fitted_consensus.csv) produced by BayesianNetwork/data_preprocessing/fit_beta_from_percentiles.py:

1. Calibration scatter: LLM mean estimates vs ground truth
2. Sequential transitions: P(i+1|i) for consecutive bins
3. Heatmap comparison: Side-by-side matrices of ground truth and LLM estimates
4. Distribution box plots: MC-aggregated distributions showing percentiles (p025, p25, p50, p75, p975)
   with mode (from KDE) and ground truth overlays

Statistical metrics (MAE, RMSE, bias, correlations) are also computed and saved.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging
import argparse
import math
from typing import Optional, Dict
from scipy.stats import spearmanr, kendalltau

# Configure logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')


def load_consensus_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """
    Load consensus estimates from fitted Beta Monte Carlo aggregation CSV.

    Args:
        csv_path: Path to detailed_estimates_fitted_consensus.csv file

    Returns:
        DataFrame with consensus data or None if loading fails
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} consensus estimates from {csv_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load CSV file {csv_path}: {e}")
        return None


def load_fitted_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """
    Load individual expert estimates from fitted Beta CSV.

    Args:
        csv_path: Path to detailed_estimates_fitted.csv file

    Returns:
        DataFrame with per-expert data or None if loading fails
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} expert estimates from {csv_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load CSV file {csv_path}: {e}")
        return None


def plot_calibration_scatter(df: pd.DataFrame, output_path: Optional[Path] = None, show: bool = False) -> plt.Figure:
    """
    Plot 1: Calibration scatter plot (P_truth vs P_llm).

    Args:
        df: DataFrame with calibration data
        output_path: Optional path to save figure
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Compute jump size for colouring
    jump_size = df['bin_j'] - df['bin_i']

    # Create scatter plot with colour by jump size
    scatter = ax.scatter(
        df['ground_truth_p_j_given_i'],
        df['mean'],
        c=jump_size,
        s=150,
        alpha=0.7,
        cmap='viridis',
        edgecolors='black',
        linewidths=1.5
    )

    # Add error bars - 25th and 75th percentiles
    ax.errorbar(
        df['ground_truth_p_j_given_i'],
        df['mean'],
        yerr=[df['mean'] - df['p25'], df['p75'] - df['mean']],
        fmt='none',
        ecolor='gray',
        alpha=0.5,
        capsize=5,
        capthick=2
    )

    # Add diagonal reference line (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=2, label='Perfect Calibration (y=x)')

    # Add colourbar for jump size
    cbar = plt.colorbar(scatter, ax=ax, label='Bin Jump Size |j-i|')
    cbar.ax.tick_params(labelsize=10)

    # Labels and formatting
    ax.set_xlabel('Ground Truth P(j|i)', fontsize=14, fontweight='bold')
    ax.set_ylabel('LLM Estimate P(j|i)', fontsize=14, fontweight='bold')
    ax.set_title('Calibration: LLM Estimates vs Ground Truth', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved calibration scatter plot to {output_path}")

    if show:
        plt.show()

    return fig


def plot_sequential_transitions(df: pd.DataFrame, output_path: Optional[Path] = None, show: bool = False) -> plt.Figure:
    """
    Plot 2: Sequential transitions (marginal difficulty) for consecutive bins only.

    Args:
        df: DataFrame with calibration data
        output_path: Optional path to save figure
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    # Compute jump size and filter for consecutive bins only (j = i+1)
    df_with_jump = df.copy()
    df_with_jump['jump_size'] = df['bin_j'] - df['bin_i']
    df_seq = df_with_jump[df_with_jump['jump_size'] == 1].sort_values('bin_i')

    if df_seq.empty:
        logger.warning("No consecutive bin transitions found in data")
        return None

    fig, ax = plt.subplots(figsize=(12, 7))

    x = df_seq['bin_i'].values
    gt = df_seq['ground_truth_p_j_given_i'].values
    llm = df_seq['mean'].values
    llm_25th = df_seq['p25'].values
    llm_75th = df_seq['p75'].values

    # Plot ground truth line
    ax.plot(x, gt, 'o-', color='#2E86AB', linewidth=2.5, markersize=10,
            label='Ground Truth', markeredgecolor='black', markeredgewidth=1.5)

    # Plot LLM estimate line
    ax.plot(x, llm, 's--', color='#A23B72', linewidth=2.5, markersize=10,
            label='LLM Estimate', markeredgecolor='black', markeredgewidth=1.5)

    # Add shaded region for LLM 25% - 75% quantiles
    ax.fill_between(x, llm_25th, llm_75th,
                     color='#A23B72', alpha=0.2, label='LLM 25% - 75% Quantiles')

    # Labels and formatting
    ax.set_xlabel('Starting Bin Index i', fontsize=14, fontweight='bold')
    ax.set_ylabel('P(next bin | current bin)', fontsize=14, fontweight='bold')
    ax.set_title('Sequential Transitions: P(i+1|i) for Consecutive Bins',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(x)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved sequential transitions plot to {output_path}")

    if show:
        plt.show()

    return fig


def plot_heatmap_comparison(df: pd.DataFrame, output_path: Optional[Path] = None, show: bool = False) -> plt.Figure:
    """
    Plot 3: Side-by-side heatmaps comparing ground truth and LLM estimates.

    Args:
        df: DataFrame with calibration data
        output_path: Optional path to save figure
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    # Determine matrix size
    n_bins = int(df[['bin_i', 'bin_j']].max().max()) + 1

    # Initialise matrices with NaN
    gt_matrix = np.full((n_bins, n_bins), np.nan)
    llm_matrix = np.full((n_bins, n_bins), np.nan)

    # Fill matrices (upper triangle only, j > i)
    for _, row in df.iterrows():
        i, j = int(row['bin_i']), int(row['bin_j'])
        gt_matrix[i, j] = row['ground_truth_p_j_given_i']
        llm_matrix[i, j] = row['mean']

    # Create side-by-side heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Shared colourmap and range
    vmin, vmax = 0, 1
    cmap = 'RdYlGn'

    # Plot ground truth heatmap
    sns.heatmap(gt_matrix, annot=True, fmt='.2f', cmap=cmap, vmin=vmin, vmax=vmax,
                cbar=False, ax=ax1, linewidths=1, linecolor='gray',
                mask=(np.triu(np.ones_like(gt_matrix, dtype=bool), k=0) is False),
                annot_kws={'size': 11, 'weight': 'bold'})
    ax1.set_title('Ground Truth P(j|i)', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('Target Bin j', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Starting Bin i', fontsize=13, fontweight='bold')

    # Plot LLM estimate heatmap
    im = sns.heatmap(llm_matrix, annot=True, fmt='.2f', cmap=cmap, vmin=vmin, vmax=vmax,
                     cbar=False, ax=ax2, linewidths=1, linecolor='gray',
                     mask=(np.triu(np.ones_like(llm_matrix, dtype=bool), k=0) is False),
                     annot_kws={'size': 11, 'weight': 'bold'})
    ax2.set_title('LLM Estimate P(j|i)', fontsize=16, fontweight='bold', pad=15)
    ax2.set_xlabel('Target Bin j', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Starting Bin i', fontsize=13, fontweight='bold')

    # Adjust subplot spacing to make room for colourbar
    fig.subplots_adjust(right=0.88)

    # Add shared colourbar with proper spacing
    cbar = fig.colorbar(im.collections[0], ax=[ax1, ax2],
                        location='right', shrink=0.6, pad=0.05)
    cbar.set_label('Probability', fontsize=13, fontweight='bold', labelpad=15)

    plt.suptitle('Conditional Probability Matrices: Ground Truth vs LLM Estimates',
                 fontsize=18, fontweight='bold', y=1.02)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved heatmap comparison to {output_path}")

    if show:
        plt.show()

    return fig


def plot_expert_distribution_boxes(df_consensus: pd.DataFrame, output_path: Optional[Path] = None, show: bool = False) -> plt.Figure:
    """
    Plot 4: Box plots showing MC-aggregated distribution statistics per bin pair.
    
    Uses the percentile data from Monte Carlo aggregation to show distribution properties.

    Args:
        df_consensus: Consensus DataFrame with percentile columns
        output_path: Optional path to save figure
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    # Sort by bin pair for consistent ordering
    df_plot = df_consensus.sort_values(['bin_i', 'bin_j']).copy()
    df_plot['pair_label'] = df_plot.apply(lambda r: f"({int(r['bin_i'])},{int(r['bin_j'])})", axis=1)
    
    fig, ax = plt.subplots(figsize=(14, 8))

    positions = range(len(df_plot))
    
    # Create box plot data structure from percentiles
    # box plot expects: [min, q1, median, q3, max] or similar
    box_data = []
    for _, row in df_plot.iterrows():
        # Use the MC percentiles: p025, p25, p50, p75, p975
        box_data.append({
            'med': row['p50'],
            'q1': row['p25'],
            'q3': row['p75'],
            'whislo': row['p025'],
            'whishi': row['p975'],
            'fliers': []
        })
    
    # Create box plots manually using bxp
    bp = ax.bxp(box_data, positions=positions, widths=0.6, showfliers=False,
                patch_artist=True, manage_ticks=False)
    
    # Colour boxes
    for patch in bp['boxes']:
        patch.set_facecolor('#8E44AD')
        patch.set_alpha(0.6)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    # Style medians, whiskers, caps
    for element in ['medians', 'whiskers', 'caps']:
        for line in bp[element]:
            line.set_color('black')
            line.set_linewidth(1.5)
    
    # Overlay ground truth markers
    ax.scatter(positions, df_plot['ground_truth_p_j_given_i'].values,
              marker='D', s=150, color='#E74C3C', edgecolors='black',
              linewidths=2, label='Ground Truth', zorder=10)
    
    # Overlay mode markers (from KDE)
    ax.scatter(positions, df_plot['mode'].values,
              marker='*', s=200, color='#F39C12', edgecolors='black',
              linewidths=1, label='Mode (KDE)', zorder=9)

    # Labels and formatting
    ax.set_xlabel('Bin Pair (i,j)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability P(j|i)', fontsize=14, fontweight='bold')
    ax.set_title('MC-Aggregated Distributions per Bin Pair (from Fitted Beta Samples)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(positions)
    ax.set_xticklabels(df_plot['pair_label'], rotation=45, ha='right')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved distribution box plot to {output_path}")

    if show:
        plt.show()

    return fig


def plot_expert_calibration_grid(df_fitted: pd.DataFrame, output_path: Optional[Path] = None, show: bool = False) -> plt.Figure:
    """
    Plot calibration scatter plots for each expert on a grid.

    Args:
        df_fitted: DataFrame from detailed_estimates_fitted.csv with per-expert data
        output_path: Optional path to save figure
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    # Get unique experts
    experts = sorted(df_fitted['expert'].unique())
    n_experts = len(experts)
    
    if n_experts == 0:
        logger.warning("No experts found in data")
        return None
    
    # Calculate grid dimensions (as square as possible)
    n_cols = math.ceil(math.sqrt(n_experts))
    n_rows = math.ceil(n_experts / n_cols)
    
    logger.info(f"Creating {n_rows}x{n_cols} grid for {n_experts} experts")
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    
    # Flatten axes for easy iteration
    axes_flat = axes.flatten()
    
    # Calculate global jump size range for consistent colouring
    all_jump_sizes = df_fitted['bin_j'] - df_fitted['bin_i']
    vmin_jump = all_jump_sizes.min()
    vmax_jump = all_jump_sizes.max()
    
    for idx, expert in enumerate(experts):
        ax = axes_flat[idx]
        df_expert = df_fitted[df_fitted['expert'] == expert].copy()
        
        # Compute jump size for colouring
        df_expert['jump_size'] = df_expert['bin_j'] - df_expert['bin_i']
        
        # Scatter plot with colour by jump size
        ax.scatter(
            df_expert['ground_truth_p_j_given_i'],
            df_expert['mean'],
            c=df_expert['jump_size'],
            s=100,
            alpha=0.7,
            cmap='viridis',
            vmin=vmin_jump,
            vmax=vmax_jump,
            edgecolors='black',
            linewidths=1
        )
        
        # Add error bars (25th-75th percentile from low_ci and high_ci)
        ax.errorbar(
            df_expert['ground_truth_p_j_given_i'],
            df_expert['mean'],
            yerr=[df_expert['mean'] - df_expert['low_ci'], df_expert['high_ci'] - df_expert['mean']],
            fmt='none',
            ecolor='gray',
            alpha=0.4,
            capsize=3,
            capthick=1
        )
        
        # Diagonal reference line (perfect calibration)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1.5)
        
        # Formatting
        ax.set_xlabel('Ground Truth P(j|i)', fontsize=10)
        ax.set_ylabel('LLM Estimate P(j|i)', fontsize=10)
        ax.set_title(f'{expert}', fontsize=12, fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Hide unused subplots
    for idx in range(n_experts, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    # Add a shared colourbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin_jump, vmax=vmax_jump))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location='right', shrink=0.6, pad=0.02)
    cbar.set_label('Bin Jump Size |j-i|', fontsize=12)
    
    plt.suptitle('Per-Expert Calibration: LLM Estimates vs Ground Truth', 
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved expert calibration grid to {output_path}")
    
    if show:
        plt.show()
    
    return fig


def compute_statistics(df: pd.DataFrame) -> Dict:
    """
    Compute statistical metrics comparing LLM estimates to ground truth.

    Args:
        df: DataFrame with calibration data

    Returns:
        Dictionary with statistical metrics
    """
    gt = df['ground_truth_p_j_given_i'].values
    llm = df['mean'].values

    # Remove any NaN values
    mask = ~(np.isnan(gt) | np.isnan(llm))
    gt = gt[mask]
    llm = llm[mask]

    if len(gt) < 2:
        logger.warning("Insufficient data for statistical analysis")
        return {}

    # Compute metrics
    mae = np.mean(np.abs(gt - llm))
    rmse = np.sqrt(np.mean((gt - llm) ** 2))

    # Correlation metrics
    spearman_corr, spearman_p = spearmanr(gt, llm)
    kendall_corr, kendall_p = kendalltau(gt, llm)

    # Bias (systematic over/under-estimation)
    bias = np.mean(llm - gt)

    stats = {
        'n_pairs': len(gt),
        'mae': mae,
        'rmse': rmse,
        'bias': bias,
        'spearman_r': spearman_corr,
        'spearman_p': spearman_p,
        'kendall_tau': kendall_corr,
        'kendall_p': kendall_p
    }

    return stats


def print_statistics_table(stats: Dict, output_path: Optional[Path] = None):
    """
    Print and optionally save statistics table.

    Args:
        stats: Dictionary with statistical metrics
        output_path: Optional path to save statistics as text file
    """
    if not stats:
        logger.warning("No statistics to print")
        return

    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("CALIBRATION STATISTICS: LLM ESTIMATES vs GROUND TRUTH")
    output_lines.append("(Using Monte Carlo Beta Aggregation)")
    output_lines.append("=" * 70)
    output_lines.append("")
    output_lines.append(f"Number of bin pairs analysed:        {stats['n_pairs']}")
    output_lines.append("")
    output_lines.append("--- Error Metrics ---")
    output_lines.append(f"Mean Absolute Error (MAE):           {stats['mae']:.4f}")
    output_lines.append(f"Root Mean Squared Error (RMSE):      {stats['rmse']:.4f}")
    output_lines.append(f"Bias (LLM - GT, mean):               {stats['bias']:+.4f}")
    output_lines.append("")
    output_lines.append("--- Correlation Metrics ---")
    output_lines.append(f"Spearman's ρ:                        {stats['spearman_r']:.4f}  (p={stats['spearman_p']:.4f})")
    output_lines.append(f"Kendall's τ:                         {stats['kendall_tau']:.4f}  (p={stats['kendall_p']:.4f})")
    output_lines.append("")
    output_lines.append("=" * 70)
    output_lines.append("")
    output_lines.append("Interpretation:")
    output_lines.append("  - MAE/RMSE: Lower values indicate better calibration (0 = perfect)")
    output_lines.append("  - Bias: Positive = overestimation, Negative = underestimation")
    output_lines.append("  - Spearman/Kendall: Rank correlation (1 = perfect, 0 = no correlation)")
    output_lines.append("=" * 70)

    # Print to console
    for line in output_lines:
        print(line)

    # Save to file if requested
    if output_path:
        with open(output_path, 'w') as f:
            f.write('\n'.join(output_lines))
        logger.info(f"Saved statistics to {output_path}")


def get_latest_run_path(results_root: Path, benchmark_name: str) -> Optional[Path]:
    """
    Find the most recent run directory for a given benchmark.

    Args:
        results_root: Results root directory (intra_benchmark_calibration/results)
        benchmark_name: Name of benchmark (e.g., 'swebench_verified')

    Returns:
        Path to most recent run directory or None if not found
    """
    runs_dir = results_root / benchmark_name

    if not runs_dir.is_dir():
        logger.error(f"Runs directory not found: {runs_dir}")
        return None

    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        logger.error(f"No run directories found in {runs_dir}")
        return None

    # Sort by name (assuming YYYYMMDD_HHMMSS format)
    latest_run = sorted(run_dirs, key=lambda p: p.name)[-1]
    logger.info(f"Found latest run: {latest_run}")

    return latest_run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate calibration plots and statistics for intra-benchmark experiments (MC aggregation version)."
    )
    parser.add_argument(
        "--consensus_csv", "-c",
        type=Path,
        help="Path to detailed_estimates_fitted_consensus.csv file. If not provided, uses latest run."
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=Path,
        help="Directory to save plots. Defaults to run_dir/plots/"
    )
    parser.add_argument(
        "--benchmark", "-b",
        type=str,
        default="swebench_verified",
        help="Benchmark name (default: swebench_verified)"
    )
    parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="Display plots interactively"
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save plots to disk"
    )

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).resolve().parent
    results_root = script_dir / "results"

    if args.consensus_csv:
        consensus_csv_path = args.consensus_csv.resolve()
        run_dir = consensus_csv_path.parent
    else:
        logger.info("No consensus CSV specified, searching for latest run...")
        run_dir = get_latest_run_path(results_root, args.benchmark)
        if not run_dir:
            logger.error("Could not find any runs")
            exit(1)
        consensus_csv_path = run_dir / "detailed_estimates_fitted_consensus.csv"

    if not consensus_csv_path.is_file():
        logger.error(f"Consensus CSV not found: {consensus_csv_path}")
        exit(1)

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        output_dir = run_dir / "plots"

    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

    # Load data
    df_consensus = load_consensus_csv(consensus_csv_path)
    if df_consensus is None:
        logger.error("Failed to load consensus data")
        exit(1)

    # Extract run_id from directory name
    run_id = run_dir.name
    logger.info(f"Processing run: {run_id}")

    # Generate plots
    print("\nGenerating plots...\n")

    # Plot 1: Calibration scatter
    plot_path_1 = output_dir / f"{run_id}_calibration_scatter.png" if not args.no_save else None
    fig1 = plot_calibration_scatter(df_consensus, output_path=plot_path_1, show=args.show)

    # Plot 2: Sequential transitions
    plot_path_2 = output_dir / f"{run_id}_sequential_transitions.png" if not args.no_save else None
    fig2 = plot_sequential_transitions(df_consensus, output_path=plot_path_2, show=args.show)

    # Plot 3: Heatmap comparison
    plot_path_3 = output_dir / f"{run_id}_heatmap_comparison.png" if not args.no_save else None
    fig3 = plot_heatmap_comparison(df_consensus, output_path=plot_path_3, show=args.show)

    # Plot 4: Distribution visualization (box plots from MC percentiles)
    plot_path_4 = output_dir / f"{run_id}_distributions.png" if not args.no_save else None
    fig4 = plot_expert_distribution_boxes(df_consensus, output_path=plot_path_4, show=args.show)

    # Plot 5: Per-expert calibration grid
    #fitted_csv_path = run_dir / "detailed_estimates_fitted.csv"
    fitted_csv_path = run_dir / "20251125_022240_swebench_verified_nbins10_claude-sonnet-4-5-20250929_nexp10_nrnd1_tmp08_estimates_fitted.csv"
    if fitted_csv_path.is_file():
        df_fitted = load_fitted_csv(fitted_csv_path)
        if df_fitted is not None:
            plot_path_5 = output_dir / f"{run_id}_calibration_scatter_per_expert.png" if not args.no_save else None
            fig5 = plot_expert_calibration_grid(df_fitted, output_path=plot_path_5, show=args.show)
    else:
        logger.warning(f"Fitted CSV not found: {fitted_csv_path} - skipping per-expert plots")

    # Compute and print statistics
    print("\nComputing statistical metrics...\n")
    stats = compute_statistics(df_consensus)
    stats_path = output_dir / f"{run_id}_calibration_statistics.txt" if not args.no_save else None
    print_statistics_table(stats, output_path=stats_path)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    if not args.no_save:
        print(f"All outputs saved to: {output_dir}")
    print(f"{'='*70}\n")

    plt.close('all')
