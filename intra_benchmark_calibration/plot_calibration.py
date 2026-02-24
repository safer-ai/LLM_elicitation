#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging
import argparse
import json
from typing import Optional, Dict
from scipy.stats import spearmanr, kendalltau

# Configure logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')


def load_intra_benchmark_results(json_path: Path) -> Optional[Dict]:
    """
    Load intra-benchmark results from full_results.json.

    Args:
        json_path: Path to full_results.json file

    Returns:
        Dictionary with run data or None if loading fails
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded results from {json_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON file {json_path}: {e}")
        return None


def extract_calibration_data(results: Dict) -> pd.DataFrame:
    """
    Extract calibration data from results into a DataFrame.

    Args:
        results: Full results dictionary

    Returns:
        DataFrame with columns: bin_i, bin_j, jump_size, ground_truth, llm_mean, llm_std, llm_median, sufficient_sample
    """
    predictions = results.get('predictions', [])

    data_rows = []
    for pred in predictions:
        row = {
            'bin_i': pred['bin_i'],
            'bin_j': pred['bin_j'],
            'jump_size': pred['bin_j'] - pred['bin_i'],
            'ground_truth': pred.get('ground_truth_p_j_given_i'),
            'llm_mean': pred.get('final_aggregated_probability'),
            'llm_std': pred.get('final_std_dev'),
            'llm_median': pred.get('final_median'),
            'bin_i_range': pred.get('bin_i_range', ''),
            'bin_j_range': pred.get('bin_j_range', ''),
            'n_reaching_i': pred.get('ground_truth_n_reaching_i'),
            'sufficient_sample': pred.get('ground_truth_n_reaching_i', 0) >= 5  # Assuming min_sample_size=5
        }
        data_rows.append(row)

    df = pd.DataFrame(data_rows)
    logger.info(f"Extracted {len(df)} predictions")
    return df


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

    # Create scatter plot with color by jump size
    scatter = ax.scatter(
        df['ground_truth'],
        df['llm_mean'],
        c=df['jump_size'],
        s=150,
        alpha=0.7,
        cmap='viridis',
        edgecolors='black',
        linewidths=1.5
    )

    # Add error bars (±1 std)
    ax.errorbar(
        df['ground_truth'],
        df['llm_mean'],
        yerr=df['llm_std'],
        fmt='none',
        ecolor='gray',
        alpha=0.5,
        capsize=5,
        capthick=2
    )

    # Add diagonal reference line (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=2, label='Perfect Calibration (y=x)')

    # Add colorbar for jump size
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
    # Filter for consecutive bins only (j = i+1)
    df_seq = df[df['jump_size'] == 1].sort_values('bin_i')

    if df_seq.empty:
        logger.warning("No consecutive bin transitions found in data")
        return None

    fig, ax = plt.subplots(figsize=(12, 7))

    x = df_seq['bin_i'].values
    gt = df_seq['ground_truth'].values
    llm = df_seq['llm_mean'].values
    llm_std = df_seq['llm_std'].values
    sufficient = df_seq['sufficient_sample'].values

    # Plot ground truth line
    ax.plot(x, gt, 'o-', color='#2E86AB', linewidth=2.5, markersize=10,
            label='Ground Truth', markeredgecolor='black', markeredgewidth=1.5)

    # Plot LLM estimate line
    ax.plot(x, llm, 's--', color='#A23B72', linewidth=2.5, markersize=10,
            label='LLM Estimate', markeredgecolor='black', markeredgewidth=1.5)

    # Add shaded region for LLM ±1 std
    ax.fill_between(x, llm - llm_std, llm + llm_std,
                     color='#A23B72', alpha=0.2, label='LLM ±1 std')

    # Mark insufficient samples with different markers
    insufficient_mask = ~sufficient
    if insufficient_mask.any():
        ax.scatter(x[insufficient_mask], llm[insufficient_mask],
                  marker='x', s=200, color='red', linewidths=3,
                  label='Insufficient Sample (n<5)', zorder=10)

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

    # Initialize matrices with NaN
    gt_matrix = np.full((n_bins, n_bins), np.nan)
    llm_matrix = np.full((n_bins, n_bins), np.nan)

    # Fill matrices (upper triangle only, j > i)
    for _, row in df.iterrows():
        i, j = int(row['bin_i']), int(row['bin_j'])
        gt_matrix[i, j] = row['ground_truth']
        llm_matrix[i, j] = row['llm_mean']

    # Create side-by-side heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Shared colormap and range
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

    # Adjust subplot spacing to make room for colorbar
    fig.subplots_adjust(right=0.88)

    # Add shared colorbar with proper spacing
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


def plot_violin_plots(results: Dict, output_path: Optional[Path] = None, show: bool = False) -> plt.Figure:
    """
    Plot violin plots showing distribution of expert estimates per bin pair.

    Args:
        results: Full results dictionary
        output_path: Optional path to save figure
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    predictions = results.get('predictions', [])

    # Extract individual expert estimates from final round
    data_rows = []
    for pred in predictions:
        pair_label = f"({pred['bin_i']},{pred['bin_j']})"
        ground_truth = pred.get('ground_truth_p_j_given_i')

        # Get final round estimates
        delphi_rounds = pred.get('delphi_rounds', [])
        if delphi_rounds:
            final_round = delphi_rounds[-1]
            expert_estimates = final_round.get('expert_estimates', [])

            for expert_data in expert_estimates:
                if 'error' not in expert_data and expert_data.get('estimate') is not None:
                    data_rows.append({
                        'pair': pair_label,
                        'estimate': expert_data['estimate'],
                        'ground_truth': ground_truth,
                        'bin_i': pred['bin_i'],
                        'bin_j': pred['bin_j']
                    })

    df_violin = pd.DataFrame(data_rows)

    if df_violin.empty:
        logger.warning("No expert estimates found for violin plot")
        return None

    # Sort by bin pair for consistent ordering
    df_violin = df_violin.sort_values(['bin_i', 'bin_j'])
    unique_pairs = df_violin.groupby(['pair', 'ground_truth']).size().reset_index()[['pair', 'ground_truth']]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Create violin plot
    parts = ax.violinplot(
        [df_violin[df_violin['pair'] == pair]['estimate'].values
         for pair in unique_pairs['pair']],
        positions=range(len(unique_pairs)),
        widths=0.7,
        showmeans=True,
        showextrema=True
    )

    # Color violins
    for pc in parts['bodies']:
        pc.set_facecolor('#8E44AD')
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)

    # Overlay ground truth markers
    ax.scatter(range(len(unique_pairs)), unique_pairs['ground_truth'].values,
              marker='D', s=150, color='#E74C3C', edgecolors='black',
              linewidths=2, label='Ground Truth', zorder=10)

    # Labels and formatting
    ax.set_xlabel('Bin Pair (i,j)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability P(j|i)', fontsize=14, fontweight='bold')
    ax.set_title('Distribution of Expert Estimates per Bin Pair (Final Round)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(len(unique_pairs)))
    ax.set_xticklabels(unique_pairs['pair'], rotation=45, ha='right')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved violin plot to {output_path}")

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
    gt = df['ground_truth'].values
    llm = df['llm_mean'].values

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
    output_lines.append("=" * 70)
    output_lines.append("")
    output_lines.append(f"Number of bin pairs analyzed:        {stats['n_pairs']}")
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


def get_latest_run_path(project_root: Path, benchmark_name: str = 'cybench') -> Optional[Path]:
    """
    Find the most recent run directory for a given benchmark.

    Args:
        project_root: Project root directory
        benchmark_name: Name of benchmark (e.g., 'cybench')

    Returns:
        Path to most recent run directory or None if not found
    """
    runs_dir = project_root / "output_data" / "intra_benchmark" / benchmark_name

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
        description="Generate calibration plots and statistics for intra-benchmark experiments."
    )
    parser.add_argument(
        "--json_file", "-j",
        type=Path,
        help="Path to full_results.json file. If not provided, uses latest run."
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=Path,
        help="Directory to save plots. Defaults to run_dir/plots/"
    )
    parser.add_argument(
        "--benchmark", "-b",
        type=str,
        default="cybench",
        help="Benchmark name (default: cybench)"
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
    project_root = Path(__file__).resolve().parent.parent.parent

    if args.json_file:
        json_path = args.json_file.resolve()
        run_dir = json_path.parent
    else:
        logger.info("No JSON file specified, searching for latest run...")
        run_dir = get_latest_run_path(project_root, args.benchmark)
        if not run_dir:
            logger.error("Could not find any runs")
            exit(1)
        json_path = run_dir / "full_results.json"

    if not json_path.is_file():
        logger.error(f"JSON file not found: {json_path}")
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
    results = load_intra_benchmark_results(json_path)
    if not results:
        logger.error("Failed to load results")
        exit(1)

    run_id = results['run_metadata']['run_id']
    logger.info(f"Processing run: {run_id}")

    # Extract calibration data
    df = extract_calibration_data(results)

    # Generate plots
    print("\nGenerating plots...\n")

    # Plot 1: Calibration scatter
    plot_path_1 = output_dir / f"{run_id}_calibration_scatter.png" if not args.no_save else None
    fig1 = plot_calibration_scatter(df, output_path=plot_path_1, show=args.show)

    # Plot 2: Sequential transitions
    plot_path_2 = output_dir / f"{run_id}_sequential_transitions.png" if not args.no_save else None
    fig2 = plot_sequential_transitions(df, output_path=plot_path_2, show=args.show)

    # Plot 3: Heatmap comparison
    plot_path_3 = output_dir / f"{run_id}_heatmap_comparison.png" if not args.no_save else None
    fig3 = plot_heatmap_comparison(df, output_path=plot_path_3, show=args.show)

    # Plot 4: Violin plots
    plot_path_4 = output_dir / f"{run_id}_expert_distributions.png" if not args.no_save else None
    fig4 = plot_violin_plots(results, output_path=plot_path_4, show=args.show)

    # Compute and print statistics
    print("\nComputing statistical metrics...\n")
    stats = compute_statistics(df)
    stats_path = output_dir / f"{run_id}_calibration_statistics.txt" if not args.no_save else None
    print_statistics_table(stats, output_path=stats_path)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    if not args.no_save:
        print(f"All outputs saved to: {output_dir}")
    print(f"{'='*70}\n")

    plt.close('all')
