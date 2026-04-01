#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plotting and statistics for inter-benchmark calibration experiments.

Generates: calibration scatter, transfer curves per source bin,
heatmap (source bins x target percentiles), expert violin plots, and statistics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging
import argparse
import json
from typing import Optional, Dict, List
from scipy.stats import spearmanr, kendalltau

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')


def _pair_label(results: Dict) -> str:
    """Build a 'source → target' label from run metadata."""
    meta = results.get('run_metadata', {})
    src = meta.get('source_benchmark', '')
    tgt = meta.get('target_benchmark', '')
    if src and tgt:
        return f"{src} → {tgt}"
    return ''


def load_results(json_path: Path) -> Optional[Dict]:
    """Load inter-benchmark results from JSON."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded results from {json_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON: {e}")
        return None


def extract_calibration_data(results: Dict) -> pd.DataFrame:
    """Extract calibration data into a DataFrame."""
    predictions = results.get('predictions', [])
    rows = []
    for pred in predictions:
        rows.append({
            'source_bin': pred['source_bin'],
            'source_bin_range_str': pred.get('source_bin_range_str', ''),
            'target_percentile': pred['target_percentile'],
            'target_task_name': pred.get('target_task_name', ''),
            'ground_truth': pred.get('ground_truth_p_solve'),
            'llm_mean': pred.get('final_aggregated_probability'),
            'llm_std': pred.get('final_std_dev'),
            'llm_median': pred.get('final_median'),
            'n_in_source_bin': pred.get('n_in_source_bin', 0),
        })
    df = pd.DataFrame(rows)
    logger.info(f"Extracted {len(df)} predictions")
    return df


def plot_calibration_scatter(df: pd.DataFrame, output_path: Optional[Path] = None, show: bool = False, benchmark_pair: str = '') -> plt.Figure:
    """Calibration scatter: predicted P(solve) vs ground truth, coloured by source bin."""
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        df['ground_truth'], df['llm_mean'],
        c=df['source_bin'], s=150, alpha=0.7,
        cmap='viridis', edgecolors='black', linewidths=1.5
    )

    if df['llm_std'].notna().any():
        ax.errorbar(
            df['ground_truth'], df['llm_mean'],
            yerr=df['llm_std'].fillna(0),
            fmt='none', ecolor='gray', alpha=0.5, capsize=5, capthick=2
        )

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=2, label='Perfect Calibration')
    cbar = plt.colorbar(scatter, ax=ax, label='Source Bin')
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel('Ground Truth P(solve)', fontsize=14, fontweight='bold')
    ax.set_ylabel('LLM Estimate P(solve)', fontsize=14, fontweight='bold')
    title = 'Inter-Benchmark Calibration: LLM vs Ground Truth'
    if benchmark_pair:
        title += f'\n({benchmark_pair})'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved calibration scatter to {output_path}")
    if show:
        plt.show()
    return fig


def plot_transfer_curves(df: pd.DataFrame, output_path: Optional[Path] = None, show: bool = False, benchmark_pair: str = '') -> plt.Figure:
    """Transfer curves: P(solve) vs target percentile, one line per source bin."""
    source_bins = sorted(df['source_bin'].unique())
    n_bins = len(source_bins)

    fig, ax = plt.subplots(figsize=(12, 7))
    colours = plt.cm.viridis(np.linspace(0, 0.9, n_bins))

    for i, sb in enumerate(source_bins):
        sub = df[df['source_bin'] == sb].sort_values('target_percentile')
        label_str = sub['source_bin_range_str'].iloc[0] if len(sub) > 0 else f"Bin {sb}"

        ax.plot(sub['target_percentile'], sub['ground_truth'],
                'o-', color=colours[i], linewidth=2, markersize=8,
                label=f'{label_str} (GT)', alpha=0.8)
        ax.plot(sub['target_percentile'], sub['llm_mean'],
                's--', color=colours[i], linewidth=2, markersize=8,
                label=f'{label_str} (LLM)', alpha=0.6)

        if sub['llm_std'].notna().any():
            ax.fill_between(
                sub['target_percentile'],
                (sub['llm_mean'] - sub['llm_std']).clip(0, 1),
                (sub['llm_mean'] + sub['llm_std']).clip(0, 1),
                color=colours[i], alpha=0.15
            )

    ax.set_xlabel('Target Task Percentile', fontsize=14, fontweight='bold')
    ax.set_ylabel('P(solve)', fontsize=14, fontweight='bold')
    title = 'Transfer Curves: P(solve target) vs Target Difficulty per Source Bin'
    if benchmark_pair:
        title += f'\n({benchmark_pair})'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=9, loc='best', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved transfer curves to {output_path}")
    if show:
        plt.show()
    return fig


def plot_heatmap_comparison(df: pd.DataFrame, output_path: Optional[Path] = None, show: bool = False, benchmark_pair: str = '') -> plt.Figure:
    """Side-by-side heatmaps: source bins (rows) x target percentiles (cols)."""
    source_bins = sorted(df['source_bin'].unique())
    target_pcts = sorted(df['target_percentile'].unique())

    gt_matrix = np.full((len(source_bins), len(target_pcts)), np.nan)
    llm_matrix = np.full((len(source_bins), len(target_pcts)), np.nan)

    sb_map = {sb: i for i, sb in enumerate(source_bins)}
    tp_map = {tp: j for j, tp in enumerate(target_pcts)}

    for _, row in df.iterrows():
        i = sb_map.get(row['source_bin'])
        j = tp_map.get(row['target_percentile'])
        if i is not None and j is not None:
            gt_matrix[i, j] = row['ground_truth']
            llm_matrix[i, j] = row['llm_mean']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    cmap = 'RdYlGn'

    bin_labels = [df[df['source_bin'] == sb]['source_bin_range_str'].iloc[0]
                  if len(df[df['source_bin'] == sb]) > 0 else f"Bin {sb}"
                  for sb in source_bins]
    pct_labels = [f"{p}%" for p in target_pcts]

    sns.heatmap(gt_matrix, annot=True, fmt='.2f', cmap=cmap, vmin=0, vmax=1,
                cbar=False, ax=ax1, linewidths=1, linecolor='gray',
                xticklabels=pct_labels, yticklabels=bin_labels,
                annot_kws={'size': 11, 'weight': 'bold'})
    ax1.set_title('Ground Truth P(solve)', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('Target Percentile', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Source Bin', fontsize=13, fontweight='bold')

    im = sns.heatmap(llm_matrix, annot=True, fmt='.2f', cmap=cmap, vmin=0, vmax=1,
                     cbar=False, ax=ax2, linewidths=1, linecolor='gray',
                     xticklabels=pct_labels, yticklabels=bin_labels,
                     annot_kws={'size': 11, 'weight': 'bold'})
    ax2.set_title('LLM Estimate P(solve)', fontsize=16, fontweight='bold', pad=15)
    ax2.set_xlabel('Target Percentile', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Source Bin', fontsize=13, fontweight='bold')

    fig.subplots_adjust(right=0.88)
    cbar = fig.colorbar(im.collections[0], ax=[ax1, ax2],
                        location='right', shrink=0.6, pad=0.05)
    cbar.set_label('Probability', fontsize=13, fontweight='bold', labelpad=15)

    suptitle = 'Inter-Benchmark: Ground Truth vs LLM Estimates'
    if benchmark_pair:
        suptitle += f'\n({benchmark_pair})'
    plt.suptitle(suptitle, fontsize=18, fontweight='bold', y=1.02)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved heatmap to {output_path}")
    if show:
        plt.show()
    return fig


def plot_violin_plots(results: Dict, output_path: Optional[Path] = None, show: bool = False, benchmark_pair: str = '') -> plt.Figure:
    """Violin plots: expert estimate distributions per prediction, with GT diamond."""
    predictions = results.get('predictions', [])
    data_rows = []

    for pred in predictions:
        pair_label = f"Bin{pred['source_bin']}->{pred['target_percentile']}%"
        gt = pred.get('ground_truth_p_solve')

        delphi_rounds = pred.get('delphi_rounds', [])
        if delphi_rounds:
            final_round = delphi_rounds[-1]
            for expert_data in final_round.get('expert_estimates', []):
                if 'error' not in expert_data and expert_data.get('estimate') is not None:
                    data_rows.append({
                        'pair': pair_label,
                        'estimate': expert_data['estimate'],
                        'ground_truth': gt,
                        'source_bin': pred['source_bin'],
                        'target_percentile': pred['target_percentile']
                    })

    df_violin = pd.DataFrame(data_rows)
    if df_violin.empty:
        logger.warning("No expert estimates for violin plot")
        return None

    df_violin = df_violin.sort_values(['source_bin', 'target_percentile'])
    unique_pairs = df_violin.groupby(['pair', 'ground_truth']).size().reset_index()[['pair', 'ground_truth']]

    fig, ax = plt.subplots(figsize=(max(14, len(unique_pairs) * 1.5), 8))

    parts = ax.violinplot(
        [df_violin[df_violin['pair'] == pair]['estimate'].values for pair in unique_pairs['pair']],
        positions=range(len(unique_pairs)),
        widths=0.7, showmeans=True, showextrema=True
    )

    for pc in parts['bodies']:
        pc.set_facecolor('#8E44AD')
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)

    ax.scatter(range(len(unique_pairs)), unique_pairs['ground_truth'].values,
              marker='D', s=150, color='#E74C3C', edgecolors='black',
              linewidths=2, label='Ground Truth', zorder=10)

    ax.set_xlabel('Prediction (Source Bin -> Target Percentile)', fontsize=14, fontweight='bold')
    ax.set_ylabel('P(solve)', fontsize=14, fontweight='bold')
    title = 'Expert Estimate Distributions (Final Round)'
    if benchmark_pair:
        title += f'\n({benchmark_pair})'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
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


def plot_source_target_solve_rates(results: Dict, output_path: Optional[Path] = None, show: bool = False, benchmark_pair: str = '') -> Optional[plt.Figure]:
    """
    Scatter plot of P(solve source task) [x] vs P(solve target task) [y].

    Two series per point:
      - Empirical ground truth (no y error bar)
      - LLM estimate (y error bar = std across experts)

    X error bar = std across the representative source tasks shown to the LLM.
    For old results files lacking 'source_task_solve_rates', falls back to
    source_bin_range midpoint with half-range as x error bar.
    """
    predictions = results.get('predictions', [])
    if not predictions:
        logger.warning("No predictions for source-target solve rate plot")
        return None

    n_preds = len(predictions)
    colours = plt.cm.tab20(np.linspace(0, 1, max(n_preds, 1)))

    using_fallback = not any('source_task_solve_rates' in p for p in predictions)

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, pred in enumerate(predictions):
        colour = colours[i]
        gt_p = pred.get('ground_truth_p_solve')
        llm_p = pred.get('final_aggregated_probability')
        llm_std = pred.get('final_std_dev') or 0.0

        src_rates = pred.get('source_task_solve_rates')
        if src_rates:
            rates = [t['solve_rate'] for t in src_rates]
            x_mean = float(np.mean(rates))
            x_err = float(np.std(rates)) if len(rates) > 1 else 0.0
        else:
            lo, hi = pred['source_bin_range'][0], pred['source_bin_range'][1]
            x_mean = (lo + hi) / 2.0
            x_err = (hi - lo) / 2.0

        tgt_pct = pred.get('target_percentile', '')
        pair_label = f"bin{pred['source_bin']} → {tgt_pct}%"

        if gt_p is not None:
            ax.errorbar(
                x_mean, gt_p,
                xerr=x_err if x_err > 0 else None,
                fmt='o', color=colour, markersize=9,
                ecolor=colour, elinewidth=1.5, capsize=4, capthick=1.5,
                alpha=0.9, markeredgecolor='black', markeredgewidth=0.8
            )

        if llm_p is not None:
            ax.errorbar(
                x_mean, llm_p,
                xerr=x_err if x_err > 0 else None,
                yerr=llm_std if llm_std > 0 else None,
                fmt='s', color=colour, markersize=9,
                ecolor=colour, elinewidth=1.5, capsize=4, capthick=1.5,
                alpha=0.55, markeredgecolor='black', markeredgewidth=0.8
            )

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.35, linewidth=1.5)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='gray', linestyle='None',
               markersize=9, markeredgecolor='black', label='Empirical solve rate'),
        Line2D([0], [0], marker='s', color='gray', linestyle='None',
               markersize=9, markeredgecolor='black', alpha=0.6, label='LLM estimate (±1 std, experts)'),
        Line2D([0], [0], linestyle='--', color='black', alpha=0.5, label='y = x'),
    ], fontsize=10, loc='upper left', framealpha=0.9)

    x_label = 'P(solve source task) — bin midpoint ± half-range (approx.)' if using_fallback \
        else 'P(solve source task) — mean ± std of representative tasks'
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('P(solve target task)', fontsize=12, fontweight='bold')
    title = 'Source vs Target Solve Rate: Empirical and LLM Estimate'
    if benchmark_pair:
        title += f'\n({benchmark_pair})'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved source-target solve rate plot to {output_path}")
    if show:
        plt.show()
    return fig


def compute_statistics(df: pd.DataFrame) -> Dict:
    """Compute MAE, RMSE, bias, Spearman, Kendall."""
    gt = df['ground_truth'].values
    llm = df['llm_mean'].values
    mask = ~(np.isnan(gt) | np.isnan(llm))
    gt, llm = gt[mask], llm[mask]

    if len(gt) < 2:
        logger.warning("Insufficient data for statistics")
        return {}

    mae = float(np.mean(np.abs(gt - llm)))
    rmse = float(np.sqrt(np.mean((gt - llm) ** 2)))
    bias = float(np.mean(llm - gt))
    sp_r, sp_p = spearmanr(gt, llm)
    kt_r, kt_p = kendalltau(gt, llm)

    return {
        'n_pairs': len(gt),
        'mae': mae, 'rmse': rmse, 'bias': bias,
        'spearman_r': float(sp_r), 'spearman_p': float(sp_p),
        'kendall_tau': float(kt_r), 'kendall_p': float(kt_p)
    }


def print_statistics_table(stats: Dict, output_path: Optional[Path] = None):
    """Print and optionally save statistics."""
    if not stats:
        return

    lines = [
        "=" * 70,
        "INTER-BENCHMARK CALIBRATION STATISTICS",
        "=" * 70,
        "",
        f"Number of predictions analysed:       {stats['n_pairs']}",
        "",
        "--- Error Metrics ---",
        f"Mean Absolute Error (MAE):            {stats['mae']:.4f}",
        f"Root Mean Squared Error (RMSE):       {stats['rmse']:.4f}",
        f"Bias (LLM - GT, mean):                {stats['bias']:+.4f}",
        "",
        "--- Correlation Metrics ---",
        f"Spearman's rho:                       {stats['spearman_r']:.4f}  (p={stats['spearman_p']:.4f})",
        f"Kendall's tau:                        {stats['kendall_tau']:.4f}  (p={stats['kendall_p']:.4f})",
        "",
        "=" * 70,
    ]

    for line in lines:
        print(line)

    if output_path:
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        logger.info(f"Saved statistics to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate inter-benchmark calibration plots and statistics.")
    parser.add_argument("--json_file", "-j", type=Path, required=True, help="Path to results JSON file")
    parser.add_argument("--output_dir", "-o", type=Path, help="Output directory for plots (default: same dir as JSON /plots/)")
    parser.add_argument("--show", "-s", action="store_true", help="Display plots interactively")
    parser.add_argument("--no_save", action="store_true", help="Don't save plots")
    args = parser.parse_args()

    json_path = args.json_file.resolve()
    if not json_path.is_file():
        logger.error(f"JSON file not found: {json_path}")
        exit(1)

    output_dir = args.output_dir.resolve() if args.output_dir else json_path.parent / "plots"
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(json_path)
    if not results:
        exit(1)

    run_id = results['run_metadata']['run_id']
    pair = _pair_label(results)
    df = extract_calibration_data(results)

    print("\nGenerating plots...\n")

    p1 = output_dir / f"{run_id}_calibration_scatter.png" if not args.no_save else None
    plot_calibration_scatter(df, output_path=p1, show=args.show, benchmark_pair=pair)

    p2 = output_dir / f"{run_id}_transfer_curves.png" if not args.no_save else None
    plot_transfer_curves(df, output_path=p2, show=args.show, benchmark_pair=pair)

    p3 = output_dir / f"{run_id}_heatmap_comparison.png" if not args.no_save else None
    plot_heatmap_comparison(df, output_path=p3, show=args.show, benchmark_pair=pair)

    p4 = output_dir / f"{run_id}_expert_distributions.png" if not args.no_save else None
    plot_violin_plots(results, output_path=p4, show=args.show, benchmark_pair=pair)

    p5 = output_dir / f"{run_id}_source_target_solve_rates.png" if not args.no_save else None
    plot_source_target_solve_rates(results, output_path=p5, show=args.show, benchmark_pair=pair)

    print("\nComputing statistics...\n")
    stats = compute_statistics(df)
    sp = output_dir / f"{run_id}_statistics.txt" if not args.no_save else None
    print_statistics_table(stats, output_path=sp)

    if not args.no_save:
        print(f"\nAll outputs saved to: {output_dir}")

    plt.close('all')
