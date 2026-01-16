# src/plot_fct/scatter_plots.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
from pathlib import Path
import logging
import argparse # For command-line arguments
import glob # For finding files
from typing import Optional

# Configure logger for this module
logger = logging.getLogger(__name__) # Gets logger instance from root config if run as part of app
# If run directly, this basicConfig will apply only if no root config already set up.
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')


def get_latest_run_csv_path(project_root: Path) -> Optional[Path]:
    """
    Finds the 'detailed_estimates.csv' file from the most recent run directory.

    Args:
        project_root (Path): The root directory of the project.

    Returns:
        Optional[Path]: Path to the CSV file if found, else None.
    """
    runs_dir = project_root / "output_data" / "runs"
    if not runs_dir.is_dir():
        logger.error(f"Runs directory not found: {runs_dir}")
        return None

    # List all subdirectories in runs_dir (these are the run_ids)
    run_id_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not run_id_dirs:
        logger.error(f"No run directories found in {runs_dir}")
        return None

    # Sort by name (assuming YYYYMMDD_HHMMSS format for natural sort)
    latest_run_dir = sorted(run_id_dirs, key=lambda p: p.name)[-1]
    
    csv_path = latest_run_dir / "detailed_estimates.csv"
    if not csv_path.is_file():
        logger.error(f"detailed_estimates.csv not found in latest run directory: {latest_run_dir}")
        return None
    
    logger.info(f"Found latest run CSV: {csv_path}")
    return csv_path


def plot_uncertainty_bounds(
    run_id: str,
    csv_file_path: Path,
    metrics_to_plot: Optional[list] = None,
    metric_scales: Optional[list] = None,
    show_task_names: bool = False,
) -> Optional[plt.Figure]:
    """
    Generates a matplotlib Figure object with scatter subplots of 
    selected metric vs. Mean Estimate/Value with Std Dev error bars for a given run.

    Each subplot corresponds to a unique 'step_name'.
    - X-axis: Selected metric (or task rank/order if metric unavailable)
    - Y-axis: Mean of 'probability_estimate' column (which holds probabilities
              or other estimated values like number of actors) from the last 
              round across experts for each task.
    - Y-error bars: Standard deviation of 'probability_estimate' from the 
                    last round across experts.

    Args:
        run_id (str): The ID of the run to process, used for titles.
        csv_file_path (Path): The direct path to the 'detailed_estimates.csv' file.
        metrics_to_plot (Optional[list]): List of metric column names to try in order of preference.
                                         Uses the first available metric for each step.
                                         If None, auto-detects task_metric_* column.
                                         Always falls back to task rank/order.
        metric_scales (Optional[list]): List of scale types ('linear' or 'log') corresponding to metrics_to_plot.
                                       If fewer scales than metrics, remaining use 'linear'.
        show_task_names (bool): If True, displays task names next to data points.

    Returns:
        Optional[plt.Figure]: The generated matplotlib Figure, or None if plotting fails.
    """
    try:
        if not csv_file_path.is_file():
            logger.error(f"CSV file not found: {csv_file_path}")
            return None
        df = pd.read_csv(csv_file_path)
        logger.info(f"Loaded data from {csv_file_path} with {len(df)} rows for run_id: {run_id}.")
    except Exception as e:
        logger.error(f"Error loading or parsing CSV {csv_file_path}: {e}")
        return None

    # --- Data Preprocessing ---
    # Convert has_error to boolean if it's stored as string
    if df['has_error'].dtype == 'object':
        df['has_error'] = df['has_error'].map({'True': True, 'False': False, True: True, False: False})
    df_valid = df[df['has_error'] == False].copy()

    # Define the required columns for the uncertainty plot
    estimate_cols = ['most_likely_estimate', 'minimum_estimate', 'maximum_estimate', 'confidence_in_range']
    
    # Check if all required columns exist
    if not all(col in df_valid.columns for col in estimate_cols):
        logger.error(f"One or more required columns for uncertainty plotting are missing in the CSV. Expected: {estimate_cols}")
        return None

    # Convert all estimate columns to numeric
    for col in estimate_cols:
        df_valid[col] = pd.to_numeric(df_valid[col], errors='coerce')
    
    # Drop rows where the primary estimate is missing, but keep rows that might have partial range data
    df_valid.dropna(subset=['most_likely_estimate'], inplace=True)

    if df_valid.empty:
        logger.warning(f"No valid (error-free, non-NaN most_likely_estimate) data rows found for run {run_id}. Cannot generate plot.")
        return None

    df_valid['last_round'] = df_valid.groupby(['step_name', 'task_name'])['round'].transform('max')
    df_last_round = df_valid[df_valid['round'] == df_valid['last_round']].copy()

    if df_last_round.empty:
        logger.warning(f"No data found for the last round of estimations for run {run_id}. Cannot generate plot.")
        return None

    # Determine preferred metrics to try in order
    preferred_metrics = []
    if metrics_to_plot:
        # Use user-specified metrics in order, filtering out ones not in the data
        preferred_metrics = [metric for metric in metrics_to_plot if metric in df.columns]
        if not preferred_metrics:
            logger.warning(f"None of the specified metrics {metrics_to_plot} found in data columns. Will try auto-detection.")
    
    # If no user metrics or none found, auto-detect task_metric_* columns
    if not preferred_metrics:
        auto_detected = [col for col in df.columns if col.startswith('task_metric_')]
        preferred_metrics.extend(auto_detected)
    
    # Always add task_rank as final fallback
    if 'task_rank' not in preferred_metrics:
        preferred_metrics.append('task_rank')
    
    # Create scale mapping for metrics
    metric_scale_map = {}
    if metric_scales:
        # Map provided scales to metrics
        for i, metric in enumerate(preferred_metrics):
            if i < len(metric_scales):
                scale = metric_scales[i].lower()
                if scale in ['log', 'linear']:
                    metric_scale_map[metric] = scale
                else:
                    logger.warning(f"Invalid scale '{metric_scales[i]}' for metric '{metric}'. Using 'linear'.")
                    metric_scale_map[metric] = 'linear'
            else:
                metric_scale_map[metric] = 'linear'
    else:
        # Default all metrics to linear scale
        for metric in preferred_metrics:
            metric_scale_map[metric] = 'linear'
    
    # Create task rank for all data (as fallback)
    df_last_round = df_last_round.copy()
    unique_tasks = df_last_round['task_name'].unique()  # Preserves order from benchmark
    task_rank_map = {task: idx + 1 for idx, task in enumerate(unique_tasks)}
    df_last_round['task_rank'] = df_last_round['task_name'].map(task_rank_map)
    
    # Prepare data for each step separately, choosing the best available metric per step
    step_summaries = []
    unique_step_names = sorted(df_last_round['step_name'].unique())
    
    for step_name in unique_step_names:
        step_data = df_last_round[df_last_round['step_name'] == step_name].copy()
        
        # Try each preferred metric in order until we find one with valid data
        metric_col_name = 'task_rank'  # Default fallback
        use_task_rank = True
        
        for metric in preferred_metrics:
            if metric == 'task_rank':
                # Always use task_rank if we get to it (guaranteed to work)
                metric_col_name = 'task_rank'
                use_task_rank = True
                break
            else:
                # Try this metric - check if it has valid data for this step
                step_data[metric] = pd.to_numeric(step_data[metric], errors='coerce')
                valid_metric_data = step_data.dropna(subset=[metric])
                if not valid_metric_data.empty:
                    metric_col_name = metric
                    use_task_rank = False
                    step_data = valid_metric_data
                    break
        
        # Group and aggregate for this step
        step_summary = step_data.groupby(
            ['step_name', 'task_name', metric_col_name]
        ).agg(
            # Average each of the four points across all experts for a given task
            mean_most_likely=('most_likely_estimate', 'mean'),
            mean_minimum=('minimum_estimate', 'mean'),
            mean_maximum=('maximum_estimate', 'mean'),
            mean_confidence=('confidence_in_range', 'mean'),
            num_experts=('expert_name', 'nunique')
        ).reset_index()
        
        # Add metadata about which metric was used for this step
        step_summary['metric_used'] = metric_col_name
        step_summary['uses_task_rank'] = use_task_rank
        step_summary['metric_scale'] = metric_scale_map.get(metric_col_name, 'linear')
        
        if not step_summary.empty:
            step_summaries.append(step_summary)
    
    # Combine all step summaries
    if not step_summaries:
        logger.warning(f"No data available after processing all steps for run {run_id}. Cannot generate plot.")
        return None
    
    summary_df = pd.concat(step_summaries, ignore_index=True)

    if summary_df.empty:
        logger.warning(f"No data available after summarization (grouping by step, task, FST) for run {run_id}. Cannot generate plot.")
        return None

    # --- Plotting ---
    unique_step_names = sorted(summary_df['step_name'].unique())
    num_steps = len(unique_step_names)

    if num_steps == 0:
        logger.warning(f"No unique step names found in the summarized data for run {run_id}. Cannot generate plot.")
        return None

    ncols = 2 if num_steps > 1 else 1
    nrows = (num_steps + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 8, nrows * 6), squeeze=False)
    axes = axes.flatten()
    sns.set_theme(style="whitegrid")

    model_name_for_title = df['model'].iloc[0] if not df.empty and 'model' in df.columns else 'N/A'


    for i, step_name in enumerate(unique_step_names):
        ax = axes[i]
        # Filter summary_df for the current step_name
        step_data = summary_df[summary_df['step_name'] == step_name]

        if step_data.empty:
            # This means that even if the step_name was unique, there was no data for it in summary_df
            logger.info(f"No plottable data for step: {step_name} in run {run_id} (data was empty after grouping).")
            ax.set_title(f"Step: {step_name}\n(No plottable data)", fontsize=12)
            ax.text(0.5, 0.5, "No data available for this step",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=10, color='grey')
            ax.set_xlabel("Task Rank/Order", fontsize=10)
            ax.set_ylabel("Mean Estimated Value/Probability", fontsize=10) 
            continue
        
        # Get the metric information for this step
        step_metric_used = step_data['metric_used'].iloc[0]
        step_uses_task_rank = step_data['uses_task_rank'].iloc[0]
        step_metric_scale = step_data['metric_scale'].iloc[0]
        
        # Normalize confidence (assuming 0-1 range) to an alpha value (e.g., 0.3-1.0)
        # This makes points with higher confidence more opaque.
        min_alpha, max_alpha = 0.3, 1.0
        # Fill missing confidence values with a default (e.g., 0.75) for plotting
        confidences = step_data['mean_confidence'].fillna(0.75)
        alphas = min_alpha + confidences * (max_alpha - min_alpha)

        # Plot each point individually to assign it a unique alpha value
        for idx, row in step_data.iterrows():
            # Error bar lengths are the difference from the 'most likely' point
            y_err_lower = row['mean_most_likely'] - row['mean_minimum']
            y_err_upper = row['mean_maximum'] - row['mean_most_likely']
            
            # Use a single label for the legend, only on the first point
            label = 'Mean Estimate (min-max range)' if idx == step_data.index[0] else ""

            ax.errorbar(
                x=row[step_metric_used],
                y=row['mean_most_likely'],
                yerr=[[y_err_lower], [y_err_upper]], # Format for asymmetric error bars
                fmt='o',
                markersize=8,
                color='darkblue',
                alpha=alphas.loc[idx], # Apply the specific alpha for this point
                ecolor='cornflowerblue',
                elinewidth=2,
                capsize=5,
                label=label
            )
            
            # Add task name annotation if requested
            if show_task_names:
                ax.annotate(
                    row['task_name'],
                    xy=(row[step_metric_used], row['mean_most_likely']),
                    xytext=(5, 5), # Offset from the point
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray')
                )
            
        ax.set_title(f"Step: {step_name}", fontsize=14, pad=10)
        # Use the step-specific metric information for the label
        if step_uses_task_rank:
            ax.set_xlabel("Task Rank/Order", fontsize=12)
        else:
            ax.set_xlabel(f"Task Metric: {step_metric_used.replace('task_metric_', '').upper()}", fontsize=12)

        is_probability_step = not step_name.startswith("ScenarioLevelMetric_")
        if is_probability_step:
            ax.set_ylabel("Mean Probability Estimate", fontsize=12)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        else: 
            ax.set_ylabel("Mean Estimated Value", fontsize=12)
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

        # Apply the appropriate scale for this step's metric
        if step_metric_scale == 'log' and not step_uses_task_rank:
            # Only apply log scale to actual metrics (not task rank) and if values are positive
            x_values = step_data[step_metric_used].values
            if np.all(x_values > 0):
                ax.set_xscale('log')
            else:
                logger.warning(f"Cannot use log scale for step '{step_name}' metric '{step_metric_used}' due to non-positive values. Using linear scale.")

        ax.grid(True, which="both", linestyle='--', alpha=0.6)
        
        # Add a legend if any labels were produced
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=9)

    for j in range(num_steps, nrows * ncols):
        fig.delaxes(axes[j])

    # Determine overall title based on metrics used across steps
    metrics_used = summary_df['metric_used'].unique()
    uses_task_rank = summary_df['uses_task_rank'].any()
    uses_actual_metric = not summary_df['uses_task_rank'].all()
    
    if uses_task_rank and uses_actual_metric:
        x_axis_label = "Mixed Metrics (Task Rank/Order + FST)"
    elif uses_task_rank:
        x_axis_label = "Task Rank/Order"
    else:
        # All steps use the same actual metric
        x_axis_label = f"{metrics_used[0].replace('task_metric_', '').upper()}"
    
    fig.suptitle(f"{x_axis_label} vs. Mean Estimate/Value by Step\n(Run ID: {run_id}, Model: {model_name_for_title})", fontsize=16, y=1.03)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    
    
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate FST vs. Mean Estimate scatter plots for pipeline runs.")
    parser.add_argument(
        "--csv_file", "-f",
        type=Path,
        help="Optional path to the 'detailed_estimates.csv' file. If not provided, the latest run's CSV will be used."
    )
    parser.add_argument(
        "--output_path", "-o",
        type=Path,
        help="Optional full path (including filename, e.g., 'my_plot.png') to save the plot. "
             "If not provided, defaults to 'output_data/runs/<run_id>/plots/<run_id>_fst_vs_mean_estimate_by_step.png'."
    )
    parser.add_argument(
        "--no_save", "-n",
        action="store_true",
        help="If set, the plot will not be saved to a file."
    )
    parser.add_argument(
        "--show_plot", "-s",
        action="store_true",
        help="If set, display the plot interactively."
    )
    parser.add_argument(
        "--metric", "-m",
        type=str,
        nargs='+',
        help="Specify metric column(s) to plot against in order of preference. Can specify multiple metrics (e.g., --metric task_metric_fst task_metric_cvss task_rank). Uses the first available metric for each step. If not provided, auto-detects task_metric_* column. Always falls back to task rank/order."
    )
    parser.add_argument(
        "--metric_scale", "-ms",
        type=str,
        nargs='+',
        help="Specify scale (linear/log) for each metric in the same order as --metric. Format: 'linear' or 'log' for each metric (e.g., --metric_scale log linear log). If fewer scales than metrics are provided, remaining metrics use linear scale. Default: all linear."
    )
    parser.add_argument(
        "--show_task_names", "-t",
        action="store_true",
        help="If set, display task names next to data points on the plot."
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    actual_csv_path: Optional[Path] = None
    run_id: Optional[str] = None
    csv_parent_dir: Optional[Path] = None

    if args.csv_file:
        if args.csv_file.is_file():
            actual_csv_path = args.csv_file.resolve()
            try:
                # Try to infer run_id: output_data/runs/<run_id>/detailed_estimates.csv
                # csv_parent_dir is the <run_id> directory
                csv_parent_dir = actual_csv_path.parent
                run_id = csv_parent_dir.name
                # Check if the parent of csv_parent_dir is 'runs'
                if csv_parent_dir.parent.name != "runs" or \
                   csv_parent_dir.parent.parent.name != "output_data" or \
                   csv_parent_dir.parent.parent.parent != project_root:
                    logger.warning(f"Provided CSV path '{actual_csv_path}' may not be in the standard 'project_root/output_data/runs/<run_id>/' structure. Run ID inference might be incorrect.")
            except (IndexError, AttributeError):
                 logger.warning(f"Could not robustly infer run_id from CSV path: {actual_csv_path}. Plot titles might be affected.")
                 run_id = "unknown_run" # Fallback run_id
                 csv_parent_dir = actual_csv_path.parent if actual_csv_path else project_root / "output_data" / "runs" / "unknown_run"
        else:
            logger.error(f"Provided CSV file not found: {args.csv_file}")
            exit(1)
    else:
        logger.info("No CSV file provided, attempting to find the latest run's CSV...")
        actual_csv_path = get_latest_run_csv_path(project_root)
        if actual_csv_path:
            csv_parent_dir = actual_csv_path.parent
            run_id = csv_parent_dir.name
        else:
            logger.error("Could not automatically find a CSV file to process.")
            exit(1)

    if not actual_csv_path or not run_id or not csv_parent_dir:
        logger.error("Failed to determine necessary path information. Exiting.")
        exit(1)
    
    logger.info(f"Processing run_id: {run_id} from CSV: {actual_csv_path}")

    fig = plot_uncertainty_bounds(
        run_id=run_id,
        csv_file_path=actual_csv_path,
        metrics_to_plot=args.metric,
        metric_scales=args.metric_scale,
        show_task_names=args.show_task_names
    )

    if fig:
        if not args.no_save:
            if args.output_path:
                actual_plot_save_path = args.output_path.resolve()
                actual_plot_save_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                default_plot_dir = csv_parent_dir / "plots"
                default_plot_dir.mkdir(parents=True, exist_ok=True)
                actual_plot_save_path = default_plot_dir / f"{run_id}_fst_vs_mean_estimate_by_step.png"
            
            try:
                fig.savefig(actual_plot_save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {actual_plot_save_path}")
            except Exception as e:
                logger.error(f"Error saving plot to {actual_plot_save_path}: {e}")

        if args.show_plot:
            logger.info("Displaying plot...")
            plt.show()
        
        plt.close(fig) 
        logger.info("Plotting process complete.")
    else:
        logger.error(f"Failed to generate plot for run {run_id}.")