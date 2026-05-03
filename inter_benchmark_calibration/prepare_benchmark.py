#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocessing script to sort benchmark tasks by a specified metric in ascending order.

This script reads a benchmark file, sorts all tasks by a specified difficulty metric
in ascending order (lowest metric value first), and writes the output to a new file
for use in intra-benchmark calibration experiments.

Usage:
    python prepare_benchmark.py <input_file> [--metric METRIC]
"""

import yaml
from pathlib import Path
import sys
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def sort_benchmark_by_metric(
    input_path: str,
    metric: str = "fst",
    descending: bool = False,
    metrics_to_use_for_estimation: list = None
) -> None:
    """
    Sort benchmark tasks by a specified metric.

    Args:
        input_path: Path to the benchmark file
        metric: Name of the metric to sort by (default: "fst")
        descending: If True, sort in descending order (default: False)
        metrics_to_use_for_estimation: List of metrics to use for estimation (default: None -> [])
    """
    input_file = Path(input_path)

    benchmark_name = input_file.stem
    sort_order = "descending" if descending else "ascending"
    output_filename = f"{benchmark_name}_{sort_order}_{metric}.yaml"
    output_file = input_file.parent / output_filename

    # Check if input file exists
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    logger.info(f"Reading benchmark from: {input_file}")
    if descending:
        logger.info(f"Sorting in DESCENDING order (highest {metric} first)")

    # Load the benchmark YAML
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load YAML file: {e}")
        sys.exit(1)

    # Validate structure
    if not isinstance(data, dict) or 'tasks' not in data:
        logger.error("Invalid benchmark file format. Expected dictionary with 'tasks' key.")
        sys.exit(1)

    tasks = data.get('tasks', [])
    if not tasks:
        logger.warning("No tasks found in benchmark file.")
        sys.exit(1)

    logger.info(f"Found {len(tasks)} tasks in benchmark")

    # Validate all tasks have the specified metric
    tasks_with_metric = []
    tasks_without_metric = []

    for task in tasks:
        if isinstance(task, dict):
            metrics = task.get('metrics', {})
            if isinstance(metrics, dict) and metric in metrics:
                metric_value = float(metrics[metric])
                if isinstance(metric_value, (int, float)):
                    tasks_with_metric.append(task)
                else:
                    logger.warning(f"Task '{task.get('name', 'Unknown')}' has non-numeric {metric}: {metric_value}")
                    tasks_without_metric.append(task)
            else:
                logger.warning(f"Task '{task.get('name', 'Unknown')}' missing {metric} metric")
                tasks_without_metric.append(task)
        else:
            logger.warning(f"Invalid task format: {task}")

    if tasks_without_metric:
        logger.warning(f"{len(tasks_without_metric)} tasks do not have valid {metric} metrics and will be excluded")

    if not tasks_with_metric:
        logger.error(f"No tasks with valid {metric} metrics found")
        sys.exit(1)

    # Sort tasks by metric
    sorted_tasks = sorted(tasks_with_metric, key=lambda t: float(t['metrics'][metric]), reverse=descending)

    logger.info(f"Sorted {len(sorted_tasks)} tasks by {metric}")
    logger.info(f"{metric} range: {sorted_tasks[0]['metrics'][metric]} (min) to {sorted_tasks[-1]['metrics'][metric]} (max)")

    # Validate metrics_to_use_for_estimation
    if metrics_to_use_for_estimation is None:
        metrics_to_use = []
        logger.info("No metrics_to_use_for_estimation specified. Setting to [].")
    else:
        # Validate that all specified metrics exist in the first task's metrics
        first_task_metrics = sorted_tasks[0].get('metrics', {})
        invalid_metrics = [m for m in metrics_to_use_for_estimation if m not in first_task_metrics]
        
        if invalid_metrics:
            logger.error(f"Invalid metrics specified in --metrics_to_use_for_estimation: {invalid_metrics}")
            logger.error(f"Available metrics in tasks: {list(first_task_metrics.keys())}")
            sys.exit(1)
        
        metrics_to_use = metrics_to_use_for_estimation
        logger.info(f"Using metrics for estimation: {metrics_to_use}")

    output_data = {
        'metrics_to_use_for_estimation': metrics_to_use,
        'benchmark_description': data.get('benchmark_description', f'{benchmark_name} benchmark sorted by {metric} in {sort_order} order'),
        'tasks': sorted_tasks
    }

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write sorted benchmark
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info(f"Successfully wrote sorted benchmark to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to write output file: {e}")
        sys.exit(1)

    # Print summary
    print("\n" + "="*60)
    print("SORTING SUMMARY")
    print("="*60)
    print(f"Input file:  {input_file}")
    print(f"Output file: {output_file}")
    print(f"Metric:      {metric}")
    print(f"Total tasks: {len(sorted_tasks)}")
    print(f"{metric} range:   {sorted_tasks[0]['metrics'][metric]} to {sorted_tasks[-1]['metrics'][metric]}")
    print("="*60)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sort benchmark tasks by a specified metric"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the benchmark YAML file"
    )
    parser.add_argument(
        "--metric-to-sort-by",
        type=str,
        default="fst",
        help="Name of the metric to sort by (default: fst)"
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort in descending order (highest values first). Default is ascending order."
    )
    parser.add_argument(
        "--metrics_to_use_for_estimation",
        type=str,
        default=None,
        help="Comma-separated list of metrics to use for estimation (e.g., 'borda_score,estimated_difficulty'). These metrics must exist in the task metrics."
    )

    args = parser.parse_args()

    # Parse comma-separated metrics into a list
    metrics_list = None
    if args.metrics_to_use_for_estimation:
        metrics_list = [m.strip() for m in args.metrics_to_use_for_estimation.split(',')]

    sort_benchmark_by_metric(args.input_file, args.metric_to_sort_by, args.descending, metrics_list)
