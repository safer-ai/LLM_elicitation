#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task selection logic for inter-benchmark calibration experiments.

Selects representative source tasks (capability ceiling) and target tasks
at specific percentile positions in the sorted benchmark.
"""

import math
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def get_task_at_percentile(
    percentile: float,
    sorted_tasks: List[Any],
) -> Optional[Any]:
    """
    Get the task at a given percentile position in a sorted task list.

    Args:
        percentile: Score percentile (e.g. 40.0 means "40% through the benchmark").
                    Interpreted as a fraction of total tasks.
        sorted_tasks: Tasks sorted by ascending difficulty.

    Returns:
        The task at the given percentile, or None if out of range.
    """
    if not sorted_tasks:
        logger.error("Empty task list provided")
        return None

    n = len(sorted_tasks)
    idx = math.floor(n * percentile / 100.0)
    idx = max(0, min(idx, n - 1))

    task = sorted_tasks[idx]
    logger.debug(f"Percentile {percentile}%: selected task at index {idx}/{n} -- '{task.name}'")
    return task


def get_representative_source_tasks(
    bin_bounds: List[float],
    sorted_tasks: List[Any],
    n_tasks: int = 3
) -> List[Any]:
    """
    Get representative tasks from a source benchmark bin (hardest N tasks from the bin).

    The bin_bounds are [lo, hi) representing model score ranges. These are interpreted
    as percentages of the benchmark (mapping score -> task position). Accepts both
    percentage-scale bounds (e.g. [10.0, 20.0]) and fraction-scale bounds (e.g. [0.1, 0.2]).

    Args:
        bin_bounds: [lower, upper] score range
        sorted_tasks: All tasks sorted by ascending difficulty
        n_tasks: Number of representative tasks to select

    Returns:
        List of task objects (up to n_tasks, from the hardest end of the bin)
    """
    if not sorted_tasks:
        logger.warning("Empty task list")
        return []

    lo, hi = bin_bounds[0], bin_bounds[1]
    # Fraction-scale scores (0-1) -> convert to percentage for index calculation
    if hi <= 1.0:
        lo, hi = lo * 100.0, hi * 100.0

    n = len(sorted_tasks)
    lo_idx = math.floor(n * lo / 100.0)
    hi_idx = math.floor(n * hi / 100.0)

    lo_idx = max(0, min(lo_idx, n - 1))
    hi_idx = max(0, min(hi_idx, n))

    if hi_idx <= lo_idx:
        logger.debug(f"Bin {bin_bounds}: empty range (indices {lo_idx}-{hi_idx})")
        return []

    tasks_in_bin = sorted_tasks[lo_idx:hi_idx]

    if not tasks_in_bin:
        return []

    representative = tasks_in_bin[-n_tasks:]

    logger.debug(f"Bin {bin_bounds}: selected {len(representative)} of {len(tasks_in_bin)} tasks "
                 f"(indices {max(lo_idx, hi_idx - n_tasks)}-{hi_idx - 1})")
    return representative


def format_task_for_prompt(task: Any, metrics_to_use: Optional[List[str]] = None) -> str:
    """Format a task into a human-readable string for prompts."""
    formatted = f"- Task Name: {task.name}\n  Description: {task.description}"

    if metrics_to_use:
        metric_parts = []
        for metric_name in metrics_to_use:
            if metric_name in task.metrics:
                metric_parts.append(f"{metric_name}={task.metrics[metric_name]}")
        if metric_parts:
            formatted += f"\n  Difficulty Metrics: {', '.join(metric_parts)}"

    return formatted


def format_tasks_list_for_prompt(tasks: List[Any], metrics_to_use: Optional[List[str]] = None) -> str:
    """Format a list of tasks into a readable string for prompts."""
    if not tasks:
        return "(No tasks to display)"

    formatted_tasks = [format_task_for_prompt(task, metrics_to_use) for task in tasks]
    return "\n\n".join(formatted_tasks)
