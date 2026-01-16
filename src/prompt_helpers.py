# src/prompt_helpers.py

import logging
from typing import List
from data_models import BenchmarkTask, Benchmark
from config import AppConfig

logger = logging.getLogger(__name__)


def _get_easier_example_tasks(
    all_tasks: List[BenchmarkTask],
    current_task: BenchmarkTask,
    num_example_tasks: int
) -> List[BenchmarkTask]:
    """
    Returns easier example tasks to show as context for a given task.

    Tasks are assumed to be ordered easiest first. This function returns:
    - The previous task (one position before current)
    - Uniformly spaced tasks from earlier positions

    Args:
        all_tasks: Full list of tasks ordered easiest first
        current_task: The task being evaluated
        num_example_tasks: Total number of examples to return (including previous task)

    Returns:
        List of easier example tasks, ordered easiest to hardest
    """
    if num_example_tasks <= 0 or not all_tasks:
        return []

    # Find current task's position in the full list
    try:
        current_idx = all_tasks.index(current_task)
    except ValueError:
        logger.warning(f"Task '{current_task.name}' not found in task list. Cannot determine easier examples.")
        return []

    if current_idx == 0:
        # First task, no easier tasks available
        return []

    # Available easier tasks are at indices 0 to (current_idx - 1)
    available_count = current_idx
    actual_num_examples = min(num_example_tasks, available_count)

    if actual_num_examples == 1:
        # Just return the previous task
        return [all_tasks[current_idx - 1]]

    # Calculate uniform spacing starting from hardest (previous) working backwards to easiest
    # We want actual_num_examples evenly distributed across [0, current_idx-1]
    example_indices = []

    if actual_num_examples >= current_idx:
        # Want more examples than available positions, take all
        example_indices = list(range(current_idx))
    else:
        # Uniform spacing from hardest to easiest
        # Step size between samples
        step = (current_idx - 1) / (actual_num_examples - 1)

        for i in range(actual_num_examples):
            # Start from current_idx - 1 (hardest) and work backwards
            idx = int(round((current_idx - 1) - (i * step)))
            if idx not in example_indices and 0 <= idx < current_idx:
                example_indices.append(idx)

    # Sort indices to maintain easiest-to-hardest order in returned list
    example_indices = sorted(set(example_indices))
    return [all_tasks[idx] for idx in example_indices]


def format_example_tasks_context(
    benchmark: Benchmark,
    current_task: BenchmarkTask,
    config: AppConfig
) -> str:
    """
    Formats easier example tasks into a context string for prompt inclusion.

    This is the single source of truth for how example tasks are formatted.

    Args:
        benchmark: The benchmark containing all tasks
        current_task: The current task being evaluated
        config: Application configuration

    Returns:
        Formatted string containing example tasks context (empty string if not configured or no examples)
    """
    if not config.workflow_settings.include_easier_tasks or not config.workflow_settings.num_example_tasks:
        return ""

    easier_tasks = _get_easier_example_tasks(
        all_tasks=benchmark.tasks,
        current_task=current_task,
        num_example_tasks=config.workflow_settings.num_example_tasks
    )

    if not easier_tasks:
        return ""

    example_lines = ["For context, here are some easier tasks from this benchmark that you should assume the LLM can solve:\n"]
    for i, ex_task in enumerate(easier_tasks, 1):
        example_lines.append(f"Example Task {i}: {ex_task.name}")
        example_lines.append(f"Description: {ex_task.description}")
        # Include metrics if available
        if benchmark.metrics_to_use and ex_task.metrics:
            metrics_parts = []
            for metric_key in benchmark.metrics_to_use:
                if metric_key in ex_task.metrics:
                    metrics_parts.append(f"{metric_key}: {ex_task.metrics[metric_key]}")
            if metrics_parts:
                example_lines.append(f"Metrics: {', '.join(metrics_parts)}")
        example_lines.append("")  # Empty line between examples

    return "\n".join(example_lines)
