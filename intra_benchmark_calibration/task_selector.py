#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task selection logic for intra-benchmark calibration experiments.

This module provides functions to select tasks corresponding to bins based on
their position in the sorted benchmark. It maps continuous score percentiles
(bin ranges) to discrete tasks using midpoint approximation.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import math

logger = logging.getLogger(__name__)


def parse_bin_range(bin_range: str) -> Tuple[float, float]:
    """
    Parse a bin range string to extract lower and upper bounds.

    Args:
        bin_range: String like "[9.4, 13.8)" or "[5.0, 9.4)"

    Returns:
        Tuple of (lower_bound, upper_bound)

    Raises:
        ValueError: If bin_range format is invalid
    """
    # Pattern to match: "[number, number)" or "(number, number]"
    pattern = r'[\[\(]\s*([0-9.]+)\s*,\s*([0-9.]+)\s*[\]\)]'
    match = re.match(pattern, bin_range.strip())

    if not match:
        raise ValueError(f"Invalid bin range format: {bin_range}. Expected format like '[9.4, 13.8)'")

    lower = float(match.group(1))
    upper = float(match.group(2))

    if lower >= upper:
        raise ValueError(f"Invalid bin range: lower bound ({lower}) must be < upper bound ({upper})")

    return lower, upper


def get_bin_midpoint(bin_range: str) -> float:
    """
    Calculate the midpoint of a bin range.

    Args:
        bin_range: String like "[9.4, 13.8)"

    Returns:
        Midpoint value (e.g., 11.6 for "[9.4, 13.8)")
    """
    lower, upper = parse_bin_range(bin_range)
    midpoint = (lower + upper) / 2.0
    return midpoint


def get_task_cutoff_index(bin_range: str, total_tasks: int, use_upper_bound: bool = True) -> int:
    """
    Calculate the task index cutoff for a given bin range.

    The cutoff is calculated as: floor(total_tasks * percentile / 100)
    where percentile is either the midpoint or upper bound of the bin.

    Args:
        bin_range: String like "[9.4, 13.8)"
        total_tasks: Total number of tasks in the sorted benchmark
        use_upper_bound: If True, uses upper bound instead of midpoint (default: True)

    Returns:
        Cutoff index k (0-indexed). 
        - If use_upper_bound=False: Tasks 0 to k-1 are "solved", task k is at the midpoint
        - If use_upper_bound=True: Tasks 0 to k are all within the bin

    Example:
        bin_range = "[9.4, 13.8)" (midpoint = 11.6%, upper = 13.8%)
        total_tasks = 40
        use_upper_bound=False: Returns floor(40 * 0.116) = 4
        use_upper_bound=True: Returns floor(40 * 0.138) = 5
    """
    if use_upper_bound:
        lower, upper = parse_bin_range(bin_range)
        percentile = upper
    else:
        percentile = get_bin_midpoint(bin_range)

    # Convert percentage to proportion
    proportion = percentile / 100.0

    # Calculate cutoff index
    k = math.floor(total_tasks * proportion)

    # Ensure k is within valid range [0, total_tasks-1]
    k = max(0, min(k, total_tasks - 1))

    logger.debug(f"Bin range {bin_range}: percentile={percentile:.2f}% (upper_bound={use_upper_bound}), "
                 f"proportion={proportion:.4f}, total_tasks={total_tasks}, cutoff_index={k}")

    return k


def get_tasks_for_bin(
    bin_range: str,
    ascending_tasks: List[Dict[str, Any]],
    total_tasks: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    LEGACY FUNCTION!
    
    Get all tasks that a model "solved" for a given bin.

    A model in bin i has solved all tasks up to the bin's midpoint.
    This function returns tasks 0 to k-1, where k is the cutoff index.

    Args:
        bin_range: String like "[9.4, 13.8)"
        ascending_tasks: List of all tasks sorted by difficulty metric in ascending order.
        total_tasks: Total number of tasks. If None, uses len(ascending_tasks).

    Returns:
        List of task dicts representing "solved tasks" for this bin.
        Returns empty list if cutoff is 0.
    """
    if total_tasks is None:
        total_tasks = len(ascending_tasks)

    if not ascending_tasks:
        logger.warning("Empty task list provided to get_tasks_for_bin")
        return []

    k = get_task_cutoff_index(bin_range, total_tasks)

    # Return tasks 0 to k-1 (all tasks "solved" by models in this bin)
    solved_tasks = ascending_tasks[:k]

    logger.debug(f"Bin {bin_range}: {len(solved_tasks)} solved tasks (indices 0 to {k-1})")

    return solved_tasks


def get_representative_tasks_for_bin(
    bin_range: str,
    ascending_tasks: List[Dict[str, Any]],
    total_tasks: Optional[int] = None,
    n_tasks: int = 3
) -> List[Dict[str, Any]]:
    """
    Get a fixed number of representative tasks from the hardest part of bin i.
    
    Args:
        bin_range: String like "[9.4, 13.8)"
        ascending_tasks: List of all tasks sorted by difficulty in ascending order.
                        NOTE: Selection is based on position in this list, NOT on metric values.
        total_tasks: Total number of tasks. If None, uses len(ascending_tasks).
        n_tasks: Number of representative tasks to select (default: 3).

    Returns:
        List of task dicts representing the hardest "solved tasks" for this bin.
        If fewer than n_tasks are available, returns all available tasks.
    """
    if total_tasks is None:
        total_tasks = len(ascending_tasks)

    if not ascending_tasks:
        logger.warning("Empty task list provided to get_representative_tasks_for_bin")
        return []

    k_upper = get_task_cutoff_index(bin_range, total_tasks, use_upper_bound=True)

    if k_upper == 0:
        logger.debug(f"Bin {bin_range}: No tasks in this bin (upper bound cutoff index k=0)")
        return []

    all_tasks_in_bin = ascending_tasks[:k_upper]
    
    if not all_tasks_in_bin:
        logger.debug(f"Bin {bin_range}: No tasks available (cutoff results in empty list)")
        return []
    
    # Select the last n_tasks (hardest) from the bin
    representative_tasks = all_tasks_in_bin[-n_tasks:]
    
    start_idx = max(0, k_upper - n_tasks)
    logger.debug(f"Bin {bin_range}: Selected {len(representative_tasks)} representative tasks "
                 f"from {len(all_tasks_in_bin)} total tasks in bin (indices {start_idx} to {k_upper-1})")

    return representative_tasks


def get_target_task_for_bin(
    bin_range: str,
    ascending_tasks: List[Dict[str, Any]],
    total_tasks: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Get the target task at the midpoint of a given bin.

    This represents the task we're trying to predict solvability for.

    Args:
        bin_range: String like "[9.4, 13.8)"
        ascending_tasks: List of all tasks sorted by difficulty metric in ascending order.
        total_tasks: Total number of tasks. If None, uses len(ascending_tasks).

    Returns:
        Single task dict representing the "target task" at the bin's midpoint.
        Returns None if cutoff index is out of range.
    """
    if total_tasks is None:
        total_tasks = len(ascending_tasks)

    if not ascending_tasks:
        logger.error("Empty task list provided to get_target_task_for_bin")
        return None

    # Use the midpoint for target task selection (not upper bound)
    # This is a design choice - we want to give the estimator the capability ceiling
    # of the model, but provide a representative task from the target bin
    k = get_task_cutoff_index(bin_range, total_tasks, use_upper_bound=False)

    # The target task is at index k
    if k >= len(ascending_tasks):
        logger.error(f"Cutoff index {k} exceeds task list length {len(ascending_tasks)}")
        return None

    target_task = ascending_tasks[k]
    
    logger.debug(f"Bin {bin_range}: target task at index {k} (midpoint) is '{target_task.name}'")

    return target_task


def format_task_for_prompt(task: Dict[str, Any], metrics_to_use: Optional[List[str]] = None) -> str:
    """
    Format a task into a human-readable string for prompts.

    Args:
        task: BenchmarkTask object
        metrics_to_use: List of metric names to use (e.g., ['fst', 'difficulty_score']).
                           If None or empty, no metrics are used.

    Returns:
        Formatted string like:
        "- Task Name: XYZ
         Description: ...
         Difficulty Metrics: fst=42, difficulty_score=0.8"
    """
    name = task.name
    description = task.description
    
    # Build the base format
    formatted = f"- Task Name: {name}\n  Description: {description}"
    
    # Add metrics if specified
    if metrics_to_use:
        metric_parts = []
        for metric_name in metrics_to_use:
            if metric_name in task.metrics:
                metric_value = task.metrics[metric_name]
                metric_parts.append(f"{metric_name}={metric_value}")
        
        if metric_parts:
            metrics_str = ", ".join(metric_parts)
            formatted += f"\n  Difficulty Metrics: {metrics_str}"
    
    return formatted


def format_tasks_list_for_prompt(tasks: List[Dict[str, Any]], metrics_to_use: Optional[List[str]] = None) -> str:
    """
    Format a list of tasks into a readable string for prompts.

    Args:
        tasks: List of task dicts
        metrics_to_use: List of metric names to use (e.g., ['fst', 'difficulty_score']).
                           If None or empty, no metrics are used.

    Returns:
        Formatted string with all tasks, one per section
    """
    if not tasks:
        return "(No tasks to display)"

    formatted_tasks = [format_task_for_prompt(task, metrics_to_use) for task in tasks]
    return "\n\n".join(formatted_tasks)


def validate_task_selection(
    bin_i_range: str,
    bin_j_range: str,
    ascending_tasks: List[Dict[str, Any]],
    total_tasks: Optional[int] = None
) -> Tuple[bool, str]:
    """
    Validate that task selection for a (i,j) pair will work correctly.

    Args:
        bin_i_range: Source bin range string
        bin_j_range: Target bin range string
        ascending_tasks: List of all tasks sorted by difficulty metric in ascending order.
        total_tasks: Total number of tasks

    Returns:
        Tuple of (is_valid, error_message)
        is_valid is True if selection will work, False otherwise.
        error_message describes the problem if not valid.
    """
    if total_tasks is None:
        total_tasks = len(ascending_tasks)

    try:
        # Parse bin ranges
        lower_i, upper_i = parse_bin_range(bin_i_range)
        lower_j, upper_j = parse_bin_range(bin_j_range)

        # Check that j > i (target bin is harder)
        if lower_j <= lower_i:
            return False, f"Target bin {bin_j_range} must be higher than source bin {bin_i_range}"

        # Get cutoff indices
        k_i = get_task_cutoff_index(bin_i_range, total_tasks)
        k_j = get_task_cutoff_index(bin_j_range, total_tasks)

        # Check that we have enough tasks
        if k_j >= total_tasks:
            return False, f"Target bin cutoff {k_j} exceeds total tasks {total_tasks}"

        # Check that cutoffs are ordered correctly
        if k_j <= k_i:
            return False, (f"Target bin cutoff ({k_j}) must be greater than "
                          f"source bin cutoff ({k_i})")

        # Check that tasks exist at these indices
        if k_i >= len(ascending_tasks) or k_j >= len(ascending_tasks):
            return False, f"Task list length {len(ascending_tasks)} insufficient for cutoffs"

        return True, "Validation passed"

    except ValueError as e:
        return False, f"Invalid bin range: {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


if __name__ == "__main__":
    # Test the task selector
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

    print("Testing task selector...")
    print("="*60)

    # Create mock tasks
    mock_tasks = []
    fst_values = [4, 7, 11, 42, 65, 78, 120, 123, 132, 159, 244, 330, 368]
    for i, fst in enumerate(fst_values):
        mock_tasks.append({
            'name': f'Task_{i+1}',
            'description': f'Mock task {i+1} with FST {fst}',
            'metrics': {'fst': fst}
        })

    print(f"\nMock benchmark: {len(mock_tasks)} tasks")
    print(f"FST range: {mock_tasks[0]['metrics']['fst']} to {mock_tasks[-1]['metrics']['fst']}")

    # Test bin range parsing
    print("\n1. Testing bin range parsing:")
    test_range = "[9.4, 13.8)"
    lower, upper = parse_bin_range(test_range)
    midpoint = get_bin_midpoint(test_range)
    print(f"   Bin range: {test_range}")
    print(f"   Parsed: lower={lower}, upper={upper}, midpoint={midpoint}")

    # Test cutoff calculation
    print("\n2. Testing cutoff calculation:")
    k = get_task_cutoff_index(test_range, len(mock_tasks))
    print(f"   Total tasks: {len(mock_tasks)}")
    print(f"   Bin {test_range} → cutoff index k={k}")

    # Test task selection
    print("\n3. Testing task selection for bin_i:")
    bin_i_range = "[5.0, 9.4)"
    solved_tasks = get_tasks_for_bin(bin_i_range, mock_tasks)
    print(f"   Bin {bin_i_range}: {len(solved_tasks)} solved tasks")
    for task in solved_tasks:
        print(f"     - {task['name']}: FST={task['metrics']['fst']}")

    print("\n4. Testing target task selection for bin_j:")
    bin_j_range = "[9.4, 13.8)"
    target_task = get_target_task_for_bin(bin_j_range, mock_tasks)
    if target_task:
        print(f"   Bin {bin_j_range}: target task = {target_task['name']} "
              f"(FST={target_task['metrics']['fst']})")

    # Test formatting
    print("\n5. Testing task formatting:")
    if solved_tasks:
        formatted = format_task_for_prompt(solved_tasks[0])
        print(f"   Single task format:\n{formatted}")

    # Test validation
    print("\n6. Testing validation:")
    is_valid, msg = validate_task_selection(bin_i_range, bin_j_range, mock_tasks)
    print(f"   Pair ({bin_i_range}, {bin_j_range}): {msg}")

    print("="*60)
