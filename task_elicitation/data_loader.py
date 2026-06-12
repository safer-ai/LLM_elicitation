#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data loader for task-level elicitation using github_data.

Loads:
- model_runs.parquet: Binary pass/fail per (agent, task_id)
- task_difficulties.parquet: Minutes-based difficulty
- tasks/*.jsonl: Task descriptions
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TaskData:
    """Container for all task-related data."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.model_runs: pd.DataFrame = None
        self.difficulties: pd.DataFrame = None
        self.task_descriptions: Dict[str, dict] = {}
        self._load_data()

    def _load_data(self):
        """Load all data files."""
        # Load model runs
        runs_path = self.data_dir / "model_runs.parquet"
        self.model_runs = pd.read_parquet(runs_path)
        logger.info(f"Loaded {len(self.model_runs)} model runs for {self.model_runs['agent'].nunique()} agents")

        # Load difficulties
        diff_path = self.data_dir / "task_difficulties.parquet"
        self.difficulties = pd.read_parquet(diff_path)

        # Create difficulty_for_binning column
        self.difficulties['difficulty_minutes'] = (
            self.difficulties['best_available_minutes']
            .fillna(self.difficulties['model_estimate_minutes'])
        )
        logger.info(f"Loaded difficulties for {len(self.difficulties)} tasks")

        # Load task descriptions from JSONL files
        tasks_dir = self.data_dir / "tasks"
        if tasks_dir.exists():
            for jsonl_file in tasks_dir.glob("*.jsonl"):
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        task = json.loads(line)
                        task_id = task.get('task_id')
                        if task_id:
                            self.task_descriptions[task_id] = task
            logger.info(f"Loaded {len(self.task_descriptions)} task descriptions")

    def get_agents(self) -> List[str]:
        """Get list of all agents."""
        return self.model_runs['agent'].unique().tolist()

    def get_task_ids(self) -> List[str]:
        """Get list of all task IDs that appear in model_runs."""
        return self.model_runs['task_id'].unique().tolist()

    def get_agent_results(self, agent: str) -> pd.DataFrame:
        """Get all results for a specific agent."""
        return self.model_runs[self.model_runs['agent'] == agent].copy()

    def get_task_result(self, agent: str, task_id: str) -> Optional[int]:
        """Get binary result (0/1) for agent on task."""
        mask = (self.model_runs['agent'] == agent) & (self.model_runs['task_id'] == task_id)
        rows = self.model_runs[mask]
        if len(rows) == 0:
            return None
        return int(rows.iloc[0]['score_binarized'])

    def get_task_difficulty(self, task_id: str) -> Optional[float]:
        """Get difficulty in minutes for a task."""
        mask = self.difficulties['task_id'] == task_id
        rows = self.difficulties[mask]
        if len(rows) == 0:
            return None
        return rows.iloc[0]['difficulty_minutes']

    def get_task_description(self, task_id: str) -> Optional[str]:
        """Get task description/prompt."""
        task = self.task_descriptions.get(task_id, {})
        # Try various fields for description
        metadata = task.get('dataset_task_metadata', {})
        prompt = metadata.get('prompt', '')
        if not prompt:
            prompt = task.get('prompt', '')
        return prompt if prompt else None

    def get_task_info(self, task_id: str) -> Dict:
        """Get full task info including description and difficulty."""
        return {
            'task_id': task_id,
            'difficulty_minutes': self.get_task_difficulty(task_id),
            'description': self.get_task_description(task_id),
            'task_family': self.task_descriptions.get(task_id, {}).get('task_family', '')
        }


def get_common_tasks(task_data: TaskData, agents: List[str]) -> List[str]:
    """
    Get tasks that ALL specified agents have completed.

    This ensures anchor and target consistency across models.

    Args:
        task_data: TaskData instance
        agents: List of agent names

    Returns:
        List of task_ids that all agents have results for
    """
    if not agents:
        return []

    # Get tasks for each agent
    agent_tasks = []
    for agent in agents:
        agent_results = task_data.get_agent_results(agent)
        agent_tasks.append(set(agent_results['task_id'].tolist()))

    # Find intersection
    common = agent_tasks[0]
    for tasks in agent_tasks[1:]:
        common = common & tasks

    common_list = sorted(list(common))
    logger.info(f"Found {len(common_list)} tasks common to all {len(agents)} agents")

    return common_list


def assign_difficulty_bins(
    task_data: TaskData,
    boundaries: List[float] = None,
    labels: List[str] = None,
    restrict_to_tasks: List[str] = None
) -> pd.DataFrame:
    """
    Assign tasks to difficulty bins based on minutes.

    Args:
        task_data: TaskData instance
        boundaries: Bin boundaries in minutes, e.g. [0, 5, 60, inf]
        labels: Bin labels, e.g. ['easy', 'medium', 'hard']
        restrict_to_tasks: If provided, only bin these tasks

    Returns:
        DataFrame with task_id, difficulty_minutes, difficulty_bin columns
    """
    if boundaries is None:
        boundaries = [0, 5, 60, float('inf')]
    if labels is None:
        labels = ['easy', 'medium', 'hard']

    # Get tasks to bin
    if restrict_to_tasks is not None:
        task_ids_to_bin = set(restrict_to_tasks)
    else:
        task_ids_to_bin = set(task_data.get_task_ids())

    # Filter difficulties to only include tasks to bin
    df = task_data.difficulties[
        task_data.difficulties['task_id'].isin(task_ids_to_bin)
    ][['task_id', 'difficulty_minutes']].copy()

    # Assign bins
    df['difficulty_bin'] = pd.cut(
        df['difficulty_minutes'],
        bins=boundaries,
        labels=labels,
        include_lowest=True
    )

    logger.info(f"Assigned {len(df)} tasks to difficulty bins:")
    for label in labels:
        count = (df['difficulty_bin'] == label).sum()
        logger.info(f"  {label}: {count} tasks")

    return df


def select_anchors(
    task_data: TaskData,
    binned_tasks: pd.DataFrame,
    n_per_bin: int = 2,
    percentiles: List[float] = None,
    seed: int = 42
) -> List[str]:
    """
    Select anchor tasks - deterministic selection within each bin.

    Args:
        task_data: TaskData instance
        binned_tasks: DataFrame with task_id, difficulty_minutes, difficulty_bin
        n_per_bin: Number of anchors per bin
        percentiles: Percentile positions within each bin (e.g., [0.33, 0.67])
        seed: Random seed (not used if percentiles provided)

    Returns:
        List of anchor task_ids
    """
    if percentiles is None:
        percentiles = [0.33, 0.67]

    anchors = []
    bins = binned_tasks['difficulty_bin'].cat.categories

    for bin_label in bins:
        bin_tasks = binned_tasks[binned_tasks['difficulty_bin'] == bin_label].copy()
        bin_tasks = bin_tasks.sort_values('difficulty_minutes')

        if len(bin_tasks) == 0:
            continue

        # Select at percentile positions
        for p in percentiles[:n_per_bin]:
            idx = int(p * (len(bin_tasks) - 1))
            idx = min(idx, len(bin_tasks) - 1)
            task_id = bin_tasks.iloc[idx]['task_id']
            if task_id not in anchors:
                anchors.append(task_id)

    logger.info(f"Selected {len(anchors)} anchor tasks across {len(bins)} bins")
    return anchors


def select_target_tasks(
    task_data: TaskData,
    binned_tasks: pd.DataFrame,
    anchor_ids: List[str],
    n_targets: int = 30,
    seed: int = 42
) -> List[str]:
    """
    Select target tasks (excluding anchors), balanced across bins.

    Args:
        task_data: TaskData instance
        binned_tasks: DataFrame with task_id, difficulty_minutes, difficulty_bin
        anchor_ids: List of anchor task_ids to exclude
        n_targets: Number of target tasks to select
        seed: Random seed

    Returns:
        List of target task_ids
    """
    np.random.seed(seed)

    # Exclude anchors
    available = binned_tasks[~binned_tasks['task_id'].isin(anchor_ids)].copy()
    bins = available['difficulty_bin'].cat.categories

    # Balance across bins
    per_bin = n_targets // len(bins)
    remainder = n_targets % len(bins)

    targets = []
    for i, bin_label in enumerate(bins):
        bin_tasks = available[available['difficulty_bin'] == bin_label]
        n = per_bin + (1 if i < remainder else 0)
        n = min(n, len(bin_tasks))

        if n > 0:
            sampled = bin_tasks.sample(n=n, random_state=seed+i)
            targets.extend(sampled['task_id'].tolist())

    logger.info(f"Selected {len(targets)} target tasks (excluding anchors)")
    return targets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test the loader
    data_dir = Path("ground-truth/github_data")
    task_data = TaskData(data_dir)

    print(f"\nAgents: {task_data.get_agents()[:3]}...")
    print(f"Total tasks: {len(task_data.get_task_ids())}")

    # Test binning
    binned = assign_difficulty_bins(task_data)
    print(f"\nBinned tasks: {len(binned)}")

    # Test anchor selection
    anchors = select_anchors(task_data, binned)
    print(f"\nAnchors: {anchors}")

    # Show anchor info
    for aid in anchors[:3]:
        info = task_data.get_task_info(aid)
        print(f"\n  {aid}: {info['difficulty_minutes']:.1f} min, {info['task_family']}")
