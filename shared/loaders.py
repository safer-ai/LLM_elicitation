#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Input data loading utilities shared across experiments.

Functions to load prompts, expert profiles, and benchmark files.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional
from shared.data_models import ExpertProfile, BenchmarkTask, Benchmark
from shared.benchmark_adapter import get_adapter

logger = logging.getLogger(__name__)


def load_prompts(prompts_dir: Path) -> Optional[Dict[str, str]]:
    """
    Loads all prompt template files from the specified directory.

    Returns:
        Dictionary mapping prompt names (without .txt extension) to their content strings.
        Returns None if the directory doesn't exist or can't be read.
    """
    if not prompts_dir.exists():
        logger.error(f"Prompts directory not found: {prompts_dir}")
        return None

    if not prompts_dir.is_dir():
        logger.error(f"Prompts path is not a directory: {prompts_dir}")
        return None

    prompts = {}
    prompt_files = list(prompts_dir.glob("*.txt"))

    if not prompt_files:
        logger.warning(f"No .txt files found in prompts directory: {prompts_dir}")
        return {}

    for prompt_file in prompt_files:
        prompt_name = prompt_file.stem
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            prompts[prompt_name] = content
            logger.debug(f"Loaded prompt template: {prompt_name} ({len(content)} characters)")
        except Exception as e:
            logger.error(f"Failed to read prompt file {prompt_file}: {e}", exc_info=True)
            return None

    logger.info(f"Loaded {len(prompts)} prompt templates from {prompts_dir}")
    return prompts


def load_experts(expert_profiles_file: Path) -> Optional[List[ExpertProfile]]:
    """
    Loads expert profiles from a YAML file.

    Returns:
        List of ExpertProfile objects, or None if loading fails.
    """
    if not expert_profiles_file.exists():
        logger.error(f"Expert profiles file not found: {expert_profiles_file}")
        return None

    try:
        with open(expert_profiles_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse expert profiles YAML: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Failed to read expert profiles file: {e}", exc_info=True)
        return None

    if not isinstance(data, dict) or 'experts' not in data:
        logger.error("Expert profiles file must contain an 'experts' key at the top level")
        return None

    experts_data = data['experts']
    if not isinstance(experts_data, list):
        logger.error("'experts' must be a list of expert profile dictionaries")
        return None

    experts = []
    for idx, expert_dict in enumerate(experts_data):
        try:
            expert = ExpertProfile(
                name=expert_dict.get('name', f'Expert{idx+1}'),
                background=expert_dict.get('background', ''),
                focus=expert_dict.get('focus', ''),
                key_trait=expert_dict.get('key_trait', ''),
                bias=expert_dict.get('bias', ''),
                analytical_approach=expert_dict.get('analytical_approach')
            )
            experts.append(expert)
            logger.debug(f"Loaded expert: {expert.name}")
        except Exception as e:
            logger.error(f"Failed to parse expert profile at index {idx}: {e}", exc_info=True)
            return None

    logger.info(f"Loaded {len(experts)} expert profiles from {expert_profiles_file}")
    return experts


def load_benchmark(benchmark_file: Path, benchmark_name: str) -> Optional[Benchmark]:
    """
    Loads a benchmark definition from a YAML file.

    Returns:
        Benchmark object, or None if loading fails.
    """
    if not benchmark_file.exists():
        logger.error(f"Benchmark file not found: {benchmark_file}")
        return None

    try:
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse benchmark YAML: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Failed to read benchmark file: {e}", exc_info=True)
        return None

    if not isinstance(data, dict):
        logger.error("Benchmark file must contain a YAML dictionary")
        return None

    description = data.get('benchmark_description', '')
    metrics_to_use = data.get('metrics_to_use_for_estimation', [])

    tasks_data = data.get('tasks', [])
    if not isinstance(tasks_data, list):
        logger.error("'tasks' must be a list of task dictionaries")
        return None

    if not tasks_data:
        logger.warning("No tasks found in benchmark file")
        return None

    if not benchmark_name:
        logger.error("benchmark_name parameter is required but was not provided")
        return None

    adapter = get_adapter(benchmark_name)
    logger.info(f"Using adapter for benchmark: {benchmark_name}")

    tasks = []
    for idx, task_dict in enumerate(tasks_data):
        try:
            task_name = adapter.get_task_name(task_dict)
            task_description = adapter.get_task_description(task_dict)

            task = BenchmarkTask(
                name=task_name,
                description=task_description,
                metrics=task_dict.get('metrics', {})
            )
            tasks.append(task)
            logger.debug(f"Loaded task: {task.name}")
        except Exception as e:
            logger.error(f"Failed to parse task at index {idx}: {e}", exc_info=True)
            return None

    benchmark = Benchmark(
        description=description,
        metrics_to_use=metrics_to_use,
        tasks=tasks
    )

    logger.info(f"Loaded benchmark from {benchmark_file}")
    logger.info(f"  - Tasks: {len(tasks)}")
    logger.info(f"  - Metrics to use: {metrics_to_use}")

    return benchmark
