#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration loading and validation for intra-benchmark calibration.

This module handles loading configuration from YAML files and validates
all settings for the intra-benchmark calibration workflow.
"""

import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from llm_client import LLMSettings

logger = logging.getLogger(__name__)


@dataclass
class WorkflowSettings:
    """Settings controlling the Delphi workflow execution."""
    num_experts: Optional[int] = None    # Max experts to use (None = all)
    delphi_rounds: int = 3               # Total number of Delphi rounds
    convergence_threshold: float = 0.05  # Std dev threshold for early stopping

    def __post_init__(self):
        if self.num_experts is not None and self.num_experts <= 0:
            raise ValueError("WorkflowSettings: 'num_experts' must be positive if specified.")
        if self.delphi_rounds <= 0:
            raise ValueError("WorkflowSettings: 'delphi_rounds' must be positive.")
        if self.convergence_threshold < 0:
            raise ValueError("WorkflowSettings: 'convergence_threshold' cannot be negative.")


@dataclass
class IntraBenchmarkSettings:
    """Settings specific to intra-benchmark calibration experiments."""
    benchmark_name: str             # e.g., "cybench"
    n_bins: int                     # Number of bins used in ground truth computation
    benchmark_description: str      # Description of the benchmark for prompts
    sorted_benchmark_file: Optional[str] = None  # e.g., "input_data/benchmarks/cybench_ascending_fst.yaml"

    def __post_init__(self):
        if not self.benchmark_name:
            raise ValueError("IntraBenchmarkSettings: 'benchmark_name' cannot be empty")
        if self.n_bins <= 0:
            raise ValueError("IntraBenchmarkSettings: 'n_bins' must be positive")
        if not self.benchmark_description:
            raise ValueError("IntraBenchmarkSettings: 'benchmark_description' cannot be empty")


@dataclass
class IntraBenchmarkConfig:
    """
    Complete configuration for intra-benchmark calibration experiments.
    """
    # API Keys
    api_key_anthropic: Optional[str]
    api_key_openai: Optional[str]

    # Input Paths (relative to intra_benchmark_calibration/ directory)
    prompts_dir: Path
    expert_profiles_file: Path
    benchmark_file: Path
    ground_truth_dir: Path

    # Settings
    llm_settings: LLMSettings
    workflow_settings: WorkflowSettings
    intra_benchmark_settings: IntraBenchmarkSettings

    # Output Paths
    output_dir: Path = Path("results")

    @property
    def runs_dir(self) -> Path:
        """Convenience property for the runs subdirectory."""
        return self.output_dir / self.intra_benchmark_settings.benchmark_name

    @property
    def registry_file(self) -> Path:
        """Convenience property for the run registry file path."""
        return self.output_dir / "run_registry.json"


def load_intra_benchmark_config(config_path: str, base_dir: Optional[Path] = None) -> IntraBenchmarkConfig:
    """
    Load configuration for intra-benchmark calibration mode.

    Args:
        config_path: Path to the YAML configuration file
        base_dir: Base directory for resolving relative paths (defaults to parent of config file)

    Returns:
        IntraBenchmarkConfig object with all settings

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config_file = Path(config_path).resolve()
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Determine base directory for relative paths
    if base_dir is None:
        base_dir = config_file.parent
    
    logger.info(f"Loading intra-benchmark configuration from: {config_file}")
    logger.info(f"Base directory for relative paths: {base_dir}")

    # Load YAML
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML configuration: {e}")

    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a YAML dictionary")

    # Extract API keys
    api_key_anthropic = data.get('anthropic_api_key')
    api_key_openai = data.get('openai_api_key')

    # Extract LLM settings
    llm_data = data.get('llm_settings', {})
    if not llm_data:
        raise ValueError("Missing 'llm_settings' in configuration")
    
    llm_settings = LLMSettings(
        model=str(llm_data.get('model', '')),
        temperature=float(llm_data.get('temperature', 0.8)),
        max_concurrent_calls=int(llm_data.get('max_concurrent_calls', 5)),
        rate_limit_calls=int(llm_data.get('rate_limit_calls', 45)),
        rate_limit_period=int(llm_data.get('rate_limit_period', 60))
    )

    # Extract workflow settings
    wf_data = data.get('workflow_settings', {})
    workflow_settings = WorkflowSettings(
        num_experts=wf_data.get('num_experts'),
        delphi_rounds=int(wf_data.get('delphi_rounds', 3)),
        convergence_threshold=float(wf_data.get('convergence_threshold', 0.05))
    )

    # Extract intra-benchmark specific settings
    ib_data = data.get('intra_benchmark_settings', {})
    if not ib_data:
        raise ValueError("Missing 'intra_benchmark_settings' in configuration")
    
    intra_benchmark_settings = IntraBenchmarkSettings(
        benchmark_name=str(ib_data.get('benchmark_name', '')),
        n_bins=int(ib_data.get('n_bins', 0)),
        benchmark_description=str(ib_data.get('benchmark_description', '')),
        sorted_benchmark_file=ib_data.get('sorted_benchmark_file')
    )

    # Extract and resolve paths
    def resolve_path(path_str: str, path_name: str) -> Path:
        """Helper to resolve a path relative to base_dir."""
        if not path_str:
            raise ValueError(f"Missing '{path_name}' in configuration")
        path = Path(path_str)
        if not path.is_absolute():
            path = base_dir / path
        return path

    prompts_dir = resolve_path(data.get('prompts_dir', 'prompts'), 'prompts_dir')
    expert_profiles_file = resolve_path(data.get('expert_profiles_file', 'input_data/expert_profiles/expert_profiles.yaml'), 'expert_profiles_file')
    ground_truth_dir = resolve_path(data.get('ground_truth_dir', 'input_data/ground_truth'), 'ground_truth_dir')
    
    # Determine benchmark file path
    if intra_benchmark_settings.sorted_benchmark_file:
        benchmark_file = resolve_path(intra_benchmark_settings.sorted_benchmark_file, 'sorted_benchmark_file')
    else:
        benchmark_file_str = data.get('benchmark_file', 'input_data/benchmarks/cybench_ascending_fst.yaml')
        benchmark_file = resolve_path(benchmark_file_str, 'benchmark_file')

    # Output directory
    output_dir_str = data.get('output_dir', 'results')
    output_dir = Path(output_dir_str)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir

    # Create config object
    config = IntraBenchmarkConfig(
        api_key_anthropic=api_key_anthropic,
        api_key_openai=api_key_openai,
        prompts_dir=prompts_dir,
        expert_profiles_file=expert_profiles_file,
        benchmark_file=benchmark_file,
        ground_truth_dir=ground_truth_dir,
        llm_settings=llm_settings,
        workflow_settings=workflow_settings,
        intra_benchmark_settings=intra_benchmark_settings,
        output_dir=output_dir
    )

    logger.info(f"Configuration loaded successfully:")
    logger.info(f"  - Benchmark: {config.intra_benchmark_settings.benchmark_name}")
    logger.info(f"  - Number of bins: {config.intra_benchmark_settings.n_bins}")
    logger.info(f"  - Model: {config.llm_settings.model}")
    logger.info(f"  - Num experts: {config.workflow_settings.num_experts or 'all'}")
    logger.info(f"  - Delphi rounds: {config.workflow_settings.delphi_rounds}")

    return config
