#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration loading and validation for inter-benchmark calibration.

Handles YAML config loading, source/target benchmark settings,
configurable source bins and target percentiles.
"""

import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Union

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.llm_client import LLMSettings, ThinkingSettings

logger = logging.getLogger(__name__)


@dataclass
class WorkflowSettings:
    """Settings controlling the Delphi workflow execution."""
    num_experts: Optional[int] = None
    delphi_rounds: int = 3
    convergence_threshold: float = 0.05

    def __post_init__(self):
        if self.num_experts is not None and self.num_experts <= 0:
            raise ValueError("WorkflowSettings: 'num_experts' must be positive if specified.")
        if self.delphi_rounds <= 0:
            raise ValueError("WorkflowSettings: 'delphi_rounds' must be positive.")
        if self.convergence_threshold < 0:
            raise ValueError("WorkflowSettings: 'convergence_threshold' cannot be negative.")


@dataclass
class SourceBenchmarkSettings:
    """Settings for a single source benchmark."""
    name: str
    sorted_benchmark_file: str
    benchmark_description: str
    n_easier_tasks: int = 3
    score_file: str = ""


@dataclass
class TargetBenchmarkSettings:
    """Settings for the target benchmark."""
    name: str
    sorted_benchmark_file: str
    benchmark_description: str
    score_file: str = ""
    target_percentiles: Union[List[float], str] = "auto"
    n_target_percentiles: int = 5


@dataclass
class InterBenchmarkConfig:
    """Complete configuration for inter-benchmark calibration experiments."""
    api_key_anthropic: Optional[str]
    api_key_openai: Optional[str]

    prompts_dir: Path
    expert_profiles_file: Path
    ground_truth_file: Path

    source_benchmarks: List[SourceBenchmarkSettings]
    target_benchmark: TargetBenchmarkSettings

    # Source bins: explicit list of [lo, hi] pairs, or "auto"
    source_bins: Union[List[List[float]], str]
    n_source_bins: int = 4

    llm_settings: LLMSettings = field(default_factory=lambda: LLMSettings(model="claude-sonnet-4-5-20250929"))
    workflow_settings: WorkflowSettings = field(default_factory=WorkflowSettings)

    output_dir: Path = Path("results")

    @property
    def primary_source(self) -> SourceBenchmarkSettings:
        return self.source_benchmarks[0]

    @property
    def runs_dir(self) -> Path:
        src = self.primary_source.name
        tgt = self.target_benchmark.name
        return self.output_dir / f"{src}_to_{tgt}"

    @property
    def registry_file(self) -> Path:
        return self.output_dir / "run_registry.json"


def load_inter_benchmark_config(config_path: str, base_dir: Optional[Path] = None) -> InterBenchmarkConfig:
    """
    Load configuration for inter-benchmark calibration mode.

    Args:
        config_path: Path to the YAML configuration file
        base_dir: Base directory for resolving relative paths (defaults to parent of config file)

    Returns:
        InterBenchmarkConfig object with all settings

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config_file = Path(config_path).resolve()

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if base_dir is None:
        base_dir = config_file.parent

    logger.info(f"Loading inter-benchmark configuration from: {config_file}")
    logger.info(f"Base directory for relative paths: {base_dir}")

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML configuration: {e}")

    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a YAML dictionary")

    api_key_anthropic = data.get('anthropic_api_key')
    api_key_openai = data.get('openai_api_key')

    # LLM settings
    llm_data = data.get('llm_settings', {})
    if not llm_data:
        raise ValueError("Missing 'llm_settings' in configuration")

    thinking_data = llm_data.get('thinking', {})
    thinking_settings = ThinkingSettings(
        enabled=bool(thinking_data.get('enabled', False)),
        budget_tokens=int(thinking_data.get('budget_tokens', 10000))
    )

    llm_settings = LLMSettings(
        model=str(llm_data.get('model', '')),
        temperature=float(llm_data.get('temperature', 0.8)),
        max_concurrent_calls=int(llm_data.get('max_concurrent_calls', 5)),
        rate_limit_calls=int(llm_data.get('rate_limit_calls', 45)),
        rate_limit_period=int(llm_data.get('rate_limit_period', 60)),
        thinking=thinking_settings
    )

    # Workflow settings
    wf_data = data.get('workflow_settings', {})
    workflow_settings = WorkflowSettings(
        num_experts=wf_data.get('num_experts'),
        delphi_rounds=int(wf_data.get('delphi_rounds', 3)),
        convergence_threshold=float(wf_data.get('convergence_threshold', 0.05))
    )

    # Inter-benchmark settings
    ib_data = data.get('inter_benchmark_settings', {})
    if not ib_data:
        raise ValueError("Missing 'inter_benchmark_settings' in configuration")

    # Parse source benchmarks
    src_list = ib_data.get('source_benchmarks', [])
    if not src_list:
        raise ValueError("At least one source benchmark is required")

    source_benchmarks = []
    for src in src_list:
        source_benchmarks.append(SourceBenchmarkSettings(
            name=str(src.get('name', '')),
            sorted_benchmark_file=str(src.get('sorted_benchmark_file', '')),
            benchmark_description=str(src.get('benchmark_description', '')),
            n_easier_tasks=int(src.get('n_easier_tasks', 3)),
            score_file=str(src.get('score_file', ''))
        ))

    # Parse target benchmark
    tgt_data = ib_data.get('target_benchmark', {})
    if not tgt_data:
        raise ValueError("Missing 'target_benchmark' in configuration")

    raw_target_pcts = tgt_data.get('target_percentiles', 'auto')
    if isinstance(raw_target_pcts, list):
        target_percentiles = [float(p) for p in raw_target_pcts]
    else:
        target_percentiles = str(raw_target_pcts)

    target_benchmark = TargetBenchmarkSettings(
        name=str(tgt_data.get('name', '')),
        sorted_benchmark_file=str(tgt_data.get('sorted_benchmark_file', '')),
        benchmark_description=str(tgt_data.get('benchmark_description', '')),
        score_file=str(tgt_data.get('score_file', '')),
        target_percentiles=target_percentiles,
        n_target_percentiles=int(tgt_data.get('n_target_percentiles', 5))
    )

    # Parse source bins
    raw_source_bins = ib_data.get('source_bins', 'auto')
    if isinstance(raw_source_bins, list):
        source_bins = [[float(b) for b in pair] for pair in raw_source_bins]
    else:
        source_bins = str(raw_source_bins)

    n_source_bins = int(ib_data.get('n_source_bins', 4))

    # Resolve paths
    def resolve_path(path_str: str, path_name: str) -> Path:
        if not path_str:
            raise ValueError(f"Missing '{path_name}' in configuration")
        path = Path(path_str)
        if not path.is_absolute():
            path = base_dir / path
        return path

    prompts_dir = resolve_path(data.get('prompts_dir', 'prompts'), 'prompts_dir')
    expert_profiles_file = resolve_path(
        data.get('expert_profiles_file', 'input_data/expert_profiles/expert_profiles.yaml'),
        'expert_profiles_file'
    )
    ground_truth_file = resolve_path(
        ib_data.get('ground_truth_file', 'input_data/ground_truth/ground_truth.json'),
        'ground_truth_file'
    )

    output_dir_str = data.get('output_dir', 'results')
    output_dir = Path(output_dir_str)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir

    config = InterBenchmarkConfig(
        api_key_anthropic=api_key_anthropic,
        api_key_openai=api_key_openai,
        prompts_dir=prompts_dir,
        expert_profiles_file=expert_profiles_file,
        ground_truth_file=ground_truth_file,
        source_benchmarks=source_benchmarks,
        target_benchmark=target_benchmark,
        source_bins=source_bins,
        n_source_bins=n_source_bins,
        llm_settings=llm_settings,
        workflow_settings=workflow_settings,
        output_dir=output_dir
    )

    logger.info("Configuration loaded successfully:")
    logger.info(f"  - Source benchmarks: {[s.name for s in config.source_benchmarks]}")
    logger.info(f"  - Target benchmark: {config.target_benchmark.name}")
    logger.info(f"  - Source bins: {config.source_bins}")
    logger.info(f"  - Target percentiles: {config.target_benchmark.target_percentiles}")
    logger.info(f"  - Model: {config.llm_settings.model}")
    logger.info(f"  - Num experts: {config.workflow_settings.num_experts or 'all'}")

    return config
