#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration loading for intra-benchmark calibration.

Mirrors the dataclass / YAML loader pattern used by `inter_benchmark_calibration/config.py`,
exposing only the knobs from the spec §6.

Default `forecasted_models` excludes GPT-2, GPT-3, and GPT-3.5 because in
model_runs.parquet these three models are only evaluated on a subset of the
headline tasks (172, 172, and 142 of 291 respectively), which causes most
(i, j, M) cells to be inadmissible.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.llm_client import LLMSettings, ThinkingSettings  # noqa: E402

logger = logging.getLogger(__name__)


# Sparse-coverage models excluded by default (q3 decision).
DEFAULT_DROP_MODELS: List[str] = ["GPT-2", "GPT-3", "GPT-3.5"]


@dataclass
class WorkflowSettings:
    num_experts: int = 2
    delphi_rounds: int = 1
    convergence_threshold: float = 0.05

    def __post_init__(self):
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.delphi_rounds <= 0:
            raise ValueError("delphi_rounds must be positive")
        if self.convergence_threshold < 0:
            raise ValueError("convergence_threshold cannot be negative")


@dataclass
class BinningSettings:
    n_bins: int = 5
    strategy: str = "equal_count"  # equal_count | equal_log_fst | explicit_edges
    explicit_edges: Optional[List[float]] = None


@dataclass
class TargetSelectionSettings:
    selection_mode: str = "auto_sample"  # auto_sample | explicit_target_tasks
    explicit_target_tasks: Optional[Dict[int, List[str]]] = None
    sampling_seed: int = 42
    n_target_tasks_per_cell: int = 1


@dataclass
class SourceProfileSettings:
    """Knobs that shape the per-cell source-side capability profile.

    `source_bins_to_show` controls which bins are shown to the forecaster as
    M's capability evidence. Three modes are supported:

      []                      -> "single_bin" mode (default, spec §6).
                                 Shows only the source bin i. Cells iterate
                                 over (i, j) for i != j; bidirectional pairs.

      "all_except_target"     -> "all_except_target" mode. For each cell with
                                 target bin j, shows every bin except j. The
                                 source bin i is collapsed (no longer
                                 meaningful), so cells iterate over j only.
                                 Cells = n_bins x n_models x K (4x cheaper at
                                 n_bins=5 than single_bin).

      [list of bin indices]   -> "custom_subset" mode. Shows the supplied
                                 list of bins regardless of i. Cells still
                                 iterate over (i, j) for i != j, but the
                                 source-profile content is identical across
                                 all i values for fixed j -- so per-i
                                 elicitations are mostly redundant
                                 (useful only as a temperature-stability
                                 sanity check). Persisted as
                                 source_profile_type='custom_subset'.
    """

    source_bins_to_show: Union[List[int], str] = field(default_factory=list)
    n_examples_per_source_bin: int = 2


@dataclass
class IntraBenchmarkConfig:
    """Complete configuration for an intra-benchmark calibration run."""

    api_key_anthropic: Optional[str]
    api_key_openai: Optional[str]
    api_key_gemini: Optional[str]

    # Data
    lyptus_repo_dir: Path
    forecasted_models: Optional[List[str]]  # None => all models in outcomes matrix
    drop_models: List[str]                  # excluded from outcomes matrix entirely
    benchmark_description: Optional[str]    # overrides default if set

    # Experimental design
    binning: BinningSettings
    target_selection: TargetSelectionSettings
    source_profile: SourceProfileSettings
    include_target_solution: bool

    # Workflow
    llm_settings: LLMSettings
    workflow_settings: WorkflowSettings

    # Paths
    prompts_dir: Path
    expert_profiles_file: Path
    output_dir: Path

    @property
    def runs_dir(self) -> Path:
        return self.output_dir

    @property
    def registry_file(self) -> Path:
        return self.output_dir / "run_registry.json"


def load_intra_benchmark_config(config_path: str | Path, base_dir: Optional[Path] = None) -> IntraBenchmarkConfig:
    """Load an intra-benchmark YAML config into an IntraBenchmarkConfig dataclass."""
    config_file = Path(config_path).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if base_dir is None:
        base_dir = config_file.parent

    with config_file.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML dictionary")

    def resolve_path(s: str, name: str) -> Path:
        if not s:
            raise ValueError(f"Missing path '{name}' in config")
        p = Path(s).expanduser()
        return p if p.is_absolute() else (base_dir / p).resolve()

    # API-key resolution: <intra>/.env -> <repo_root>/.env -> process env.
    repo_root = base_dir.parent
    api_key_anthropic = resolve_api_key("ANTHROPIC_API_KEY", base_dir, repo_root)
    api_key_openai = resolve_api_key("OPENAI_API_KEY", base_dir, repo_root)
    api_key_gemini = resolve_api_key("GEMINI_API_KEY", base_dir, repo_root)
    if not api_key_gemini:
        api_key_gemini = resolve_api_key("GOOGLE_API_KEY", base_dir, repo_root)

    # LLM settings
    llm_data = data.get("llm_settings") or {}
    thinking_data = llm_data.get("thinking") or {}
    llm_settings = LLMSettings(
        model=str(llm_data.get("model", "claude-sonnet-4-6")),
        temperature=float(llm_data.get("temperature", 0.8)),
        max_concurrent_calls=int(llm_data.get("max_concurrent_calls", 5)),
        rate_limit_calls=int(llm_data.get("rate_limit_calls", 45)),
        rate_limit_period=int(llm_data.get("rate_limit_period", 60)),
        thinking=ThinkingSettings(
            enabled=bool(thinking_data.get("enabled", False)),
            budget_tokens=int(thinking_data.get("budget_tokens", 10000)),
        ),
    )

    # Workflow settings
    wf_data = data.get("workflow_settings") or {}
    workflow_settings = WorkflowSettings(
        num_experts=int(wf_data.get("num_experts", 2)),
        delphi_rounds=int(wf_data.get("delphi_rounds", 1)),
        convergence_threshold=float(wf_data.get("convergence_threshold", 0.05)),
    )

    # Intra-benchmark settings
    ib = data.get("intra_benchmark_settings") or {}
    if not ib:
        raise ValueError("Missing 'intra_benchmark_settings' in config")

    bin_data = ib.get("binning") or {}
    binning = BinningSettings(
        n_bins=int(bin_data.get("n_bins", 5)),
        strategy=str(bin_data.get("strategy", "equal_count")),
        explicit_edges=([float(x) for x in bin_data["explicit_edges"]] if bin_data.get("explicit_edges") else None),
    )

    tgt_data = ib.get("target_selection") or {}
    raw_explicit = tgt_data.get("explicit_target_tasks")
    explicit_target_tasks = (
        {int(k): list(v) for k, v in raw_explicit.items()} if raw_explicit else None
    )
    target_selection = TargetSelectionSettings(
        selection_mode=str(tgt_data.get("selection_mode", "auto_sample")),
        explicit_target_tasks=explicit_target_tasks,
        sampling_seed=int(tgt_data.get("sampling_seed", 42)),
        n_target_tasks_per_cell=int(tgt_data.get("n_target_tasks_per_cell", 1)),
    )

    sp_data = ib.get("source_profile") or {}
    raw_bins = sp_data.get("source_bins_to_show")
    if raw_bins is None or (isinstance(raw_bins, list) and len(raw_bins) == 0):
        parsed_bins: Union[List[int], str] = []
    elif isinstance(raw_bins, str):
        if raw_bins.strip() != "all_except_target":
            raise ValueError(
                f"source_bins_to_show string must be 'all_except_target', got '{raw_bins}'"
            )
        parsed_bins = "all_except_target"
    elif isinstance(raw_bins, list):
        parsed_bins = [int(b) for b in raw_bins]
    else:
        raise ValueError(f"source_bins_to_show must be a list or 'all_except_target' string, got {type(raw_bins)}")
    source_profile = SourceProfileSettings(
        source_bins_to_show=parsed_bins,
        n_examples_per_source_bin=int(sp_data.get("n_examples_per_source_bin", 2)),
    )

    forecasted_models = ib.get("forecasted_models")
    forecasted_models = list(forecasted_models) if forecasted_models else None

    drop_models = list(ib.get("drop_models") or DEFAULT_DROP_MODELS)
    include_target_solution = bool(ib.get("include_target_solution", False))
    benchmark_description = ib.get("benchmark_description")

    lyptus_repo_dir = resolve_path(str(ib.get("lyptus_repo_dir", "")), "lyptus_repo_dir")

    prompts_dir = resolve_path(data.get("prompts_dir", "prompts"), "prompts_dir")
    expert_profiles_file = resolve_path(
        data.get("expert_profiles_file", "../inter_benchmark_calibration/input_data/expert_profiles/expert_profiles.yaml"),
        "expert_profiles_file",
    )
    output_dir = resolve_path(data.get("output_dir", "results"), "output_dir")

    cfg = IntraBenchmarkConfig(
        api_key_anthropic=api_key_anthropic,
        api_key_openai=api_key_openai,
        api_key_gemini=api_key_gemini,
        lyptus_repo_dir=lyptus_repo_dir,
        forecasted_models=forecasted_models,
        drop_models=drop_models,
        benchmark_description=benchmark_description,
        binning=binning,
        target_selection=target_selection,
        source_profile=source_profile,
        include_target_solution=include_target_solution,
        llm_settings=llm_settings,
        workflow_settings=workflow_settings,
        prompts_dir=prompts_dir,
        expert_profiles_file=expert_profiles_file,
        output_dir=output_dir,
    )

    logger.info("Intra-benchmark config loaded:")
    logger.info(f"  Lyptus repo: {cfg.lyptus_repo_dir}")
    logger.info(f"  n_bins: {cfg.binning.n_bins}, strategy: {cfg.binning.strategy}")
    logger.info(f"  forecasted_models: {cfg.forecasted_models or '(all in outcomes matrix)'}")
    logger.info(f"  drop_models: {cfg.drop_models}")
    logger.info(f"  K target tasks/cell: {cfg.target_selection.n_target_tasks_per_cell}")
    logger.info(f"  num_experts: {cfg.workflow_settings.num_experts}, "
                f"delphi_rounds: {cfg.workflow_settings.delphi_rounds}")
    return cfg
