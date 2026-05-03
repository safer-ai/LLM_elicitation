#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entry point for the intra-benchmark calibration experiment.

Wires together: config -> Lyptus data load -> binning -> cell planning ->
expert + prompt loading -> async LLM client -> async Delphi workflow ->
progressive results persistence.

Usage:
    python intra_benchmark_calibration/run_calibration.py \\
        -c intra_benchmark_calibration/config_full.yaml -d
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from intra_benchmark_calibration.binning import compute_bins  # noqa: E402
from intra_benchmark_calibration.config import (  # noqa: E402
    IntraBenchmarkConfig,
    load_intra_benchmark_config,
)
from intra_benchmark_calibration.lyptus_data import load_lyptus_dataset  # noqa: E402
from intra_benchmark_calibration.results_handler import (  # noqa: E402
    finalize_run,
    initialize_run,
    update_registry,
)
from intra_benchmark_calibration.task_selector import build_cell_plans  # noqa: E402
from intra_benchmark_calibration.workflow import run_intra_benchmark_workflow  # noqa: E402

from shared.llm_client import initialize_client  # noqa: E402
from shared.loaders import load_experts, load_prompts  # noqa: E402

logger = logging.getLogger("IntraBenchmarkCalibration")


def _config_snapshot(cfg: IntraBenchmarkConfig) -> dict:
    """JSON-safe snapshot of the config (without API keys)."""
    def to_serialisable(v):
        if dataclasses.is_dataclass(v):
            return {k: to_serialisable(getattr(v, k)) for k in v.__dataclass_fields__}
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, (list, tuple)):
            return [to_serialisable(x) for x in v]
        if isinstance(v, dict):
            return {k: to_serialisable(x) for k, x in v.items()}
        return v
    snap = to_serialisable(cfg)
    snap.pop("api_key_anthropic", None)
    snap.pop("api_key_openai", None)
    return snap


async def main(config_path: str) -> int:
    logger.info("--- Starting Intra-Benchmark Calibration Experiment ---")
    logger.info(f"Config: {config_path}")

    # 1. Config
    try:
        cfg = load_intra_benchmark_config(config_path)
    except Exception as e:
        logger.error(f"Failed to load config: {e}", exc_info=True)
        return 1

    # 2. Data
    dataset = load_lyptus_dataset(cfg.lyptus_repo_dir, drop_models=cfg.drop_models)
    forecasted = cfg.forecasted_models or list(dataset.outcomes.models)
    missing = [m for m in forecasted if m not in dataset.outcomes.models]
    if missing:
        logger.error(f"forecasted_models not in outcomes matrix: {missing}")
        return 1

    # 3. Bins + cell plans
    bins = compute_bins(
        dataset.fst_array(),
        n_bins=cfg.binning.n_bins,
        strategy=cfg.binning.strategy,
        explicit_edges=cfg.binning.explicit_edges,
    )

    cell_plans = build_cell_plans(
        bins=bins,
        dataset=dataset,
        forecasted_models=forecasted,
        source_bins_to_show=cfg.source_profile.source_bins_to_show,
        n_examples_per_source_bin=cfg.source_profile.n_examples_per_source_bin,
        n_target_tasks_per_cell=cfg.target_selection.n_target_tasks_per_cell,
        target_sampling_seed=cfg.target_selection.sampling_seed,
        explicit_target_tasks=cfg.target_selection.explicit_target_tasks,
    )
    if not cell_plans:
        logger.error("No admissible cell plans were produced. Aborting.")
        return 1

    # 4. Prompts + experts
    prompt_templates = load_prompts(cfg.prompts_dir)
    if not prompt_templates:
        logger.error(f"Failed to load prompts from {cfg.prompts_dir}")
        return 1
    experts_all = load_experts(cfg.expert_profiles_file)
    if not experts_all:
        logger.error(f"Failed to load experts from {cfg.expert_profiles_file}")
        return 1
    n = cfg.workflow_settings.num_experts
    experts = experts_all[:n]
    logger.info(f"Using {len(experts)}/{len(experts_all)} expert profiles: "
                f"{[e.name for e in experts]}")

    # 5. LLM client + concurrency
    try:
        client = initialize_client(
            cfg.api_key_anthropic, cfg.api_key_openai, cfg.llm_settings.model
        )
    except (ImportError, ValueError) as e:
        logger.error(f"Failed to initialise LLM client: {e}", exc_info=True)
        return 1
    semaphore = asyncio.Semaphore(cfg.llm_settings.max_concurrent_calls)

    # 6. Init run handles
    handles = initialize_run(
        output_base_dir=cfg.output_dir,
        model=cfg.llm_settings.model,
        num_experts=len(experts),
        delphi_rounds=cfg.workflow_settings.delphi_rounds,
        temperature=cfg.llm_settings.temperature,
        config_snapshot=_config_snapshot(cfg),
        dataset_provenance=dataset.provenance_dict(),
        bin_definition={
            "strategy": bins.strategy,
            "n_bins": bins.n_bins,
            "edges_minutes": bins.edges_minutes,
            "n_tasks_per_bin": [int(c) for c in
                                __import__("numpy").bincount(bins.bin_index_per_task,
                                                              minlength=bins.n_bins)],
        },
        n_cells_planned=len(cell_plans),
    )
    logger.info(f"Run ID: {handles.run_id}")
    logger.info(f"Run dir: {handles.run_dir}")

    # 7. Workflow
    try:
        result = await run_intra_benchmark_workflow(
            cfg=cfg,
            cell_plans=cell_plans,
            experts=experts,
            prompt_templates=prompt_templates,
            dataset=dataset,
            client=client,
            semaphore=semaphore,
            handles=handles,
        )
    except Exception as e:
        logger.error(f"Workflow crashed: {e}", exc_info=True)
        return 1

    # 8. Finalise
    finalize_run(
        handles,
        n_elicitations_attempted=result["n_elicitations_attempted"],
        n_elicitations_succeeded=result["n_elicitations_completed"],
    )
    update_registry(
        registry_file=cfg.registry_file,
        run_id=handles.run_id,
        model=cfg.llm_settings.model,
        num_experts=len(experts),
        delphi_rounds=cfg.workflow_settings.delphi_rounds,
        n_elicitations_attempted=result["n_elicitations_attempted"],
        n_elicitations_completed=result["n_elicitations_completed"],
        output_path=handles.run_dir,
        config_file=str(Path(config_path).resolve()),
        timestamp_start=handles.run_id,  # already in YYYYMMDD_HHMMSS form
    )

    logger.info(f"Run complete. CSV: {handles.csv_path}")
    logger.info(f"           JSON: {handles.json_path}")
    logger.info(f"  Successful elicitations: {result['n_elicitations_completed']} / {result['n_elicitations_attempted']}")
    return 0


def cli() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", required=True, help="Path to YAML config")
    p.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")
    args = p.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=log_level, format=log_format,
                        handlers=[logging.StreamHandler(sys.stdout)])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger.setLevel(log_level)

    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        return 1
    try:
        return asyncio.run(main(args.config))
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(cli())
