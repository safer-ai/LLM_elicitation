#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entry point for the intra-benchmark calibration experiment.

Wires together: config -> Lyptus data load -> binning -> cell planning ->
expert + prompt loading -> async LLM client(s) -> async Delphi workflow ->
progressive results persistence.

Usage:
    python intra_benchmark_calibration/run_calibration.py \\
        -c intra_benchmark_calibration/config_example.yaml -d
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import sys
from dataclasses import replace
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

from tqdm.auto import tqdm  # noqa: E402
from tqdm.contrib.logging import logging_redirect_tqdm  # noqa: E402

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
    snap.pop("api_key_gemini", None)
    return snap


def _config_for_forecaster(base_cfg: IntraBenchmarkConfig, model: str) -> IntraBenchmarkConfig:
    """Return a copy of `base_cfg` with `llm_settings.model` rebound to `model`.

    All other settings (rate limits, reasoning_effort, paths, API keys) are
    preserved so a single `load_intra_benchmark_config` call covers a
    multi-model run.
    """
    new_llm = replace(base_cfg.llm_settings, model=model)
    return replace(base_cfg, llm_settings=new_llm)


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

    # 5. Init run handles
    handles = initialize_run(
        output_base_dir=cfg.output_dir,
        models_run=cfg.models_to_run,
        num_experts=len(experts),
        delphi_rounds=cfg.workflow_settings.delphi_rounds,
        num_repeats=cfg.workflow_settings.num_repeats,
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

    # 6. Workflow: outer loop over forecaster models × repeats.
    num_repeats = max(1, cfg.workflow_settings.num_repeats)
    n_cells = len(cell_plans)
    n_rounds = cfg.workflow_settings.delphi_rounds
    n_experts = len(experts)
    expected_per_run = n_cells * n_experts * n_rounds
    expected_total = expected_per_run * len(cfg.models_to_run) * num_repeats
    logger.info(
        f"Total expected elicitations across {len(cfg.models_to_run)} forecaster "
        f"model(s) × {num_repeats} repeat(s) × {n_cells} cells × {n_experts} experts "
        f"× {n_rounds} rounds = up to {expected_total}"
    )

    total_attempted = 0
    total_completed = 0
    crashed_models: list = []

    with logging_redirect_tqdm():
        progress = tqdm(
            total=expected_total,
            desc="Elicitations",
            unit="call",
            smoothing=0.05,
            dynamic_ncols=True,
        )
        try:
            for forecaster_model in cfg.models_to_run:
                per_model_cfg = _config_for_forecaster(cfg, forecaster_model)
                provider = per_model_cfg.inferred_api_provider
                logger.info(
                    f"\n=== Forecaster model: {forecaster_model} (provider: {provider}, "
                    f"repeats: {num_repeats}) ==="
                )

                try:
                    client = initialize_client(
                        api_key_anthropic=per_model_cfg.api_key_anthropic,
                        api_key_openai=per_model_cfg.api_key_openai,
                        model=per_model_cfg.llm_settings.model,
                        api_key_gemini=per_model_cfg.api_key_gemini,
                    )
                    logger.info(f"API client initialised for {provider}.")
                except (ImportError, ValueError) as e:
                    logger.error(
                        f"Failed to initialise client for forecaster '{forecaster_model}': {e}",
                        exc_info=True,
                    )
                    crashed_models.append((forecaster_model, str(e)))
                    # Advance the bar past this model's expected slots so the
                    # ETA stays sensible.
                    skipped = expected_per_run * num_repeats
                    progress.update(skipped)
                    continue

                semaphore = asyncio.Semaphore(per_model_cfg.llm_settings.max_concurrent_calls)

                for repeat_index in range(1, num_repeats + 1):
                    if num_repeats > 1:
                        logger.info(
                            f"\n--- {forecaster_model}: starting repeat {repeat_index}/{num_repeats} ---"
                        )
                    try:
                        result = await run_intra_benchmark_workflow(
                            cfg=per_model_cfg,
                            cell_plans=cell_plans,
                            experts=experts,
                            prompt_templates=prompt_templates,
                            dataset=dataset,
                            client=client,
                            semaphore=semaphore,
                            handles=handles,
                            forecaster_model=forecaster_model,
                            repeat_index=repeat_index,
                            progress=progress,
                        )
                    except Exception as e:
                        logger.error(
                            f"Workflow crashed for forecaster '{forecaster_model}' "
                            f"(repeat {repeat_index}/{num_repeats}): {e}",
                            exc_info=True,
                        )
                        crashed_models.append((forecaster_model, str(e)))
                        # Best-effort progress advance for the unfinished slot.
                        continue
                    total_attempted += result["n_elicitations_attempted"]
                    total_completed += result["n_elicitations_completed"]
        finally:
            progress.close()

    # 7. Finalise (single registry entry covering all models / repeats).
    finalize_run(
        handles,
        n_elicitations_attempted=total_attempted,
        n_elicitations_succeeded=total_completed,
    )
    update_registry(
        registry_file=cfg.registry_file,
        run_id=handles.run_id,
        models_run=cfg.models_to_run,
        num_experts=len(experts),
        delphi_rounds=cfg.workflow_settings.delphi_rounds,
        num_repeats=num_repeats,
        n_elicitations_attempted=total_attempted,
        n_elicitations_completed=total_completed,
        output_path=handles.run_dir,
        config_file=str(Path(config_path).resolve()),
        timestamp_start=handles.run_id,  # already in YYYYMMDD_HHMMSS form
    )

    logger.info(f"Run complete. CSV: {handles.csv_path}")
    logger.info(f"           JSON: {handles.json_path}")
    logger.info(f"  Successful elicitations: {total_completed} / {total_attempted}")
    if crashed_models:
        logger.warning(
            f"  {len(crashed_models)} forecaster (model, repeat) slot(s) crashed; "
            f"see log above. Examples: {crashed_models[:3]}"
        )
        return 2
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
