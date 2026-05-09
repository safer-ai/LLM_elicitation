#!/usr/bin/env python3
"""
Recovery script for experiment F_full_loop_k3.

Reads the partial run's CSV, finds rows where p50 is missing (API failures),
rebuilds exactly those cell plans, re-runs only those elicitations, and appends
the new results to the existing run's CSV and JSON files.

Usage (from intra_benchmark_calibration/):
    python3 experiments/F_full_loop_k3/recover_failed.py \
        --run-dir experiments/F_full_loop_k3/results/20260504_164356 \
        --config experiments/F_full_loop_k3/config_full_loop_k3.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# ---- path setup (same as run_calibration.py) ----
EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from config import load_intra_benchmark_config          # local experiment copy
from task_selector import build_cell_plans              # local experiment copy

from intra_benchmark_calibration.binning import compute_bins
from intra_benchmark_calibration.lyptus_data import load_lyptus_dataset
from intra_benchmark_calibration.results_handler import RunHandles
from intra_benchmark_calibration.workflow import _one_elicitation
from shared.llm_client import initialize_client
from shared.loaders import load_experts, load_prompts

logger = logging.getLogger("Recovery")


async def main(run_dir: Path, config_path: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # 1. Find failed (condition_id, expert_id) pairs
    csv_path = next(run_dir.glob("*_estimates.csv"))
    json_path = next(run_dir.glob("*_results.json"))
    df = pd.read_csv(csv_path)

    failed = df[df["p50"].isna()][["condition_id", "expert_id"]]
    failed_keys = set(map(tuple, failed.values.tolist()))
    failed_condition_ids = {k[0] for k in failed_keys}

    logger.info(f"Found {len(failed_keys)} failed (condition_id, expert) pairs to retry.")
    if not failed_keys:
        logger.info("Nothing to recover. Exiting.")
        return

    # 2. Load config + data (same as original run)
    cfg = load_intra_benchmark_config(str(config_path))
    dataset = load_lyptus_dataset(cfg.lyptus_repo_dir, drop_models=cfg.drop_models)
    forecasted = cfg.forecasted_models or list(dataset.outcomes.models)

    # 3. Rebuild all cell plans (same seed → same tasks), filter to failed ones
    bins = compute_bins(
        dataset.fst_array(),
        n_bins=cfg.binning.n_bins,
        strategy=cfg.binning.strategy,
        explicit_edges=cfg.binning.explicit_edges,
    )
    all_plans = build_cell_plans(
        bins=bins,
        dataset=dataset,
        forecasted_models=forecasted,
        source_bins_to_show=cfg.source_profile.source_bins_to_show,
        n_examples_per_source_bin=cfg.source_profile.n_examples_per_source_bin,
        n_target_tasks_per_cell=cfg.target_selection.n_target_tasks_per_cell,
        target_sampling_seed=cfg.target_selection.sampling_seed,
        explicit_target_tasks=cfg.target_selection.explicit_target_tasks,
        resample_anchors_per_target=cfg.source_profile.resample_anchors_per_target,
    )
    plans_to_retry = [p for p in all_plans if p.cell_id in failed_condition_ids]
    logger.info(f"Matched {len(plans_to_retry)}/{len(all_plans)} cell plans to retry.")

    # 4. Prompts + experts
    prompt_templates = load_prompts(cfg.prompts_dir)
    all_experts = load_experts(cfg.expert_profiles_file)
    experts = all_experts[: cfg.workflow_settings.num_experts]

    # 5. LLM client
    client = initialize_client(cfg.api_key_anthropic, cfg.api_key_openai, cfg.llm_settings.model)
    semaphore = asyncio.Semaphore(cfg.llm_settings.max_concurrent_calls)

    # 6. Attach to existing run files (no new run dir)
    handles = RunHandles(
        run_id=run_dir.name,
        run_dir=run_dir,
        csv_path=csv_path,
        json_path=json_path,
    )

    ground_truth_summary = dataset.outcomes.ground_truth_summary()

    total = sum(
        1 for p in plans_to_retry for ex in experts
        if (p.cell_id, ex.name) in failed_keys
    )
    logger.info(f"Will re-run {total} elicitations (appending to existing run files).")

    # 7. Run only the failed elicitations, appending to the existing CSV/JSON
    with logging_redirect_tqdm():
        progress = tqdm(total=total, desc="Recovery", unit="call", smoothing=0.05)
        try:
            for plan in plans_to_retry:
                coros = [
                    _one_elicitation(
                        plan=plan,
                        expert=ex,
                        round_num=1,
                        prev_round_responses=None,
                        client=client,
                        semaphore=semaphore,
                        cfg=cfg,
                        prompt_templates=prompt_templates,
                        ground_truth_summary=ground_truth_summary,
                        benchmark_description=cfg.benchmark_description,
                        handles=handles,
                        dataset=dataset,
                        progress=progress,
                    )
                    for ex in experts
                    if (plan.cell_id, ex.name) in failed_keys
                ]
                if coros:
                    await asyncio.gather(*coros, return_exceptions=True)
        finally:
            progress.close()

    # 8. Report
    df_after = pd.read_csv(csv_path)
    still_missing = df_after["p50"].isna().sum()
    recovered = len(failed_keys) - still_missing + df["p50"].isna().sum() - still_missing
    logger.info(f"Done. CSV rows: {len(df_after)}.  p50 missing before: {len(failed_keys)},  after: {still_missing}.")


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()
    asyncio.run(main(args.run_dir.resolve(), args.config.resolve()))


if __name__ == "__main__":
    cli()
