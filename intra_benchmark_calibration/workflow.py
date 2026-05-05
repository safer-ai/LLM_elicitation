#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Async Delphi orchestration for intra-benchmark calibration.

For each (i, j, M, t) cell admitted by `task_selector.build_cell_plans`, we run
the Delphi process:

  - Round 1: each expert does (a) capability analysis, then (b) initial
    percentile estimation, two API calls. The stage-2 prompt sees the stage-1
    output of THE SAME expert.
  - Rounds 2+: each expert refines, one API call, given the previous-round
    responses of the OTHER experts.

Critical persistence guarantee: each (cell × expert × round) elicitation is
written to disk IMMEDIATELY after the API call returns, under an `asyncio.Lock`
in `results_handler.append_elicitation_row`. A mid-run crash loses at most one
in-flight call. We use `asyncio.gather` for parallel dispatch but each per-expert
coroutine writes its own row before returning, so the gather is just for
concurrency (no batched flush).
"""

from __future__ import annotations

import asyncio
import logging
import time
from asyncio import Semaphore
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None  # type: ignore[assignment]
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment]

from shared.data_models import ExpertProfile
from shared.llm_client import make_api_call
from shared.parsing import parse_probability_response

from intra_benchmark_calibration.config import IntraBenchmarkConfig
from intra_benchmark_calibration.lyptus_data import LyptusDataset
from intra_benchmark_calibration.prompt_builder import (
    assemble_prompts,
    format_target_task,
)
from intra_benchmark_calibration.results_handler import (
    RunHandles,
    append_cell_summary,
    append_elicitation_row,
    build_csv_row,
    build_json_record,
)
from intra_benchmark_calibration.task_selector import CellPlan

logger = logging.getLogger(__name__)


_MAX_TOKENS = 20000


async def _one_elicitation(
    *,
    plan: CellPlan,
    expert: ExpertProfile,
    round_num: int,
    prev_round_responses: Optional[List[Dict[str, Any]]],
    client: Union["AsyncAnthropic", "AsyncOpenAI"],
    semaphore: Semaphore,
    cfg: IntraBenchmarkConfig,
    prompt_templates: Dict[str, str],
    ground_truth_summary: Dict[str, float],
    benchmark_description: Optional[str],
    handles: RunHandles,
    dataset: LyptusDataset,
    forecaster_model: str,
    repeat_index: int,
    progress: Optional["tqdm"] = None,
) -> Dict[str, Any]:
    """
    Run ONE expert × ONE round elicitation, parse the response, and persist
    the row before returning.

    Returns the in-memory record (dict) so the round-aggregation logic can
    still compute means/stds across experts for the cell summary and so the
    next Delphi round can see this expert's previous response.

    `progress`: optional tqdm bar; advanced by 1 once persistence completes,
    regardless of success / failure. Tick happens in the `finally` of the
    inner _do() so it cannot be skipped on early returns.
    """
    persona_system_prompt = expert.get_persona_description()

    # Stage-1 analysis text is only generated in round 1, then carried into stage 2.
    raw_analysis: Optional[str] = None
    analysis_user_prompt: Optional[str] = None

    if round_num == 1:
        prep = assemble_prompts(
            plan=plan,
            persona_system_prompt=persona_system_prompt,
            prompt_templates=prompt_templates,
            round_num=1,
            prev_round_responses=None,
            technical_analysis=None,
            benchmark_description=benchmark_description,
            ground_truth_summary=ground_truth_summary,
            include_target_solution=cfg.include_target_solution,
        )
        analysis_user_prompt = prep.analysis_user_prompt

        analysis_text = await make_api_call(
            client, semaphore, cfg.llm_settings,
            persona_system_prompt, analysis_user_prompt or "", _MAX_TOKENS,
        )
        if analysis_text.startswith("Error:"):
            error = f"Analysis API call failed: {analysis_text}"
            return await _persist_failure(
                handles=handles, plan=plan, expert=expert, round_num=round_num,
                forecaster_model=forecaster_model, repeat_index=repeat_index,
                system_prompt=persona_system_prompt,
                analysis_user_prompt=analysis_user_prompt,
                estimation_user_prompt="(skipped — analysis failed)",
                raw_analysis=analysis_text, raw_estimation="",
                error=error, progress=progress,
            )
        raw_analysis = analysis_text

        # Now build stage-2 prompt with the just-produced analysis embedded.
        prep = assemble_prompts(
            plan=plan,
            persona_system_prompt=persona_system_prompt,
            prompt_templates=prompt_templates,
            round_num=1,
            prev_round_responses=None,
            technical_analysis=raw_analysis,
            benchmark_description=benchmark_description,
            ground_truth_summary=ground_truth_summary,
            include_target_solution=cfg.include_target_solution,
        )
        estimation_user_prompt = prep.estimation_user_prompt
    else:
        prep = assemble_prompts(
            plan=plan,
            persona_system_prompt=persona_system_prompt,
            prompt_templates=prompt_templates,
            round_num=round_num,
            prev_round_responses=prev_round_responses,
            technical_analysis=None,
            benchmark_description=benchmark_description,
            ground_truth_summary=ground_truth_summary,
            include_target_solution=cfg.include_target_solution,
        )
        estimation_user_prompt = prep.estimation_user_prompt

    estimation_text = await make_api_call(
        client, semaphore, cfg.llm_settings,
        persona_system_prompt, estimation_user_prompt, _MAX_TOKENS,
    )

    if estimation_text.startswith("Error:"):
        error = f"Estimation API call failed: {estimation_text}"
        return await _persist_failure(
            handles=handles, plan=plan, expert=expert, round_num=round_num,
            forecaster_model=forecaster_model, repeat_index=repeat_index,
            system_prompt=persona_system_prompt,
            analysis_user_prompt=analysis_user_prompt,
            estimation_user_prompt=estimation_user_prompt,
            raw_analysis=raw_analysis, raw_estimation=estimation_text,
            error=error, progress=progress,
        )

    parsed = parse_probability_response(estimation_text)
    error: Optional[str] = None
    if parsed.get("percentile_50th") is None:
        error = "Probability parsing failed (no p50 extracted)"

    return await _persist_success(
        handles=handles, plan=plan, expert=expert, round_num=round_num,
        forecaster_model=forecaster_model, repeat_index=repeat_index,
        system_prompt=persona_system_prompt,
        analysis_user_prompt=analysis_user_prompt,
        estimation_user_prompt=estimation_user_prompt,
        raw_analysis=raw_analysis, raw_estimation=estimation_text,
        parsed=parsed, error=error, progress=progress,
    )


async def _persist_success(
    *,
    handles: RunHandles,
    plan: CellPlan,
    expert: ExpertProfile,
    round_num: int,
    forecaster_model: str,
    repeat_index: int,
    system_prompt: str,
    analysis_user_prompt: Optional[str],
    estimation_user_prompt: str,
    raw_analysis: Optional[str],
    raw_estimation: str,
    parsed: Dict[str, Any],
    error: Optional[str],
    progress: Optional["tqdm"] = None,
) -> Dict[str, Any]:
    target_text = format_target_task(plan, include_solution=False)

    prompts_for_hash = [system_prompt, estimation_user_prompt]
    if analysis_user_prompt:
        prompts_for_hash.insert(1, analysis_user_prompt)

    csv_row = build_csv_row(
        handles=handles,
        plan=plan,
        forecaster_model=forecaster_model,
        repeat_index=repeat_index,
        expert_id=expert.name,
        delphi_round=round_num,
        parsed=parsed,
        prompts_for_hash=prompts_for_hash,
        target_prompt_chars=len(target_text),
    )
    json_record = build_json_record(
        csv_row=csv_row,
        plan=plan,
        forecaster_model=forecaster_model,
        repeat_index=repeat_index,
        expert_id=expert.name,
        delphi_round=round_num,
        system_prompt=system_prompt,
        analysis_user_prompt=analysis_user_prompt,
        estimation_user_prompt=estimation_user_prompt,
        raw_analysis=raw_analysis,
        raw_estimation=raw_estimation,
        error=error,
    )
    await append_elicitation_row(handles, csv_row=csv_row, json_elicitation_record=json_record)
    if progress is not None:
        progress.update(1)

    return {
        "expert": expert.name,
        "round": round_num,
        "percentile_25th": parsed.get("percentile_25th"),
        "percentile_50th": parsed.get("percentile_50th"),
        "percentile_75th": parsed.get("percentile_75th"),
        "estimate": parsed.get("estimate"),
        "rationale": (parsed.get("rationale") or "").strip(),
        "error": error,
    }


async def _persist_failure(
    *,
    handles: RunHandles,
    plan: CellPlan,
    expert: ExpertProfile,
    round_num: int,
    forecaster_model: str,
    repeat_index: int,
    system_prompt: str,
    analysis_user_prompt: Optional[str],
    estimation_user_prompt: str,
    raw_analysis: Optional[str],
    raw_estimation: str,
    error: str,
    progress: Optional["tqdm"] = None,
) -> Dict[str, Any]:
    target_text = format_target_task(plan, include_solution=False)
    prompts_for_hash = [system_prompt, estimation_user_prompt]
    if analysis_user_prompt:
        prompts_for_hash.insert(1, analysis_user_prompt)

    parsed = {"percentile_25th": None, "percentile_50th": None, "percentile_75th": None,
              "estimate": None, "rationale": ""}
    csv_row = build_csv_row(
        handles=handles,
        plan=plan,
        forecaster_model=forecaster_model,
        repeat_index=repeat_index,
        expert_id=expert.name,
        delphi_round=round_num,
        parsed=parsed,
        prompts_for_hash=prompts_for_hash,
        target_prompt_chars=len(target_text),
    )
    json_record = build_json_record(
        csv_row=csv_row,
        plan=plan,
        forecaster_model=forecaster_model,
        repeat_index=repeat_index,
        expert_id=expert.name,
        delphi_round=round_num,
        system_prompt=system_prompt,
        analysis_user_prompt=analysis_user_prompt,
        estimation_user_prompt=estimation_user_prompt,
        raw_analysis=raw_analysis,
        raw_estimation=raw_estimation,
        error=error,
    )
    await append_elicitation_row(handles, csv_row=csv_row, json_elicitation_record=json_record)
    if progress is not None:
        progress.update(1)
    return {"expert": expert.name, "round": round_num, "error": error,
            "percentile_25th": None, "percentile_50th": None,
            "percentile_75th": None, "estimate": None, "rationale": ""}


def _round_stats(responses: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    valid = [r["percentile_50th"] for r in responses
             if r.get("error") is None and isinstance(r.get("percentile_50th"), (int, float))]
    if len(valid) < 2:
        return {"n_valid": len(valid), "mean": None, "median": None, "std": None}
    return {
        "n_valid": len(valid),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "std": float(np.std(valid)),
    }


async def run_intra_benchmark_workflow(
    *,
    cfg: IntraBenchmarkConfig,
    cell_plans: List[CellPlan],
    experts: Sequence[ExpertProfile],
    prompt_templates: Dict[str, str],
    dataset: LyptusDataset,
    client: Union["AsyncAnthropic", "AsyncOpenAI"],
    semaphore: Semaphore,
    handles: RunHandles,
    forecaster_model: str,
    repeat_index: int = 1,
    progress: Optional["tqdm"] = None,
) -> Dict[str, Any]:
    """Iterate cells × Delphi rounds × experts, persisting each elicitation as it lands.

    `forecaster_model` is the LLM acting as the forecaster (rotated by the
    runner from `cfg.models_to_run`). `repeat_index` (1-based) identifies
    which independent repeat of the full pipeline this is when
    `workflow_settings.num_repeats > 1`.

    `progress`: optional shared tqdm bar owned by the runner. When None, this
    function creates its own bar sized for a single (model, repeat) chunk;
    when supplied, the runner is responsible for sizing and closing it.
    """
    run_start = time.time()
    n_cells = len(cell_plans)
    n_rounds = cfg.workflow_settings.delphi_rounds
    n_experts = len(experts)
    expected_total = n_cells * n_experts * n_rounds
    num_repeats_total = max(1, cfg.workflow_settings.num_repeats)
    rep_label = (
        f" (repeat {repeat_index}/{num_repeats_total})"
        if num_repeats_total > 1 else ""
    )
    logger.info(
        f"Workflow starting for forecaster '{forecaster_model}'{rep_label}: "
        f"{n_cells} cells × {n_experts} experts × {n_rounds} rounds = up to "
        f"{expected_total} elicitations"
    )

    benchmark_description = cfg.benchmark_description
    ground_truth_summary = dataset.outcomes.ground_truth_summary()

    n_elicitations_attempted = 0
    n_elicitations_completed = 0

    owns_progress = progress is None
    # `logging_redirect_tqdm` makes existing `logger.info` calls play nicely
    # with the bar (no shredding). The bar advances by 1 inside each
    # _persist_success / _persist_failure, so it ticks per-elicitation
    # regardless of success or API failure (with early-stop in convergence:
    # the bar may finish before reaching `expected_total`).
    if owns_progress:
        with logging_redirect_tqdm():
            progress = tqdm(
                total=expected_total,
                desc="Elicitations",
                unit="call",
                smoothing=0.05,
                dynamic_ncols=True,
            )
            try:
                n_elicitations_attempted, n_elicitations_completed = await _run_cells(
                    cfg=cfg, cell_plans=cell_plans, experts=experts,
                    prompt_templates=prompt_templates, dataset=dataset,
                    client=client, semaphore=semaphore, handles=handles,
                    ground_truth_summary=ground_truth_summary,
                    benchmark_description=benchmark_description,
                    forecaster_model=forecaster_model,
                    repeat_index=repeat_index,
                    progress=progress,
                )
            finally:
                progress.close()
    else:
        n_elicitations_attempted, n_elicitations_completed = await _run_cells(
            cfg=cfg, cell_plans=cell_plans, experts=experts,
            prompt_templates=prompt_templates, dataset=dataset,
            client=client, semaphore=semaphore, handles=handles,
            ground_truth_summary=ground_truth_summary,
            benchmark_description=benchmark_description,
            forecaster_model=forecaster_model,
            repeat_index=repeat_index,
            progress=progress,
        )

    duration = time.time() - run_start
    logger.info(
        f"Workflow done for '{forecaster_model}'{rep_label} in {duration:.1f}s | "
        f"completed {n_elicitations_completed}/{n_elicitations_attempted} elicitations "
        f"(of up to {expected_total} possible)"
    )
    return {
        "n_elicitations_attempted": n_elicitations_attempted,
        "n_elicitations_completed": n_elicitations_completed,
        "duration_seconds": duration,
    }


async def _run_cells(
    *,
    cfg: IntraBenchmarkConfig,
    cell_plans: List[CellPlan],
    experts: Sequence[ExpertProfile],
    prompt_templates: Dict[str, str],
    dataset: LyptusDataset,
    client: Union["AsyncAnthropic", "AsyncOpenAI"],
    semaphore: Semaphore,
    handles: RunHandles,
    ground_truth_summary: Dict[str, float],
    benchmark_description: Optional[str],
    forecaster_model: str,
    repeat_index: int,
    progress: "tqdm",
) -> tuple:
    """The cell × round × expert nested loop. Updates `progress` per elicitation
    via the _persist_* call sites."""
    n_cells = len(cell_plans)
    n_elicitations_attempted = 0
    n_elicitations_completed = 0

    for idx, plan in enumerate(cell_plans, 1):
        i_str = "ALL" if plan.source_bin_i is None else f"i{plan.source_bin_i}"
        progress.set_postfix_str(
            f"{forecaster_model[:14]} r{repeat_index} cell {idx}/{n_cells}: "
            f"M={plan.forecasted_model[:14]} {i_str}->j{plan.target_bin_j}"
        )
        logger.info(f"\n--- Cell {idx}/{n_cells}: {plan.cell_id} (true outcome={int(plan.target_outcome)}) ---")
        round_responses_history: List[List[Dict[str, Any]]] = []
        cell_round_stats: List[Dict[str, Any]] = []

        for round_num in range(1, cfg.workflow_settings.delphi_rounds + 1):
            round_start = time.time()
            prev = round_responses_history[-1] if round_responses_history else None
            coros = [
                _one_elicitation(
                    plan=plan, expert=ex, round_num=round_num,
                    prev_round_responses=prev,
                    client=client, semaphore=semaphore, cfg=cfg,
                    prompt_templates=prompt_templates,
                    ground_truth_summary=ground_truth_summary,
                    benchmark_description=benchmark_description,
                    handles=handles, dataset=dataset,
                    forecaster_model=forecaster_model,
                    repeat_index=repeat_index,
                    progress=progress,
                ) for ex in experts
            ]
            n_elicitations_attempted += len(coros)
            results = await asyncio.gather(*coros, return_exceptions=True)

            processed: List[Dict[str, Any]] = []
            for ex, res in zip(experts, results):
                if isinstance(res, Exception):
                    logger.error(f"Expert {ex.name} round {round_num} raised: {res}", exc_info=res)
                    processed.append({"expert": ex.name, "round": round_num,
                                      "error": f"Unhandled exception: {res}",
                                      "percentile_25th": None, "percentile_50th": None,
                                      "percentile_75th": None, "estimate": None, "rationale": ""})
                else:
                    processed.append(res)
                    if res.get("error") is None and res.get("percentile_50th") is not None:
                        n_elicitations_completed += 1

            round_responses_history.append(processed)
            stats = _round_stats(processed)
            cell_round_stats.append({"round": round_num, **stats,
                                     "duration_s": round(time.time() - round_start, 2)})

            mean_str = f"{stats['mean']:.3f}" if stats["mean"] is not None else "N/A"
            std_str = f"{stats['std']:.3f}" if stats["std"] is not None else "N/A"
            logger.info(f"  Round {round_num}: n_valid={stats['n_valid']}/{len(experts)}, "
                        f"mean p50={mean_str}, std={std_str}")

            if (cfg.workflow_settings.delphi_rounds > 1
                    and stats["std"] is not None
                    and stats["std"] < cfg.workflow_settings.convergence_threshold):
                logger.info(f"  Converged at round {round_num} (std {stats['std']:.4f} < "
                            f"{cfg.workflow_settings.convergence_threshold})")
                break

        cell_summary = {
            "cell_id": plan.cell_id,
            "forecaster_model": forecaster_model,
            "repeat_index": repeat_index,
            "source_profile_type": plan.source_profile_type,
            "source_bin_i": plan.source_bin_i,
            "source_bins_shown": plan.source_bins_to_show,
            "target_bin_j": plan.target_bin_j,
            "forecasted_model": plan.forecasted_model,
            "target_task_id": plan.target_task.task_id,
            "target_task_family": plan.target_task.task_family,
            "target_fst_minutes": plan.target_task.fst_minutes,
            "outcome": int(plan.target_outcome),
            "per_bin_anchor_task_ids": [p.anchor.task_id for p in plan.profiles],
            "per_bin_M_pass_rates": [p.pass_rate for p in plan.profiles],
            "rounds": cell_round_stats,
        }
        await append_cell_summary(handles, cell_summary)

    return n_elicitations_attempted, n_elicitations_completed
