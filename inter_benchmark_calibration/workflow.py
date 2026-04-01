#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Workflow orchestration for inter-benchmark calibration experiments.

Implements the Delphi estimation process for predicting P(model solves target task
on benchmark B | model scores in bin X on source benchmark A).
"""

import asyncio
import datetime
import logging
import time
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional, Union
from asyncio import Semaphore

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

from shared.data_models import ExpertProfile, BenchmarkTask, Benchmark
from shared.llm_client import make_api_call
from shared.parsing import parse_probability_response
from config import InterBenchmarkConfig
from task_selector import (
    get_representative_source_tasks,
    get_task_at_percentile,
    format_tasks_list_for_prompt
)
from results_handler import (
    initialize_inter_benchmark_run,
    append_prediction_to_csv,
    save_prediction_to_json,
    finalize_inter_benchmark_run,
    add_inter_benchmark_run_to_registry
)

logger = logging.getLogger(__name__)


def _build_source_description(config: InterBenchmarkConfig) -> str:
    """Build a combined description of all source benchmarks for prompts."""
    parts = []
    for src in config.source_benchmarks:
        parts.append(f"=== Source Benchmark: {src.name} ===\n{src.benchmark_description.strip()}")
    return "\n\n".join(parts)


def _build_source_tasks_text(
    source_tasks_by_benchmark: Dict[str, List[BenchmarkTask]],
    source_metrics: Dict[str, List[str]]
) -> str:
    """Build formatted source tasks text across all source benchmarks."""
    parts = []
    for bm_name, tasks in source_tasks_by_benchmark.items():
        metrics = source_metrics.get(bm_name, [])
        tasks_text = format_tasks_list_for_prompt(tasks, metrics)
        parts.append(f"--- {bm_name} (capability ceiling tasks) ---\n{tasks_text}")
    return "\n\n".join(parts)


async def _run_expert_round(
    expert: ExpertProfile,
    source_tasks_text: str,
    source_description: str,
    target_task: BenchmarkTask,
    target_benchmark_description: str,
    target_metrics_to_use: List[str],
    round_num: int,
    prev_round_responses: Optional[List[Dict[str, Any]]],
    client: Union[AsyncAnthropic, AsyncOpenAI],
    semaphore: Semaphore,
    config: InterBenchmarkConfig,
    prompts: Dict[str, str]
) -> Dict[str, Any]:
    """Run a single expert's estimation for one (source_bin, target_percentile) pair."""
    ename = expert.name
    epersona = expert.get_persona_description()
    system_prompt_base = epersona

    expert_result: Dict[str, Any] = {
        "expert": ename,
        "estimate": None,
        "percentile_25th": None,
        "percentile_50th": None,
        "percentile_75th": None,
        "rationale": ""
    }

    target_task_metrics_str = ""
    if target_metrics_to_use:
        metric_parts = []
        for metric_name in target_metrics_to_use:
            if metric_name in target_task.metrics:
                metric_parts.append(f"{metric_name}={target_task.metrics[metric_name]}")
        if metric_parts:
            target_task_metrics_str = ", ".join(metric_parts)

    prompt_data = {
        "source_benchmarks_description": source_description,
        "source_tasks_list": source_tasks_text,
        "target_benchmark_description": target_benchmark_description,
        "target_task_name": target_task.name,
        "target_task_description": target_task.description,
        "target_task_metrics": target_task_metrics_str if target_task_metrics_str else "N/A"
    }

    max_tokens = 8000

    try:
        if round_num == 1:
            # Stage 1: Cross-benchmark analysis
            analysis_template = prompts.get('cross_benchmark_analysis')
            if not analysis_template:
                raise ValueError("Missing 'cross_benchmark_analysis' prompt template")

            analysis_user_prompt = analysis_template.format(**prompt_data)
            expert_result["analysis_system_prompt"] = system_prompt_base
            expert_result["analysis_user_prompt"] = analysis_user_prompt

            analysis_text = await make_api_call(
                client, semaphore, config.llm_settings, system_prompt_base,
                analysis_user_prompt, max_tokens
            )
            expert_result["raw_analysis"] = analysis_text

            if analysis_text.startswith("Error:"):
                expert_result["error"] = f"Analysis API Call Failed: {analysis_text}"
                return expert_result

            prompt_data["technical_analysis"] = analysis_text

            # Stage 2: Initial estimation
            estimation_template = prompts.get('initial_cross_benchmark_estimation')
            if not estimation_template:
                raise ValueError("Missing 'initial_cross_benchmark_estimation' prompt template")

            estimation_user_prompt = estimation_template.format(**prompt_data)
            expert_result["estimation_system_prompt"] = system_prompt_base
            expert_result["estimation_user_prompt"] = estimation_user_prompt

            estimation_text = await make_api_call(
                client, semaphore, config.llm_settings, system_prompt_base,
                estimation_user_prompt, max_tokens
            )
            expert_result["raw_estimation"] = estimation_text

            if estimation_text.startswith("Error:"):
                expert_result["error"] = f"Initial Estimation API Call Failed: {estimation_text}"
            else:
                parsed = parse_probability_response(estimation_text)
                expert_result["estimate"] = parsed.get("estimate")
                expert_result["percentile_25th"] = parsed.get("percentile_25th")
                expert_result["percentile_50th"] = parsed.get("percentile_50th")
                expert_result["percentile_75th"] = parsed.get("percentile_75th")
                expert_result["rationale"] = parsed.get("rationale", "")

                if any(v is None for k, v in expert_result.items() if k not in ("error", "raw_analysis", "raw_estimation", "analysis_system_prompt", "analysis_user_prompt", "estimation_system_prompt", "estimation_user_prompt")):
                    logger.warning(f"R1 - Expert '{ename}' probability parsing incomplete")
                    expert_result["error"] = "Probability Parsing Failed"

        else:
            # Refinement rounds
            context_lines = ["\n---\nOther experts' estimates from the previous round:\n---\n"]

            if prev_round_responses:
                others = [r for r in prev_round_responses if r.get("expert") != ename]
                if others:
                    for r in others:
                        p25 = f"{r.get('percentile_25th', 'N/A'):.3f}" if isinstance(r.get('percentile_25th'), float) else "N/A"
                        p50 = f"{r.get('percentile_50th', 'N/A'):.3f}" if isinstance(r.get('percentile_50th'), float) else "N/A"
                        p75 = f"{r.get('percentile_75th', 'N/A'):.3f}" if isinstance(r.get('percentile_75th'), float) else "N/A"
                        rationale = r.get('rationale', 'N/A')
                        if len(rationale) > 300:
                            rationale = rationale[:300] + "..."
                        context_lines.append(
                            f"Expert: {r['expert']}\n"
                            f"Percentiles: 25th={p25}, 50th (median)={p50}, 75th={p75}\n"
                            f"Rationale: {rationale}\n"
                        )
                else:
                    context_lines.append("(No other valid responses received in the previous round)")
            else:
                context_lines.append("(No previous round responses available)")

            context_lines.append("\n---\n")
            prompt_data["context"] = "\n".join(context_lines)

            subsequent_template = prompts.get('subsequent_cross_benchmark_estimation')
            if not subsequent_template:
                raise ValueError("Missing 'subsequent_cross_benchmark_estimation' prompt template")

            subsequent_user_prompt = subsequent_template.format(**prompt_data)
            expert_result["estimation_system_prompt"] = system_prompt_base
            expert_result["estimation_user_prompt"] = subsequent_user_prompt

            estimation_text = await make_api_call(
                client, semaphore, config.llm_settings, system_prompt_base,
                subsequent_user_prompt, max_tokens
            )
            expert_result["raw_estimation"] = estimation_text

            if estimation_text.startswith("Error:"):
                expert_result["error"] = f"Subsequent Estimation API Call Failed: {estimation_text}"
            else:
                parsed = parse_probability_response(estimation_text)
                expert_result["estimate"] = parsed.get("estimate")
                expert_result["percentile_25th"] = parsed.get("percentile_25th")
                expert_result["percentile_50th"] = parsed.get("percentile_50th")
                expert_result["percentile_75th"] = parsed.get("percentile_75th")
                expert_result["rationale"] = parsed.get("rationale", "")

                if any(v is None for k, v in expert_result.items() if k not in ("error", "raw_estimation", "estimation_system_prompt", "estimation_user_prompt")):
                    logger.warning(f"R{round_num} - Expert '{ename}' probability parsing incomplete")
                    expert_result["error"] = "Probability Parsing Failed"

    except ValueError as ve:
        logger.error(f"ValueError during expert round for {ename}: {ve}")
        expert_result["error"] = f"Configuration/Prompt Error: {ve}"
    except Exception as e:
        logger.error(f"Unexpected error during expert round for {ename}: {e}", exc_info=True)
        expert_result["error"] = f"Unexpected Workflow Error: {e}"

    return expert_result


async def run_inter_benchmark_estimation(
    client: Union[AsyncAnthropic, AsyncOpenAI],
    semaphore: Semaphore,
    config: InterBenchmarkConfig,
    input_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main orchestration for inter-benchmark calibration.

    Iterates over source_bins x target_percentiles, running the full Delphi process
    for each pair.

    Args:
        client: LLM API client
        semaphore: Concurrency semaphore
        config: InterBenchmarkConfig
        input_data: Dict with prompts, experts, source_benchmarks (dict of name -> Benchmark),
                    target_benchmark (Benchmark), ground_truth
    """
    run_start_time = time.time()
    logger.info("\n\n--- Starting Inter-Benchmark Calibration Workflow ---")

    prompts = input_data.get('prompts', {})
    experts = input_data.get('experts', [])
    source_benchmarks = input_data.get('source_benchmarks', {})
    target_benchmark = input_data.get('target_benchmark')
    ground_truth_data = input_data.get('ground_truth', {})

    if not prompts or not experts or not target_benchmark:
        return {"error": "Missing required input data"}

    # Select experts
    num_experts = config.workflow_settings.num_experts
    if num_experts is not None and num_experts > 0:
        experts_to_use = experts[:num_experts]
    else:
        experts_to_use = experts
    logger.info(f"Using {len(experts_to_use)} experts")

    # Load ground truth entries
    gt_entries = ground_truth_data.get('ground_truth', [])
    gt_entries = [e for e in gt_entries if e.get('sufficient_sample', False)]

    if not gt_entries:
        return {"error": "No ground truth entries with sufficient sample"}

    source_description = _build_source_description(config)
    target_desc = config.target_benchmark.benchmark_description

    # Initialise run
    run_info = initialize_inter_benchmark_run(
        source_name=config.primary_source.name,
        target_name=config.target_benchmark.name,
        model=config.llm_settings.model,
        num_experts=len(experts_to_use),
        delphi_rounds=config.workflow_settings.delphi_rounds,
        temperature=config.llm_settings.temperature,
        output_base_dir=config.output_dir,
        n_source_bins=ground_truth_data['metadata'].get('n_source_bins', 0),
        n_target_percentiles=ground_truth_data['metadata'].get('n_target_percentiles', 0),
        ground_truth_file=str(config.ground_truth_file)
    )

    if not run_info:
        return {"error": "Failed to initialise run"}

    logger.info(f"Run ID: {run_info['run_id']}")
    logger.info(f"Output: {run_info['run_dir']}")

    predictions_completed = 0
    predictions_attempted = len(gt_entries)

    for pair_idx, gt_entry in enumerate(gt_entries, 1):
        src_bin = gt_entry['source_bin']
        src_bin_range = gt_entry['source_bin_range']
        target_pct = gt_entry['target_percentile']
        gt_p_solve = gt_entry['p_solve']
        sufficient = gt_entry['sufficient_sample']

        src_bin_range_str = gt_entry.get('source_bin_range_str', str(src_bin_range))

        logger.info(f"\n{'='*70}")
        logger.info(f"Prediction {pair_idx}/{predictions_attempted}: "
                     f"Source bin {src_bin} {src_bin_range_str} -> Target pct {target_pct}")
        logger.info(f"  Ground truth P(solve): {gt_p_solve:.3f}")
        logger.info(f"{'='*70}")

        # Select source tasks
        source_tasks_by_bm: Dict[str, List[BenchmarkTask]] = {}
        source_metrics: Dict[str, List[str]] = {}
        for src_cfg in config.source_benchmarks:
            bm = source_benchmarks.get(src_cfg.name)
            if bm:
                tasks = get_representative_source_tasks(src_bin_range, bm.tasks, src_cfg.n_easier_tasks)
                source_tasks_by_bm[src_cfg.name] = tasks
                source_metrics[src_cfg.name] = bm.metrics_to_use

        source_tasks_text = _build_source_tasks_text(source_tasks_by_bm, source_metrics)

        # Select target task
        target_task = get_task_at_percentile(target_pct, target_benchmark.tasks)
        if not target_task:
            logger.error(f"No target task found at percentile {target_pct}")
            continue

        target_task_idx = int(len(target_benchmark.tasks) * target_pct / 100.0)
        target_task_idx = max(0, min(target_task_idx, len(target_benchmark.tasks) - 1))

        logger.info(f"  Target task: {target_task.name} (index {target_task_idx})")
        for bm_name, tasks in source_tasks_by_bm.items():
            logger.info(f"  Source tasks ({bm_name}): {len(tasks)} tasks")

        # Delphi process
        delphi_rounds_data = []
        round_responses_history = []

        for current_round in range(1, config.workflow_settings.delphi_rounds + 1):
            round_start = time.time()
            logger.info(f"\n  Round {current_round}/{config.workflow_settings.delphi_rounds}")

            coroutines = [
                _run_expert_round(
                    expert=expert,
                    source_tasks_text=source_tasks_text,
                    source_description=source_description,
                    target_task=target_task,
                    target_benchmark_description=target_desc,
                    target_metrics_to_use=target_benchmark.metrics_to_use,
                    round_num=current_round,
                    prev_round_responses=round_responses_history[-1] if round_responses_history else None,
                    client=client,
                    semaphore=semaphore,
                    config=config,
                    prompts=prompts
                ) for expert in experts_to_use
            ]

            round_results = await asyncio.gather(*coroutines, return_exceptions=True)
            logger.info(f"  Round {current_round} completed in {time.time() - round_start:.2f}s")

            processed = []
            for i, res in enumerate(round_results):
                expert_name = experts_to_use[i].name
                if isinstance(res, Exception):
                    logger.error(f"    Expert '{expert_name}' failed: {res}", exc_info=res)
                    processed.append({"expert": expert_name, "error": f"Unhandled Exception: {res}"})
                else:
                    processed.append(res)

            round_responses_history.append(processed)

            valid_estimates = [r["estimate"] for r in processed
                              if "error" not in r and isinstance(r.get("estimate"), float)]

            if len(valid_estimates) >= 2:
                std_dev = float(np.std(valid_estimates))
                mean_est = float(np.mean(valid_estimates))
                median_est = float(np.median(valid_estimates))
                logger.info(f"    Stats: {len(valid_estimates)} valid, "
                            f"Mean={mean_est:.4f}, Median={median_est:.4f}, Std={std_dev:.4f}")
            else:
                std_dev = None
                mean_est = None
                median_est = None

            # Save to CSV
            append_prediction_to_csv(
                csv_path=run_info['csv_path'],
                source_bin=src_bin,
                source_bin_range=src_bin_range_str,
                target_percentile=target_pct,
                target_task_index=target_task_idx,
                target_task_name=target_task.name,
                round_num=current_round,
                expert_responses=processed,
                ground_truth_p_solve=gt_p_solve,
                sufficient_sample=sufficient
            )

            delphi_rounds_data.append({
                "round": current_round,
                "expert_estimates": processed,
                "round_mean": mean_est,
                "round_median": median_est,
                "round_std": std_dev
            })

            # Check convergence
            if std_dev is not None and std_dev < config.workflow_settings.convergence_threshold:
                logger.info(f"    Convergence at R{current_round} "
                            f"(Std {std_dev:.4f} < {config.workflow_settings.convergence_threshold})")
                break

        # Final aggregation
        if delphi_rounds_data:
            final = delphi_rounds_data[-1]
            final_valid = [r["estimate"] for r in final["expert_estimates"]
                          if "error" not in r and isinstance(r.get("estimate"), float)]
            if final_valid:
                final_mean = float(np.mean(final_valid))
                final_median = float(np.median(final_valid))
                final_std = float(np.std(final_valid))
            else:
                final_mean = final_median = final_std = None
        else:
            final_mean = final_median = final_std = None

        source_task_solve_rates = []
        for bm_name, tasks in source_tasks_by_bm.items():
            for task in tasks:
                sr = task.metrics.get('solve_rate')
                if sr is not None:
                    source_task_solve_rates.append({
                        'name': task.name,
                        'solve_rate': float(sr),
                        'benchmark': bm_name
                    })

        prediction_result = {
            "source_bin": src_bin,
            "source_bin_range": src_bin_range,
            "source_bin_range_str": src_bin_range_str,
            "target_percentile": target_pct,
            "target_task_index": target_task_idx,
            "target_task_name": target_task.name,
            "source_task_solve_rates": source_task_solve_rates,
            "delphi_rounds": delphi_rounds_data,
            "final_aggregated_probability": final_mean,
            "final_median": final_median,
            "final_std_dev": final_std,
            "ground_truth_p_solve": gt_p_solve,
            "n_in_source_bin": gt_entry.get('n_in_source_bin'),
            "n_solving_target": gt_entry.get('n_solving_target'),
            "converged": (final_std < config.workflow_settings.convergence_threshold) if final_std is not None else False,
            "total_rounds_executed": len(delphi_rounds_data)
        }

        save_prediction_to_json(run_info['json_path'], prediction_result)
        predictions_completed += 1

        if final_mean is not None:
            logger.info(f"  Final P(solve): {final_mean:.4f} (GT: {gt_p_solve:.3f})")

    # Finalise
    finalize_inter_benchmark_run(run_info['json_path'], predictions_attempted)

    add_inter_benchmark_run_to_registry(
        registry_file=config.registry_file,
        run_id=run_info['run_id'],
        source_name=config.primary_source.name,
        target_name=config.target_benchmark.name,
        model=config.llm_settings.model,
        num_experts=len(experts_to_use),
        delphi_rounds=config.workflow_settings.delphi_rounds,
        num_predictions_attempted=predictions_attempted,
        num_predictions_completed=predictions_completed,
        output_path=run_info['run_dir'],
        config_file="config_example.yaml",
        timestamp_start=datetime.datetime.now().isoformat()
    )

    duration = time.time() - run_start_time
    logger.info(f"\n{'='*70}")
    logger.info("Inter-Benchmark Calibration Workflow Completed")
    logger.info(f"  Run ID: {run_info['run_id']}")
    logger.info(f"  Predictions: {predictions_completed}/{predictions_attempted}")
    logger.info(f"  Duration: {duration:.2f}s")
    logger.info(f"  Output: {run_info['run_dir']}")
    logger.info(f"{'='*70}\n")

    return {
        "run_id": run_info['run_id'],
        "predictions_completed": predictions_completed,
        "predictions_attempted": predictions_attempted,
        "output_path": str(run_info['run_dir'])
    }
