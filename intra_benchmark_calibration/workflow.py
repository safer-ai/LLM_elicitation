#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Workflow orchestration for intra-benchmark calibration experiments.

This module implements the Delphi estimation process for predicting
conditional probabilities P(j|i) between benchmark task bins.
"""

import asyncio
import datetime
import logging
import time
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional, Union
from asyncio import Semaphore

# Conditional imports for type hinting
try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.data_models import ExpertProfile
from shared.llm_client import make_api_call
from shared.parsing import parse_probability_response
from config import IntraBenchmarkConfig
from data_loader import load_ground_truth, validate_ground_truth_data
from task_selector import (
    get_representative_tasks_for_bin,
    get_target_task_for_bin,
    format_tasks_list_for_prompt
)
from results_handler import (
    initialize_intra_benchmark_run,
    append_prediction_to_csv,
    save_prediction_to_json,
    finalize_intra_benchmark_run,
    add_intra_benchmark_run_to_registry
)

logger = logging.getLogger(__name__)


async def _run_expert_round_intra_benchmark(
    expert: ExpertProfile,
    tasks_solved: List[Dict[str, Any]],
    target_task: Dict[str, Any],
    benchmark_description: str,
    round_num: int,
    prev_round_responses: Optional[List[Dict[str, Any]]],
    client: Union[AsyncAnthropic, AsyncOpenAI],
    semaphore: Semaphore,
    config: IntraBenchmarkConfig,
    prompts: Dict[str, str],
    metrics_to_use: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run a single expert's estimation for one (i,j) bin pair in one round.

    Args:
        expert: ExpertProfile object
        tasks_solved: List of tasks that models in bin_i have solved
        target_task: The target task at bin_j's midpoint
        benchmark_description: Description of the benchmark for context
        round_num: Current Delphi round number (1-indexed)
        prev_round_responses: Previous round's responses from all experts (None for round 1)
        client: LLM API client
        semaphore: Concurrency semaphore
        config: IntraBenchmarkConfig object
        prompts: Dict of prompt templates

    Returns:
        Dict with expert's response: expert, estimate, percentile_25th, percentile_50th, percentile_75th, rationale, error
    """
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

    # Prepare prompt data
    # Format metrics for target task (only if metrics are specified)
    target_task_metrics_str = ""
    if metrics_to_use:
        metric_parts = []
        for metric_name in metrics_to_use:
            if metric_name in target_task.metrics:
                metric_value = target_task.metrics[metric_name]
                metric_parts.append(f"{metric_name}={metric_value}")
        if metric_parts:
            target_task_metrics_str = ", ".join(metric_parts)
    
    prompt_data = {
        "benchmark_description": benchmark_description,
        "solved_tasks_list": format_tasks_list_for_prompt(tasks_solved, metrics_to_use),
        "target_task_name": target_task.name,
        "target_task_description": target_task.description,
        "target_task_metrics": target_task_metrics_str if target_task_metrics_str else "N/A"
    }

    max_tokens = 8000

    try:
        if round_num == 1:
            # --- Round 1: Analysis + Initial Estimation ---
            logger.debug(f"R1 Intra-Bench - Running analysis for expert '{ename}'")

            # Stage 1: Task relationship analysis
            analysis_prompt_template = prompts.get('task_relationship_analysis')
            if not analysis_prompt_template:
                raise ValueError("Missing 'task_relationship_analysis' prompt template")

            analysis_user_prompt = analysis_prompt_template.format(**prompt_data)
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

            logger.debug(f"R1 Intra-Bench - Running initial estimation for expert '{ename}'")

            # Stage 2: Initial probability estimation
            estimation_prompt_template = prompts.get('initial_conditional_probability_estimation')
            if not estimation_prompt_template:
                raise ValueError("Missing 'initial_conditional_probability_estimation' prompt template")

            estimation_user_prompt = estimation_prompt_template.format(**prompt_data)
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
                parsed_estimation = parse_probability_response(estimation_text)
                expert_result["estimate"] = parsed_estimation.get("estimate")
                expert_result["percentile_25th"] = parsed_estimation.get("percentile_25th")
                expert_result["percentile_50th"] = parsed_estimation.get("percentile_50th")
                expert_result["percentile_75th"] = parsed_estimation.get("percentile_75th")
                expert_result["rationale"] = parsed_estimation.get("rationale", "")

                # Check if any field failed to parse (exclude 'error' which is only set on failure)
                if any(v is None for k, v in expert_result.items() if k != "error"):
                    logger.warning(f"R1 - Expert '{ename}' probability parsing failed")
                    expert_result["error"] = "Probability Parsing Failed"

        else:
            # --- Round 2+: Refinement with group feedback ---
            logger.debug(f"R{round_num} Intra-Bench - Refinement for expert '{ename}'")

            # Prepare context with other experts' previous round responses
            context_lines = [
                "\n---\nOther experts' estimates and reasoning from the previous round:\n---\n"
            ]

            if prev_round_responses:
                others_prev = [r for r in prev_round_responses if r.get("expert") != ename]
                if others_prev:
                    for r in others_prev:
                        p25_str = f"{r.get('percentile_25th', 'N/A'):.3f}" if isinstance(r.get('percentile_25th'), float) else "N/A"
                        p50_str = f"{r.get('percentile_50th', 'N/A'):.3f}" if isinstance(r.get('percentile_50th'), float) else "N/A"
                        p75_str = f"{r.get('percentile_75th', 'N/A'):.3f}" if isinstance(r.get('percentile_75th'), float) else "N/A"

                        rationale = r.get('rationale', 'N/A')
                        if len(rationale) > 300:
                            rationale = rationale[:300] + "..."

                        context_lines.append(
                            f"Expert: {r['expert']}\n"
                            f"Percentiles: 25th={p25_str}, 50th (median)={p50_str}, 75th={p75_str}\n"
                            f"Rationale: {rationale}\n"
                        )
                else:
                    context_lines.append("(No other valid responses received in the previous round)")
            else:
                context_lines.append("(No previous round responses available)")

            context_lines.append("\n---\n")
            prompt_data["context"] = "\n".join(context_lines)

            subsequent_prompt_template = prompts.get('subsequent_conditional_probability_estimation')
            if not subsequent_prompt_template:
                raise ValueError("Missing 'subsequent_conditional_probability_estimation' prompt template")

            subsequent_user_prompt = subsequent_prompt_template.format(**prompt_data)
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
                parsed_estimation = parse_probability_response(estimation_text)
                expert_result["estimate"] = parsed_estimation.get("estimate")
                expert_result["percentile_25th"] = parsed_estimation.get("percentile_25th")
                expert_result["percentile_50th"] = parsed_estimation.get("percentile_50th")
                expert_result["percentile_75th"] = parsed_estimation.get("percentile_75th")
                expert_result["rationale"] = parsed_estimation.get("rationale", "")

                # Check if any field failed to parse (exclude 'error' which is only set on failure)
                if any(v is None for k, v in expert_result.items() if k != "error"):
                    logger.warning(f"R{round_num} - Expert '{ename}' probability parsing failed")
                    expert_result["error"] = "Probability Parsing Failed"

    except ValueError as ve:
        logger.error(f"ValueError during expert round for {ename}: {ve}")
        expert_result["error"] = f"Configuration/Prompt Error: {ve}"
    except Exception as e:
        logger.error(f"Unexpected error during expert round for {ename}: {e}", exc_info=True)
        expert_result["error"] = f"Unexpected Workflow Error: {e}"

    return expert_result


async def run_intra_benchmark_estimation(
    client: Union[AsyncAnthropic, AsyncOpenAI],
    semaphore: Semaphore,
    config: IntraBenchmarkConfig,
    input_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main orchestration function for intra-benchmark calibration experiments.

    Args:
        client: LLM API client
        semaphore: Concurrency semaphore
        config: IntraBenchmarkConfig object
        input_data: Dict with 'prompts', 'experts', 'benchmark_tasks' keys

    Returns:
        Dict with run results and metadata
    """
    run_start_time = time.time()
    logger.info("\n\n--- Starting Intra-Benchmark Calibration Workflow ---")

    # Validate input data
    prompts = input_data.get('prompts', {})
    experts = input_data.get('experts', [])
    benchmark_tasks = input_data.get('benchmark_tasks', [])
    benchmark_description = input_data.get('benchmark_description', config.intra_benchmark_settings.benchmark_description)
    metrics_to_use = input_data.get('metrics_to_use', [])
    
    # Log which metrics will be used for difficulty estimation
    if metrics_to_use:
        logger.info(f"Using difficulty metrics: {', '.join(metrics_to_use)}")
    else:
        logger.info("No difficulty metrics specified in benchmark file - will not display metrics in prompts")

    if not prompts or not experts or not benchmark_tasks:
        return {"error": "Missing required input data (prompts, experts, or benchmark_tasks)"}

    # Select experts to use
    num_experts = config.workflow_settings.num_experts
    if num_experts is not None and num_experts > 0:
        experts_to_use = experts[:num_experts]
        logger.info(f"Using first {num_experts} experts")
    else:
        experts_to_use = experts
        logger.info(f"Using all {len(experts)} experts")

    if not experts_to_use:
        return {"error": "No experts selected to run"}

    # Load ground truth data
    ground_truth_data = load_ground_truth(
        config.intra_benchmark_settings.benchmark_name,
        config.intra_benchmark_settings.n_bins,
        config.ground_truth_dir
    )

    if not ground_truth_data:
        return {"error": "Failed to load ground truth data"}

    if not validate_ground_truth_data(ground_truth_data):
        return {"error": "Ground truth data validation failed"}

    # Confirm n_bins consistency (validation already passed in load_ground_truth)
    loaded_n_bins = ground_truth_data['metadata'].get('n_bins')
    config_n_bins = config.intra_benchmark_settings.n_bins
    logger.info(f"✓ Confirmed: Config n_bins={config_n_bins} matches ground truth file n_bins={loaded_n_bins}")

    ground_truth_pairs = ground_truth_data['ground_truth']
    logger.info(f"Loaded {len(ground_truth_pairs)} (i,j) pairs with sufficient sample")

    # Initialize run
    run_info = initialize_intra_benchmark_run(
        benchmark_name=config.intra_benchmark_settings.benchmark_name,
        n_bins=config.intra_benchmark_settings.n_bins,
        model=config.llm_settings.model,
        num_experts=len(experts_to_use),
        delphi_rounds=config.workflow_settings.delphi_rounds,
        temperature=config.llm_settings.temperature,
        output_base_dir=config.output_dir
    )

    if not run_info:
        return {"error": "Failed to initialize run"}

    logger.info(f"Run ID: {run_info['run_id']}")
    logger.info(f"Output directory: {run_info['run_dir']}")

    # Track overall results
    predictions_completed = 0
    predictions_attempted = len(ground_truth_pairs)

    # Process each (i,j) pair
    for pair_idx, gt_pair in enumerate(ground_truth_pairs, 1):
        bin_i = gt_pair['bin_i']
        bin_j = gt_pair['bin_j']
        bin_i_range = gt_pair['bin_i_range']
        bin_j_range = gt_pair['bin_j_range']
        ground_truth_p = gt_pair['p_j_given_i']

        logger.info(f"\n{'='*70}")
        logger.info(f"Processing prediction {pair_idx}/{predictions_attempted}: "
                    f"Bin {bin_i} → {bin_j}")
        logger.info(f"  Bin ranges: {bin_i_range} → {bin_j_range}")
        logger.info(f"  Ground truth P(j|i): {ground_truth_p:.3f}")
        logger.info(f"{'='*70}")

        # Select tasks for this bin pair
        try:
            # Use the new function to get representative tasks (default: 3 hardest tasks from bin i)
            tasks_solved = get_representative_tasks_for_bin(bin_i_range, benchmark_tasks, len(benchmark_tasks), n_tasks=3)
            target_task = get_target_task_for_bin(bin_j_range, benchmark_tasks, len(benchmark_tasks))
        except Exception as e:
            logger.error(f"Failed to select tasks for pair ({bin_i}, {bin_j}): {e}")
            continue

        if not target_task:
            logger.error(f"No target task found for bin {bin_j}")
            continue

        logger.info(f"  Representative solved tasks: {len(tasks_solved)} tasks (hardest bin is {bin_i})")
        logger.info(f"  Target task: {target_task.name}")

        # Run Delphi process for this (i,j) pair
        delphi_rounds_data = []
        round_responses_history = []

        for current_round in range(1, config.workflow_settings.delphi_rounds + 1):
            round_start_time = time.time()
            logger.info(f"\n  Round {current_round}/{config.workflow_settings.delphi_rounds} "
                        f"for pair ({bin_i}, {bin_j}) starting...")

            # Run all experts in parallel for this round
            coroutines = [
                _run_expert_round_intra_benchmark(
                    expert=expert,
                    tasks_solved=tasks_solved,
                    target_task=target_task,
                    benchmark_description=benchmark_description,
                    round_num=current_round,
                    prev_round_responses=round_responses_history[-1] if round_responses_history else None,
                    client=client,
                    semaphore=semaphore,
                    config=config,
                    prompts=prompts,
                    metrics_to_use=metrics_to_use
                ) for expert in experts_to_use
            ]

            round_api_results = await asyncio.gather(*coroutines, return_exceptions=True)
            logger.info(f"  Round {current_round} completed in {time.time() - round_start_time:.2f}s")

            # Process responses
            processed_responses = []
            for i, res in enumerate(round_api_results):
                expert_name = experts_to_use[i].name
                if isinstance(res, Exception):
                    logger.error(f"    Expert '{expert_name}' failed: {res}", exc_info=res)
                    processed_responses.append({"expert": expert_name, "error": f"Unhandled Exception: {res}"})
                else:
                    processed_responses.append(res)

            round_responses_history.append(processed_responses)

            # Calculate round statistics
            valid_estimates = [r["estimate"] for r in processed_responses
                               if "error" not in r and isinstance(r.get("estimate"), float)]

            if len(valid_estimates) >= 2:
                std_dev = np.std(valid_estimates)
                mean_est = np.mean(valid_estimates)
                median_est = np.median(valid_estimates)
                logger.info(f"    Round {current_round} Stats: Valid Estimates={len(valid_estimates)}, "
                            f"Mean={mean_est:.4f}, Median={median_est:.4f}, StdDev={std_dev:.4f}")

                # Save round to CSV
                append_prediction_to_csv(
                    csv_path=run_info['csv_path'],
                    bin_i=bin_i,
                    bin_j=bin_j,
                    bin_i_range=bin_i_range,
                    bin_j_range=bin_j_range,
                    round_num=current_round,
                    expert_responses=processed_responses,
                    ground_truth_p_j_given_i=ground_truth_p,
                    sufficient_sample=True
                )

                # Store round data for JSON
                delphi_rounds_data.append({
                    "round": current_round,
                    "expert_estimates": processed_responses,
                    "round_mean": mean_est,
                    "round_median": median_est,
                    "round_std": std_dev
                })

                # Check convergence
                if std_dev < config.workflow_settings.convergence_threshold:
                    logger.info(f"    Convergence reached at R{current_round} "
                                f"(StdDev {std_dev:.4f} < {config.workflow_settings.convergence_threshold:.4f})")
                    break
            else:
                logger.info(f"    Round {current_round} Stats: Not enough valid estimates ({len(valid_estimates)}) "
                            f"to check convergence")

                # Still save to CSV and JSON even with insufficient estimates
                append_prediction_to_csv(
                    csv_path=run_info['csv_path'],
                    bin_i=bin_i,
                    bin_j=bin_j,
                    bin_i_range=bin_i_range,
                    bin_j_range=bin_j_range,
                    round_num=current_round,
                    expert_responses=processed_responses,
                    ground_truth_p_j_given_i=ground_truth_p,
                    sufficient_sample=True
                )

                delphi_rounds_data.append({
                    "round": current_round,
                    "expert_estimates": processed_responses,
                    "round_mean": None,
                    "round_median": None,
                    "round_std": None
                })

        # Calculate final aggregated estimate
        if delphi_rounds_data:
            final_round = delphi_rounds_data[-1]
            final_responses = final_round["expert_estimates"]
            final_valid_estimates = [r["estimate"] for r in final_responses
                                      if "error" not in r and isinstance(r.get("estimate"), float)]

            if final_valid_estimates:
                final_mean = np.mean(final_valid_estimates)
                final_median = np.median(final_valid_estimates)
                final_std = np.std(final_valid_estimates)
                logger.info(f"  Final Agg. Probability: {final_mean:.4f} (Ground Truth: {ground_truth_p:.3f})")
            else:
                final_mean = None
                final_median = None
                final_std = None
                logger.warning("  No valid final estimates found")
        else:
            final_mean = None
            final_median = None
            final_std = None

        # Construct prediction result for JSON
        prediction_result = {
            "bin_i": bin_i,
            "bin_j": bin_j,
            "bin_i_range": bin_i_range,
            "bin_j_range": bin_j_range,
            "tasks_in_bin_i": [task.name for task in tasks_solved],
            "target_task": target_task.name,
            "delphi_rounds": delphi_rounds_data,
            "final_aggregated_probability": final_mean,
            "final_median": final_median,
            "final_std_dev": final_std,
            "ground_truth_p_j_given_i": ground_truth_p,
            "ground_truth_n_reaching_i": gt_pair.get('n_reaching_i'),
            "ground_truth_n_reaching_j": gt_pair.get('n_reaching_j'),
            "converged": final_std < config.workflow_settings.convergence_threshold if final_std is not None else False,
            "convergence_round": len(delphi_rounds_data) if final_std is not None and final_std < config.workflow_settings.convergence_threshold else None,
            "total_rounds_executed": len(delphi_rounds_data)
        }

        # Save prediction to JSON
        save_prediction_to_json(run_info['json_path'], prediction_result)
        predictions_completed += 1

    # Finalize run
    finalize_intra_benchmark_run(run_info['json_path'], predictions_attempted)

    # Add to registry
    timestamp_start = datetime.datetime.now().isoformat()  # Approximate - could extract from JSON
    add_intra_benchmark_run_to_registry(
        registry_file=config.registry_file,
        run_id=run_info['run_id'],
        benchmark_name=config.intra_benchmark_settings.benchmark_name,
        n_bins=config.intra_benchmark_settings.n_bins,
        model=config.llm_settings.model,
        num_experts=len(experts_to_use),
        delphi_rounds=config.workflow_settings.delphi_rounds,
        num_predictions_attempted=predictions_attempted,
        num_predictions_completed=predictions_completed,
        output_path=run_info['run_dir'],
        config_file="config_intra_benchmark.yaml",  # Could be passed as parameter
        timestamp_start=timestamp_start
    )

    run_duration = time.time() - run_start_time
    logger.info(f"\n{'='*70}")
    logger.info("Intra-Benchmark Calibration Workflow Completed")
    logger.info(f"  Run ID: {run_info['run_id']}")
    logger.info(f"  Predictions attempted: {predictions_attempted}")
    logger.info(f"  Predictions completed: {predictions_completed}")
    logger.info(f"  Duration: {run_duration:.2f}s")
    logger.info(f"  Output directory: {run_info['run_dir']}")
    logger.info(f"{'='*70}\n")

    return {
        "run_id": run_info['run_id'],
        "predictions_completed": predictions_completed,
        "predictions_attempted": predictions_attempted,
        "output_path": str(run_info['run_dir'])
    }
