import asyncio
import logging
import time
import datetime
from dataclasses import asdict
import numpy as np
from typing import List, Dict, Optional, Union, Any, Tuple
from asyncio import Semaphore
from pathlib import Path

# Conditional imports for type hinting client
try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None # type: ignore
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None # type: ignore

# Import necessary components from other modules
from config import AppConfig
from data_models import ExpertProfile, BenchmarkTask, Benchmark, Scenario, ScenarioStep, InputData, ScenarioLevelMetric
from llm_api import make_api_call
from parsing import parse_analysis_response, parse_probability_response, parse_quantity_response
from results_handler import initialize_run, append_round_to_csv, save_intermediate_json, finalize_run
from prompt_helpers import format_example_tasks_context


logger = logging.getLogger(__name__)

# --- Helper Function ---

def _select_items(items: List[Any], num_to_select: Optional[int], item_type_name: str) -> List[Any]:
    """Selects the first 'num_to_select' items from a list, handling None."""
    if num_to_select is None:
        logger.debug(f"Processing all {len(items)} {item_type_name}s.")
        return items
    elif num_to_select >= 0:
        selected = items[:num_to_select]
        logger.debug(f"Selected first {len(selected)} of {len(items)} {item_type_name}s based on config.")
        return selected
    else: # Should be caught by config validation, but defensive check
        logger.warning(f"Invalid negative number ({num_to_select}) specified for {item_type_name}s. Processing all.")
        return items


def _prepare_scenario_full_steps_description(scenario: Scenario) -> str:
    """
    Prepares a string containing the full description of all scenario steps.
    This is used as context for scenario-level metric estimations.
    """
    lines = [
        "--- Full Scenario Steps Description ---"
    ]
    if not scenario.steps:
        lines.append("No steps defined in the scenario.")
        return "\n".join(lines)

    for i, step in enumerate(scenario.steps):
        lines.append(f"\nStep {i+1}: {step.name}")
        lines.append(f"  Description: {step.description.strip()}")
        if step.assumptions and step.assumptions.strip():
            lines.append(f"  Assumptions: {step.assumptions.strip()}")
    lines.append("\n--- End of Full Scenario Steps Description ---")
    return "\n".join(lines)


async def _run_expert_round(
    expert: ExpertProfile,
    task: BenchmarkTask,
    benchmark: Benchmark, # Now step-specific
    scenario_step: ScenarioStep,
    scenario: Scenario,
    round_num: int,
    prev_round_responses: Optional[List[Dict[str, Any]]], # List of responses {expert_name, estimate, rationale,...} from previous round
    client: Union[AsyncAnthropic, AsyncOpenAI],
    semaphore: Semaphore,
    config: AppConfig,
    prompts: Dict[str, str]
) -> Dict[str, Any]:
    """
    Runs a single expert's estimation task for a specific round (FOR STEP PROBABILITY).

    Handles calling the appropriate prompts (analysis+initial or subsequent),
    formatting inputs, making API calls, and parsing results.
    The `benchmark` passed here is now specific to the `scenario_step`.
    """
    ename = expert.name
    epersona = expert.get_persona_description()
    general_system_context = prompts.get("system_context", "").strip()

    system_prompt_base = f"{general_system_context}\n\n{epersona}" if general_system_context else epersona

    expert_result: Dict[str, Any] = {"expert": ename, "estimate": None, "rationale": ""}

    # --- Prepare Prompt Placeholders ---
    prompt_data = {
        "scenario_name": scenario.name,
        "scenario_description": scenario.description,
        "task_name": task.name,
        "task_description": task.description,
        "benchmark_description": benchmark.description,
        "threat_actor_name": scenario.threat_actor.name,
        "threat_actor_description": scenario.threat_actor.description,
        "target_name": scenario.target.name,
        "target_description": scenario.target.description,
        "risk_scenario_step_name": scenario_step.name,
        "risk_scenario_step_description": scenario_step.description,
        "risk_scenario_step_assumptions": scenario_step.assumptions,
        "example_tasks_context": format_example_tasks_context(benchmark, task, config)
    }


    relevant_metrics_parts = []
    if benchmark.metrics_to_use and task.metrics:
        for metric_key in benchmark.metrics_to_use:
            if metric_key in task.metrics:
                metric_value = task.metrics[metric_key]
                relevant_metrics_parts.append(f"{metric_key.upper()}: {metric_value}")

    prompt_data["task_relevant_metrics_details"] = "Relevant Task Metrics: " + ", ".join(relevant_metrics_parts) + "." if relevant_metrics_parts else "Relevant Task Metrics: None specified or available for this task."

    max_tokens_analysis = 8000
    max_tokens_estimation = 8000

    try:
        if round_num == 1:
            # --- Round 1: Task Analysis + Initial Estimation ---
            logger.debug(f"R1 ProbEst - Running analysis for Task '{task.name}' by Expert '{ename}' for Step '{scenario_step.name}'")
            analysis_prompt_template = prompts.get('task_analysis')
            if not analysis_prompt_template:
                raise ValueError("Missing 'task_analysis' prompt template.")

            analysis_user_prompt = analysis_prompt_template.format(**prompt_data)
            expert_result["analysis_system_prompt"] = system_prompt_base
            expert_result["analysis_user_prompt"] = analysis_user_prompt

            analysis_text = await make_api_call(client, semaphore, config, system_prompt_base, analysis_user_prompt, max_tokens_analysis)
            expert_result["raw_analysis"] = analysis_text

            if analysis_text.startswith("Error:"):
                expert_result["error"] = f"Analysis API Call Failed: {analysis_text}"
                return expert_result

            parsed_analysis = parse_analysis_response(analysis_text)
            expert_result["parsed_analysis"] = parsed_analysis
            # Use the full analysis text as context for the next prompt
            prompt_data["technical_analysis"] = parsed_analysis.get("technical_capabilities", "N/A")

            logger.debug(f"R1 ProbEst - Running initial estimation for Task '{task.name}', Step '{scenario_step.name}' by Expert '{ename}'")
            estimation_prompt_template = prompts.get('initial_probability_estimation')
            if not estimation_prompt_template:
                 raise ValueError("Missing 'initial_probability_estimation' prompt template.")

            estimation_user_prompt = estimation_prompt_template.format(**prompt_data)
            expert_result["estimation_system_prompt"] = system_prompt_base
            expert_result["estimation_user_prompt"] = estimation_user_prompt

            estimation_text = await make_api_call(client, semaphore, config, system_prompt_base, estimation_user_prompt, max_tokens_estimation)
            expert_result["raw_estimation"] = estimation_text

            if estimation_text.startswith("Error:"):
                expert_result["error"] = f"Initial Probability Estimation API Call Failed: {estimation_text}"
            else:
                 parsed_estimation = parse_probability_response(estimation_text)
                 expert_result["parsed_estimation"] = parsed_estimation
                 # --- CORRECTED KEY MAPPINGS ---
                 expert_result["estimate"] = parsed_estimation.get("estimate")
                 expert_result["minimum"] = parsed_estimation.get("minimum")
                 expert_result["maximum"] = parsed_estimation.get("maximum")
                 expert_result["confidence"] = parsed_estimation.get("confidence")
                 expert_result["rationale"] = parsed_estimation.get("rationale", "")
                 if expert_result["estimate"] is None and not expert_result.get("error"):
                      logger.warning(f"R1 ProbEst - Expert '{ename}' probability estimation parsing failed for Task '{task.name}', Step '{scenario_step.name}'.")
                      expert_result["error"] = "Probability Parsing Failed"
        else:
            # --- Subsequent Rounds: Refinement ---
            logger.debug(f"R{round_num} ProbEst - Running refinement for Task '{task.name}', Step '{scenario_step.name}' by Expert '{ename}'")
            if not prev_round_responses:
                raise ValueError(f"R{round_num} ProbEst: Missing previous round data for Task '{task.name}', Step '{scenario_step.name}'.")

            own_prev = next((r for r in prev_round_responses if r.get("expert") == ename and r.get("estimate") is not None and "error" not in r), None)

            if not own_prev:
                prev_error_state = next((r for r in prev_round_responses if r.get("expert") == ename), None)
                err_msg = "Missing valid previous round estimate to refine."
                if prev_error_state and prev_error_state.get("error"):
                     err_msg = f"Carrying forward error from R{round_num-1}: {prev_error_state.get('error')}"
                logger.warning(f"R{round_num} ProbEst - Expert '{ename}' cannot refine Task '{task.name}', Step '{scenario_step.name}'. Reason: {err_msg}")
                expert_result["error"] = err_msg
                return expert_result

            others_prev = [r for r in prev_round_responses if r.get("expert") != ename and r.get("estimate") is not None and "error" not in r]

            if own_prev.get("parsed_analysis"):
                prompt_data["technical_analysis"] = own_prev["parsed_analysis"].get("technical_capabilities", "N/A")
            else:
                prompt_data["technical_analysis"] = "N/A"

            context_lines = [f"--- Your Previous Response (Round {round_num-1}) ---"]
            context_lines.append(f"Estimate: {own_prev.get('estimate', 'N/A'):.4f}")
            context_lines.append(f"Rationale: {own_prev.get('rationale', 'N/A')}")
            context_lines.append(f"\n--- Other Experts' Valid Responses (Round {round_num-1}) ---")
            if others_prev:
                 for r in others_prev:
                      other_rationale_short = r.get('rationale', 'N/A')[:150] + ('...' if len(r.get('rationale', '')) > 150 else '')
                      context_lines.append(f"- {r['expert']}: Estimate {r.get('estimate', 'N/A'):.4f}, Rationale: {other_rationale_short}")
            else:
                 context_lines.append("(No other valid responses received in the previous round)")
            context_lines.append("\n---\nReview the information and provide your updated four-point probability estimate and rationale.\n---")

            prompt_data["context"] = "\n".join(context_lines)

            subsequent_prompt_template = prompts.get('subsequent_probability_estimation')
            if not subsequent_prompt_template:
                raise ValueError("Missing 'subsequent_probability_estimation' prompt template.")

            subsequent_user_prompt = subsequent_prompt_template.format(**prompt_data)
            expert_result["estimation_system_prompt"] = system_prompt_base
            expert_result["estimation_user_prompt"] = subsequent_user_prompt

            estimation_text = await make_api_call(client, semaphore, config, system_prompt_base, subsequent_user_prompt, max_tokens_estimation)
            expert_result["raw_estimation"] = estimation_text

            if estimation_text.startswith("Error:"):
                 expert_result["error"] = f"Subsequent Probability Estimation API Call Failed: {estimation_text}"
            else:
                 parsed_estimation = parse_probability_response(estimation_text)
                 expert_result["parsed_estimation"] = parsed_estimation
                 # --- CORRECTED KEY MAPPINGS ---
                 expert_result["estimate"] = parsed_estimation.get("estimate")
                 expert_result["minimum"] = parsed_estimation.get("minimum")
                 expert_result["maximum"] = parsed_estimation.get("maximum")
                 expert_result["confidence"] = parsed_estimation.get("confidence")
                 expert_result["rationale"] = parsed_estimation.get("rationale", "")
                 if expert_result["estimate"] is None and not expert_result.get("error"):
                     logger.warning(f"R{round_num} ProbEst - Expert '{ename}' probability estimation parsing failed for Task '{task.name}', Step '{scenario_step.name}'.")
                     expert_result["error"] = "Probability Parsing Failed"

    except ValueError as ve:
         logger.error(f"ValueError during probability expert round for {ename}, Task '{task.name}', Step '{scenario_step.name}': {ve}")
         expert_result["error"] = f"Configuration/Prompt Error: {ve}"
    except Exception as e:
        logger.error(f"Unexpected error during probability expert round for {ename}, Task '{task.name}', Step '{scenario_step.name}': {e}", exc_info=True)
        expert_result["error"] = f"Unexpected Workflow Error: {e}"

    return expert_result


async def _run_expert_round_for_scenario_metric(
    expert: ExpertProfile,
    task_from_dedicated_benchmark: BenchmarkTask,
    dedicated_benchmark_details: Benchmark,
    scenario: Scenario,
    full_scenario_steps_description_str: str,
    round_num: int,
    prev_round_responses_for_this_metric_task: Optional[List[Dict[str, Any]]],
    client: Union[AsyncAnthropic, AsyncOpenAI],
    semaphore: Semaphore,
    config: AppConfig,
    prompts: Dict[str, str],
    metric_name_logging: str,
    metric_initial_prompt_key: str,
    metric_subsequent_prompt_key: str,
    metric_parser_func: callable,
    scenario_level_metric_assumptions: str
) -> Dict[str, Any]:
    """
    Runs a single expert's estimation for a scenario-level metric.
    """
    ename = expert.name
    epersona = expert.get_persona_description()
    general_system_context = prompts.get("system_context", "").strip()

    system_prompt_base = f"{general_system_context}\n\n{epersona}" if general_system_context else epersona

    expert_result: Dict[str, Any] = {"expert": ename, "estimate": None, "rationale": ""}

    prompt_data = {
        "scenario_name": scenario.name,
        "scenario_description": scenario.description,
        "task_name": task_from_dedicated_benchmark.name,
        "task_description": task_from_dedicated_benchmark.description,
        "benchmark_description": dedicated_benchmark_details.description,
        "threat_actor_name": scenario.threat_actor.name,
        "threat_actor_description": scenario.threat_actor.description,
        "target_name": scenario.target.name,
        "target_description": scenario.target.description,
        "scenario_steps_full_description": full_scenario_steps_description_str,
        "scenario_level_metric_assumptions": scenario_level_metric_assumptions,
        "example_tasks_context": format_example_tasks_context(dedicated_benchmark_details, task_from_dedicated_benchmark, config)
    }

    relevant_metrics_parts = []
    if dedicated_benchmark_details.metrics_to_use and task_from_dedicated_benchmark.metrics:
        for metric_key in dedicated_benchmark_details.metrics_to_use:
            if metric_key in task_from_dedicated_benchmark.metrics:
                metric_value = task_from_dedicated_benchmark.metrics[metric_key]
                relevant_metrics_parts.append(f"{metric_key.upper()}: {metric_value}")
    
    prompt_data["task_relevant_metrics_details"] = "Relevant Task Metrics (for this capability benchmark task): " + ", ".join(relevant_metrics_parts) + "." if relevant_metrics_parts else "Relevant Task Metrics: None specified or available for this capability benchmark task."

    max_tokens_estimation = 8000

    try:
        if round_num == 1:
            # --- Round 1: Task Analysis + Initial Estimation ---
            logger.debug(f"R1 {metric_name_logging}Est - Running analysis for MetricTask '{task_from_dedicated_benchmark.name}' by Expert '{ename}'")
            analysis_prompt_template = prompts.get('task_analysis')
            if not analysis_prompt_template:
                raise ValueError("Missing 'task_analysis' prompt template.")

            # The analysis prompt is designed for a specific step, but here we are at the scenario level.
            # We will use the scenario-level context for the analysis prompt.
            analysis_prompt_data = prompt_data.copy()
            # The analysis prompt expects step-specific keys, which we don't have.
            # We'll provide placeholder or scenario-level info.
            analysis_prompt_data["risk_scenario_step_name"] = f"Scenario-Level Estimation ({metric_name_logging})"
            analysis_prompt_data["risk_scenario_step_description"] = f"This analysis is for the scenario-level metric: {metric_name_logging}."
            analysis_prompt_data["risk_scenario_step_assumptions"] = scenario_level_metric_assumptions

            analysis_user_prompt = analysis_prompt_template.format(**analysis_prompt_data)
            expert_result["analysis_system_prompt"] = system_prompt_base
            expert_result["analysis_user_prompt"] = analysis_user_prompt

            analysis_text = await make_api_call(client, semaphore, config, system_prompt_base, analysis_user_prompt, max_tokens_estimation)
            expert_result["raw_analysis"] = analysis_text

            if analysis_text.startswith("Error:"):
                expert_result["error"] = f"Analysis API Call Failed for {metric_name_logging}: {analysis_text}"
                return expert_result

            parsed_analysis = parse_analysis_response(analysis_text)
            expert_result["parsed_analysis"] = parsed_analysis
            prompt_data["technical_analysis"] = parsed_analysis.get("technical_capabilities", "N/A")


            logger.debug(f"R1 {metric_name_logging}Est - Initial estimation for MetricTask '{task_from_dedicated_benchmark.name}' by Expert '{ename}'")
            initial_prompt_template = prompts.get(metric_initial_prompt_key)
            if not initial_prompt_template:
                raise ValueError(f"Missing initial prompt template: '{metric_initial_prompt_key}'")

            initial_user_prompt = initial_prompt_template.format(**prompt_data)
            expert_result["estimation_system_prompt"] = system_prompt_base
            expert_result["estimation_user_prompt"] = initial_user_prompt

            estimation_text = await make_api_call(client, semaphore, config, system_prompt_base, initial_user_prompt, max_tokens_estimation)
            expert_result["raw_estimation"] = estimation_text

            if estimation_text.startswith("Error:"):
                expert_result["error"] = f"Initial {metric_name_logging} Estimation API Call Failed: {estimation_text}"
            else:
                parsed_estimation = metric_parser_func(estimation_text)
                expert_result["parsed_estimation"] = parsed_estimation
                expert_result["estimate"] = parsed_estimation.get("estimate")
                expert_result["minimum"] = parsed_estimation.get("minimum")
                expert_result["maximum"] = parsed_estimation.get("maximum")
                expert_result["confidence"] = parsed_estimation.get("confidence")
                expert_result["rationale"] = parsed_estimation.get("rationale", "")
                if expert_result["estimate"] is None and not expert_result.get("error"):
                    logger.warning(f"R1 {metric_name_logging}Est - Expert '{ename}' quantity estimation parsing failed for MetricTask '{task_from_dedicated_benchmark.name}'.")
                    expert_result["error"] = f"{metric_name_logging} Value Parsing Failed"
        else: # Subsequent rounds
            logger.debug(f"R{round_num} {metric_name_logging}Est - Refinement for MetricTask '{task_from_dedicated_benchmark.name}' by Expert '{ename}'")
            if not prev_round_responses_for_this_metric_task:
                raise ValueError(f"R{round_num} {metric_name_logging}Est: Missing previous round data for MetricTask '{task_from_dedicated_benchmark.name}'.")

            own_prev = next((r for r in prev_round_responses_for_this_metric_task if r.get("expert") == ename and r.get("estimate") is not None and "error" not in r), None)
            if not own_prev:
                prev_error_state = next((r for r in prev_round_responses_for_this_metric_task if r.get("expert") == ename), None)
                err_msg = "Missing valid previous round estimate to refine."
                if prev_error_state and prev_error_state.get("error"):
                     err_msg = f"Carrying forward error from R{round_num-1}: {prev_error_state.get('error')}"
                logger.warning(f"R{round_num} {metric_name_logging}Est - Expert '{ename}' cannot refine MetricTask '{task_from_dedicated_benchmark.name}'. Reason: {err_msg}")
                expert_result["error"] = err_msg
                return expert_result

            others_prev = [r for r in prev_round_responses_for_this_metric_task if r.get("expert") != ename and r.get("estimate") is not None and "error" not in r]
            
            if own_prev.get("parsed_analysis"):
                prompt_data["technical_analysis"] = own_prev["parsed_analysis"].get("technical_capabilities", "N/A")
            else:
                prompt_data["technical_analysis"] = "N/A"

            context_lines = [f"--- Your Previous Response (Round {round_num-1}) ---"]
            context_lines.append(f"Estimated Value: {own_prev.get('estimate', 'N/A')}")
            context_lines.append(f"Rationale: {own_prev.get('rationale', 'N/A')}")
            context_lines.append(f"\n--- Other Experts' Valid Responses (Round {round_num-1}) ---")
            if others_prev:
                 for r in others_prev:
                      other_rationale_short = r.get('rationale', 'N/A')[:150] + ('...' if len(r.get('rationale', '')) > 150 else '')
                      context_lines.append(f"- {r['expert']}: Estimated Value {r.get('estimate', 'N/A')}, Rationale: {other_rationale_short}")
            else:
                 context_lines.append("(No other valid responses received in the previous round)")
            context_lines.append("\n---\nReview the information and provide your updated estimated value and rationale using 'Final Estimated Value: {integer}' and 'Rationale: ...' format.\n---")

            prompt_data["context"] = "\n".join(context_lines)

            subsequent_prompt_template = prompts.get(metric_subsequent_prompt_key)
            if not subsequent_prompt_template:
                raise ValueError(f"Missing subsequent prompt template: '{metric_subsequent_prompt_key}'")
            
            subsequent_user_prompt = subsequent_prompt_template.format(**prompt_data)
            expert_result["estimation_system_prompt"] = system_prompt_base
            expert_result["estimation_user_prompt"] = subsequent_user_prompt

            estimation_text = await make_api_call(client, semaphore, config, system_prompt_base, subsequent_user_prompt, max_tokens_estimation)
            expert_result["raw_estimation"] = estimation_text

            if estimation_text.startswith("Error:"):
                expert_result["error"] = f"Subsequent {metric_name_logging} Estimation API Call Failed: {estimation_text}"
            else:
                parsed_estimation = metric_parser_func(estimation_text)
                expert_result["parsed_estimation"] = parsed_estimation
                expert_result["estimate"] = parsed_estimation.get("estimate")
                expert_result["minimum"] = parsed_estimation.get("minimum")
                expert_result["maximum"] = parsed_estimation.get("maximum")
                expert_result["confidence"] = parsed_estimation.get("confidence")
                expert_result["rationale"] = parsed_estimation.get("rationale", "")
                if expert_result["estimate"] is None and not expert_result.get("error"):
                    logger.warning(f"R{round_num} {metric_name_logging}Est - Expert '{ename}' quantity estimation parsing failed for MetricTask '{task_from_dedicated_benchmark.name}'.")
                    expert_result["error"] = f"{metric_name_logging} Value Parsing Failed"

    except ValueError as ve:
        logger.error(f"ValueError during {metric_name_logging} expert round for {ename}, MetricTask '{task_from_dedicated_benchmark.name}': {ve}")
        expert_result["error"] = f"Configuration/Prompt Error: {ve}"
    except Exception as e:
        logger.error(f"Unexpected error during {metric_name_logging} expert round for {ename}, MetricTask '{task_from_dedicated_benchmark.name}': {e}", exc_info=True)
        expert_result["error"] = f"Unexpected Workflow Error: {e}"
    
    return expert_result


async def _run_scenario_level_metric_estimation_phase(
    client: Union[AsyncAnthropic, AsyncOpenAI],
    semaphore: Semaphore,
    config: AppConfig,
    input_data: InputData,
    full_scenario_steps_description_str: str,
    experts_to_use: List[ExpertProfile],
    project_root: Path,
    run_info: Dict[str, Any],
    overall_results: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Orchestrates Phase 2: Estimation of scenario-level metrics.
    """
    logger.info("\n\n--- Starting Phase 2: Scenario-Level Metric Estimations ---")
    phase2_results: Dict[str, List[Dict[str, Any]]] = {}

    metric_suites_config = [
        {"metric_name_logging": "NumActors", "config_flag_name": "estimate_num_actors_per_task_benchmark", "benchmark_key_in_scenario_yaml": "num_actors_estimation", "initial_prompt_key": "scenario_num_actors_estimation_initial", "subsequent_prompt_key": "scenario_num_actors_estimation_subsequent", "parser_func": parse_quantity_response, "output_pseudo_step_name": "ScenarioLevelMetric_NumActors"},
        {"metric_name_logging": "NumAttacks", "config_flag_name": "estimate_num_attacks_per_task_benchmark", "benchmark_key_in_scenario_yaml": "num_attacks_estimation", "initial_prompt_key": "scenario_num_attacks_estimation_initial", "subsequent_prompt_key": "scenario_num_attacks_estimation_subsequent", "parser_func": parse_quantity_response, "output_pseudo_step_name": "ScenarioLevelMetric_NumAttacks"},
        {"metric_name_logging": "Damage", "config_flag_name": "estimate_damage_per_task_benchmark", "benchmark_key_in_scenario_yaml": "damage_estimation", "initial_prompt_key": "scenario_damage_estimation_initial", "subsequent_prompt_key": "scenario_damage_estimation_subsequent", "parser_func": parse_quantity_response, "output_pseudo_step_name": "ScenarioLevelMetric_Damage"},
    ]

    total_metric_tasks_to_process_phase2 = 0
    for suite_cfg in metric_suites_config:
        if getattr(config.workflow_settings, suite_cfg["config_flag_name"], False):
            metric_config = input_data.scenario.scenario_level_metrics.get(suite_cfg["benchmark_key_in_scenario_yaml"])
            if metric_config:
                benchmark_file_path_normalized = metric_config.benchmark_file
                current_dedicated_benchmark = input_data.loaded_benchmarks.get(benchmark_file_path_normalized)
                if current_dedicated_benchmark:
                    tasks_for_this_metric_benchmark = _select_items(current_dedicated_benchmark.tasks, config.workflow_settings.num_tasks, f"task for {suite_cfg['metric_name_logging']}")
                    total_metric_tasks_to_process_phase2 += len(tasks_for_this_metric_benchmark)
    
    processed_metric_tasks_phase2 = 0

    for suite_cfg in metric_suites_config:
        metric_name_log = suite_cfg["metric_name_logging"]
        pseudo_step_name = suite_cfg["output_pseudo_step_name"]

        if not getattr(config.workflow_settings, suite_cfg["config_flag_name"], False):
            logger.info(f"Skipping {metric_name_log} estimation (disabled in config).")
            phase2_results[pseudo_step_name] = []
            continue

        logger.info(f"\nEstimating Scenario-Level Metric: {metric_name_log} ...")
        metric_config = input_data.scenario.scenario_level_metrics.get(suite_cfg["benchmark_key_in_scenario_yaml"])
        if not metric_config:
            logger.error(f"No configuration found for {metric_name_log} in scenario YAML (key: {suite_cfg['benchmark_key_in_scenario_yaml']}). Skipping this metric.")
            phase2_results[pseudo_step_name] = [{"error": f"Metric config missing for {metric_name_log}"}]
            continue
        
        benchmark_file_path_normalized = metric_config.benchmark_file
        metric_assumptions = metric_config.assumptions
        if metric_assumptions:
            logger.info(f"  Using assumptions for {metric_name_log}: '{metric_assumptions[:100]}...'")

        current_dedicated_benchmark = input_data.loaded_benchmarks.get(benchmark_file_path_normalized)
        if not current_dedicated_benchmark:
            logger.error(f"Dedicated benchmark '{benchmark_file_path_normalized}' for {metric_name_log} not found. Skipping this metric.")
            phase2_results[pseudo_step_name] = [{"error": f"Benchmark file {benchmark_file_path_normalized} not loaded"}]
            continue
        
        logger.info(f"  Using Benchmark for {metric_name_log}: '{benchmark_file_path_normalized}' ({current_dedicated_benchmark.description[:30]}...)")
        
        tasks_for_this_metric_benchmark = _select_items(current_dedicated_benchmark.tasks, config.workflow_settings.num_tasks, f"task for {metric_name_log}")
        if not tasks_for_this_metric_benchmark:
            logger.warning(f"No tasks selected for {metric_name_log} from benchmark '{benchmark_file_path_normalized}'.")
            phase2_results[pseudo_step_name] = []
            continue

        metric_suite_task_results_list: List[Dict[str, Any]] = []

        for task_from_bench in tasks_for_this_metric_benchmark:
            processed_metric_tasks_phase2 += 1
            logger.info(f"\n\n  Processing Metric Combo {processed_metric_tasks_phase2}/{total_metric_tasks_to_process_phase2}: Metric='{metric_name_log}', BenchmarkTask='{task_from_bench.name}'")
            
            task_metric_result_dict: Dict[str, Any] = {"task_name": task_from_bench.name, "task_description": task_from_bench.description, "task_metrics": task_from_bench.metrics, "benchmark_source_for_task": benchmark_file_path_normalized, "rounds_data": [], "final_aggregated_estimate": None, "converged_at_round": None}
            round_responses_history: List[List[Dict[str, Any]]] = []

            for current_round in range(1, config.workflow_settings.delphi_rounds + 1):
                round_start_time = time.time()
                logger.info(f"    Round {current_round}/{config.workflow_settings.delphi_rounds} for {metric_name_log}/Task='{task_from_bench.name}' starting...")

                expert_coroutines = [
                    _run_expert_round_for_scenario_metric(
                        expert=expert, task_from_dedicated_benchmark=task_from_bench,
                        dedicated_benchmark_details=current_dedicated_benchmark, scenario=input_data.scenario,
                        full_scenario_steps_description_str=full_scenario_steps_description_str, round_num=current_round,
                        prev_round_responses_for_this_metric_task=round_responses_history[-1] if round_responses_history else None,
                        client=client, semaphore=semaphore, config=config, prompts=input_data.prompts,
                        metric_name_logging=metric_name_log, metric_initial_prompt_key=suite_cfg["initial_prompt_key"],
                        metric_subsequent_prompt_key=suite_cfg["subsequent_prompt_key"], metric_parser_func=suite_cfg["parser_func"],
                        scenario_level_metric_assumptions=metric_assumptions
                    ) for expert in experts_to_use
                ]
                current_round_api_results = await asyncio.gather(*expert_coroutines, return_exceptions=True)
                round_duration = time.time() - round_start_time
                logger.info(f"    Round {current_round} for {metric_name_log}/Task='{task_from_bench.name}' completed in {round_duration:.2f}s.")

                current_round_processed_responses = []
                for i, res in enumerate(current_round_api_results):
                    expert_name = experts_to_use[i].name
                    if isinstance(res, Exception):
                        logger.error(f"      Expert '{expert_name}' failed in R{current_round} for {metric_name_log}/Task='{task_from_bench.name}': {res}", exc_info=res)
                        current_round_processed_responses.append({"expert": expert_name, "error": f"Unhandled Exception: {res}"})
                    elif isinstance(res, dict):
                        current_round_processed_responses.append(res)
                    else: 
                        current_round_processed_responses.append({"expert": expert_name, "error": "Unexpected result type from expert round function"})
                
                round_responses_history.append(current_round_processed_responses)
                task_metric_result_dict["rounds_data"].append({"round": current_round, "responses": current_round_processed_responses})

                append_round_to_csv(
                    csv_path=run_info["csv_path"], step_name=pseudo_step_name, task_name=task_from_bench.name,
                    task_metrics=task_from_bench.metrics, round_num=current_round, responses=current_round_processed_responses,
                    run_id=run_info["run_id"], model=overall_results["run_metadata"]["config_used"]["llm_settings"]["model"],
                    temperature=overall_results["run_metadata"]["config_used"]["llm_settings"]["temperature"],
                    timestamp_start=overall_results["run_metadata"]["timestamp_start"]
                )

                valid_estimates = [r["estimate"] for r in current_round_processed_responses if "error" not in r and r.get("estimate") is not None]
                if len(valid_estimates) >= 2:
                    std_dev, mean_est = np.std(valid_estimates), np.mean(valid_estimates)
                    logger.info(f"    Round {current_round} Stats for {metric_name_log}/Task='{task_from_bench.name}': Valid Estimates={len(valid_estimates)}, Mean={mean_est:.2f}, StdDev={std_dev:.2f}")
                    # Convergence threshold might need to be different for quantities vs. probabilities
                    if std_dev < config.workflow_settings.convergence_threshold: 
                        logger.info(f"    Convergence reached for {metric_name_log}/Task='{task_from_bench.name}' at Round {current_round} (StdDev {std_dev:.4f} < {config.workflow_settings.convergence_threshold:.4f})")
                        task_metric_result_dict["converged_at_round"] = current_round
                        break 
                else: 
                    logger.info(f"    Round {current_round} Stats for {metric_name_log}/Task='{task_from_bench.name}': Not enough valid estimates ({len(valid_estimates)}) to check convergence.")

            final_round_num_metric = len(task_metric_result_dict["rounds_data"])
            if task_metric_result_dict["rounds_data"]:
                final_round_responses_metric = task_metric_result_dict["rounds_data"][-1]["responses"]
                final_valid_estimates = [r["estimate"] for r in final_round_responses_metric if "error" not in r and r.get("estimate") is not None]
                if final_valid_estimates:
                    task_metric_result_dict["final_aggregated_estimate"] = np.mean(final_valid_estimates) 
                    logger.info(f"    Final Agg. Value for {metric_name_log}/Task='{task_from_bench.name}': {task_metric_result_dict['final_aggregated_estimate']:.2f} (from {len(final_valid_estimates)} estimates in R{final_round_num_metric})")
                else:
                    task_metric_result_dict["final_aggregated_estimate"] = None
                    logger.warning(f"    No valid final quantity estimates found for {metric_name_log}/Task='{task_from_bench.name}' after R{final_round_num_metric}.")
            
            metric_suite_task_results_list.append(task_metric_result_dict)
        
        phase2_results[pseudo_step_name] = metric_suite_task_results_list
        
        avg_metric_val, count_valid_metric_tasks = None, 0
        if metric_suite_task_results_list:
            all_final_agg_estimates = [res["final_aggregated_estimate"] for res in metric_suite_task_results_list if res.get("final_aggregated_estimate") is not None]
            if all_final_agg_estimates:
                avg_metric_val = np.mean(all_final_agg_estimates)
                count_valid_metric_tasks = len(all_final_agg_estimates)

        val_str = f"{avg_metric_val:.2f}" if avg_metric_val is not None else "N/A"
        logger.info(f"--- Metric Suite Summary ({metric_name_log}) ---")
        logger.info(f"    Benchmark: {benchmark_file_path_normalized}")
        logger.info(f"    Average aggregated '{metric_name_log}' value across {count_valid_metric_tasks} benchmark tasks: {val_str}")
        logger.info(f"--------------------------------------")

    logger.info("--- Phase 2: Scenario-Level Metric Estimations Completed ---")
    return phase2_results


# --- Main Workflow Function ---

async def run_delphi_estimation(
    client: Union[AsyncAnthropic, AsyncOpenAI],
    semaphore: Semaphore,
    config: AppConfig,
    input_data: InputData
) -> Dict[str, Any]:
    """
    Runs the main Delphi estimation workflow.
    """
    run_start_time = time.time()
    logger.info("\n\n--- Starting Full Delphi Estimation Workflow ---")
    project_root = Path.cwd() 

    if not all([input_data.prompts, input_data.experts, input_data.scenario, input_data.loaded_benchmarks]):
        return {"error": "One or more critical input data components are missing."}

    experts_to_use = _select_items(input_data.experts, config.workflow_settings.num_experts, "expert")
    if not experts_to_use: return {"error": "No experts selected to run."}

    run_info = initialize_run(config)
    if not run_info:
        return {"error": "Failed to initialize run"}

    overall_results: Dict[str, Any] = {
        "run_metadata": {
            "timestamp_start": datetime.datetime.now().isoformat(),
            "config_used": {
                 "llm_settings": asdict(config.llm_settings),
                 "workflow_settings": asdict(config.workflow_settings),
                 "provider": config.inferred_api_provider,
                 "benchmark_file": str(config.default_benchmark_file.relative_to(project_root).as_posix()),
                 "scenario_file": str(config.scenario_file.relative_to(project_root).as_posix()),
                 "num_experts_run": len(experts_to_use),
                 "num_steps_run": 0, 
                 "num_tasks_run": config.workflow_settings.num_tasks
            },
        },
        "results_per_step": [] 
    }

    # PHASE 1: Step-Specific Probability Estimation
    logger.info("\n\n--- Starting Phase 1: Step-Specific Probability Estimations ---")
    
    steps_to_run_phase1 = []
    if not input_data.scenario.steps:
        logger.warning("No scenario steps defined. Skipping Phase 1.")
    else:
        all_step_names = [step.name for step in input_data.scenario.steps]
        if config.workflow_settings.scenario_steps is None:
            steps_to_run_phase1 = input_data.scenario.steps
        else:
            specified_steps = config.workflow_settings.scenario_steps
            if isinstance(specified_steps, list):
                steps_to_run_phase1 = [s for s in input_data.scenario.steps if s.name in specified_steps]
                missing_steps = set(specified_steps) - set(s.name for s in steps_to_run_phase1)
                if missing_steps:
                    logger.warning(f"Configured steps not found in scenario: {list(missing_steps)}. Available: {all_step_names}")
            else:
                 logger.error("Config error: 'scenario_steps' must be a list or null. Skipping Phase 1.")
    
    if steps_to_run_phase1:
        logger.info(f"Phase 1: Processing probability for steps: {[s.name for s in steps_to_run_phase1]}")
    overall_results["run_metadata"]["config_used"]["num_steps_run"] = len(steps_to_run_phase1)
    
    approx_total_tasks_phase1 = 0
    for scenario_step in steps_to_run_phase1:
        benchmark_key = scenario_step.benchmark_file or overall_results["run_metadata"]["config_used"]["benchmark_file"]
        benchmark = input_data.loaded_benchmarks.get(benchmark_key)
        if benchmark:
            tasks = _select_items(benchmark.tasks, config.workflow_settings.num_tasks, f"task from {benchmark_key}")
            approx_total_tasks_phase1 += len(tasks)
    
    processed_tasks_phase1 = 0

    for scenario_step in steps_to_run_phase1:
        step_results_p1: Dict[str, Any] = {"step_name": scenario_step.name, "step_description": scenario_step.description, "step_type": "ProbabilityEstimation", "results_per_task": []}
        
        benchmark_key = scenario_step.benchmark_file or overall_results["run_metadata"]["config_used"]["benchmark_file"]
        current_benchmark_p1 = input_data.loaded_benchmarks.get(benchmark_key)

        if not current_benchmark_p1:
            logger.error(f"Phase 1: Benchmark (key: '{benchmark_key}') for step '{scenario_step.name}' not found. Skipping.")
            step_results_p1["error"] = f"Benchmark '{benchmark_key}' could not be loaded."
            overall_results["results_per_step"].append(step_results_p1)
            continue
        
        logger.info(f"\nPhase 1: Processing Step '{scenario_step.name}' using Benchmark '{benchmark_key}'")
        step_results_p1["benchmark_used"] = benchmark_key

        tasks_to_run_for_step = _select_items(current_benchmark_p1.tasks, config.workflow_settings.num_tasks, f"task from '{benchmark_key}'")

        for task_p1 in tasks_to_run_for_step:
            processed_tasks_phase1 += 1
            logger.info(f"\n\n  Processing Prob. Combo {processed_tasks_phase1}/{approx_total_tasks_phase1}: Step='{scenario_step.name}', Task='{task_p1.name}'")
            
            task_step_result_p1: Dict[str, Any] = {"task_name": task_p1.name, "task_description": task_p1.description, "task_metrics": task_p1.metrics, "benchmark_source_for_task": benchmark_key, "rounds_data": [], "final_aggregated_probability": None, "converged_at_round": None}
            round_responses_history_p1: List[List[Dict[str, Any]]] = []

            for current_round_p1 in range(1, config.workflow_settings.delphi_rounds + 1):
                round_start_time = time.time()
                logger.info(f"    Round {current_round_p1}/{config.workflow_settings.delphi_rounds} for ProbEst/Step='{scenario_step.name}'/Task='{task_p1.name}' starting...")

                coroutines = [
                    _run_expert_round(
                        expert=expert, task=task_p1, benchmark=current_benchmark_p1,
                        scenario_step=scenario_step, scenario=input_data.scenario,
                        round_num=current_round_p1, prev_round_responses=round_responses_history_p1[-1] if round_responses_history_p1 else None,
                        client=client, semaphore=semaphore, config=config, prompts=input_data.prompts
                    ) for expert in experts_to_use
                ]
                round_api_results = await asyncio.gather(*coroutines, return_exceptions=True)
                logger.info(f"    Round {current_round_p1} completed in {time.time() - round_start_time:.2f}s.")
                
                processed_responses = []
                for i, res in enumerate(round_api_results):
                    expert_name = experts_to_use[i].name
                    if isinstance(res, Exception):
                        logger.error(f"      Expert '{expert_name}' failed: {res}", exc_info=res)
                        processed_responses.append({"expert": expert_name, "error": f"Unhandled Exception: {res}"})
                    else:
                        processed_responses.append(res)

                round_responses_history_p1.append(processed_responses)
                task_step_result_p1["rounds_data"].append({"round": current_round_p1, "responses": processed_responses})

                append_round_to_csv(
                    csv_path=run_info["csv_path"], step_name=scenario_step.name, task_name=task_p1.name,
                    task_metrics=task_p1.metrics, round_num=current_round_p1, responses=processed_responses,
                    run_id=run_info["run_id"], model=overall_results["run_metadata"]["config_used"]["llm_settings"]["model"],
                    temperature=overall_results["run_metadata"]["config_used"]["llm_settings"]["temperature"],
                    timestamp_start=overall_results["run_metadata"]["timestamp_start"]
                )

                valid_estimates = [r["estimate"] for r in processed_responses if "error" not in r and isinstance(r.get("estimate"), float)]
                if len(valid_estimates) >= 2:
                    std_dev, mean_est = np.std(valid_estimates), np.mean(valid_estimates)
                    logger.info(f"    Round {current_round_p1} Stats: Valid Estimates={len(valid_estimates)}, Mean={mean_est:.4f}, StdDev={std_dev:.4f}")
                    if std_dev < config.workflow_settings.convergence_threshold:
                        logger.info(f"    Convergence reached at R{current_round_p1} (StdDev {std_dev:.4f} < {config.workflow_settings.convergence_threshold:.4f})")
                        task_step_result_p1["converged_at_round"] = current_round_p1
                        break 
                else:
                     logger.info(f"    Round {current_round_p1} Stats: Not enough valid estimates ({len(valid_estimates)}) to check convergence.")

            if task_step_result_p1["rounds_data"]:
                final_round_responses = task_step_result_p1["rounds_data"][-1]["responses"]
                final_valid_estimates = [r["estimate"] for r in final_round_responses if "error" not in r and isinstance(r.get("estimate"), float)]
                if final_valid_estimates:
                    task_step_result_p1["final_aggregated_probability"] = np.mean(final_valid_estimates)
                    logger.info(f"    Final Agg. Prob.: {task_step_result_p1['final_aggregated_probability']:.4f}")
                else:
                    logger.warning("    No valid final probability estimates found.")

            step_results_p1["results_per_task"].append(task_step_result_p1)
            save_intermediate_json(run_info["json_path"], overall_results)
            
        overall_results["results_per_step"].append(step_results_p1)
    
    logger.info("--- Phase 1: Step-Specific Probability Estimations Completed ---")

    # PHASE 2: Scenario-Level Metric Estimation
    full_scenario_desc_str = _prepare_scenario_full_steps_description(input_data.scenario)
    phase2_results = await _run_scenario_level_metric_estimation_phase(
        client, semaphore, config, input_data, full_scenario_desc_str, experts_to_use, project_root,
        run_info, overall_results
    )

    for pseudo_step_name, tasks_list in phase2_results.items():
        overall_results["results_per_step"].append({
            "step_name": pseudo_step_name, 
            "step_description": f"Estimation for {pseudo_step_name.replace('ScenarioLevelMetric_', '')}",
            "step_type": "ScenarioLevelMetricEstimation", 
            "results_per_task": tasks_list 
        })
        save_intermediate_json(run_info["json_path"], overall_results)

    # Finalization
    run_duration = time.time() - run_start_time
    overall_results["run_metadata"]["timestamp_end"] = datetime.datetime.now().isoformat()
    overall_results["run_metadata"]["duration_seconds"] = round(run_duration, 2)
    
    finalize_run(config, run_info["run_id"], run_info["run_dir"], overall_results)
    
    logger.info(f"--- Full Delphi Estimation Workflow Completed in {run_duration:.2f} seconds ---")
    return overall_results