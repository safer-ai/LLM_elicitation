#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import datetime
import logging
import argparse
import statistics
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, replace
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Tuple
import yaml

logger = logging.getLogger("PipelineRunner")

try:
    from config import load_config, AppConfig
    from data_loader import load_all_inputs, InputData
    from shared.llm_client import initialize_client
    from workflow import run_delphi_estimation
    from results_handler import initialize_run, finalize_run, save_intermediate_json
except ImportError as e:
    logger.error(f"Failed to import necessary project modules: {e}")
    logger.error("Ensure you have run 'pip install -r requirements.txt' (if applicable) and that the script is run from the project root directory.")
    sys.exit(1)

def _print_summary_table(results: Dict[str, Any]):
    """Formats and prints an ASCII summary table of results to the console."""
    logger.info("Attempting to generate run summary table...")

    if "error" in results:
        logger.warning(f"Skipping summary table generation due to workflow error: {results['error']}")
        return

    step_results_list = results.get("results_per_step", [])
    if not step_results_list:
        logger.warning("No step results found; cannot generate summary table.")
        return

    table_data_with_duplicates = []
    for step_res in step_results_list:
        step_name = step_res.get("step_name", "Unknown Step")
        step_type = step_res.get("step_type", "ProbabilityEstimation")
        task_results_list = step_res.get("results_per_task", [])

        for task_res in task_results_list:
            task_name = task_res.get("task_name", "Unknown Task")
            
            val_str = "N/A"
            if step_type == "ScenarioLevelMetricEstimation":
                final_val = task_res.get("final_aggregated_estimate")
                if final_val is not None:
                    val_str = f"{final_val:.2f}"
            elif step_type == "ProbabilityEstimation":
                final_prob = task_res.get("final_aggregated_probability")
                if final_prob is not None:
                    val_str = f"{final_prob:.4f}"
            else:
                logger.warning(f"Unknown step_type '{step_type}' for step '{step_name}'. Cannot determine value field for task '{task_name}'.")

            table_data_with_duplicates.append((step_name, task_name, val_str))

    # Workaround: Remove duplicate rows from table_data.
    # The root cause of duplicate data likely lies in workflow.py's results generation.
    table_data = []
    seen_rows = set()
    for row_tuple in table_data_with_duplicates:
        if row_tuple not in seen_rows:
            table_data.append(row_tuple)
            seen_rows.add(row_tuple)
    
    if not table_data_with_duplicates and not table_data: # Handles if original was empty
        logger.warning("No task results with data found after processing; cannot generate summary table.")
        return
    elif len(table_data_with_duplicates) > len(table_data):
        logger.info(f"Removed {len(table_data_with_duplicates) - len(table_data)} duplicate row(s) for summary table display.")


    # --- Column Width Calculation ---
    try:
        headers = ("Step Name", "Task Name", "Agg. Value")
        header_str_list = [str(h) for h in headers]
        
        if not table_data:
            step_width = len(header_str_list[0])
            task_width = len(header_str_list[1])
            val_width = len(header_str_list[2])
        else:
            # Ensure all items in rows are strings for len()
            data_str_lists = [[str(item) for item in row] for row in table_data]
            step_width = max(len(header_str_list[0]), max(len(row[0]) for row in data_str_lists))
            task_width = max(len(header_str_list[1]), max(len(row[1]) for row in data_str_lists))
            val_width = max(len(header_str_list[2]), max(len(row[2]) for row in data_str_lists))

    except Exception as e:
         logger.error(f"Error calculating table column widths: {e}. Skipping table generation.", exc_info=True)
         return

    # --- Table Formatting Strings ---
    header_format = f"| {{:<{step_width}}} | {{:<{task_width}}} | {{:>{val_width}}} |"
    row_format = f"| {{:<{step_width}}} | {{:<{task_width}}} | {{:>{val_width}}} |"
    separator = f"+-{'-'*step_width}-+-{'-'*task_width}-+-{'-'*val_width}-+"
    border_line = "=" * len(separator)

    # --- Printing the Table (ensuring it prints only once correctly) ---
    print(f"\n{border_line}")
    print(f"{' ' * ((len(separator) - len('Run Summary Table')) // 2)}Run Summary Table")
    print(border_line)
    print(separator)
    print(header_format.format(*headers))
    print(separator)
    
    if not table_data:
        no_data_message = "No data to display in table."
        print(f"| {no_data_message:<{step_width + task_width + val_width + 4}} |") # +4 for separators
    else:
        for row in table_data:
            print(row_format.format(row[0], row[1], row[2]))
    
    print(separator)
    print(border_line)
    print()

def _print_aggregated_summary_table(entries_for_model: List[Dict[str, Any]]):
    """Aggregates per-(step, task) final values across repeats of one model.

    Skips entries that carry an 'error'. Probability values are formatted to
    4 d.p., quantity (scenario-level metric) values to 2 d.p. Std is sample
    std (ddof=1); rows with a single repeat show '-' for std.
    """
    agg: "OrderedDict[Tuple[str, str, str], List[float]]" = OrderedDict()
    for entry in entries_for_model:
        if "error" in entry:
            continue
        for step_res in entry.get("results_per_step", []):
            step_name = step_res.get("step_name", "Unknown Step")
            step_type = step_res.get("step_type", "ProbabilityEstimation")
            for task_res in step_res.get("results_per_task", []):
                task_name = task_res.get("task_name", "Unknown Task")
                if step_type == "ScenarioLevelMetricEstimation":
                    val = task_res.get("final_aggregated_estimate")
                else:
                    val = task_res.get("final_aggregated_probability")
                if val is None:
                    continue
                try:
                    val_f = float(val)
                except (TypeError, ValueError):
                    continue
                agg.setdefault((step_name, task_name, step_type), []).append(val_f)

    if not agg:
        logger.warning("No data to aggregate; skipping aggregated summary table.")
        return

    rows: List[Tuple[str, str, str, str, str]] = []
    for (step_name, task_name, step_type), values in agg.items():
        n = len(values)
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if n > 1 else None
        if step_type == "ScenarioLevelMetricEstimation":
            mean_str = f"{mean:.2f}"
            std_str = f"{std:.2f}" if std is not None else "-"
        else:
            mean_str = f"{mean:.4f}"
            std_str = f"{std:.4f}" if std is not None else "-"
        rows.append((step_name, task_name, mean_str, std_str, str(n)))

    headers = ("Step Name", "Task Name", "Mean", "Std", "N")
    widths = [
        max(len(h), max((len(r[i]) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    row_fmt = (
        f"| {{:<{widths[0]}}} | {{:<{widths[1]}}} "
        f"| {{:>{widths[2]}}} | {{:>{widths[3]}}} | {{:>{widths[4]}}} |"
    )
    sep = (
        f"+-{'-'*widths[0]}-+-{'-'*widths[1]}-+-"
        f"{'-'*widths[2]}-+-{'-'*widths[3]}-+-{'-'*widths[4]}-+"
    )
    border = "=" * len(sep)
    title = "Aggregated Summary (mean / std across repeats)"

    print(border)
    print(f"{' ' * max(0, (len(sep) - len(title)) // 2)}{title}")
    print(border)
    print(sep)
    print(row_fmt.format(*headers))
    print(sep)
    for r in rows:
        print(row_fmt.format(*r))
    print(sep)
    print(border)
    print()


# --- Main Orchestration Function ---

def _config_for_model(base_config: AppConfig, model: str) -> AppConfig:
    """Returns a shallow copy of `base_config` with `llm_settings.model` rebound to `model`.

    All other settings (rate limits, reasoning_effort, paths, API keys) are preserved
    so that one call to `load_config` is enough for a multi-model run.
    """
    new_llm_settings = replace(base_config.llm_settings, model=model)
    return replace(base_config, llm_settings=new_llm_settings)


async def main(config_path: str):
    """Orchestrates the LLM Delphi estimation pipeline.

    A single `run_id` is created for the whole process. All models loop into
    the same CSV (one row per expert/round/task, with a `model` column) and
    the same JSON (a top-level `results_per_model` list). Intermediate JSON
    snapshots are written after each task so a partial run is recoverable.
    """
    logger.info(f"--- Starting LLM Estimator Pipeline using config: {config_path} ---")

    # 1. Load Configuration
    try:
        config = load_config(config_path)
        logger.info(
            f"Configuration loaded. Models to run: {config.models_to_run}. "
            f"Required providers: {sorted(config.required_providers)}."
        )
    except (FileNotFoundError, ValueError, TypeError, yaml.YAMLError) as e:
        logger.error(f"Failed to load or validate configuration: {e}", exc_info=True)
        return

    # 2. Load Input Data
    try:
        input_data = load_all_inputs(config)
        if input_data is None:
            logger.error("Exiting due to critical input data loading failure.")
            return
        logger.info("All input data loaded successfully.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during input data loading: {e}", exc_info=True)
        return

    # 3. Initialize Run Output
    # One run for the whole process: one timestamp / run_id / directory / CSV / JSON.
    run_info = initialize_run(config)
    if not run_info:
        logger.error("Failed to initialise run output directory; aborting.")
        return
    project_root = Path.cwd()

    # 4. Run Core Workflow
    process_start = time.time()
    combined_state: Dict[str, Any] = {
        "run_metadata": {
            "run_id": run_info["run_id"],
            "timestamp_start": datetime.datetime.now().isoformat(),
            "models_run": list(config.models_to_run),
            "scenario_file": str(config.scenario_file.relative_to(project_root).as_posix()),
            "default_benchmark_file": str(config.default_benchmark_file.relative_to(project_root).as_posix()),
            "workflow_settings": asdict(config.workflow_settings),
            "shared_llm_settings": {
                # Settings that are constant across all model runs (everything
                # except `model` itself). Captured here so per-model entries
                # don't have to repeat them.
                "temperature": config.llm_settings.temperature,
                "max_concurrent_calls": config.llm_settings.max_concurrent_calls,
                "rate_limit_calls": config.llm_settings.rate_limit_calls,
                "rate_limit_period": config.llm_settings.rate_limit_period,
                "reasoning_effort": config.llm_settings.reasoning_effort,
            },
            "num_repeats": config.workflow_settings.num_repeats,
        },
        "results_per_model": [],
    }
    save_intermediate_json(run_info["json_path"], combined_state)

    num_repeats = max(1, config.workflow_settings.num_repeats)

    # Sequential per-model runs. The semaphore and rate-limit state are
    # per-provider in practice (different providers don't share state), but
    # running sequentially keeps log output and CSV ordering predictable.
    # When num_repeats > 1 the inner loop runs the entire pipeline `num_repeats`
    # times for each model, producing independent samples (the analysis cache
    # is reset on every call) tagged with `repeat_index` in CSV/JSON.
    for model in config.models_to_run:
        per_model_config = _config_for_model(config, model)
        provider = per_model_config.inferred_api_provider
        logger.info(
            f"\n=== Running model: {model} (provider: {provider}, "
            f"repeats: {num_repeats}) ==="
        )

        try:
            client = initialize_client(
                api_key_anthropic=per_model_config.api_key_anthropic,
                api_key_openai=per_model_config.api_key_openai,
                model=per_model_config.llm_settings.model,
                api_key_gemini=per_model_config.api_key_gemini,
            )
            logger.info(f"API client initialised for {provider}.")
        except (ImportError, ValueError) as e:
            logger.error(f"Failed to initialize API client for model '{model}': {e}", exc_info=True)
            for repeat_index in range(1, num_repeats + 1):
                combined_state["results_per_model"].append({
                    "model": model, "provider": provider,
                    "repeat_index": repeat_index,
                    "error": f"Client init failed: {e}",
                })
            save_intermediate_json(run_info["json_path"], combined_state)
            continue

        semaphore = asyncio.Semaphore(per_model_config.llm_settings.max_concurrent_calls)

        for repeat_index in range(1, num_repeats + 1):
            if num_repeats > 1:
                logger.info(f"\n--- {model}: starting repeat {repeat_index}/{num_repeats} ---")
            try:
                await run_delphi_estimation(
                    client, semaphore, per_model_config, input_data,
                    run_info=run_info, combined_state=combined_state,
                    repeat_index=repeat_index,
                )
            except Exception as e:
                logger.error(
                    f"Critical error during Delphi workflow for model '{model}' "
                    f"(repeat {repeat_index}/{num_repeats}): {e}",
                    exc_info=True,
                )
                # The workflow may already have appended a partial entry to
                # `results_per_model`; record the crash on the entry whose
                # (model, repeat_index) matches and that doesn't already
                # carry an error.
                for entry in combined_state["results_per_model"]:
                    if (entry.get("model") == model
                            and entry.get("repeat_index") == repeat_index
                            and "error" not in entry):
                        entry["error"] = f"Unexpected workflow crash: {e}"
                        break
                else:
                    combined_state["results_per_model"].append({
                        "model": model, "provider": provider,
                        "repeat_index": repeat_index,
                        "error": f"Unexpected workflow crash: {e}",
                    })
                save_intermediate_json(run_info["json_path"], combined_state)

    # Finalise once: stamp end times, write the canonical JSON, register the run.
    process_duration = time.time() - process_start
    combined_state["run_metadata"]["timestamp_end"] = datetime.datetime.now().isoformat()
    combined_state["run_metadata"]["duration_seconds"] = round(process_duration, 2)
    finalize_run(config, run_info["run_id"], run_info["run_dir"], combined_state)

    logger.info("--- LLM Estimator Pipeline Finished ---")
    logger.info(f"Results saved to: {run_info['run_dir']}")
    logger.info(f"  - CSV : {run_info['csv_path'].name}")
    logger.info(f"  - JSON: {run_info['json_path'].name}")

    for entry in combined_state["results_per_model"]:
        model_name = entry.get("model", "<unknown>")
        rep = entry.get("repeat_index", 1)
        rep_label = f" (repeat {rep}/{num_repeats})" if num_repeats > 1 else ""
        if "error" in entry:
            print(f"\n>>> Summary for model: {model_name}{rep_label} -- ERROR: {entry['error']}")
            continue
        print(f"\n>>> Summary for model: {model_name}{rep_label}")
        _print_summary_table(entry)

    # When num_repeats > 1, also print one aggregated table per model that
    # combines all repeats: per (step_name, task_name) we report mean, std and
    # n. Models are kept in their original config order.
    if num_repeats > 1:
        entries_by_model: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
        for entry in combined_state["results_per_model"]:
            entries_by_model.setdefault(entry.get("model", "<unknown>"), []).append(entry)
        for model_name, entries in entries_by_model.items():
            print(f"\n>>> Aggregated summary across {num_repeats} repeats for model: {model_name}")
            _print_aggregated_summary_table(entries)


# --- Script Entry Point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LLM Delphi Estimation Pipeline.")
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to the configuration YAML file (default: config.yaml)"
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug level logging for detailed output."
    )
    args = parser.parse_args()

    if args.debug:
        log_level = logging.DEBUG
        print("DEBUG logging enabled.", file=sys.stderr)
    else:
        log_level = logging.INFO
    
    log_format = '%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s'

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Set specific logger levels if needed, otherwise they inherit from root.
    project_loggers = ["PipelineRunner", "config", "data_loader", "shared.llm_client", "shared.parsing", "results_handler", "workflow"]
    for proj_logger_name in project_loggers:
        logging.getLogger(proj_logger_name).setLevel(log_level)
    
    if args.debug:
        logger.info("DEBUG logging enabled (confirmed by logger).")
    else:
        logger.info(f"Standard logging enabled (level: {logging.getLevelName(log_level)}). Use -d for DEBUG.")

    try:
        if not Path(args.config).is_file():
             logger.error(f"Configuration file specified does not exist: {args.config}")
             sys.exit(1)
        asyncio.run(main(config_path=args.config))
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user (Ctrl+C).")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"A critical unexpected error occurred at the top level: {e}", exc_info=True)
        sys.exit(1)