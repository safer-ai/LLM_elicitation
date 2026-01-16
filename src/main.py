#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import logging
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Tuple
import yaml # Added for type hinting in main for config error

# --- Logger Setup (Get logger instance, level set later) ---
logger = logging.getLogger("PipelineRunner")

# --- Import Project Modules ---
try:
    from config import load_config, AppConfig
    from data_loader import load_all_inputs, InputData
    from llm_api import initialize_client
    from workflow import run_delphi_estimation
    # No longer need save_run_results since it's done progressively
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

# --- Main Orchestration Function ---

async def main(config_path: str):
    """
    Orchestrates the LLM Delphi estimation pipeline.

    Args:
        config_path: Path to the configuration YAML file.
    """
    logger.info(f"--- Starting LLM Estimator Pipeline using config: {config_path} ---")

    # 1. Load Configuration
    try:
        config = load_config(config_path)
        logger.info(f"Configuration loaded successfully. API Provider: {config.inferred_api_provider}, Model: {config.llm_settings.model}")
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

    # 3. Initialize API Client
    try:
        client = initialize_client(config)
        logger.info(f"API client initialized successfully for {config.inferred_api_provider}.")
    except (ImportError, ValueError) as e:
        logger.error(f"Failed to initialize API client: {e}", exc_info=True)
        logger.error("Please ensure the correct API library is installed and the API key is set in the configuration file.")
        return

    # 4. Create Semaphore
    semaphore = asyncio.Semaphore(config.llm_settings.max_concurrent_calls)
    logger.debug(f"Semaphore created with limit: {config.llm_settings.max_concurrent_calls}")

    # 5. Run Core Workflow
    results = {}
    run_id = None
    try:
        logger.info("Starting Delphi estimation workflow...")
        logger.info("Results will be saved progressively during execution.")
        results = await run_delphi_estimation(client, semaphore, config, input_data)
        if "error" in results:
            logger.error(f"Workflow completed with an error: {results['error']}")
        else:
            logger.info("Delphi estimation workflow completed successfully.")
            
            # Extract run_id from the timestamp (same format as _generate_run_id)
            timestamp_start = results.get("run_metadata", {}).get("timestamp_start", "")
            if timestamp_start:
                # Convert ISO format to run_id format
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp_start)
                run_id = dt.strftime("%Y%m%d_%H%M%S")
                logger.info(f"Results saved to: {config.runs_dir / run_id}")
                logger.info(f"  - CSV: detailed_estimates.csv")
                logger.info(f"  - JSON: full_results.json")
            
    except Exception as e:
        logger.error(f"An unexpected critical error occurred during the Delphi workflow: {e}", exc_info=True)
        results = {"error": f"Unexpected workflow crash: {e}"}

    logger.info("--- LLM Estimator Pipeline Finished ---")

    # 6. Print Summary Table
    _print_summary_table(results)


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
    project_loggers = ["PipelineRunner", "config", "data_loader", "llm_api", "parsing", "results_handler", "workflow"]
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