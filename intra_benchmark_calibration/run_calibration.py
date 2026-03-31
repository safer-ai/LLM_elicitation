#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main entry point for intra-benchmark calibration experiments.

This script orchestrates the LLM-based Delphi estimation process for predicting
conditional probabilities P(j|i) between benchmark task bins.
"""

import asyncio
import logging
import argparse
import sys
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_intra_benchmark_config
from shared.llm_client import initialize_client
from shared.loaders import load_prompts, load_experts, load_benchmark
from workflow import run_intra_benchmark_estimation

logger = logging.getLogger("IntraBenchmarkCalibration")

async def main(config_path: str):
    """
    Orchestrates the intra-benchmark calibration experiment.

    Args:
        config_path: Path to the configuration YAML file.
    """
    logger.info("--- Starting Intra-Benchmark Calibration Experiment ---")
    logger.info(f"Config file: {config_path}")

    # 1. Load Configuration
    try:
        # Get the directory where this script resides (intra_benchmark_calibration/)
        script_dir = Path(__file__).parent.resolve()
        config = load_intra_benchmark_config(config_path, base_dir=script_dir)
        logger.info("Configuration loaded successfully")
        logger.info(f"  Benchmark: {config.intra_benchmark_settings.benchmark_name}")
        logger.info(f"  N_bins: {config.intra_benchmark_settings.n_bins}")
        logger.info(f"  Model: {config.llm_settings.model}")
    except (FileNotFoundError, ValueError, TypeError) as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        return

    # 2. Load Input Data
    try:
        # Load prompts
        prompts = load_prompts(config.prompts_dir)
        if not prompts:
            logger.error("Failed to load prompts")
            return

        # Load experts
        experts = load_experts(config.expert_profiles_file)
        if not experts:
            logger.error("Failed to load experts")
            return

        # Load sorted benchmark
        benchmark = load_benchmark(config.benchmark_file, config.intra_benchmark_settings.benchmark_name)
        if not benchmark:
            logger.error(f"Failed to load benchmark from {config.benchmark_file}")
            return

        logger.info(f"Loaded {len(benchmark.tasks)} benchmark tasks")
        logger.info(f"Loaded {len(experts)} expert profiles")
        logger.info(f"Loaded {len(prompts)} prompt templates")

        # Package input data
        input_data = {
            "prompts": prompts,
            "experts": experts,
            "benchmark_tasks": benchmark.tasks,
            "benchmark_description": benchmark.description,
            "metrics_to_use": benchmark.metrics_to_use,
        }

    except Exception as e:
        logger.error(
            f"An unexpected error occurred during input data loading: {e}",
            exc_info=True,
        )
        return

    # 3. Initialize API Client
    try:
        client = initialize_client(
            config.api_key_anthropic, config.api_key_openai, config.llm_settings.model
        )
        model_lower = config.llm_settings.model.lower()
        provider = "anthropic" if "claude" in model_lower else "openai"
        logger.info(f"API client initialized successfully for {provider}")
    except (ImportError, ValueError) as e:
        logger.error(f"Failed to initialize API client: {e}", exc_info=True)
        logger.error(
            "Please ensure the correct API library is installed and the API key is set in the configuration file."
        )
        return

    # 4. Create Semaphore
    semaphore = asyncio.Semaphore(config.llm_settings.max_concurrent_calls)
    logger.debug(
        f"Semaphore created with limit: {config.llm_settings.max_concurrent_calls}"
    )

    # 5. Run Intra-Benchmark Workflow
    try:
        logger.info("Starting intra-benchmark calibration workflow...")
        results = await run_intra_benchmark_estimation(
            client, semaphore, config, input_data
        )

        if "error" in results:
            logger.error(f"Workflow completed with an error: {results['error']}")
        else:
            logger.info("Intra-benchmark calibration workflow completed successfully")
            logger.info(f"Run ID: {results.get('run_id')}")
            logger.info(
                f"Predictions completed: {results.get('predictions_completed')}/{results.get('predictions_attempted')}"
            )
            logger.info(f"Output path: {results.get('output_path')}")

    except Exception as e:
        logger.error(
            f"An unexpected critical error occurred during the workflow: {e}",
            exc_info=True,
        )

    logger.info("--- Intra-Benchmark Calibration Experiment Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Intra-Benchmark Calibration Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python run_calibration.py -c config_example.yaml
  python run_calibration.py -c config_example.yaml -d
        """,
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config_example.yaml",
        help="Path to the configuration YAML file (default: config_example.yaml)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug level logging for detailed output",
    )
    args = parser.parse_args()

    # Set up logging
    if args.debug:
        log_level = logging.DEBUG
        print("DEBUG logging enabled.", file=sys.stderr)
    else:
        log_level = logging.INFO

    log_format = "%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s"

    logging.basicConfig(
        level=log_level, format=log_format, handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Suppress overly verbose logs from HTTP libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Set logger level
    logger.setLevel(log_level)

    if args.debug:
        logger.info("DEBUG logging enabled (confirmed by logger).")
    else:
        logger.info(
            f"Standard logging enabled (level: {logging.getLevelName(log_level)}). Use -d for DEBUG."
        )

    try:
        config_file = Path(args.config)
        if not config_file.exists():
            logger.error(f"Configuration file does not exist: {args.config}")
            sys.exit(1)

        asyncio.run(main(config_path=args.config))

    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user (Ctrl+C).")
        sys.exit(0)
    except Exception as e:
        logger.critical(
            f"A critical unexpected error occurred at the top level: {e}", exc_info=True
        )
        sys.exit(1)
