#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main entry point for inter-benchmark calibration experiments.

Orchestrates the LLM-based Delphi estimation process for predicting
P(model solves target task on B | model scores in bin X on source A).
"""

import asyncio
import logging
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_inter_benchmark_config
from shared.llm_client import initialize_client
from shared.loaders import load_prompts, load_experts, load_benchmark
from workflow import run_inter_benchmark_estimation
from data_loader import load_ground_truth, validate_ground_truth_data

logger = logging.getLogger("InterBenchmarkCalibration")


async def main(config_path: str):
    """Orchestrates the inter-benchmark calibration experiment."""
    logger.info("--- Starting Inter-Benchmark Calibration Experiment ---")
    logger.info(f"Config file: {config_path}")

    # 1. Load Configuration
    try:
        script_dir = Path(__file__).parent.resolve()
        config = load_inter_benchmark_config(config_path, base_dir=script_dir)
        logger.info("Configuration loaded successfully")
        logger.info(f"  Source(s): {[s.name for s in config.source_benchmarks]}")
        logger.info(f"  Target: {config.target_benchmark.name}")
        logger.info(f"  Model: {config.llm_settings.model}")
    except (FileNotFoundError, ValueError, TypeError) as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        return

    # 2. Load Input Data
    try:
        prompts = load_prompts(config.prompts_dir)
        if not prompts:
            logger.error("Failed to load prompts")
            return

        experts = load_experts(config.expert_profiles_file)
        if not experts:
            logger.error("Failed to load experts")
            return

        # Load source benchmarks
        source_benchmarks = {}
        for src_cfg in config.source_benchmarks:
            src_file = Path(src_cfg.sorted_benchmark_file)
            if not src_file.is_absolute():
                src_file = script_dir / src_file
            bm = load_benchmark(src_file, src_cfg.name)
            if not bm:
                logger.error(f"Failed to load source benchmark: {src_cfg.name}")
                return
            source_benchmarks[src_cfg.name] = bm
            logger.info(f"Loaded source benchmark '{src_cfg.name}': {len(bm.tasks)} tasks")

        # Load target benchmark
        tgt_file = Path(config.target_benchmark.sorted_benchmark_file)
        if not tgt_file.is_absolute():
            tgt_file = script_dir / tgt_file
        target_benchmark = load_benchmark(tgt_file, config.target_benchmark.name)
        if not target_benchmark:
            logger.error(f"Failed to load target benchmark: {config.target_benchmark.name}")
            return
        logger.info(f"Loaded target benchmark '{config.target_benchmark.name}': {len(target_benchmark.tasks)} tasks")

        # Load ground truth
        ground_truth = load_ground_truth(config.ground_truth_file)
        if not ground_truth:
            logger.error("Failed to load ground truth data")
            return

        if not validate_ground_truth_data(ground_truth):
            logger.error("Ground truth validation failed")
            return

        logger.info(f"Loaded {len(experts)} expert profiles")
        logger.info(f"Loaded {len(prompts)} prompt templates")

        input_data = {
            "prompts": prompts,
            "experts": experts,
            "source_benchmarks": source_benchmarks,
            "target_benchmark": target_benchmark,
            "ground_truth": ground_truth
        }

    except Exception as e:
        logger.error(f"Error during input data loading: {e}", exc_info=True)
        return

    # 3. Initialise API Client
    try:
        client = initialize_client(
            config.api_key_anthropic, config.api_key_openai, config.llm_settings.model
        )
        model_lower = config.llm_settings.model.lower()
        provider = "anthropic" if "claude" in model_lower else "openai"
        logger.info(f"API client initialised for {provider}")
    except (ImportError, ValueError) as e:
        logger.error(f"Failed to initialise API client: {e}", exc_info=True)
        return

    # 4. Create Semaphore
    semaphore = asyncio.Semaphore(config.llm_settings.max_concurrent_calls)

    # 5. Run Workflow
    try:
        results = await run_inter_benchmark_estimation(client, semaphore, config, input_data)

        if "error" in results:
            logger.error(f"Workflow error: {results['error']}")
        else:
            logger.info("Workflow completed successfully")
            logger.info(f"  Run ID: {results.get('run_id')}")
            logger.info(f"  Predictions: {results.get('predictions_completed')}/{results.get('predictions_attempted')}")
            logger.info(f"  Output: {results.get('output_path')}")

    except Exception as e:
        logger.error(f"Critical error during workflow: {e}", exc_info=True)

    logger.info("--- Inter-Benchmark Calibration Experiment Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Inter-Benchmark Calibration Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python run_calibration.py -c config_example.yaml
  python run_calibration.py -c config_example.yaml -d
        """
    )
    parser.add_argument("-c", "--config", default="config_example.yaml",
                        help="Path to configuration YAML file")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    log_format = "%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s"
    logging.basicConfig(level=log_level, format=log_format,
                        handlers=[logging.StreamHandler(sys.stdout)])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger.setLevel(log_level)

    try:
        config_file = Path(args.config)
        if not config_file.exists():
            logger.error(f"Configuration file does not exist: {args.config}")
            sys.exit(1)

        asyncio.run(main(config_path=args.config))

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Critical error: {e}", exc_info=True)
        sys.exit(1)
