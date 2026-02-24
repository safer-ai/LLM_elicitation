#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Results handling for intra-benchmark calibration experiments.

This module provides progressive result saving functionality adapted
from the main results_handler.py for the intra-benchmark use case.
"""

import json
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def _generate_run_id() -> str:
    """Generates a unique run ID based on the current timestamp."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _create_descriptive_filename(
    base_name: str,
    timestamp: str,
    benchmark_name: str,
    n_bins: int,
    model: str,
    num_experts: int,
    delphi_rounds: int,
    temperature: float,
    extension: str
) -> str:
    """
    Create a descriptive filename with abbreviated parameter names.
    
    Args:
        base_name: Base name for the file (e.g., "estimates", "results")
        timestamp: Timestamp string (e.g., "20251123_011829")
        benchmark_name: Name of the benchmark
        n_bins: Number of bins
        model: Model name
        num_experts: Number of experts
        delphi_rounds: Number of Delphi rounds
        temperature: Temperature value
        extension: File extension (e.g., "csv", "json")
    
    Returns:
        Descriptive filename string
    """
    # Format temperature to remove unnecessary decimals
    temp_str = f"tmp{temperature:.1f}".replace(".", "")
    
    # Build filename components
    components = [
        timestamp,
        benchmark_name,
        f"nbins{n_bins}",
        model,
        f"nexp{num_experts}",
        f"nrnd{delphi_rounds}",
        temp_str,
        base_name
    ]
    
    return "_".join(components) + f".{extension}"


def initialize_intra_benchmark_run(
    benchmark_name: str,
    n_bins: int,
    model: str,
    num_experts: int,
    delphi_rounds: int,
    temperature: float,
    output_base_dir: Path
) -> Optional[Dict[str, Any]]:
    """
    Initialize directory structure and files for an intra-benchmark run.

    Args:
        benchmark_name: Name of benchmark (e.g., "cybench")
        n_bins: Number of bins used
        model: Model name
        num_experts: Number of experts
        delphi_rounds: Max number of Delphi rounds
        temperature: LLM temperature setting
        output_base_dir: Base output directory (e.g., output_data/intra_benchmark)

    Returns:
        Dict with run info: run_id, csv_path, json_path, run_dir
        Returns None if initialization fails
    """
    run_id = _generate_run_id()

    # Create directory structure: output_base_dir/benchmark_name/run_id/
    benchmark_dir = output_base_dir / benchmark_name
    run_dir = benchmark_dir / run_id

    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created run directory: {run_dir}")
    except Exception as e:
        logger.error(f"Failed to create run directory {run_dir}: {e}")
        return None

    # Generate descriptive filenames
    csv_filename = _create_descriptive_filename(
        base_name="estimates",
        timestamp=run_id,
        benchmark_name=benchmark_name,
        n_bins=n_bins,
        model=model,
        num_experts=num_experts,
        delphi_rounds=delphi_rounds,
        temperature=temperature,
        extension="csv"
    )
    
    json_filename = _create_descriptive_filename(
        base_name="results",
        timestamp=run_id,
        benchmark_name=benchmark_name,
        n_bins=n_bins,
        model=model,
        num_experts=num_experts,
        delphi_rounds=delphi_rounds,
        temperature=temperature,
        extension="json"
    )

    # Initialize CSV file with headers
    csv_path = run_dir / csv_filename
    csv_headers = [
        "bin_i",
        "bin_j",
        "bin_i_range",
        "bin_j_range",
        "expert",
        "round",
        "percentile_25th",
        "percentile_50th",
        "percentile_75th",
        "rationale",
        "ground_truth_p_j_given_i",
        "sufficient_sample"
    ]

    try:
        df = pd.DataFrame(columns=csv_headers)
        df.to_csv(csv_path, index=False)
        logger.info(f"Initialized CSV file: {csv_path}")
    except Exception as e:
        logger.error(f"Failed to initialize CSV file {csv_path}: {e}")
        return None

    # Initialize JSON file with metadata
    json_path = run_dir / json_filename
    initial_json = {
        "run_metadata": {
            "run_id": run_id,
            "timestamp_start": datetime.datetime.now().isoformat(),
            "timestamp_end": None,
            "mode": "intra_benchmark",
            "benchmark_name": benchmark_name,
            "n_bins": n_bins,
            "model": model,
            "num_experts": num_experts,
            "delphi_rounds_max": delphi_rounds,
            "temperature": temperature,
            "total_predictions": 0,
            "predictions_computed": 0
        },
        "predictions": []
    }

    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(initial_json, f, indent=2, ensure_ascii=False)
        logger.info(f"Initialized JSON file: {json_path}")
    except Exception as e:
        logger.error(f"Failed to initialize JSON file {json_path}: {e}")
        return None

    return {
        "run_id": run_id,
        "csv_path": csv_path,
        "json_path": json_path,
        "run_dir": run_dir,
        "benchmark_name": benchmark_name,
        "n_bins": n_bins
    }


def append_prediction_to_csv(
    csv_path: Path,
    bin_i: int,
    bin_j: int,
    bin_i_range: str,
    bin_j_range: str,
    round_num: int,
    expert_responses: List[Dict[str, Any]],
    ground_truth_p_j_given_i: float,
    sufficient_sample: bool
):
    """
    Append expert responses for one (i,j) pair and round to the CSV file.

    Args:
        csv_path: Path to the CSV file
        bin_i: Source bin ID
        bin_j: Target bin ID
        bin_i_range: Source bin range string (e.g., "[5.0, 9.4)")
        bin_j_range: Target bin range string
        round_num: Current Delphi round number
        expert_responses: List of dicts with expert, estimate, rationale, etc.
        ground_truth_p_j_given_i: Ground truth conditional probability
        sufficient_sample: Whether this pair has sufficient sample
    """
    try:
        rows = []
        for response in expert_responses:
            row = {
                "bin_i": bin_i,
                "bin_j": bin_j,
                "bin_i_range": bin_i_range,
                "bin_j_range": bin_j_range,
                "expert": response.get("expert", "Unknown"),
                "round": round_num,
                "percentile_25th": response.get("percentile_25th"),
                "percentile_50th": response.get("percentile_50th"),
                "percentile_75th": response.get("percentile_75th"),
                "rationale": response.get("rationale", ""),
                "ground_truth_p_j_given_i": ground_truth_p_j_given_i,
                "sufficient_sample": sufficient_sample
            }
            rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            # Append to existing CSV
            df.to_csv(csv_path, mode='a', header=False, index=False)
            logger.debug(f"Appended {len(rows)} rows to CSV for bin pair ({bin_i}, {bin_j}) round {round_num}")
    except Exception as e:
        logger.error(f"Failed to append to CSV {csv_path}: {e}", exc_info=True)


def save_prediction_to_json(
    json_path: Path,
    prediction_result: Dict[str, Any]
):
    """
    Save a complete prediction result to the JSON file.

    This loads the existing JSON, appends the new prediction, and saves it back.

    Args:
        json_path: Path to the JSON file
        prediction_result: Dict with complete prediction data for one (i,j) pair
    """
    try:
        # Load existing JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Append new prediction
        if "predictions" not in data:
            data["predictions"] = []
        data["predictions"].append(prediction_result)

        # Update metadata
        data["run_metadata"]["predictions_computed"] = len(data["predictions"])

        # Save back
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        logger.debug(f"Saved prediction for bin pair ({prediction_result.get('bin_i')}, "
                     f"{prediction_result.get('bin_j')}) to JSON")
    except Exception as e:
        logger.error(f"Failed to save prediction to JSON {json_path}: {e}", exc_info=True)


def finalize_intra_benchmark_run(
    json_path: Path,
    total_predictions_attempted: int
):
    """
    Finalize the run by updating metadata with completion time.

    Args:
        json_path: Path to the JSON file
        total_predictions_attempted: Total number of (i,j) pairs attempted
    """
    try:
        # Load existing JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Update metadata
        data["run_metadata"]["timestamp_end"] = datetime.datetime.now().isoformat()
        data["run_metadata"]["total_predictions"] = total_predictions_attempted

        # Calculate duration if both timestamps exist
        start_str = data["run_metadata"].get("timestamp_start")
        end_str = data["run_metadata"].get("timestamp_end")
        if start_str and end_str:
            try:
                start_dt = datetime.datetime.fromisoformat(start_str)
                end_dt = datetime.datetime.fromisoformat(end_str)
                duration_seconds = (end_dt - start_dt).total_seconds()
                data["run_metadata"]["duration_seconds"] = duration_seconds
            except Exception:
                pass

        # Save back
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Finalized run metadata in {json_path}")
    except Exception as e:
        logger.error(f"Failed to finalize run {json_path}: {e}", exc_info=True)


def init_intra_benchmark_registry(registry_file: Path) -> Dict[str, Any]:
    """
    Initialize or load the intra-benchmark run registry.

    Args:
        registry_file: Path to run_registry.json

    Returns:
        Registry dict with structure: {"runs": [...], "last_updated": "..."}
    """
    registry: Dict[str, Any] = {"runs": [], "last_updated": None}

    try:
        if registry_file.is_file():
            with open(registry_file, 'r', encoding='utf-8') as f:
                registry_data = json.load(f)
                if isinstance(registry_data, dict) and isinstance(registry_data.get("runs"), list):
                    registry = registry_data
                    logger.info(f"Loaded existing intra-benchmark registry from {registry_file}")
                else:
                    logger.warning(f"Registry file {registry_file} has invalid format. Re-initializing.")
        else:
            logger.info(f"Registry file not found. Initializing new registry at {registry_file}")
    except Exception as e:
        logger.error(f"Error loading registry {registry_file}: {e}. Re-initializing.", exc_info=True)

    # Ensure essential keys exist
    if "runs" not in registry or not isinstance(registry["runs"], list):
        registry["runs"] = []
    if "last_updated" not in registry:
        registry["last_updated"] = datetime.datetime.now().isoformat()

    # Ensure parent directory exists
    registry_file.parent.mkdir(parents=True, exist_ok=True)

    return registry


def add_intra_benchmark_run_to_registry(
    registry_file: Path,
    run_id: str,
    benchmark_name: str,
    n_bins: int,
    model: str,
    num_experts: int,
    delphi_rounds: int,
    num_predictions_attempted: int,
    num_predictions_completed: int,
    output_path: Path,
    config_file: str,
    timestamp_start: str
):
    """
    Add a new intra-benchmark run to the registry and save it.

    Args:
        registry_file: Path to run_registry.json
        run_id: Unique run ID
        benchmark_name: Name of benchmark
        n_bins: Number of bins
        model: Model name
        num_experts: Number of experts used
        delphi_rounds: Max number of rounds
        num_predictions_attempted: Total (i,j) pairs attempted
        num_predictions_completed: Successfully completed pairs
        output_path: Path to run directory
        config_file: Config file used
        timestamp_start: ISO format timestamp
    """
    # Load registry
    registry = init_intra_benchmark_registry(registry_file)

    # Create entry
    run_entry = {
        "run_id": run_id,
        "benchmark_name": benchmark_name,
        "n_bins": n_bins,
        "timestamp": timestamp_start,
        "model": model,
        "num_experts": num_experts,
        "delphi_rounds": delphi_rounds,
        "num_predictions_attempted": num_predictions_attempted,
        "num_predictions_completed": num_predictions_completed,
        "output_path": str(output_path.relative_to(Path.cwd())),
        "config_file": config_file
    }

    # Check for duplicates
    if any(r.get("run_id") == run_id for r in registry["runs"]):
        logger.warning(f"Run ID {run_id} already exists in registry. Skipping add.")
        return

    # Add entry
    registry["runs"].append(run_entry)
    registry["runs"].sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    registry["last_updated"] = datetime.datetime.now().isoformat()

    # Save registry
    try:
        with open(registry_file, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Added run {run_id} to registry at {registry_file}")
    except Exception as e:
        logger.error(f"Failed to save registry {registry_file}: {e}", exc_info=True)


if __name__ == "__main__":
    # Test the results handler
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("Testing intra-benchmark results handler...")
    print("="*60)

    # Test initialization
    test_output_dir = Path("output_data/intra_benchmark")
    run_info = initialize_intra_benchmark_run(
        benchmark_name="cybench",
        n_bins=4,
        model="claude-sonnet-4-5-20250929",
        num_experts=5,
        delphi_rounds=3,
        temperature=0.8,
        output_base_dir=test_output_dir
    )

    if run_info:
        print(f"✓ Run initialized: {run_info['run_id']}")
        print(f"  Directory: {run_info['run_dir']}")
        print(f"  CSV: {run_info['csv_path']}")
        print(f"  JSON: {run_info['json_path']}")

        # Test CSV append (mock data)
        mock_responses = [
            {"expert": "Expert 1", "percentile_25th": 0.55, "percentile_50th": 0.65, "percentile_75th": 0.75, "rationale": "Test reasoning"},
            {"expert": "Expert 2", "percentile_25th": 0.60, "percentile_50th": 0.70, "percentile_75th": 0.80, "rationale": "Test reasoning 2"}
        ]

        append_prediction_to_csv(
            csv_path=run_info['csv_path'],
            bin_i=0,
            bin_j=1,
            bin_i_range="[5.0, 9.4)",
            bin_j_range="[9.4, 13.8)",
            round_num=1,
            expert_responses=mock_responses,
            ground_truth_p_j_given_i=0.6667,
            sufficient_sample=True
        )
        print("✓ CSV append test passed")

        # Clean up test files
        import shutil
        if run_info['run_dir'].exists():
            shutil.rmtree(run_info['run_dir'])
            print("✓ Cleaned up test files")
    else:
        print("✗ Failed to initialize run")

    print("="*60)
