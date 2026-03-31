#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Results handling for inter-benchmark calibration experiments.

Progressive result saving: CSV rows are appended after each Delphi round,
JSON predictions are appended after each (source_bin, target_percentile) pair.
"""

import json
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def _generate_run_id() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _create_descriptive_filename(
    base_name: str,
    timestamp: str,
    source_name: str,
    target_name: str,
    model: str,
    num_experts: int,
    delphi_rounds: int,
    temperature: float,
    extension: str
) -> str:
    temp_str = f"tmp{temperature:.1f}".replace(".", "")
    components = [
        timestamp,
        f"{source_name}_to_{target_name}",
        model,
        f"nexp{num_experts}",
        f"nrnd{delphi_rounds}",
        temp_str,
        base_name
    ]
    return "_".join(components) + f".{extension}"


def initialize_inter_benchmark_run(
    source_name: str,
    target_name: str,
    model: str,
    num_experts: int,
    delphi_rounds: int,
    temperature: float,
    output_base_dir: Path,
    n_source_bins: int = 0,
    n_target_percentiles: int = 0
) -> Optional[Dict[str, Any]]:
    """
    Initialise directory structure and files for an inter-benchmark run.

    Returns:
        Dict with run_id, csv_path, json_path, run_dir, or None on failure.
    """
    run_id = _generate_run_id()
    pair_name = f"{source_name}_to_{target_name}"
    run_dir = output_base_dir / pair_name / run_id

    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created run directory: {run_dir}")
    except Exception as e:
        logger.error(f"Failed to create run directory {run_dir}: {e}")
        return None

    csv_filename = _create_descriptive_filename(
        "estimates", run_id, source_name, target_name,
        model, num_experts, delphi_rounds, temperature, "csv"
    )
    json_filename = _create_descriptive_filename(
        "results", run_id, source_name, target_name,
        model, num_experts, delphi_rounds, temperature, "json"
    )

    csv_path = run_dir / csv_filename
    csv_headers = [
        "source_bin", "source_bin_range",
        "target_percentile", "target_task_index", "target_task_name",
        "expert", "round",
        "percentile_25th", "percentile_50th", "percentile_75th",
        "rationale",
        "ground_truth_p_solve", "sufficient_sample"
    ]

    try:
        df = pd.DataFrame(columns=csv_headers)
        df.to_csv(csv_path, index=False)
        logger.info(f"Initialised CSV file: {csv_path}")
    except Exception as e:
        logger.error(f"Failed to initialise CSV file {csv_path}: {e}")
        return None

    json_path = run_dir / json_filename
    initial_json = {
        "run_metadata": {
            "run_id": run_id,
            "timestamp_start": datetime.datetime.now().isoformat(),
            "timestamp_end": None,
            "mode": "inter_benchmark",
            "source_benchmark": source_name,
            "target_benchmark": target_name,
            "n_source_bins": n_source_bins,
            "n_target_percentiles": n_target_percentiles,
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
        logger.info(f"Initialised JSON file: {json_path}")
    except Exception as e:
        logger.error(f"Failed to initialise JSON file {json_path}: {e}")
        return None

    return {
        "run_id": run_id,
        "csv_path": csv_path,
        "json_path": json_path,
        "run_dir": run_dir,
        "source_name": source_name,
        "target_name": target_name
    }


def append_prediction_to_csv(
    csv_path: Path,
    source_bin: int,
    source_bin_range: str,
    target_percentile: float,
    target_task_index: int,
    target_task_name: str,
    round_num: int,
    expert_responses: List[Dict[str, Any]],
    ground_truth_p_solve: float,
    sufficient_sample: bool
):
    """Append expert responses for one prediction and round to CSV."""
    try:
        rows = []
        for response in expert_responses:
            row = {
                "source_bin": source_bin,
                "source_bin_range": source_bin_range,
                "target_percentile": target_percentile,
                "target_task_index": target_task_index,
                "target_task_name": target_task_name,
                "expert": response.get("expert", "Unknown"),
                "round": round_num,
                "percentile_25th": response.get("percentile_25th"),
                "percentile_50th": response.get("percentile_50th"),
                "percentile_75th": response.get("percentile_75th"),
                "rationale": response.get("rationale", ""),
                "ground_truth_p_solve": ground_truth_p_solve,
                "sufficient_sample": sufficient_sample
            }
            rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, mode='a', header=False, index=False)
            logger.debug(f"Appended {len(rows)} rows for bin {source_bin}, pct {target_percentile}, round {round_num}")
    except Exception as e:
        logger.error(f"Failed to append to CSV {csv_path}: {e}", exc_info=True)


def save_prediction_to_json(json_path: Path, prediction_result: Dict[str, Any]):
    """Save a complete prediction result to the JSON file (progressive)."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "predictions" not in data:
            data["predictions"] = []
        data["predictions"].append(prediction_result)
        data["run_metadata"]["predictions_computed"] = len(data["predictions"])

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        logger.debug(f"Saved prediction for bin {prediction_result.get('source_bin')}, "
                     f"pct {prediction_result.get('target_percentile')} to JSON")
    except Exception as e:
        logger.error(f"Failed to save prediction to JSON {json_path}: {e}", exc_info=True)


def finalize_inter_benchmark_run(json_path: Path, total_predictions_attempted: int):
    """Finalise the run by updating metadata with completion time."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        data["run_metadata"]["timestamp_end"] = datetime.datetime.now().isoformat()
        data["run_metadata"]["total_predictions"] = total_predictions_attempted

        start_str = data["run_metadata"].get("timestamp_start")
        end_str = data["run_metadata"].get("timestamp_end")
        if start_str and end_str:
            try:
                start_dt = datetime.datetime.fromisoformat(start_str)
                end_dt = datetime.datetime.fromisoformat(end_str)
                data["run_metadata"]["duration_seconds"] = (end_dt - start_dt).total_seconds()
            except Exception:
                pass

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Finalised run metadata in {json_path}")
    except Exception as e:
        logger.error(f"Failed to finalise run {json_path}: {e}", exc_info=True)


def add_inter_benchmark_run_to_registry(
    registry_file: Path,
    run_id: str,
    source_name: str,
    target_name: str,
    model: str,
    num_experts: int,
    delphi_rounds: int,
    num_predictions_attempted: int,
    num_predictions_completed: int,
    output_path: Path,
    config_file: str,
    timestamp_start: str
):
    """Add a new inter-benchmark run to the registry."""
    registry: Dict[str, Any] = {"runs": [], "last_updated": None}

    try:
        if registry_file.is_file():
            with open(registry_file, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                if isinstance(loaded, dict) and isinstance(loaded.get("runs"), list):
                    registry = loaded
    except Exception as e:
        logger.error(f"Error loading registry {registry_file}: {e}. Re-initialising.")

    run_entry = {
        "run_id": run_id,
        "source_benchmark": source_name,
        "target_benchmark": target_name,
        "timestamp": timestamp_start,
        "model": model,
        "num_experts": num_experts,
        "delphi_rounds": delphi_rounds,
        "num_predictions_attempted": num_predictions_attempted,
        "num_predictions_completed": num_predictions_completed,
        "output_path": str(output_path),
        "config_file": config_file
    }

    if any(r.get("run_id") == run_id for r in registry["runs"]):
        logger.warning(f"Run ID {run_id} already exists in registry. Skipping.")
        return

    registry["runs"].append(run_entry)
    registry["runs"].sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    registry["last_updated"] = datetime.datetime.now().isoformat()

    try:
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        with open(registry_file, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Added run {run_id} to registry at {registry_file}")
    except Exception as e:
        logger.error(f"Failed to save registry {registry_file}: {e}", exc_info=True)
