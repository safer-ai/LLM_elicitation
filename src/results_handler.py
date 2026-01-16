# src/results_handler.py

import json
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd # Added dependency for CSV output

# Import necessary configuration and data models (if needed for typing)
from config import AppConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Run ID Generation ---

def _generate_run_id() -> str:
    """Generates a unique run ID based on the current timestamp."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Run Registry Management ---

def init_registry(registry_file: Path) -> Dict[str, Any]:
    """
    Initializes or loads the run registry file.

    Args:
        registry_file: Path to the run_registry.json file.

    Returns:
        The loaded or initialized registry dictionary.
    """
    registry: Dict[str, Any] = {"runs": [], "last_updated": None}
    try:
        if registry_file.is_file():
            with open(registry_file, 'r', encoding='utf-8') as f:
                registry_data = json.load(f)
                # Basic validation
                if isinstance(registry_data, dict) and isinstance(registry_data.get("runs"), list):
                    registry = registry_data
                    logger.info(f"Loaded existing run registry from {registry_file}")
                else:
                    logger.warning(f"Registry file {registry_file} has invalid format. Re-initializing.")
                    # Optionally back up the invalid file here
        else:
            logger.info(f"Registry file not found. Initializing new registry at {registry_file}")

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from registry file {registry_file}: {e}. Re-initializing.")
    except IOError as e:
        logger.error(f"Error reading registry file {registry_file}: {e}. Re-initializing.")
    except Exception as e:
         logger.error(f"Unexpected error loading registry {registry_file}: {e}. Re-initializing.", exc_info=True)

    # Ensure essential keys exist even if loaded partially or re-initialized
    if "runs" not in registry or not isinstance(registry["runs"], list):
        registry["runs"] = []
    if "last_updated" not in registry: # Or if it's invalid type
         registry["last_updated"] = datetime.datetime.now().isoformat()

    # Ensure parent directory exists before attempting to save later
    registry_file.parent.mkdir(parents=True, exist_ok=True)

    return registry

def save_registry(registry_file: Path, registry: Dict[str, Any]):
    """
    Saves the registry dictionary back to the JSON file.

    Args:
        registry_file: Path to the run_registry.json file.
        registry: The registry dictionary to save.
    """
    try:
        registry["last_updated"] = datetime.datetime.now().isoformat()
        # Ensure parent directory exists (redundant if init_registry called, but safe)
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        with open(registry_file, 'w', encoding='utf-8') as f:
            # Use default=str for better serialization of potential odd types
            json.dump(registry, f, indent=2, ensure_ascii=False, default=str)
        logger.debug(f"Run registry saved successfully to {registry_file}")
    except TypeError as e:
         logger.error(f"TypeError saving registry to {registry_file} (potential non-serializable data): {e}", exc_info=True)
    except IOError as e:
        logger.error(f"Error writing registry file {registry_file}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving registry to {registry_file}: {e}", exc_info=True)


def add_run_to_registry(
    registry_file: Path,
    registry: Dict[str, Any],
    run_id: str,
    run_metadata: Dict[str, Any], # Extracted from workflow results['run_metadata']
    output_path: Path
):
    """
    Adds a new run entry to the registry and saves it.

    Args:
        registry_file: Path to the run_registry.json file.
        registry: The current registry dictionary (will be modified).
        run_id: The unique ID for this run.
        run_metadata: Metadata dictionary from the workflow results.
        output_path: Path object for the run's output directory.
    """
    # Extract key metadata for the registry entry
    config_used = run_metadata.get("config_used", {})
    entry_metadata = {
        "provider": config_used.get("provider", "Unknown"),
        "model": config_used.get("llm_settings", {}).get("model", "Unknown"),
        "temperature": config_used.get("llm_settings", {}).get("temperature"),
        "delphi_rounds": config_used.get("workflow_settings", {}).get("delphi_rounds"),
        "num_experts": config_used.get("num_experts_run"),
        "num_tasks": config_used.get("num_tasks_run"),
        "num_steps": config_used.get("num_steps_run"),
        "benchmark_file": config_used.get("benchmark_file"),
        "scenario_file": config_used.get("scenario_file"),
        "duration_seconds": run_metadata.get("duration_seconds"),
    }


    run_entry = {
        "run_id": run_id,
        "timestamp_start": run_metadata.get("timestamp_start"),
        "timestamp_end": run_metadata.get("timestamp_end"),
        "output_path": str(output_path.relative_to(Path.cwd())), # Store relative path
        "metadata": entry_metadata
    }

    if not isinstance(registry.get("runs"), list):
        logger.error("Registry 'runs' list is invalid or missing. Cannot add run.")
        # Optionally re-initialize or raise error
        return

    # Avoid duplicate run_ids if somehow generated identically (highly unlikely with timestamp)
    if any(r.get("run_id") == run_id for r in registry["runs"]):
        logger.warning(f"Run ID {run_id} already exists in registry. Skipping add.")
        return

    registry["runs"].append(run_entry)
    # Sort runs by timestamp_start for better readability (optional)
    registry["runs"].sort(key=lambda x: x.get("timestamp_start", ""), reverse=True)

    save_registry(registry_file, registry)
    logger.info(f"Added run {run_id} to registry.")


# --- Results Flattening ---

def _flatten_results_for_csv(results: Dict[str, Any], run_id: str) -> List[Dict[str, Any]]:
    """
    Flattens the nested results dictionary into a list of dictionaries,
    suitable for creating a pandas DataFrame and saving as CSV.

    Each row represents one expert's response in one round for one task/step combo.

    Args:
        results: The results dictionary returned by run_delphi_estimation.
        run_id: The unique ID for this run.

    Returns:
        A list of flat dictionaries (rows).
    """
    rows: List[Dict[str, Any]] = []
    run_meta = results.get("run_metadata", {})
    run_config_meta = run_meta.get("config_used", {})
    llm_meta = run_config_meta.get("llm_settings", {})

    steps_data = results.get("results_per_step", [])

    for step_result in steps_data:
        step_name = step_result.get("step_name", "Unknown Step")
        tasks_data = step_result.get("results_per_task", [])

        for task_result in tasks_data:
            task_name = task_result.get("task_name", "Unknown Task")
            task_metrics = task_result.get("task_metrics", {}) # Get metrics dict
            rounds_data = task_result.get("rounds_data", [])

            for round_data in rounds_data:
                round_num = round_data.get("round", -1)
                responses = round_data.get("responses", [])

                for response in responses:
                    expert_name = response.get("expert", "Unknown Expert")
                    most_likely_estimate = response.get("estimate")
                    min_val = response.get("minimum")
                    max_val = response.get("maximum")
                    confidence_val = response.get("confidence")
                    rationale = response.get("rationale", "")
                    error = response.get("error") # Can be None

                    row_data = {
                        "run_id": run_id,
                        "timestamp_start": run_meta.get("timestamp_start"),
                        "model": llm_meta.get("model"),
                        "temperature": llm_meta.get("temperature"),
                        "step_name": step_name,
                        "task_name": task_name,
                        "round": round_num,
                        "expert_name": expert_name,
                        "most_likely_estimate": most_likely_estimate,
                        "minimum_estimate": min_val,
                        "maximum_estimate": max_val,
                        "confidence_in_range": confidence_val,
                        "rationale": rationale,
                        "has_error": error is not None,
                        "error_message": error if error else "",
                        # Include task metrics dynamically
                        **{f"task_metric_{k}": v for k, v in task_metrics.items()}
                    }
                    rows.append(row_data)

    logger.info(f"Flattened results into {len(rows)} rows for CSV export.")
    return rows


# --- NEW: Progressive Writing Functions ---

def initialize_run(config: AppConfig) -> Optional[Dict[str, Any]]:
    """
    Initializes a new run at the beginning of the workflow.
    Creates run directory and initializes CSV file with headers.
    
    Args:
        config: The AppConfig object with output paths.
        
    Returns:
        A dictionary containing run_id, run_dir, csv_path, and csv_file handle,
        or None if initialization fails.
    """
    try:
        run_id = _generate_run_id()
        run_dir = config.runs_dir / run_id
        
        logger.info(f"Initializing run_id: {run_id} in directory: {run_dir}")
        
        # Create run directory
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create run directory {run_dir}: {e}", exc_info=True)
            return None
            
        # Initialize CSV file
        csv_path = run_dir / "detailed_estimates.csv"
        
        # Define CSV headers based on the flattened structure
        # Note: We cannot know all task metrics in advance, so we'll let pandas handle it
        # when appending rows with additional columns
        csv_headers = [
            "run_id", "timestamp_start", "model", "temperature",
            "step_name", "task_name", "round", "expert_name",
            "most_likely_estimate", "minimum_estimate", "maximum_estimate", "confidence_in_range", "rationale", "has_error", "error_message", "task_metric"
        ]
        
        try:
            # Create empty DataFrame with headers and save
            df = pd.DataFrame(columns=csv_headers)
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"Initialized CSV file at {csv_path}")
            logger.debug("Note: Task metric columns will be added dynamically as data is appended")
            
            return {
                "run_id": run_id,
                "run_dir": run_dir,
                "csv_path": csv_path,
                "json_path": run_dir / "full_results.json"
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize CSV file at {csv_path}: {e}", exc_info=True)
            return None
            
    except Exception as e:
        logger.error(f"Unexpected error during run initialization: {e}", exc_info=True)
        return None

def append_round_to_csv(
    csv_path: Path,
    step_name: str,
    task_name: str,
    task_metrics: Dict[str, Any],
    round_num: int,
    responses: List[Dict[str, Any]],
    run_id: str,
    model: str,
    temperature: float,
    timestamp_start: str
) -> bool:
    """
    Appends results from a single round to the CSV file in a robust way that
    handles dynamically added columns.
    """
    try:
        new_rows = []
        for response in responses:
            row_data = {
                "run_id": run_id,
                "timestamp_start": timestamp_start,
                "model": model,
                "temperature": temperature,
                "step_name": step_name,
                "task_name": task_name,
                "round": round_num,
                "expert_name": response.get("expert", "Unknown Expert"),
                "most_likely_estimate": response.get("estimate"),
                "minimum_estimate": response.get("minimum"),
                "maximum_estimate": response.get("maximum"),
                "confidence_in_range": response.get("confidence"),
                "rationale": response.get("rationale", ""),
                "has_error": response.get("error") is not None,
                "error_message": response.get("error", ""),
                # This part dynamically adds the task_metric_* columns
                **{f"task_metric_{k}": v for k, v in task_metrics.items()}
            }
            new_rows.append(row_data)

        if not new_rows:
            return True # Nothing to append

        new_df = pd.DataFrame(new_rows)

        # Robust append: read existing, concat, and overwrite.
        if csv_path.exists() and csv_path.stat().st_size > 0:
            try:
                existing_df = pd.read_csv(csv_path)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            except pd.errors.EmptyDataError:
                # If file exists but is empty, just use the new data
                combined_df = new_df
        else:
            # If file doesn't exist or is empty, the new data is all there is
            combined_df = new_df
        
        # Overwrite the file with the combined data
        combined_df.to_csv(csv_path, index=False, encoding='utf-8')
        
        logger.debug(f"Saved/Appended {len(new_rows)} rows to CSV for {step_name}/{task_name}/Round{round_num}")
        return True

    except Exception as e:
        logger.error(f"Failed to append round data to CSV at {csv_path}: {e}", exc_info=True)
        return False

def save_intermediate_json(json_path: Path, results: Dict[str, Any]) -> bool:
    """
    Saves the current state of results to JSON file.
    Overwrites the existing file to maintain a single JSON with current state.
    
    Args:
        json_path: Path to the JSON file
        results: Current results dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        logger.debug(f"Saved intermediate results to {json_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save intermediate JSON to {json_path}: {e}", exc_info=True)
        return False


def finalize_run(
    config: AppConfig,
    run_id: str,
    run_dir: Path,
    final_results: Dict[str, Any]
) -> bool:
    """
    Finalizes the run by saving final JSON and updating registry.
    
    Args:
        config: The AppConfig object
        run_id: The run identifier
        run_dir: Path to the run directory
        final_results: Final results dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Save final JSON (one last time with complete metadata)
        json_path = run_dir / "full_results.json"
        save_intermediate_json(json_path, final_results)
        
        # Update run registry
        registry_file = config.registry_file
        registry = init_registry(registry_file)
        run_metadata = final_results.get("run_metadata", {})
        add_run_to_registry(registry_file, registry, run_id, run_metadata, run_dir)
        
        logger.info(f"Finalized run {run_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to finalize run {run_id}: {e}", exc_info=True)
        return False


# --- Main Saving Function (Kept for backward compatibility) ---

def save_run_results(
    config: AppConfig,
    results: Dict[str, Any] # The dictionary returned by run_delphi_estimation
) -> Optional[str]:
    """
    Orchestrates saving the results of a completed Delphi run.

    - Generates a run ID.
    - Creates the run output directory.
    - Saves the full results as JSON.
    - Flattens results and saves as CSV.
    - Updates the central run registry.

    Args:
        config: The AppConfig object with output paths.
        results: The results dictionary from the workflow.

    Returns:
        The generated run_id if successful, otherwise None.
    """
    try:
        if "error" in results:
            logger.error(f"Workflow reported an error, results not saved: {results['error']}")
            return None

        run_id = _generate_run_id()
        run_dir = config.runs_dir / run_id
        registry_file = config.registry_file

        logger.info(f"Saving results for run_id: {run_id} to directory: {run_dir}")

        # 1. Create run directory
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create run directory {run_dir}: {e}", exc_info=True)
            return None # Cannot proceed without directory

        # 2. Save Full Results (JSON)
        json_path = run_dir / "full_results.json"
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                # Use default=str to handle potential non-serializable types like Path
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Full results saved to {json_path}")
        except Exception as e:
            logger.error(f"Failed to save full results JSON to {json_path}: {e}", exc_info=True)
            # Continue to try saving CSV and registry, but log the failure

        # 3. Flatten and Save Numeric Data (CSV)
        csv_path = run_dir / "detailed_estimates.csv"
        try:
            flat_data = _flatten_results_for_csv(results, run_id)
            if flat_data:
                df = pd.DataFrame(flat_data)
                df.to_csv(csv_path, index=False, encoding='utf-8')
                logger.info(f"Detailed estimates saved to {csv_path} ({len(df)} rows)")
            else:
                logger.warning("No data rows generated for CSV export.")
        except ImportError:
             logger.warning("Pandas library not found ('pip install pandas'). Cannot save CSV results.")
        except Exception as e:
            logger.error(f"Failed to save detailed estimates CSV to {csv_path}: {e}", exc_info=True)

        # 4. Update Run Registry
        try:
            registry = init_registry(registry_file) # Load current registry
            run_metadata = results.get("run_metadata", {})
            add_run_to_registry(registry_file, registry, run_id, run_metadata, run_dir)
            # Registry saving happens within add_run_to_registry
        except Exception as e:
            logger.error(f"Failed to update run registry {registry_file} for run {run_id}: {e}", exc_info=True)
            # Run results are saved, but registry might be inconsistent

        return run_id

    except Exception as e:
        # Catch any unexpected errors during the saving orchestration
        logger.error(f"Unexpected error occurred during save_run_results: {e}", exc_info=True)
        return None


# --- Test Execution Block ---
if __name__ == "__main__":
    print("--- Running Results Handler Tests ---")
    # Create a dummy results structure similar to workflow output
    from config import load_config
    dummy_run_id_for_test = _generate_run_id()
    dummy_results = {
        "run_metadata": {
            "timestamp_start": datetime.datetime.now().isoformat(),
            "config_used": {
                "llm_settings": {"model": "test-model-v1", "temperature": 0.5},
                "workflow_settings": {"delphi_rounds": 2, "num_experts": 2, "num_tasks": 1, "scenario_steps": ["Step One"]},
                "provider": "test-provider",
                "benchmark_file": "input_data/benchmark/dummy_bench.yaml",
                "scenario_file": "input_data/scenario/dummy_scene.yaml",
                "num_experts_run": 2,
                "num_tasks_run": 1,
                "num_steps_run": 1,
            },
            "timestamp_end": datetime.datetime.now().isoformat(),
            "duration_seconds": 12.34,
        },
        "results_per_step": [
            {
                "step_name": "Step One",
                "step_description": "The first step.",
                "results_per_task": [
                    {
                        "task_name": "Task Alpha",
                        "task_description": "An easy task.",
                        "task_metrics": {"fst": 10, "category": "web"},
                        "rounds_data": [
                            { # Round 1
                                "round": 1,
                                "responses": [
                                    {"expert": "Expert A", "estimate": 0.3, "rationale": "Initial guess.", "raw_analysis": "Analysis A1", "raw_estimation": "Est A1", "parsed_analysis": {}, "parsed_estimation": {"probability": 0.3, "rationale": "Initial guess."}},
                                    {"expert": "Expert B", "estimate": 0.4, "rationale": "Slightly higher.", "raw_analysis": "Analysis B1", "raw_estimation": "Est B1", "parsed_analysis": {}, "parsed_estimation": {"probability": 0.4, "rationale": "Slightly higher."}},
                                ]
                            },
                            { # Round 2
                                "round": 2,
                                "responses": [
                                    {"expert": "Expert A", "estimate": 0.32, "rationale": "Revised slightly.", "raw_estimation": "Est A2", "parsed_estimation": {"probability": 0.32, "rationale": "Revised slightly."}},
                                    {"expert": "Expert B", "estimate": None, "rationale": "Parsing failed.", "raw_estimation": "Est B2", "parsed_estimation": {"probability": None, "rationale": "Parsing failed."}, "error": "Probability Parsing Failed"},
                                ]
                            }
                        ],
                        "final_aggregated_probability": 0.32, # Aggregated from valid R2 estimate
                        "converged_at_round": None,
                    }
                ]
            }
        ]
    }

    try:
        # Assuming running from project root
        config = load_config("config.yaml")

        print(f"\nAttempting to save dummy results (Run ID will be generated like: {dummy_run_id_for_test})")
        # Ensure output directory exists for the test
        config.output_dir.mkdir(parents=True, exist_ok=True)

        saved_run_id = save_run_results(config, dummy_results)

        if saved_run_id:
            print(f"Successfully saved results for run_id: {saved_run_id}")
            run_output_dir = config.runs_dir / saved_run_id
            print(f"  Output directory: {run_output_dir}")
            json_file = run_output_dir / "full_results.json"
            csv_file = run_output_dir / "detailed_estimates.csv"
            registry_file = config.registry_file

            print(f"  Checking existence:")
            print(f"    Directory: {run_output_dir.is_dir()}")
            print(f"    JSON file: {json_file.is_file()}")
            print(f"    CSV file: {csv_file.is_file()}")
            print(f"    Registry file: {registry_file.is_file()}")

            # Basic check if run ID is in registry
            if registry_file.is_file():
                try:
                    with open(registry_file, 'r') as f: reg_data = json.load(f)
                    found_in_registry = any(r.get("run_id") == saved_run_id for r in reg_data.get("runs",[]))
                    print(f"    Run ID in registry: {found_in_registry}")
                except Exception as e:
                    print(f"    Error reading registry for check: {e}")

        else:
            print("Failed to save results.")

    except FileNotFoundError as e:
        print(f"\nERROR: Prerequisite file/directory not found for test: {e}")
    except ImportError as e:
         print(f"\nERROR: Missing dependency (likely pandas for CSV): {e}. Install with 'pip install pandas'")
    except Exception as e:
        print(f"\nUNEXPECTED ERROR during results handler test: {e}")
        logger.exception("Unexpected error in results_handler test:")

    print("\n--- Results Handler Tests Complete ---")