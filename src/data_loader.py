# src/data_loader.py

import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


# Import required configurations and data models
from config import AppConfig, load_config # Need load_config for testing
from data_models import ExpertProfile, BenchmarkTask, Benchmark, ThreatActor, Target, ScenarioStep, Scenario, InputData, ScenarioLevelMetric

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Data Loading Functions ---

def load_prompts(prompts_dir: Path) -> Dict[str, str]:
    """
    Loads all .txt prompt files from the specified directory.

    Args:
        prompts_dir: The Path object pointing to the directory containing prompt files.

    Returns:
        A dictionary where keys are prompt filenames (without extension)
        and values are the content of the files. Returns an empty dict if
        the directory doesn't exist or no .txt files are found.
    """
    prompts: Dict[str, str] = {}
    if not prompts_dir.is_dir():
        logger.error(f"Prompts directory not found or is not a directory: {prompts_dir}")
        return prompts

    try:
        for prompt_file in prompts_dir.glob('*.txt'):
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompts[prompt_file.stem] = f.read()
            except IOError as e:
                logger.warning(f"Could not read prompt file {prompt_file.name}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error reading prompt file {prompt_file.name}: {e}", exc_info=True)

        if not prompts:
            logger.warning(f"No .txt prompt files found in directory: {prompts_dir}")
        else:
            logger.info(f"Loaded {len(prompts)} prompts from {prompts_dir}")


    except Exception as e:
        logger.error(f"Error scanning prompts directory {prompts_dir}: {e}", exc_info=True)

    return prompts


def load_experts(expert_file: Path) -> List[ExpertProfile]:
    """
    Loads expert persona definitions from the specified YAML file.

    Args:
        expert_file: The Path object pointing to the expert profiles YAML file.

    Returns:
        A list of ExpertProfile objects. Returns an empty list if the file
        cannot be read, parsed, or contains invalid data.
    """
    expert_profiles: List[ExpertProfile] = []
    if not expert_file.is_file():
        logger.error(f"Expert profiles file not found: {expert_file}")
        return expert_profiles

    try:
        with open(expert_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict) or 'experts' not in data:
            logger.error(f"Invalid format in expert file {expert_file.name}. Expected a dictionary with an 'experts' key.")
            return expert_profiles

        raw_experts = data['experts']
        if not isinstance(raw_experts, list):
            logger.error(f"Invalid format in expert file {expert_file.name}. 'experts' key should contain a list.")
            return expert_profiles

        for i, expert_data in enumerate(raw_experts):
            if not isinstance(expert_data, dict):
                logger.warning(f"Skipping item {i} in expert file: expected a dictionary, got {type(expert_data)}")
                continue
            try:
                profile = ExpertProfile(
                    name=str(expert_data.get('name', f'Unnamed Expert {i+1}')),
                    background=str(expert_data.get('background', 'N/A')),
                    focus=str(expert_data.get('focus', 'N/A')),
                    key_trait=str(expert_data.get('key_trait', 'N/A')),
                    bias=str(expert_data.get('bias', 'N/A')),
                    analytical_approach=expert_data.get('analytical_approach') # Optional
                )
                expert_profiles.append(profile)
            except (TypeError, KeyError, ValueError) as e:
                logger.warning(f"Skipping expert entry {i} due to invalid data: {expert_data}. Error: {e}")

        if not expert_profiles:
            logger.warning(f"No valid expert profiles loaded from {expert_file.name}")
        else:
            logger.info(f"Loaded {len(expert_profiles)} expert profiles from {expert_file.name}")

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML in expert file {expert_file.name}: {e}")
    except IOError as e:
         logger.error(f"Error reading expert file {expert_file.name}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading experts from {expert_file.name}: {e}", exc_info=True)

    return expert_profiles


def load_benchmark(benchmark_file: Path) -> Optional[Benchmark]:
    """
    Loads a benchmark definition from the specified YAML file.

    Args:
        benchmark_file: The Path object pointing to the benchmark definition file.

    Returns:
        A Benchmark object, or None if the file cannot be loaded or parsed correctly.
    """
    if not benchmark_file.is_file():
        logger.error(f"Benchmark file not found: {benchmark_file}")
        return None

    try:
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            logger.error(f"Invalid format in benchmark file {benchmark_file.name}. Expected a dictionary.")
            return None

        # Extract top-level info
        description = str(data.get('benchmark_description', 'No description provided.'))
        metrics_to_use = data.get('metrics_to_use_for_estimation', [])
        if not isinstance(metrics_to_use, list) or not all(isinstance(m, str) for m in metrics_to_use):
             logger.warning(f"Invalid 'metrics_to_use_for_estimation' in {benchmark_file.name}. Defaulting to empty list.")
             metrics_to_use = []

        # Extract and parse tasks
        raw_tasks = data.get('tasks', [])
        if not isinstance(raw_tasks, list):
            logger.error(f"Invalid format in benchmark file {benchmark_file.name}. 'tasks' key should contain a list.")
            return None

        loaded_tasks: List[BenchmarkTask] = []
        for i, task_data in enumerate(raw_tasks):
            if not isinstance(task_data, dict):
                logger.warning(f"Skipping task item {i} in benchmark file: expected a dictionary, got {type(task_data)}")
                continue
            try:
                task = BenchmarkTask(
                    name=str(task_data.get('name', f'Unnamed Task {i+1}')),
                    description=str(task_data.get('description', 'No description.')),
                    metrics=dict(task_data.get('metrics', {})) # Ensure metrics is a dict
                )
                loaded_tasks.append(task)
            except (TypeError, KeyError, ValueError) as e:
                logger.warning(f"Skipping benchmark task entry {i} due to invalid data: {task_data}. Error: {e}")

        if not loaded_tasks:
             logger.warning(f"No valid tasks loaded from benchmark file {benchmark_file.name}")

        benchmark = Benchmark(
            description=description,
            metrics_to_use=metrics_to_use,
            tasks=loaded_tasks
        )
        logger.info(f"Loaded benchmark '{description[:50]}...' ({len(loaded_tasks)} tasks) from {benchmark_file.name}")
        return benchmark

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML in benchmark file {benchmark_file.name}: {e}")
    except IOError as e:
         logger.error(f"Error reading benchmark file {benchmark_file.name}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading benchmark from {benchmark_file.name}: {e}", exc_info=True)

    return None


def load_scenario(scenario_file: Path) -> Optional[Scenario]:
    """
    Loads a risk scenario definition from the specified YAML file.
    Now supports the new scenario_level_metrics structure with assumptions.

    Args:
        scenario_file: The Path object pointing to the scenario definition file.

    Returns:
        A Scenario object, or None if the file cannot be loaded or parsed correctly.
    """
    if not scenario_file.is_file():
        logger.error(f"Scenario file not found: {scenario_file}")
        return None

    try:
        with open(scenario_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            logger.error(f"Invalid format in scenario file {scenario_file.name}. Expected a dictionary.")
            return None

        # Load Threat Actor
        actor_data = data.get('threat_actor')
        if not isinstance(actor_data, dict):
            logger.error(f"Missing or invalid 'threat_actor' section in {scenario_file.name}")
            return None
        try:
            threat_actor = ThreatActor(
                name=str(actor_data.get('name', 'Unnamed Actor')),
                description=str(actor_data.get('description', 'No description.'))
            )
        except (TypeError, KeyError) as e:
             logger.error(f"Invalid threat_actor data in {scenario_file.name}: {e}")
             return None

        # Load Target
        target_data = data.get('target')
        if not isinstance(target_data, dict):
            logger.error(f"Missing or invalid 'target' section in {scenario_file.name}")
            return None
        try:
            target = Target(
                name=str(target_data.get('name', 'Unnamed Target')),
                description=str(target_data.get('description', 'No description.'))
            )
        except (TypeError, KeyError) as e:
             logger.error(f"Invalid target data in {scenario_file.name}: {e}")
             return None

        # Load Steps
        raw_steps = data.get('steps', [])
        if not isinstance(raw_steps, list):
            logger.error(f"Invalid format in scenario file {scenario_file.name}. 'steps' key should contain a list.")
            return None

        loaded_steps: List[ScenarioStep] = []
        for i, step_data in enumerate(raw_steps):
            if not isinstance(step_data, dict):
                logger.warning(f"Skipping step item {i} in scenario file: expected a dictionary, got {type(step_data)}")
                continue
            try:
                step = ScenarioStep(
                    name=str(step_data.get('name', f'Unnamed Step {i+1}')),
                    description=str(step_data.get('description', 'No description.')),
                    assumptions=str(step_data.get('assumptions', '')), # Default to empty string
                    # Normalize benchmark_file path to use forward slashes for consistency
                    benchmark_file=Path(bf).as_posix() if (bf := step_data.get('benchmark_file')) else None
                )
                loaded_steps.append(step)
            except (TypeError, KeyError, ValueError) as e:
                logger.warning(f"Skipping scenario step entry {i} due to invalid data: {step_data}. Error: {e}")

        if not loaded_steps:
             logger.warning(f"No valid steps loaded from scenario file {scenario_file.name}")

        # Load Scenario Info
        scenario_name = str(data.get('scenario_name', 'Unnamed Scenario'))
        scenario_description = str(data.get('scenario_description', 'No description.'))

        # NEW: Load scenario_level_metrics with the new structure
        scenario_level_metrics_data = data.get('scenario_level_metrics', {})
        
        # Also check for old structure for backward compatibility
        if not scenario_level_metrics_data and 'scenario_level_metrics_benchmarks' in data:
            logger.info(f"Found legacy 'scenario_level_metrics_benchmarks' in {scenario_file.name}. Converting to new format.")
            old_benchmarks = data.get('scenario_level_metrics_benchmarks', {})
            scenario_level_metrics_data = {}
            for key, benchmark_path in old_benchmarks.items():
                if isinstance(benchmark_path, str):
                    scenario_level_metrics_data[key] = {
                        'benchmark_file': benchmark_path,
                        'assumptions': ''  # No assumptions in old format
                    }
        
        if not isinstance(scenario_level_metrics_data, dict):
            logger.warning(f"Invalid 'scenario_level_metrics' in {scenario_file.name}. Expected a dictionary. Defaulting to empty.")
            scenario_level_metrics_data = {}
        
        # Parse scenario level metrics
        valid_scenario_level_metrics: Dict[str, ScenarioLevelMetric] = {}
        for metric_key, metric_data in scenario_level_metrics_data.items():
            if isinstance(metric_data, dict):
                benchmark_file = metric_data.get('benchmark_file')
                if not benchmark_file:
                    logger.warning(f"Missing 'benchmark_file' for metric '{metric_key}'. Skipping.")
                    continue
                
                # Normalize path to use forward slashes
                normalized_benchmark_path = Path(benchmark_file).as_posix()
                
                # Get assumptions, default to empty string
                assumptions = str(metric_data.get('assumptions', ''))
                
                valid_scenario_level_metrics[str(metric_key)] = ScenarioLevelMetric(
                    benchmark_file=normalized_benchmark_path,
                    assumptions=assumptions
                )
            elif isinstance(metric_data, str):
                # Handle simple string format (just benchmark path) for backward compatibility
                logger.info(f"Metric '{metric_key}' uses simple string format. Converting to new structure with empty assumptions.")
                valid_scenario_level_metrics[str(metric_key)] = ScenarioLevelMetric(
                    benchmark_file=Path(metric_data).as_posix(),
                    assumptions=''
                )
            else:
                logger.warning(f"Invalid data for metric '{metric_key}'. Expected dict or string, got {type(metric_data)}. Skipping.")

        scenario = Scenario(
            name=scenario_name,
            description=scenario_description,
            threat_actor=threat_actor,
            target=target,
            steps=loaded_steps,
            scenario_level_metrics=valid_scenario_level_metrics
        )
        logger.info(f"Loaded scenario '{scenario.name}' ({len(loaded_steps)} steps) from {scenario_file.name}")
        if scenario.scenario_level_metrics:
            logger.info(f"  Scenario-level metrics configured: {list(scenario.scenario_level_metrics.keys())}")
        else:
            logger.info(f"  No scenario-level metrics configured.")

        return scenario

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML in scenario file {scenario_file.name}: {e}")
    except IOError as e:
         logger.error(f"Error reading scenario file {scenario_file.name}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading scenario from {scenario_file.name}: {e}", exc_info=True)

    return None


def load_all_inputs(config: AppConfig) -> Optional[InputData]:
    """
    Loads all necessary input data based on the provided AppConfig.
    This includes loading the scenario, prompts, experts, and all referenced
    benchmark files (default, step-specific, and scenario-level metric-specific).
    Ensures path strings used as keys for loaded_benchmarks are POSIX-style.

    Args:
        config: The AppConfig object containing paths and settings.

    Returns:
        An InputData object containing all loaded data, or None if
        a critical component (like scenario or a required benchmark) failed to load.
    """
    logger.info("Loading all input data...")
    prompts = load_prompts(config.prompts_dir)
    experts = load_experts(config.expert_profiles_file)
    scenario = load_scenario(config.scenario_file) # Scenario loader now also normalizes its paths

    if scenario is None:
        logger.critical(f"Failed to load critical scenario data from {config.scenario_file}. Aborting input load.")
        return None
    if not prompts:
        logger.warning("Prompt dictionary is empty. Workflow might fail.")
    if not experts:
         logger.warning("Expert list is empty. Workflow might fail.")

    loaded_benchmarks: Dict[str, Benchmark] = {}
    project_root = Path.cwd() # For resolving relative benchmark paths

    # Helper to load a benchmark if not already loaded
    def _load_benchmark_if_needed(benchmark_path_str_from_yaml: str, purpose_description: str):
        nonlocal loaded_benchmarks 
        
        # Normalize the path string from YAML to POSIX style for use as a key
        # This assumes benchmark_path_str_from_yaml is relative to project root,
        # or becomes so after Path() if it was, e.g. "input_data/file.yaml"
        normalized_key = Path(benchmark_path_str_from_yaml).as_posix()
        
        if normalized_key not in loaded_benchmarks:
            logger.info(f"Loading benchmark ({purpose_description}): '{normalized_key}' (original YAML str: '{benchmark_path_str_from_yaml}')")
            
            # Resolve the original path string fully from project root to load the file
            full_benchmark_path = (project_root / benchmark_path_str_from_yaml).resolve()
            
            benchmark_obj = load_benchmark(full_benchmark_path)
            if benchmark_obj is None:
                logger.error(f"Failed to load benchmark '{full_benchmark_path}' (key: '{normalized_key}'). This may cause errors or skips.")
            else:
                loaded_benchmarks[normalized_key] = benchmark_obj
                logger.info(f"Successfully loaded benchmark using key: '{normalized_key}'")
        else:
            logger.debug(f"Benchmark for key '{normalized_key}' ({purpose_description}) already loaded.")


    # 1. Load default benchmark first
    # config.default_benchmark_file is already an absolute Path object
    # We need its string representation relative to project_root, POSIX-style for the key
    default_benchmark_path_str_rel = config.default_benchmark_file.relative_to(project_root).as_posix()
    _load_benchmark_if_needed(default_benchmark_path_str_rel, "default for steps")
    if default_benchmark_path_str_rel not in loaded_benchmarks:
        logger.critical(f"Failed to load critical default benchmark data from {config.default_benchmark_file} (key: {default_benchmark_path_str_rel}). Aborting input load.")
        return None 

    # 2. Load step-specific benchmarks
    if scenario and scenario.steps:
        for step in scenario.steps:
            if step.benchmark_file: # This is already POSIX-normalized by load_scenario
                # step.benchmark_file is assumed to be a relative POSIX path string
                _load_benchmark_if_needed(step.benchmark_file, f"for step '{step.name}'")

    # 3. Load benchmarks specified for scenario-level metrics (NEW STRUCTURE)
    if scenario and scenario.scenario_level_metrics:
        for metric_key, metric_config in scenario.scenario_level_metrics.items():
            # metric_config.benchmark_file is already POSIX-normalized by load_scenario
            if metric_config.benchmark_file: 
                _load_benchmark_if_needed(metric_config.benchmark_file, f"for scenario-level metric '{metric_key}'")

    logger.debug(f"Final keys in loaded_benchmarks: {list(loaded_benchmarks.keys())}") # DEBUG LINE
    
    logger.info(f"All input data loaded. Total unique benchmarks loaded: {len(loaded_benchmarks)}.")
    return InputData(
        prompts=prompts,
        experts=experts,
        scenario=scenario,
        loaded_benchmarks=loaded_benchmarks
    )


# --- Test Execution Block ---
if __name__ == "__main__":
    print("--- Running Data Loader Tests ---")
    try:
        # Assuming this script is run from the project root directory
        # where config.yaml resides.
        
        app_config = load_config("config.yaml")
        print("Configuration loaded successfully for testing.")

        # Test loading each component individually
        print("\nTesting load_prompts...")
        prompts = load_prompts(app_config.prompts_dir)
        if prompts:
            print(f"  Loaded {len(prompts)} prompts. First prompt key: '{next(iter(prompts.keys()))}'")
        else:
            print("  Failed to load prompts or directory empty.")

        print("\nTesting load_experts...")
        experts = load_experts(app_config.expert_profiles_file)
        if experts:
            print(f"  Loaded {len(experts)} experts. First expert name: '{experts[0].name}'")
        else:
            print("  Failed to load experts or file empty/invalid.")

        print("\nTesting load_benchmark (default)...")
        # Test loading the default benchmark directly
        default_benchmark = load_benchmark(app_config.default_benchmark_file)
        if default_benchmark:
            print(f"  Loaded default benchmark: '{default_benchmark.description[:60]}...'")
            print(f"  Number of tasks: {len(default_benchmark.tasks)}")
            if default_benchmark.tasks:
                 print(f"  First task name: '{default_benchmark.tasks[0].name}'")
        else:
            print(f"  Failed to load default benchmark from {app_config.default_benchmark_file}.")


        print("\nTesting load_scenario...")
        scenario = load_scenario(app_config.scenario_file)
        if scenario:
            print(f"  Loaded scenario: '{scenario.name}'")
            print(f"  Threat Actor: {scenario.threat_actor.name}")
            print(f"  Target: {scenario.target.name}")
            print(f"  Number of steps: {len(scenario.steps)}")
            if scenario.steps:
                 print(f"  First step name: '{scenario.steps[0].name}', benchmark_file: {scenario.steps[0].benchmark_file}")
                 if len(scenario.steps) > 1:
                     print(f"  Second step name: '{scenario.steps[1].name}', benchmark_file: {scenario.steps[1].benchmark_file}")
            print(f"  Scenario-level metrics:")
            for metric_key, metric_config in scenario.scenario_level_metrics.items():
                print(f"    {metric_key}: benchmark='{metric_config.benchmark_file}', assumptions='{metric_config.assumptions[:50]}...'")
        else:
            print("  Failed to load scenario.")

        # Test loading all together
        print("\nTesting load_all_inputs...")
        all_data = load_all_inputs(app_config)
        if all_data:
            print("  Successfully loaded all input data using load_all_inputs.")
            print(f"  Confirm scenario loaded: {all_data.scenario is not None}")
            if all_data.scenario:
                print(f"  Scenario-level metrics in loaded scenario: {len(all_data.scenario.scenario_level_metrics)}")
            print(f"  Number of unique benchmarks loaded into InputData: {len(all_data.loaded_benchmarks)}")
            for path_key, bench_obj in all_data.loaded_benchmarks.items():
                print(f"    Benchmark Path Key (Normalized): {path_key}, Description: {bench_obj.description[:30]}..., Tasks: {len(bench_obj.tasks)}")
        else:
            print("  Failed to load all input data (check logs for critical failures).")


    except FileNotFoundError as e:
        print(f"\nERROR: File not found during testing: {e}")
        print("Ensure 'config.yaml' and all referenced input files (prompts, experts, benchmarks, scenario) exist.")
    except (yaml.YAMLError, ValueError, TypeError) as e:
        print(f"\nERROR: Validation or parsing error during testing: {e}")
    except Exception as e:
        print(f"\nUNEXPECTED ERROR during testing: {e}")
        logger.exception("Unexpected error in test block:")

    print("\n--- Data Loader Tests Complete ---")