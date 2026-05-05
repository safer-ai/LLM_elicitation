# src/config.py

import sys
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, List, Set, Union

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from shared.api_keys import parse_num_repeats as _parse_num_repeats  # noqa: E402,F401
from shared.api_keys import resolve_api_key as _resolve_api_key  # noqa: E402
from shared.llm_client import (  # noqa: E402
    REASONING_EFFORT_VALUES,
    parse_reasoning_effort,
    provider_for_model as get_provider_for_model,
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Data Structure Definitions ---


@dataclass
class LLMSettings:
    """Settings related to the LLM and API interaction."""
    model: str                    # Model identifier (e.g., "claude-3-5-sonnet-20240620")
    temperature: float = 0.8      # LLM sampling temperature
    max_concurrent_calls: int = 5 # Max parallel API requests
    rate_limit_calls: int = 45    # Max calls per rate_limit_period
    rate_limit_period: int = 60   # Time window for rate limit (seconds)
    # Provider-agnostic reasoning/thinking knob. One of REASONING_EFFORT_VALUES.
    # "off" means: don't request extended reasoning. For OpenAI reasoning models
    # (gpt-5*, o-series) this is translated to reasoning_effort="minimal" because
    # those models always reason internally and "minimal" is the smallest budget.
    reasoning_effort: str = "off"

    def __post_init__(self):
        if not self.model:
            raise ValueError("LLMSettings: 'model' cannot be empty.")
        if not 0.0 <= self.temperature <= 2.0:
            logger.warning(f"LLMSettings: 'temperature' ({self.temperature}) is outside the typical range [0.0, 2.0].")
        if self.max_concurrent_calls <= 0:
            raise ValueError("LLMSettings: 'max_concurrent_calls' must be positive.")
        if self.rate_limit_calls <= 0:
            raise ValueError("LLMSettings: 'rate_limit_calls' must be positive.")
        if self.rate_limit_period <= 0:
            raise ValueError("LLMSettings: 'rate_limit_period' must be positive.")
        if self.reasoning_effort not in REASONING_EFFORT_VALUES:
            raise ValueError(
                f"LLMSettings: 'reasoning_effort' must be one of "
                f"{list(REASONING_EFFORT_VALUES)}, got {self.reasoning_effort!r}."
            )

@dataclass
class WorkflowSettings:
    """Settings controlling the Delphi workflow execution."""
    # num_tasks accepts three modes:
    #   - None: process all tasks for every benchmark.
    #   - int: process the first N tasks of every benchmark.
    #   - List[str]: an explicit list of task names (e.g. ["Paddle", "Labyrinth Linguist"]).
    #     For each benchmark, if any of the listed names match its tasks, only those
    #     tasks are run. Benchmarks whose names are not present in the list fall back
    #     to running their full task set
    num_tasks: Union[None, int, List[str]] = None
    num_experts: Optional[int] = None
    scenario_steps: Optional[List[str]] = None
    delphi_rounds: int = 3
    convergence_threshold: float = 0.05
    # Number of independent repeats of the full Delphi pipeline per model.
    # rows are distinguished by # the `repeat_index` column (1-based).
    num_repeats: int = 1

    include_easier_tasks: bool = True
    num_example_tasks: Optional[int] = 3

    estimate_num_actors_per_task_benchmark: bool = False
    estimate_num_attacks_per_task_benchmark: bool = False
    estimate_damage_per_task_benchmark: bool = False


    def __post_init__(self):
        if isinstance(self.num_tasks, bool):
            raise TypeError("WorkflowSettings: 'num_tasks' must be null, an int, or a list of strings (got bool).")
        if isinstance(self.num_tasks, int):
            if self.num_tasks < 0:
                raise ValueError("WorkflowSettings: 'num_tasks' cannot be negative.")
        elif isinstance(self.num_tasks, list):
            if not all(isinstance(s, str) for s in self.num_tasks):
                raise TypeError("WorkflowSettings: when 'num_tasks' is a list, all entries must be strings.")
        elif self.num_tasks is not None:
            raise TypeError("WorkflowSettings: 'num_tasks' must be null, an int, or a list of strings.")
        if self.num_experts is not None and self.num_experts <= 0:
            raise ValueError("WorkflowSettings: 'num_experts' must be positive if specified.")
        if self.delphi_rounds <= 0:
            raise ValueError("WorkflowSettings: 'delphi_rounds' must be positive.")
        if self.convergence_threshold < 0:
            raise ValueError("WorkflowSettings: 'convergence_threshold' cannot be negative.")
        if self.scenario_steps is not None and not isinstance(self.scenario_steps, list):
            raise TypeError("WorkflowSettings: 'scenario_steps' must be a list of strings or null.")
        if isinstance(self.num_repeats, bool) or not isinstance(self.num_repeats, int):
            raise TypeError("WorkflowSettings: 'num_repeats' must be a positive integer.")
        if self.num_repeats < 1:
            raise ValueError("WorkflowSettings: 'num_repeats' must be >= 1 (1 = run once).")


@dataclass
class AppConfig:
    """Root configuration object holding all settings and paths."""
    # --- Fields WITHOUT default values ---
    # Input Paths (resolved to absolute paths during loading)
    prompts_dir: Path
    default_benchmark_file: Path # Renamed from benchmark_file
    scenario_file: Path
    expert_profiles_file: Path

    # Nested Settings
    llm_settings: LLMSettings
    workflow_settings: WorkflowSettings

    # --- Fields WITH default values ---
    output_dir: Path = Path("output_data")

    # API keys for each supported provider. Resolved from .env > process env.
    api_key_anthropic: Optional[str] = None
    api_key_openai: Optional[str] = None
    api_key_gemini: Optional[str] = None

    models_to_run: List[str] = field(default_factory=list)

    @property
    def inferred_api_provider(self) -> str:
        """Provider for the currently active single model in `llm_settings.model`."""
        return get_provider_for_model(self.llm_settings.model)

    @property
    def required_providers(self) -> Set[str]:
        """Set of providers needed across `models_to_run`."""
        return {get_provider_for_model(m) for m in self.models_to_run}

    @property
    def runs_dir(self) -> Path:
        return self.output_dir / "runs"

    @property
    def registry_file(self) -> Path:
        return self.output_dir / "run_registry.json"


# --- Configuration Loading Function ---

def load_config(config_path: str = "config.yaml") -> AppConfig:
    """
    Loads configuration from the specified YAML file, performs validation,
    resolves paths, and returns an AppConfig object.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        An AppConfig instance with loaded and validated settings.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.
        IOError: If there's an error reading the file.
        ValueError: If the configuration contains invalid values or structure.
        TypeError: If configuration values have incorrect types.
    """
    config_file = Path(config_path).resolve() # Resolve path immediately
    if not config_file.is_file():
        logger.error(f"Configuration file not found: {config_file}")
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    logger.info(f"Loading configuration from: {config_file}")

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_file}: {e}")
        raise yaml.YAMLError(f"Error parsing YAML file {config_file}: {e}") from e
    except Exception as e:
        logger.error(f"Error reading configuration file {config_file}: {e}")
        raise IOError(f"Error reading configuration file {config_file}: {e}") from e

    if not isinstance(raw_config, dict):
        raise ValueError(f"Configuration file content must be a dictionary. Found type: {type(raw_config)}")

    try:
        # --- Load Nested Settings ---
        llm_settings_raw = raw_config.get("llm_settings", {})
        if not isinstance(llm_settings_raw, dict):
            raise ValueError("'llm_settings' must be a dictionary.")

        workflow_settings_raw = raw_config.get("workflow_settings", {})
        if not isinstance(workflow_settings_raw, dict):
            raise ValueError("'workflow_settings' must be a dictionary.")
        
        reasoning_effort = parse_reasoning_effort(llm_settings_raw, logger_=logger)

        # `model` may be a single string or a list of strings. Normalise to a list
        # internally; LLMSettings.model holds the active model for the current run
        # (main.py rotates it through `models_to_run`).
        raw_model = llm_settings_raw.get("model")
        if raw_model is None:
            raise ValueError("'llm_settings.model' is required (string or list of strings).")
        if isinstance(raw_model, str):
            models_to_run: List[str] = [raw_model.strip()]
        elif isinstance(raw_model, list):
            if not raw_model:
                raise ValueError("'llm_settings.model' is an empty list; provide at least one model.")
            if not all(isinstance(m, str) and m.strip() for m in raw_model):
                raise TypeError("'llm_settings.model' list must contain non-empty strings only.")
            models_to_run = [m.strip() for m in raw_model]
        else:
            raise TypeError(f"'llm_settings.model' must be a string or a list of strings (got {type(raw_model).__name__}).")

        # Instantiate nested settings (validation happens in __post_init__)
        # The active `model` defaults to the first entry; main.py overrides per run.
        llm_settings = LLMSettings(
            model=models_to_run[0],
            temperature=float(llm_settings_raw.get("temperature", LLMSettings.temperature)),
            max_concurrent_calls=int(llm_settings_raw.get("max_concurrent_calls", LLMSettings.max_concurrent_calls)),
            rate_limit_calls=int(llm_settings_raw.get("rate_limit_calls", LLMSettings.rate_limit_calls)),
            rate_limit_period=int(llm_settings_raw.get("rate_limit_period", LLMSettings.rate_limit_period)),
            reasoning_effort=reasoning_effort,
        )

        workflow_settings = WorkflowSettings(
            # num_tasks: pass through None / list as-is; coerce numeric scalars to int.
            num_tasks=(
                None if (n := workflow_settings_raw.get("num_tasks")) is None
                else (list(n) if isinstance(n, list)
                      else int(n))
            ),
            num_experts=int(n) if (n := workflow_settings_raw.get("num_experts")) is not None else None,
            scenario_steps=workflow_settings_raw.get("scenario_steps"), # Keep as None or list
            delphi_rounds=int(workflow_settings_raw.get("delphi_rounds", WorkflowSettings.delphi_rounds)),
            convergence_threshold=float(workflow_settings_raw.get("convergence_threshold", WorkflowSettings.convergence_threshold)),
            # Load example task settings
            include_easier_tasks=bool(workflow_settings_raw.get("include_easier_tasks", WorkflowSettings.include_easier_tasks)),
            num_example_tasks=int(n) if (n := workflow_settings_raw.get("num_example_tasks")) is not None else WorkflowSettings.num_example_tasks,
            # Load new boolean flags for scenario-level metrics
            estimate_num_actors_per_task_benchmark=bool(workflow_settings_raw.get("estimate_num_actors_per_task_benchmark", WorkflowSettings.estimate_num_actors_per_task_benchmark)),
            estimate_num_attacks_per_task_benchmark=bool(workflow_settings_raw.get("estimate_num_attacks_per_task_benchmark", WorkflowSettings.estimate_num_attacks_per_task_benchmark)),
            estimate_damage_per_task_benchmark=bool(workflow_settings_raw.get("estimate_damage_per_task_benchmark", WorkflowSettings.estimate_damage_per_task_benchmark)),
            num_repeats=_parse_num_repeats(workflow_settings_raw.get("num_repeats", WorkflowSettings.num_repeats)),
        )


        # --- Resolve and Validate Paths ---
        project_root = Path.cwd() # Assume script is run from project root

        def resolve_validate_path(key: str, default: str, is_dir: bool = False, is_file: bool = False) -> Path:
            path_str = raw_config.get(key, default)
            if not path_str: raise ValueError(f"Configuration key '{key}' cannot be empty.")
            resolved_path = (project_root / Path(str(path_str))).resolve() # Resolve relative to root

            if is_dir and not resolved_path.is_dir():
                raise ValueError(f"Path for '{key}' not found or not a directory: {resolved_path}")
            if is_file and not resolved_path.is_file():
                raise ValueError(f"Path for '{key}' not found or not a file: {resolved_path}")
            if not is_dir and not is_file and not resolved_path.exists(): # Generic existence check if not dir/file
                 logger.warning(f"Path for '{key}' does not exist: {resolved_path}. Proceeding, but ensure it's created if needed.")
                 # Allow non-existent output dir, but raise for inputs
                 if is_dir and key != "output_dir": raise ValueError(f"Input directory '{key}' does not exist: {resolved_path}")
                 if is_file : raise ValueError(f"Input file '{key}' does not exist: {resolved_path}")

            return resolved_path

        prompts_dir = resolve_validate_path("prompts_dir", "input_data/prompts", is_dir=True)
        # Renamed 'benchmark_file' key to 'default_benchmark_file'
        default_benchmark_file = resolve_validate_path("default_benchmark_file", "input_data/benchmark/cybench_reordered.yaml", is_file=True)
        scenario_file = resolve_validate_path("scenario_file", "input_data/scenario/dummy_scenario.yaml", is_file=True)
        expert_profiles_file = resolve_validate_path("expert_profiles_file", "input_data/expert_profiles.yaml", is_file=True)

        # --- Output Directory ---
        # Resolve path, but don't require it to exist yet. It will be created by the results handler.
        output_dir_str = raw_config.get("output_dir", "output_data")
        output_dir = (project_root / Path(str(output_dir_str))).resolve()


        # --- Load API Keys (<root>/.env > process env) ---
        api_key_anthropic = _resolve_api_key("ANTHROPIC_API_KEY", project_root)
        api_key_openai = _resolve_api_key("OPENAI_API_KEY", project_root)
        # Accept either GEMINI_API_KEY or GOOGLE_API_KEY as the env-var name.
        api_key_gemini = _resolve_api_key("GEMINI_API_KEY", project_root)
        if not api_key_gemini:
            api_key_gemini = _resolve_api_key("GOOGLE_API_KEY", project_root)

        app_config = AppConfig(
            prompts_dir=prompts_dir,
            default_benchmark_file=default_benchmark_file,
            scenario_file=scenario_file,
            expert_profiles_file=expert_profiles_file,
            llm_settings=llm_settings,
            workflow_settings=workflow_settings,
            output_dir=output_dir,
            api_key_anthropic=api_key_anthropic,
            api_key_openai=api_key_openai,
            api_key_gemini=api_key_gemini,
            models_to_run=models_to_run,
        )

        # --- Validate API key presence based on the union of providers ---
        required = app_config.required_providers
        if 'anthropic' in required and not app_config.api_key_anthropic:
            raise ValueError(
                "At least one configured model is from Anthropic, but no Anthropic API key found "
                "(checked <root>/.env, ANTHROPIC_API_KEY env var)."
            )
        if 'openai' in required and not app_config.api_key_openai:
            raise ValueError(
                "At least one configured model is from OpenAI, but no OpenAI API key found "
                "(checked <root>/.env, OPENAI_API_KEY env var)."
            )
        if 'google' in required and not app_config.api_key_gemini:
            raise ValueError(
                "At least one configured model is from Google (Gemini), but no Gemini API key found "
                "(checked <root>/.env, GEMINI_API_KEY/GOOGLE_API_KEY env vars)."
            )

        logger.info(
            f"Configuration loaded. Models to run ({len(models_to_run)}): {models_to_run}. "
            f"Providers required: {sorted(required)}."
        )
        return app_config

    # Catch potential errors during processing/validation
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        logger.error(f"Configuration validation failed: {e}", exc_info=True)
        # Re-raise as ValueError for consistent error handling upstream
        raise ValueError(f"Configuration validation failed: {e}") from e

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    try:
        # Assumes config.yaml is in the project root when running from src/
        # To test, create a config.yaml from config_example.yaml and fill in keys.
        # Also, add the new flags to config.yaml under workflow_settings to test their loading.
        # e.g.,
        # workflow_settings:
        #   ...
        #   estimate_num_actors_per_task_benchmark: true

        config = load_config("config.yaml") # Use a config.yaml for testing

        print("--- Configuration Loaded Successfully ---")
        print(f"API Provider: {config.inferred_api_provider}")
        print(f"Model: {config.llm_settings.model}")
        print(f"Prompts Dir: {config.prompts_dir}")
        print(f"Default Benchmark File: {config.default_benchmark_file}")
        print(f"Scenario File: {config.scenario_file}")
        print(f"Experts File: {config.expert_profiles_file}")
        print(f"Output Dir: {config.output_dir}")
        print(f"Run Dir: {config.runs_dir}")
        print(f"Registry File: {config.registry_file}")
        print(f"LLM Settings: {config.llm_settings}")
        print(f"Workflow Settings: {config.workflow_settings}") # Should show new flags
        print(f"  Estimate Num Actors: {config.workflow_settings.estimate_num_actors_per_task_benchmark}")
        print(f"  Estimate Num Attacks: {config.workflow_settings.estimate_num_attacks_per_task_benchmark}")
        print(f"  Estimate Damage: {config.workflow_settings.estimate_damage_per_task_benchmark}")
        print(f"Anthropic Key Present: {bool(config.api_key_anthropic)}")
        print(f"OpenAI Key Present: {bool(config.api_key_openai)}")

    except (FileNotFoundError, yaml.YAMLError, ValueError, TypeError) as err:
        print(f"\n--- Error loading configuration ---")
        print(err)
        print("Ensure you have a 'config.yaml' in the project root for testing this script directly.")
    except Exception as general_err:
         print(f"\n--- An unexpected error occurred ---")
         print(general_err)
         import traceback
         traceback.print_exc() # Print full traceback for unexpected errors