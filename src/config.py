# src/config.py

import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

# Basic logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Data Structure Definitions ---
@dataclass
class ThinkingSettings:
    """Settings to control Anthropic's extended thinking feature."""
    enabled: bool = False
    budget_tokens: int = 4000

    def __post_init__(self):
        if self.budget_tokens <= 0:
            raise ValueError("ThinkingSettings: 'budget_tokens' must be positive.")

@dataclass
class LLMSettings:
    """Settings related to the LLM and API interaction."""
    model: str                    # Model identifier (e.g., "claude-3-5-sonnet-20240620")
    temperature: float = 0.8      # LLM sampling temperature
    max_concurrent_calls: int = 5 # Max parallel API requests
    rate_limit_calls: int = 45    # Max calls per rate_limit_period
    rate_limit_period: int = 60   # Time window for rate limit (seconds)
    thinking: ThinkingSettings = field(default_factory=ThinkingSettings)

    def __post_init__(self):
        # Basic validation within the dataclass
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

@dataclass
class WorkflowSettings:
    """Settings controlling the Delphi workflow execution."""
    num_tasks: Optional[int] = None     # Max tasks to process (None = all)
    num_experts: Optional[int] = None   # Max experts to use (None = all)
    scenario_steps: Optional[List[str]] = None # Specific scenario steps to run (None = all)
    delphi_rounds: int = 3              # Total number of Delphi rounds
    convergence_threshold: float = 0.05 # Std dev threshold for early stopping

    # Example Task Settings
    include_easier_tasks: bool = True  # Include easier example tasks alongside hardest task
    num_example_tasks: Optional[int] = 3 # Number of example tasks to include (including previous task)

    # NEW Scenario-Level Metric Estimation Flags
    estimate_num_actors_per_task_benchmark: bool = False
    estimate_num_attacks_per_task_benchmark: bool = False
    estimate_damage_per_task_benchmark: bool = False


    def __post_init__(self):
        if self.num_tasks is not None and self.num_tasks < 0:
            raise ValueError("WorkflowSettings: 'num_tasks' cannot be negative.")
        if self.num_experts is not None and self.num_experts <= 0:
            raise ValueError("WorkflowSettings: 'num_experts' must be positive if specified.")
        if self.delphi_rounds <= 0:
            raise ValueError("WorkflowSettings: 'delphi_rounds' must be positive.")
        if self.convergence_threshold < 0:
            raise ValueError("WorkflowSettings: 'convergence_threshold' cannot be negative.")
        if self.scenario_steps is not None and not isinstance(self.scenario_steps, list):
            raise TypeError("WorkflowSettings: 'scenario_steps' must be a list of strings or null.")
        # No specific validation needed for the new boolean flags, as they default to False.


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
    # Output Paths (base directories)
    output_dir: Path = Path("output_data")

    # API Keys (loaded but potentially sensitive)
    api_key_anthropic: Optional[str] = None
    api_key_openai: Optional[str] = None

    # --- Properties (methods acting like attributes) ---
    @property
    def inferred_api_provider(self) -> str:
        """Infers the API provider based on the model name."""
        model_lower = self.llm_settings.model.lower()
        if 'claude' in model_lower:
            return 'anthropic'
        elif 'gpt-' in model_lower or 'o' in model_lower: # Broader check for OpenAI models
            return 'openai'
        else:
            logger.error(f"Could not infer API provider from model name: '{self.llm_settings.model}'. Add specific check if needed.")
            raise ValueError(f"Could not infer API provider from model name: {self.llm_settings.model}")

    @property
    def runs_dir(self) -> Path:
        """Convenience property for the runs subdirectory."""
        return self.output_dir / "runs"

    @property
    def registry_file(self) -> Path:
        """Convenience property for the run registry file path."""
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
        
        thinking_settings_raw = llm_settings_raw.get("thinking", {})
        if not isinstance(thinking_settings_raw, dict):
            raise ValueError("'llm_settings.thinking' must be a dictionary.")

        thinking_settings = ThinkingSettings(
            enabled=bool(thinking_settings_raw.get("enabled", ThinkingSettings.enabled)),
            budget_tokens=int(thinking_settings_raw.get("budget_tokens", ThinkingSettings.budget_tokens)),
        )

        # Instantiate nested settings (validation happens in __post_init__)
        llm_settings = LLMSettings(
            model=str(llm_settings_raw.get("model")), # Required, let potential None raise error later
            temperature=float(llm_settings_raw.get("temperature", LLMSettings.temperature)),
            max_concurrent_calls=int(llm_settings_raw.get("max_concurrent_calls", LLMSettings.max_concurrent_calls)),
            rate_limit_calls=int(llm_settings_raw.get("rate_limit_calls", LLMSettings.rate_limit_calls)),
            rate_limit_period=int(llm_settings_raw.get("rate_limit_period", LLMSettings.rate_limit_period)),
            thinking=thinking_settings
        )

        workflow_settings = WorkflowSettings(
            num_tasks=int(n) if (n := workflow_settings_raw.get("num_tasks")) is not None else None,
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


        # --- Load API Keys ---
        api_key_anthropic = raw_config.get("anthropic_api_key")
        api_key_openai = raw_config.get("openai_api_key")

        # --- Instantiate Final AppConfig ---
        # Order fields here according to the corrected dataclass definition
        app_config = AppConfig(
            # Non-defaults first
            prompts_dir=prompts_dir,
            default_benchmark_file=default_benchmark_file, # Use new name
            scenario_file=scenario_file,
            expert_profiles_file=expert_profiles_file,
            llm_settings=llm_settings,
            workflow_settings=workflow_settings,
            # Defaults last
            output_dir=output_dir,
            api_key_anthropic=str(api_key_anthropic) if api_key_anthropic else None,
            api_key_openai=str(api_key_openai) if api_key_openai else None,
        )

        # --- Validate API Key Presence based on inferred provider ---
        provider = app_config.inferred_api_provider # This might raise ValueError if model is unknown
        if provider == 'anthropic' and not app_config.api_key_anthropic:
            raise ValueError("Model indicates Anthropic provider, but 'anthropic_api_key' is missing or empty in config.")
        if provider == 'openai' and not app_config.api_key_openai:
            raise ValueError("Model indicates OpenAI provider, but 'openai_api_key' is missing or empty in config.")

        logger.info("Configuration loaded and validated successfully.")
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