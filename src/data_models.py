# src/data_models.py
import yaml
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# --- Data Models for Input Files ---

@dataclass(frozen=True) # Use frozen=True for immutable data records
class ExpertProfile:
    """Represents a simulated expert persona loaded from expert_profiles.yaml."""
    name: str
    background: str
    focus: str
    key_trait: str
    bias: str
    # Added analytical_approach based on your yaml structure
    analytical_approach: Optional[str] = None

    def get_persona_description(self) -> str:
        """Generates a descriptive string for the expert's persona."""
        parts = [
            f"You are {self.name}.",
            f"Background: {self.background}.",
            f"Focus: {self.focus}.",
            f"Trait: {self.key_trait}.",
            f"Bias/Approach: {self.bias}."
        ]
        if self.analytical_approach:
            parts.append(f"Analytical Method: {self.analytical_approach}.")
        return " ".join(parts)


@dataclass(frozen=True)
class BenchmarkTask:
    """Represents a single task within a benchmark set."""
    name: str
    description: str
    metrics: Dict[str, Any] = field(default_factory=dict) # Flexible metrics (e.g., {'fst': 7})


@dataclass(frozen=True)
class Benchmark:
    """Represents a benchmark definition loaded from a benchmark file."""
    description: str
    metrics_to_use: List[str] = field(default_factory=list) # e.g., ['fst']
    tasks: List[BenchmarkTask] = field(default_factory=list)


@dataclass(frozen=True)
class ThreatActor:
    """Represents the threat actor profile in a scenario."""
    name: str
    description: str


@dataclass(frozen=True)
class Target:
    """Represents the target profile in a scenario."""
    name: str
    description: str


@dataclass(frozen=True)
class ScenarioStep:
    """Represents a single step within a risk scenario."""
    name: str
    description: str
    assumptions: str # Can be multi-line string from YAML
    benchmark_file: Optional[str] = None # Path to the benchmark YAML for this step


@dataclass(frozen=True)
class ScenarioLevelMetric:
    """Represents configuration for a single scenario-level metric estimation."""
    benchmark_file: str  # Path to the benchmark YAML for this metric
    assumptions: str = "" # Assumptions specific to this metric estimation


@dataclass(frozen=True)
class Scenario:
    """Represents a complete risk scenario loaded from a scenario file."""
    name: str
    description: str
    threat_actor: ThreatActor
    target: Target
    steps: List[ScenarioStep] = field(default_factory=list)
    # UPDATED FIELD: Changed from Dict[str, str] to Dict[str, ScenarioLevelMetric]
    scenario_level_metrics: Optional[Dict[str, ScenarioLevelMetric]] = field(default_factory=dict)


# --- Convenience Function to Load All Inputs ---
@dataclass
class InputData:
    """Container for all loaded input data."""
    prompts: Dict[str, str]
    experts: List[ExpertProfile]
    scenario: Optional[Scenario] # Scenario object itself will now contain scenario_level_metrics
    # Store loaded benchmarks by their file path string
    loaded_benchmarks: Dict[str, Benchmark] = field(default_factory=dict)