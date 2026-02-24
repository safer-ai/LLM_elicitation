#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data models for intra-benchmark calibration.

Contains dataclass definitions for expert profiles, benchmarks,
and other data structures used in the calibration workflow.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExpertProfile:
    """Represents a simulated expert persona loaded from expert_profiles.yaml."""
    name: str
    background: str
    focus: str
    key_trait: str
    bias: str
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
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Benchmark:
    """Represents a benchmark definition loaded from a benchmark file."""
    description: str
    metrics_to_use: List[str] = field(default_factory=list)
    tasks: List[BenchmarkTask] = field(default_factory=list)

