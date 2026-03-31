#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark-specific adapters for calibration experiments.

Provides a clean abstraction for handling different benchmark formats
(CyBench, SWEbench, etc.) with their specific field names and description formats.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BenchmarkAdapter(ABC):
    """Abstract interface for benchmark-specific logic"""

    @abstractmethod
    def get_task_name(self, task_dict: Dict[str, Any]) -> str:
        """Extract the unique task identifier/name"""
        pass

    @abstractmethod
    def get_task_description(self, task_dict: Dict[str, Any]) -> str:
        """Extract and format the task description for prompts"""
        pass


class CyBenchAdapter(BenchmarkAdapter):
    """Adapter for CyBench (Capture the Flag) tasks"""

    def get_task_name(self, task_dict: Dict[str, Any]) -> str:
        return task_dict.get('name', '')

    def get_task_description(self, task_dict: Dict[str, Any]) -> str:
        return task_dict.get('description', '')


class SWEBenchAdapter(BenchmarkAdapter):
    """Adapter for SWEbench (Software Engineering Benchmark) tasks"""

    def get_task_name(self, task_dict: Dict[str, Any]) -> str:
        return task_dict.get('instance_id', '')

    def get_task_description(self, task_dict: Dict[str, Any]) -> str:
        problem_stmt = task_dict.get('problem_statement', '')
        patch = task_dict.get('patch', '')
        test_patch = task_dict.get('test_patch', '')

        desc_parts = []
        if problem_stmt:
            desc_parts.append(f"\n=== Problem Statement ===\n{problem_stmt.strip()}")
        if patch:
            desc_parts.append(f"\n=== Reference Solution (Patch) ===\n{patch.strip()}")
        if test_patch:
            desc_parts.append(f"\n=== Test Patch ===\n{test_patch.strip()}")

        return "\n\n".join(desc_parts) if desc_parts else ""


class LiveBenchLCBGenerationAdapter(BenchmarkAdapter):
    """Adapter for LiveBench LCB_generation (code generation) tasks."""

    def get_task_name(self, task_dict: Dict[str, Any]) -> str:
        return task_dict.get('question_title', task_dict.get('question_id', ''))

    def get_task_description(self, task_dict: Dict[str, Any]) -> str:
        turns = task_dict.get('turns', [])
        prompt = turns[0] if isinstance(turns, list) and turns else str(turns)
        test_cases = task_dict.get('public_test_cases', '')

        parts = [f"=== Problem Prompt ===\n{prompt.strip()}"]
        if test_cases:
            parts.append(f"=== Public Test Cases ===\n{test_cases.strip()}")
        return "\n\n".join(parts)


class LiveBenchCodingCompletionAdapter(BenchmarkAdapter):
    """Adapter for LiveBench coding_completion tasks."""

    def get_task_name(self, task_dict: Dict[str, Any]) -> str:
        return task_dict.get('question_title', task_dict.get('question_id', ''))

    def get_task_description(self, task_dict: Dict[str, Any]) -> str:
        turns = task_dict.get('turns', [])
        prompt = turns[0] if isinstance(turns, list) and turns else str(turns)
        test_cases = task_dict.get('public_test_cases', '')
        partial = task_dict.get('partial_solution', '')
        remainder = task_dict.get('remainder', '')

        parts = [f"=== Problem Prompt (includes partial solution to complete) ===\n{prompt.strip()}"]
        if partial:
            parts.append(f"=== Partial Solution Provided ===\n{partial.strip()}")
        if remainder:
            parts.append(f"=== Expected Completion (remainder) ===\n{remainder.strip()}")
        if test_cases:
            parts.append(f"=== Public Test Cases ===\n{test_cases.strip()}")
        return "\n\n".join(parts)


class GenericAdapter(BenchmarkAdapter):
    """Fallback adapter for benchmarks with simple name+description format."""

    def get_task_name(self, task_dict: Dict[str, Any]) -> str:
        return task_dict.get('name', task_dict.get('id', ''))

    def get_task_description(self, task_dict: Dict[str, Any]) -> str:
        return task_dict.get('description', '')


_ADAPTER_REGISTRY = {
    'cybench': CyBenchAdapter,
    'swebench_verified': SWEBenchAdapter,
    'swebench': SWEBenchAdapter,
    'livebench_lcb_generation': LiveBenchLCBGenerationAdapter,
    'livebench_coding_completion': LiveBenchCodingCompletionAdapter,
}


def register_adapter(benchmark_name: str, adapter_class: type):
    """Register a new adapter for a benchmark name."""
    _ADAPTER_REGISTRY[benchmark_name.lower().strip()] = adapter_class


def get_adapter(benchmark_name: str) -> BenchmarkAdapter:
    """
    Factory function to get the appropriate adapter for a benchmark.

    Falls back to GenericAdapter if benchmark_name is not recognised.
    """
    normalized_name = benchmark_name.lower().strip()
    adapter_class = _ADAPTER_REGISTRY.get(normalized_name, GenericAdapter)
    return adapter_class()
