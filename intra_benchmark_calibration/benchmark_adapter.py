#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark-specific adapters for intra-benchmark calibration.

This module provides a clean abstraction for handling different benchmark formats
(CyBench, SWEbench, etc.) with their specific field names and description formats.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


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
        """CyBench uses 'name' field"""
        return task_dict.get('name', '')

    def get_task_description(self, task_dict: Dict[str, Any]) -> str:
        """CyBench uses simple 'description' field"""
        return task_dict.get('description', '')


class SWEBenchAdapter(BenchmarkAdapter):
    """Adapter for SWEbench (Software Engineering Benchmark) tasks"""

    def get_task_name(self, task_dict: Dict[str, Any]) -> str:
        """SWEbench uses 'instance_id' field (e.g., 'django__django-11099')"""
        return task_dict.get('instance_id', '')

    def get_task_description(self, task_dict: Dict[str, Any]) -> str:
        """
        SWEbench combines problem_statement, patch, and test_patch.
        
        Returns a formatted description with clearly labeled sections.
        """
        problem_stmt = task_dict.get('problem_statement', '')
        patch = task_dict.get('patch', '')
        test_patch = task_dict.get('test_patch', '')
        
        # Build description with sections
        desc_parts = []
        
        if problem_stmt:
            desc_parts.append(f"\n=== Problem Statement ===\n{problem_stmt.strip()}")
        
        if patch:
            desc_parts.append(f"\n=== Reference Solution (Patch) ===\n{patch.strip()}")
        
        if test_patch:
            desc_parts.append(f"\n=== Test Patch ===\n{test_patch.strip()}")
        
        return "\n\n".join(desc_parts) if desc_parts else ""


def get_adapter(benchmark_name: str) -> BenchmarkAdapter:
    """
    Factory function to get the appropriate adapter for a benchmark.
    
    Args:
        benchmark_name: Name of the benchmark (e.g., 'cybench', 'swebench_verified')
    
    Returns:
        BenchmarkAdapter instance for the specified benchmark
    
    Raises:
        ValueError: If benchmark_name is not recognized
    """
    normalized_name = benchmark_name.lower().strip()
    
    adapters = {
        'cybench': CyBenchAdapter,
        'swebench_verified': SWEBenchAdapter,
        'swebench': SWEBenchAdapter,
    }
    
    adapter_class = adapters.get(normalized_name)
    
    if not adapter_class:
        raise ValueError(
            f"Unknown benchmark: '{benchmark_name}'. "
            f"Available benchmarks: {list[str](adapters.keys())}"
        )
    
    return adapter_class()
