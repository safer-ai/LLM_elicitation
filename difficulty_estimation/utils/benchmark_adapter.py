from abc import ABC, abstractmethod
from typing import Optional
import os


class BenchmarkAdapter(ABC):
    """Abstract interface for benchmark-specific logic"""

    @abstractmethod
    def get_task_id(self, task: dict) -> str:
        """Extract unique task identifier"""
        pass

    @abstractmethod
    def get_description(self, task: dict) -> str:
        """Extract task description"""
        pass

    @abstractmethod
    def format_code_section(self, task: dict, cache_dir: Optional[str] = None) -> str:
        """Format code section for prompt (returns empty string if no code)"""
        pass


class BountyBenchAdapter(BenchmarkAdapter):
    def get_task_id(self, task: dict) -> str:
        source_url = task.get('source_url', '')
        if not source_url:
            return None
        return source_url.split('/')[-1]

    def get_description(self, task: dict) -> str:
        return task.get('description', '')

    def format_code_section(self, task: dict, cache_dir: Optional[str] = None) -> str:
        if not cache_dir:
            raise ValueError("cache_dir must be set for BountyBench when include_code=True")

        task_id = self.get_task_id(task)
        if not task_id:
            return ""

        sections = []
        bounty_cache_dir = os.path.join(cache_dir, task_id)

        unpatched_path = os.path.join(bounty_cache_dir, 'unpatched_code.txt')
        if os.path.exists(unpatched_path):
            with open(unpatched_path, 'r') as f:
                unpatched_code = f.read()
                if unpatched_code != '# Unpatched file not found':
                    sections.append(f"Unpatched Code:\n```\n{unpatched_code}\n```")

        patched_path = os.path.join(bounty_cache_dir, 'patched_code.txt')
        if os.path.exists(patched_path):
            with open(patched_path, 'r') as f:
                patched_code = f.read()
                if patched_code != '# Patched file not found':
                    sections.append(f"Patched Code:\n```\n{patched_code}\n```")

        return "\n\n".join(sections)


class SWEBenchAdapter(BenchmarkAdapter):
    def get_task_id(self, task: dict) -> str:
        return task.get('instance_id', '')

    def get_description(self, task: dict) -> str:
        return task.get('problem_statement', '')

    def format_code_section(self, task: dict, cache_dir: Optional[str] = None) -> str:
        sections = []

        if patch := task.get('patch'):
            sections.append(f"**Code Changes ('git diff') representing a reference solution to the problem:**\n```diff\n{patch}\n```")

        if test_patch := task.get('test_patch'):
            sections.append(f"**Unseen tests for checking if a task was solved:**\n```diff\n{test_patch}\n```")

        if hints := task.get('hints_text'):
            sections.append(f"**Hints/Comments from Issue:**\n{hints}")

        return "\n\n".join(sections)


class BigCodeBenchFullCompleteAdapter(BenchmarkAdapter):
    def get_task_id(self, task: dict) -> str:
        return task.get('task_id', '').replace('/', '_')

    def get_description(self, task: dict) -> str:
        return task.get('complete_prompt', '')

    def format_code_section(self, task: dict, cache_dir: Optional[str] = None) -> str:
        sections = []

        if canonical_solution := task.get('canonical_solution'):
            sections.append(f"**Reference implementation (canonical solution):**\n```python\n{canonical_solution}\n```")

        if test := task.get('test'):
            sections.append(f"**Test cases:**\n```python\n{test}\n```")

        return "\n\n".join(sections)


def get_adapter(benchmark_type: str) -> BenchmarkAdapter:
    """Factory function to get the right adapter"""
    adapters = {
        'bountybench': BountyBenchAdapter,
        'swebench_verified': SWEBenchAdapter,
        'bigcodebench_full_complete': BigCodeBenchFullCompleteAdapter,
    }
    adapter_class = adapters.get(benchmark_type)
    if not adapter_class:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}. Available: {list(adapters.keys())}")
    return adapter_class()
