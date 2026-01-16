from typing import Optional
from utils.benchmark_adapter import BenchmarkAdapter


def format_task_list(tasks: list[dict], adapter: BenchmarkAdapter) -> tuple[str, dict]:
    """
    Generates a list of tasks and returns a single prompt representing the list of tasks, and a mapping from task index
    to the task ID corresponding to that index.
    """
    task_map = {}
    task_prompts = []
    for i, task in enumerate(tasks):
        task_id = adapter.get_task_id(task)
        description = adapter.get_description(task)
        task_section = f"""
TASK {i}:
================================================================================
{description}
================================================================================
"""
        task_map[i] = task_id
        task_prompts.append(task_section)

    return "\n\n\n".join(task_prompts), task_map


def format_single_task_prompt(template: str, task: dict, adapter: BenchmarkAdapter,
                  include_code: bool, cache_dir: Optional[str] = None,
                  prev_task: Optional[dict] = None) -> str:
    """
    Formats the prompt with task data. If include_code is True, includes
    code sections based on the benchmark adapter's implementation.
    """
    tasks = [task]
    prefixes = [""]

    if prev_task is not None:
        tasks = [prev_task, task]
        prefixes = ["task1_", "task2_"]

    fill_dict = {}
    for prefix, task in zip(prefixes, tasks):
        code_section = ""
        if include_code:
            code_section = adapter.format_code_section(task, cache_dir)

        fill_dict[f"{prefix}task_description"] = adapter.get_description(task)
        fill_dict[f"{prefix}code_section"] = code_section

    template = template.format(**fill_dict)

    return template
