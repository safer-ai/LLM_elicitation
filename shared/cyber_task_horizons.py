#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helpers for loading data from the cyber-task-horizons submodule.

This keeps the rest of the repository insulated from the upstream repo's
internal directory layout and file naming quirks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMODULE_ROOT = REPO_ROOT / "external" / "cyber-task-horizons-data"


def get_submodule_root() -> Path:
    """Return the submodule root, or raise with setup guidance."""
    if not SUBMODULE_ROOT.exists():
        raise FileNotFoundError(
            "cyber-task-horizons submodule is missing. Run "
            "'git submodule update --init --recursive'."
        )
    return SUBMODULE_ROOT


def get_data_root() -> Path:
    """Return the submodule's data directory."""
    return get_submodule_root() / "data"


def list_benchmarks() -> List[str]:
    """List benchmark directories shipped in the upstream dataset."""
    tasks_root = get_data_root() / "tasks"
    return sorted(path.name for path in tasks_root.iterdir() if path.is_dir())


def load_task_metadata() -> "pd.DataFrame":
    """Load the upstream cross-benchmark task metadata table."""
    return _read_csv(get_data_root() / "tasks" / "task_metadata.csv")


def load_human_completions() -> "pd.DataFrame":
    """Load anonymized human completion outcomes."""
    return _read_csv(get_data_root() / "human" / "completions.csv")


def load_human_estimations() -> "pd.DataFrame":
    """Load human task-duration estimates."""
    return _read_csv(get_data_root() / "human" / "estimations.csv")


def load_provider_models(provider: str) -> Dict[str, Any]:
    """Load one provider model catalog, e.g. 'openai' or 'anthropic'."""
    with open(get_data_root() / "models" / f"{provider}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_tasks(benchmark: str) -> List[Dict[str, Any]]:
    """Load task definitions for one benchmark."""
    return _load_jsonl(_resolve_benchmark_file(benchmark, "_tasks.jsonl"))


def load_model_estimates(benchmark: str) -> List[Dict[str, Any]]:
    """Load frontier-model duration estimates for one benchmark."""
    return _load_jsonl(_resolve_benchmark_file(benchmark, "_model_estimates.jsonl"))


def load_human_runs(benchmark: str) -> List[Dict[str, Any]]:
    """Load per-task human run logs if the benchmark ships them."""
    human_runs = _resolve_benchmark_file(benchmark, "_human_runs.jsonl", required=False)
    if human_runs is None:
        return []
    return _load_jsonl(human_runs)


def get_prebuilt_analysis_path(*relative_parts: str) -> Path:
    """Resolve a path under the submodule's shipped analysis artifacts."""
    return get_submodule_root() / "analysis" / Path(*relative_parts)


def _resolve_benchmark_dir(benchmark: str) -> Path:
    benchmark_dir = get_data_root() / "tasks" / benchmark
    if not benchmark_dir.is_dir():
        available = ", ".join(list_benchmarks())
        raise FileNotFoundError(
            f"Unknown benchmark '{benchmark}'. Available benchmarks: {available}"
        )
    return benchmark_dir


def _resolve_benchmark_file(
    benchmark: str,
    suffix: str,
    required: bool = True,
) -> Optional[Path]:
    benchmark_dir = _resolve_benchmark_dir(benchmark)
    matches = sorted(benchmark_dir.glob(f"*{suffix}"))

    if not matches:
        if required:
            raise FileNotFoundError(
                f"No file matching '*{suffix}' found in {benchmark_dir}"
            )
        return None

    if len(matches) > 1:
        raise ValueError(
            f"Expected one file matching '*{suffix}' in {benchmark_dir}, found {len(matches)}"
        )

    return matches[0]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_csv(path: Path) -> "pd.DataFrame":
    import pandas as pd

    return pd.read_csv(path)
