"""
ground-truth/data.py
====================
Loader for the Lyptus Research "Offensive Cyber Task Horizons" dataset.

Source
------
  Paper : https://lyptusresearch.org/research/offensive-cyber-time-horizons
  HF    : https://huggingface.co/datasets/lyptus-research/cyber-task-horizons
  GitHub: https://github.com/lyptus-research/cyber-task-horizons-data

Dataset subsets
---------------
  completions     – one row per expert task attempt (174 rows)
  estimations     – one row per expert time estimate
  human_baselines – per-task human difficulty labels
  model_estimates – frontier-model difficulty estimates for all evaluated tasks

Relevance to this project
--------------------------
This dataset provides *empirical* ground truth for:
  1. Human expert task-completion times  (completions / human_baselines)
  2. LLM-elicited difficulty estimates compared with human ground truth (model_estimates)
  3. Per-task pass rates across 630 tasks × 15 models × 7 cyber benchmarks

This lets us directly evaluate our LLM-elicitation methodology:
  • Elicit using tasks whose difficulty + pass rate we know.
  • Ask the model to predict pass rate on held-out tasks.
  • Compare against empirical pass rates from this dataset.

Usage
-----
  from ground_truth.data import load_all, get_completions, get_human_baselines, ...

  # or run directly for a summary:
  python ground-truth/data.py
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


HF_DATASET_ID = "lyptus-research/cyber-task-horizons"

SUBSETS = {
    "completions":     "completions",
    "estimations":     "estimations",
    "human_baselines": "human_baselines",
    "model_estimates": "model_estimates",
}


COLUMN_DOCS: dict[str, dict[str, str]] = {
    "completions": {
        "expert_id":        "Anonymised expert label (expert_01 … expert_10)",
        "task_id":          "Benchmark task identifier",
        "benchmark":        "Parent benchmark (cybench, cvebench, cybashbench, nl2bash, "
                            "intercode-ctf, nyuctf, cybergym)",
        "passed":           "Expert solved the task",
        "censored":         "Right-censored: expert gave up; elapsed_minutes is a lower bound",
        "elapsed_minutes":  "Wall-clock time in minutes (timing corrections applied)",
        "submitted_at":     "ISO 8601 timestamp",
        "skip_reason":      "Only for censored rows: too_difficult or time_constraint",
    },
    "estimations": {
        "expert_id":          "Anonymised expert label",
        "task_id":            "Benchmark task identifier",
        "benchmark":          "Parent benchmark",
        "estimated_minutes":  "Point estimate of task difficulty in minutes",
        "confidence":         "Self-reported: high (~2×), medium (2-3×), low (5×+)",
        "notes":              "Optional reasoning or caveats",
        "review_minutes":     "Time spent reviewing the task and solution",
        "submitted_at":       "ISO 8601 timestamp",
    },
    "human_baselines": {
        "task_id":                   "Benchmark task identifier",
        "task_family":               "Benchmark name",
        "completion_minutes":        "Median expert completion time (if available)",
        "n_completions":             "Number of expert completions",
        "estimate_minutes":          "Geometric mean of expert estimates (if available)",
        "n_estimates":               "Number of expert estimates",
        "firstblood_minutes":        "CTF first-blood time (CyBench only)",
        "censored_lower_minutes":    "Lower bound from right-censored attempts",
        "best_available_minutes":    "Difficulty label used in IRT analysis",
        "best_available_source":     "Source that provided the label "
                                     "(completions > first-blood > estimates)",
    },
    "model_estimates": {
        "task_id":               "Benchmark task identifier",
        "task_family":           "Benchmark name",
        "model_estimate_minutes": "Model's point estimate of task difficulty in minutes",
    },
}


def _ensure_datasets() -> None:
    """Raise a clear ImportError with install instructions if `datasets` is missing."""
    try:
        import datasets  # noqa: F401
    except ImportError:
        raise ImportError(
            "The `datasets` package is required to load from Hugging Face.\n"
            "Install it with:\n\n"
            "    pip install datasets\n\n"
            "or add it to requirements.txt and re-run `pip install -r requirements.txt`."
        )


def _load_subset(subset_name: str, split: str = "train") -> pd.DataFrame:
    """
    Load a single HuggingFace subset and return as a pandas DataFrame.

    Parameters
    ----------
    subset_name : str
        One of the keys in SUBSETS.
    split : str
        HuggingFace split to load (default "train").

    Returns
    -------
    pd.DataFrame
    """
    _ensure_datasets()
    from datasets import load_dataset  

    logger.info(f"Loading HuggingFace subset '{subset_name}' from '{HF_DATASET_ID}' …")
    ds = load_dataset(HF_DATASET_ID, subset_name, split=split)
    df = ds.to_pandas()
    logger.info(f"  → {len(df):,} rows, {len(df.columns)} columns")
    return df



def get_completions(split: str = "train") -> pd.DataFrame:
    """
    Load the **completions** subset.

    One row per expert task attempt, including successful completions,
    failed attempts, and right-censored sessions.

    Key columns: expert_id, task_id, benchmark, passed, censored,
                 elapsed_minutes, submitted_at, skip_reason.
    """
    return _load_subset("completions", split)


def get_estimations(split: str = "train") -> pd.DataFrame:
    """
    Load the **estimations** subset.

    One row per expert time estimate. Experts reviewed each task and its
    reference solution, then estimated how long a cold-start practitioner
    would take.

    Key columns: expert_id, task_id, benchmark, estimated_minutes,
                 confidence, notes, review_minutes, submitted_at.
    """
    return _load_subset("estimations", split)


def get_human_baselines(split: str = "train") -> pd.DataFrame:
    """
    Load the **human_baselines** subset.

    Per-task human-derived difficulty labels for tasks with at least one
    human data source (expert completions, expert estimates, or CTF
    first-blood times).

    The ``best_available_minutes`` column is the difficulty label used in
    the paper's IRT analysis (source hierarchy: completions > first-blood
    > estimates).

    Key columns: task_id, task_family, completion_minutes, n_completions,
                 estimate_minutes, n_estimates, firstblood_minutes,
                 censored_lower_minutes, best_available_minutes,
                 best_available_source.
    """
    return _load_subset("human_baselines", split)


def get_model_estimates(split: str = "train") -> pd.DataFrame:
    """
    Load the **model_estimates** subset.

    Frontier-model difficulty estimates for all evaluated tasks, generated
    by a frontier LLM estimating how long a skilled practitioner would take
    (same materials as human experts).

    Key columns: task_id, task_family, model_estimate_minutes.
    """
    return _load_subset("model_estimates", split)


def load_all(split: str = "train") -> dict[str, pd.DataFrame]:
    """
    Load all four subsets and return them as a dictionary of DataFrames.

    Returns
    -------
    dict with keys: "completions", "estimations", "human_baselines",
                    "model_estimates"

    Example
    -------
    >>> from ground_truth.data import load_all
    >>> data = load_all()
    >>> baselines = data["human_baselines"]
    >>> baselines.sort_values("best_available_minutes").head(10)
    """
    return {
        "completions":     get_completions(split),
        "estimations":     get_estimations(split),
        "human_baselines": get_human_baselines(split),
        "model_estimates": get_model_estimates(split),
    }



def expert_pass_rate(completions: Optional[pd.DataFrame] = None) -> pd.Series:
    """
    Compute per-benchmark expert pass rate (excluding censored attempts).

    Parameters
    ----------
    completions : pd.DataFrame, optional
        Pre-loaded completions DataFrame. Loaded automatically if not provided.

    Returns
    -------
    pd.Series indexed by benchmark name
    """
    if completions is None:
        completions = get_completions()
    uncensored = completions[~completions["censored"]]
    return uncensored.groupby("benchmark")["passed"].mean().rename("expert_pass_rate")


def difficulty_comparison(
    human_baselines: Optional[pd.DataFrame] = None,
    model_estimates: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Join human baseline difficulty and model-estimated difficulty on task_id
    for direct comparison.

    Returns a DataFrame with columns:
        task_id, task_family, best_available_minutes (human),
        best_available_source, model_estimate_minutes, ratio (model/human)
    """
    if human_baselines is None:
        human_baselines = get_human_baselines()
    if model_estimates is None:
        model_estimates = get_model_estimates()

    merged = human_baselines.merge(
        model_estimates[["task_id", "model_estimate_minutes"]],
        on="task_id",
        how="inner",
    )
    merged["ratio_model_over_human"] = (
        merged["model_estimate_minutes"] / merged["best_available_minutes"]
    )
    return merged


def task_pass_rates_by_difficulty_bin(
    completions: Optional[pd.DataFrame] = None,
    human_baselines: Optional[pd.DataFrame] = None,
    n_bins: int = 4,
) -> pd.DataFrame:
    """
    Compute per-difficulty-bin pass rates, mirroring the METR / Lyptus IRT
    approach.

    Bins tasks by ``best_available_minutes`` (log-uniform), then computes
    the expert pass rate within each bin.

    Parameters
    ----------
    completions : pd.DataFrame, optional
    human_baselines : pd.DataFrame, optional
    n_bins : int
        Number of difficulty bins (default 4).

    Returns
    -------
    pd.DataFrame with columns: difficulty_bin, n_tasks, expert_pass_rate
    """
    import numpy as np

    if completions is None:
        completions = get_completions()
    if human_baselines is None:
        human_baselines = get_human_baselines()

    uncensored = completions[~completions["censored"]]
    merged = uncensored.merge(
        human_baselines[["task_id", "best_available_minutes"]],
        on="task_id",
        how="inner",
    )

    merged["log_difficulty"] = np.log10(merged["best_available_minutes"].clip(lower=0.01))
    merged["difficulty_bin"] = pd.cut(merged["log_difficulty"], bins=n_bins, labels=False)

    result = (
        merged.groupby("difficulty_bin")
        .agg(
            n_tasks=("task_id", "nunique"),
            expert_pass_rate=("passed", "mean"),
            min_minutes=("best_available_minutes", "min"),
            max_minutes=("best_available_minutes", "max"),
        )
        .reset_index()
    )
    return result


def describe_column(subset: str, column: str) -> str:
    """Return a human-readable description of a column in a given subset."""
    docs = COLUMN_DOCS.get(subset, {})
    return docs.get(column, f"No description available for '{subset}.{column}'")



def _print_summary(data: dict[str, pd.DataFrame]) -> None:
    sep = "=" * 70

    print(f"\n{sep}")
    print("  Lyptus Research – Offensive Cyber Task Horizons  |  Ground Truth")
    print(sep)

    for subset_name, df in data.items():
        print(f"\n[{subset_name.upper()}]  {len(df):,} rows × {len(df.columns)} columns")
        print(f"  Columns: {list(df.columns)}")
        if "benchmark" in df.columns or "task_family" in df.columns:
            key = "benchmark" if "benchmark" in df.columns else "task_family"
            counts = df[key].value_counts()
            print(f"  Breakdown by {key}:")
            for name, cnt in counts.items():
                print(f"    {name:<35} {cnt:>4}")

    # Expert pass rate
    print(f"\n{sep}")
    print("  Expert Pass Rate by Benchmark (uncensored attempts only)")
    print(sep)
    rates = expert_pass_rate(data["completions"])
    for bm, rate in rates.sort_values(ascending=False).items():
        print(f"  {bm:<35} {rate:.1%}")

    # Overall pass rate
    completions = data["completions"]
    uncensored = completions[~completions["censored"]]
    overall = uncensored["passed"].mean()
    print(f"\n  Overall expert pass rate (all benchmarks): {overall:.1%}")
    print(f"  (based on {len(uncensored):,} uncensored attempts by "
          f"{completions['expert_id'].nunique()} experts across "
          f"{completions['task_id'].nunique()} unique tasks)\n")

    # Difficulty comparison (if both subsets have matching tasks)
    try:
        comp = difficulty_comparison(data["human_baselines"], data["model_estimates"])
        if not comp.empty:
            print(f"{sep}")
            print("  Model vs. Human Difficulty: summary statistics")
            print(sep)
            print(comp[["best_available_minutes", "model_estimate_minutes",
                         "ratio_model_over_human"]].describe().round(2).to_string())
            print()
    except Exception as exc:
        logger.debug(f"Difficulty comparison skipped: {exc}")



if __name__ == "__main__":
    try:
        data = load_all()
        _print_summary(data)
    except ImportError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error")
        sys.exit(1)
