#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lyptus offensive-cyber-time-horizons data loading.

Loads the two parquet files plus the per-benchmark task JSONLs, restricts to the
291 headline tasks (those with `best_available_minutes` AND model-run coverage),
joins everything by `task_id`, and exposes:

  - `LyptusTask`: per-task record with id, family, FST, estimation_instructions,
    optional solution_walkthrough.
  - `LyptusOutcomes`: an outcome matrix indexed by (model_alias, task_id) with NaN
    for unevaluated cells.
  - `load_lyptus_dataset(...)`: convenience entry point returning both.
  - `read_repo_commit_sha(repo_dir)`: records the git SHA of the data checkout for
    run metadata (reproducibility).
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


LYPTUS_TASK_FAMILIES = (
    "cybashbench",
    "nl2bash",
    "intercode-ctf",
    "nyuctf",
    "cybench",
    "cvebench",
    "cybergym",
)

# Mapping from on-disk subdirectory under data/tasks/ to the JSONL filename pattern
# used inside that subdirectory. Lyptus is mostly consistent (`<dir>_tasks.jsonl`)
# except intercode-ctf which keeps the hyphen.
_TASK_JSONL_NAME = {
    "cybashbench": "cybashbench_tasks.jsonl",
    "nl2bash": "nl2bash_tasks.jsonl",
    "intercode-ctf": "intercode-ctf_tasks.jsonl",
    "nyuctf": "nyuctf_tasks.jsonl",
    "cybench": "cybench_tasks.jsonl",
    "cvebench": "cvebench_tasks.jsonl",
    "cybergym": "cybergym_tasks.jsonl",
}


@dataclass(frozen=True)
class LyptusTask:
    """One headline task with the fields needed by the experiment."""

    task_id: str
    task_family: str  # high-level family from task_difficulties (e.g. 'cvebench')
    fst_minutes: float
    fst_source: Optional[str]
    estimation_instructions: str
    solution_walkthrough: Optional[str] = None


@dataclass
class LyptusOutcomes:
    """
    Per-(model, task) binary outcome matrix.

    `frame` is a DataFrame indexed by model_alias with columns = task_ids and
    values in {0.0, 1.0, NaN}. NaN means the model was not evaluated on that task.
    """

    frame: pd.DataFrame  # index=alias, columns=task_id, values float in {0,1,NaN}

    @property
    def models(self) -> List[str]:
        return list(self.frame.index)

    @property
    def task_ids(self) -> List[str]:
        return list(self.frame.columns)

    def outcome(self, model_alias: str, task_id: str) -> Optional[float]:
        """Return 0.0/1.0 if evaluated, None if not."""
        try:
            v = self.frame.at[model_alias, task_id]
        except KeyError:
            return None
        return None if pd.isna(v) else float(v)

    def evaluated_mask(self, model_alias: str, task_ids: Sequence[str]) -> np.ndarray:
        """Boolean mask: True if model has an outcome for that task."""
        row = self.frame.loc[model_alias, list(task_ids)]
        return ~row.isna().to_numpy()

    def pass_rate(self, model_alias: str, task_ids: Sequence[str]) -> Optional[Dict[str, float]]:
        """
        Pass rate for `model_alias` over `task_ids`, computed on the evaluated
        subset only. Returns dict with keys 'rate', 'n_evaluated', 'n_in_subset',
        'n_solved', or None if no tasks were evaluated.
        """
        row = self.frame.loc[model_alias, list(task_ids)]
        evaluated = row.dropna()
        if len(evaluated) == 0:
            return None
        return {
            "rate": float(evaluated.mean()),
            "n_evaluated": int(len(evaluated)),
            "n_in_subset": int(len(task_ids)),
            "n_solved": int(evaluated.sum()),
        }

    def ground_truth_summary(self) -> Dict[str, float]:
        """
        Aggregate stats across the (model, task) outcome matrix, used to give the
        forecaster a calibration-relevant base rate in its prompt.

        Per-task pass rate is computed across the (post-drop) model panel only,
        skipping NaNs. So if a forecaster sees this summary at temperature 0,
        it should see numbers consistent with the panel actually used.
        """
        per_task = self.frame.mean(axis=0, skipna=True)
        n_models = int(self.frame.shape[0])
        n_tasks = int(self.frame.shape[1])
        n_zero = int((per_task == 0.0).sum())
        n_one = int((per_task == 1.0).sum())
        return {
            "n_models": n_models,
            "n_tasks": n_tasks,
            "mean_per_task_pass_rate": float(per_task.mean()),
            "median_per_task_pass_rate": float(per_task.median()),
            "n_tasks_zero_pass": n_zero,
            "frac_tasks_zero_pass": float(n_zero / max(n_tasks, 1)),
            "n_tasks_full_pass": n_one,
            "frac_tasks_full_pass": float(n_one / max(n_tasks, 1)),
        }


@dataclass
class LyptusDataset:
    """Bundle of headline tasks, outcomes, and reproducibility metadata."""

    tasks: List[LyptusTask]
    outcomes: LyptusOutcomes
    repo_dir: Path
    commit_sha: Optional[str]
    model_aliases_full: List[str] = field(default_factory=list)
    sparse_models: List[str] = field(default_factory=list)
    # Provenance for the headline-task filter:
    n_headline_in_parquet: int = 0
    n_dropped_no_estimation_instructions: int = 0
    dropped_task_ids_no_estimation_instructions: List[str] = field(default_factory=list)

    @property
    def task_ids(self) -> List[str]:
        return [t.task_id for t in self.tasks]

    @property
    def task_by_id(self) -> Dict[str, LyptusTask]:
        return {t.task_id: t for t in self.tasks}

    def fst_array(self) -> np.ndarray:
        return np.array([t.fst_minutes for t in self.tasks], dtype=float)

    def provenance_dict(self) -> Dict[str, object]:
        """Compact dict for run metadata: who, what, when, how many, why-some-dropped."""
        return {
            "lyptus_repo_dir": str(self.repo_dir),
            "lyptus_commit_sha": self.commit_sha,
            "n_headline_tasks_in_parquet": self.n_headline_in_parquet,
            "n_headline_tasks_used": len(self.tasks),
            "n_dropped_no_estimation_instructions": self.n_dropped_no_estimation_instructions,
            "dropped_task_ids_no_estimation_instructions": list(self.dropped_task_ids_no_estimation_instructions),
            "models_in_outcomes_matrix": list(self.outcomes.models),
            "n_models_in_outcomes_matrix": len(self.outcomes.models),
            "all_model_aliases_in_parquet": list(self.model_aliases_full),
            "sparse_models_at_load": list(self.sparse_models),
            "design_observations": [
                "Per-task binary outcomes for the forecasted model are shown alongside "
                "anchor and easier tasks in the prompt (more discriminative than the "
                "bin-level pass rate alone). See prompt_builder._outcome_tag.",
                "Anchor selection is M-independent in practice: with 10/12 models having "
                "full coverage of the 269 headline tasks, the representativeness "
                "heuristic |task_panel_pass_rate - bin_panel_mean| picks the same task "
                "across forecasted models within a bin. GLM-5 / Opus 4 are missing "
                "1-2 tasks each in the harder bins but this never changes the top "
                "candidate (verified for n_bins=5, see production_anchors.csv).",
                "Anchor distribution skews to NYUCTF (3 of 5 bins under default "
                "n_bins=5 / equal_count): NYUCTF tasks tend to have tight per-task "
                "pass-rate distributions that match bin means well. cybashbench "
                "anchors bin 0; cybergym (arvo) anchors bin 3. Documented design "
                "observation, not a bug.",
            ],
        }


def read_repo_commit_sha(repo_dir: Path) -> Optional[str]:
    """Return the HEAD SHA of `repo_dir`, or None if not a git checkout."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _load_task_jsonl(tasks_dir: Path) -> Dict[str, dict]:
    """Load every per-benchmark JSONL into a single {task_id: row} dict."""
    by_id: Dict[str, dict] = {}
    for subdir, fname in _TASK_JSONL_NAME.items():
        path = tasks_dir / subdir / fname
        if not path.exists():
            logger.warning(f"JSONL missing: {path} — skipping family '{subdir}'")
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                by_id[row["task_id"]] = row
    logger.info(f"Loaded JSONL records for {len(by_id)} tasks across {len(_TASK_JSONL_NAME)} families")
    return by_id


def load_lyptus_dataset(
    repo_dir: Path,
    *,
    drop_models: Sequence[str] = (),
    parquet_subdir: str = "analysis/figures/data",
    tasks_subdir: str = "data/tasks",
) -> LyptusDataset:
    """
    Load and assemble the Lyptus headline-task dataset.

    Args:
        repo_dir: path to the cyber-task-horizons-data repo checkout.
        drop_models: model aliases to drop from the outcomes matrix entirely.
            Default is empty; callers (e.g. the config layer) typically pass
            sparse-coverage models like ['GPT-2', 'GPT-3', 'GPT-3.5'].
        parquet_subdir: location of the two parquet files within `repo_dir`.
        tasks_subdir: location of the per-family JSONLs within `repo_dir`.
    """
    repo_dir = repo_dir.resolve()
    pq_dir = repo_dir / parquet_subdir
    td_path = pq_dir / "task_difficulties.parquet"
    mr_path = pq_dir / "model_runs.parquet"
    for p in (td_path, mr_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Required Lyptus parquet missing: {p}. "
                "Run `python intra_benchmark_calibration/scripts/fetch_lyptus_data.py`."
            )

    td = pd.read_parquet(td_path)
    mr = pd.read_parquet(mr_path)
    logger.info(f"task_difficulties: {td.shape}, model_runs: {mr.shape}")

    # Headline = best_available_minutes present AND covered by model_runs.
    td_h = td.dropna(subset=["best_available_minutes"]).copy()
    mr_task_ids = set(mr["task_id"].unique())
    td_h = td_h[td_h["task_id"].isin(mr_task_ids)].copy()
    headline_ids = list(td_h["task_id"].astype(str))
    logger.info(f"Headline tasks (best_available_minutes set AND in model_runs): {len(headline_ids)}")

    # Per-task JSONL fields (estimation_instructions, solution_walkthrough)
    jsonl = _load_task_jsonl(repo_dir / tasks_subdir)

    tasks: List[LyptusTask] = []
    missing_ei: List[str] = []
    n_headline_in_parquet = len(headline_ids)
    for _, row in td_h.iterrows():
        tid = str(row["task_id"])
        jl = jsonl.get(tid)
        meta = (jl or {}).get("dataset_task_metadata", {}) if jl else {}
        ei = (meta.get("estimation_instructions") or "").strip()
        if not ei:
            missing_ei.append(tid)
            continue
        tasks.append(
            LyptusTask(
                task_id=tid,
                task_family=str(row["task_family"]),
                fst_minutes=float(row["best_available_minutes"]),
                fst_source=(str(row["best_available_source"]) if pd.notna(row["best_available_source"]) else None),
                estimation_instructions=ei,
                solution_walkthrough=(meta.get("solution_walkthrough") or None),
            )
        )
    if missing_ei:
        logger.warning(
            f"Dropped {len(missing_ei)}/{n_headline_in_parquet} headline tasks lacking "
            f"`estimation_instructions` (first 5: {missing_ei[:5]}). "
            "Recorded on dataset.dropped_task_ids_no_estimation_instructions."
        )

    # Outcomes matrix on the surviving headline tasks
    keep_ids = [t.task_id for t in tasks]
    mr_h = mr[mr["task_id"].isin(keep_ids)].copy()
    pivot = mr_h.pivot_table(
        index="alias", columns="task_id", values="score_binarized", aggfunc="first"
    )
    # Reindex so columns are always the headline-task order
    pivot = pivot.reindex(columns=keep_ids)

    all_models = sorted(pivot.index.tolist())
    sparse_models = [
        m for m in all_models if pivot.loc[m].notna().sum() < len(keep_ids)
    ]
    logger.info(f"Models in outcomes matrix: {len(all_models)} ({all_models})")
    logger.info(
        f"Models with sparse coverage (<{len(keep_ids)} tasks evaluated): "
        f"{[(m, int(pivot.loc[m].notna().sum())) for m in sparse_models]}"
    )

    if drop_models:
        present = [m for m in drop_models if m in pivot.index]
        if present:
            logger.info(f"Dropping models from outcomes matrix per config: {present}")
            pivot = pivot.drop(index=present)

    outcomes = LyptusOutcomes(frame=pivot)
    sha = read_repo_commit_sha(repo_dir)
    return LyptusDataset(
        tasks=tasks,
        outcomes=outcomes,
        repo_dir=repo_dir,
        commit_sha=sha,
        model_aliases_full=all_models,
        sparse_models=sparse_models,
        n_headline_in_parquet=n_headline_in_parquet,
        n_dropped_no_estimation_instructions=len(missing_ei),
        dropped_task_ids_no_estimation_instructions=missing_ei,
    )
