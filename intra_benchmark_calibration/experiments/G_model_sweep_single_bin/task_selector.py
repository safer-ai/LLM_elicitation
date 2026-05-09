#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Anchor + easier-task selection and target-task sampling.

The forecaster's source-side prompt shows a per-bin capability profile for the
forecasted model M:

  - The model's empirical pass rate on bin i (over evaluated subset).
  - One anchor task (representative of bin i's mid-range pass rate, evaluated
    by M, never trivially {0%, 100%} unless the whole bin is).
  - `n_examples_per_source_bin` easier tasks from bin i, evaluated by M.

The target side picks K target tasks from bin j either deterministically (when
`explicit_target_tasks` is provided) or via stratified-by-log-FST sampling.

Cell admissibility (q3 caveat):
    For cell (i, j, M, t), require M to have a non-NaN outcome on:
      - the target task t, AND
      - every task that will be SHOWN in the source profile (anchor + easier
        tasks for every bin in source_bins_to_show).
    If anchor/easier selection cannot find any evaluated tasks for M in a shown
    bin, the cell is dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from intra_benchmark_calibration.binning import BinAssignment
from intra_benchmark_calibration.lyptus_data import LyptusDataset, LyptusTask, LyptusOutcomes

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceBinProfile:
    """One bin's slice of a model's capability profile, ready for the prompt."""

    bin_index: int
    pass_rate: float
    n_evaluated: int
    n_in_bin: int
    n_solved: int
    anchor: LyptusTask
    easier_tasks: List[LyptusTask]
    # Per-task binary outcome of M on the shown tasks (1.0 = solved, 0.0 = failed,
    # None = M not evaluated). By construction these are non-None whenever the
    # profile was admissible, but we keep Optional in case the heuristic is
    # called outside the cell-admissibility check.
    anchor_outcome: Optional[float] = None
    easier_outcomes: List[Optional[float]] = None  # type: ignore[assignment]


@dataclass(frozen=True)
class CellPlan:
    """Everything needed to assemble the prompt for one elicitation.

    `source_bin_i` is None when `source_profile_type == 'all_except_target'`
    (the i index is collapsed in that mode).
    """

    source_bin_i: Optional[int]
    target_bin_j: int
    forecasted_model: str
    target_task: LyptusTask
    target_outcome: float  # 0.0 or 1.0
    source_bins_to_show: List[int]
    profiles: List[SourceBinProfile]  # parallel to source_bins_to_show
    source_profile_type: str = "single_bin"  # single_bin | all_except_target | custom_subset

    @property
    def cell_id(self) -> str:
        if self.source_profile_type == "all_except_target":
            i_part = "ALL"
        elif self.source_profile_type == "custom_subset":
            i_part = f"i{self.source_bin_i}_cs[{','.join(map(str, self.source_bins_to_show))}]"
        else:
            i_part = f"i{self.source_bin_i}"
        return f"{i_part}_j{self.target_bin_j}_M={self.forecasted_model}_t={self.target_task.task_id}"


def _bin_task_ids_evaluated_by(
    bin_idx: int,
    bins: BinAssignment,
    task_ids: Sequence[str],
    model: str,
    outcomes: LyptusOutcomes,
) -> List[str]:
    in_bin = [tid for tid, b in zip(task_ids, bins.bin_index_per_task) if b == bin_idx]
    mask = outcomes.evaluated_mask(model, in_bin)
    return [tid for tid, ok in zip(in_bin, mask) if ok]


def _representativeness_score(task_pass_rate: float, bin_mean_pass_rate: float) -> float:
    """Higher is more representative. Closer to bin mean → higher."""
    return -abs(task_pass_rate - bin_mean_pass_rate)


def select_anchor_and_easier(
    bin_idx: int,
    bins: BinAssignment,
    dataset: LyptusDataset,
    model: str,
    n_easier: int,
) -> Optional[SourceBinProfile]:
    """
    Pick anchor + n_easier tasks from `bin_idx` that are evaluated by `model`.

    Heuristic:
      - Anchor: the task in the bin (evaluated by M, with pass rate not in {0, 1}
        across the FULL model panel — Jeff's caveat) whose pass rate across the
        FULL panel is closest to the bin's mean pass rate. Falls back to allowing
        {0, 1} pass-rate tasks if the bin has none in (0, 1).
      - Easier tasks: from the easier half of the bin by FST, the n_easier tasks
        with the highest representativeness score (ties broken by ascending FST).

    Returns None if M has no evaluated tasks in this bin.
    """
    all_ids = dataset.task_ids
    by_id = dataset.task_by_id
    outcomes = dataset.outcomes

    evaluated_in_bin = _bin_task_ids_evaluated_by(bin_idx, bins, all_ids, model, outcomes)
    n_in_bin = sum(1 for b in bins.bin_index_per_task if b == bin_idx)
    if not evaluated_in_bin:
        return None

    # Per-task pass rate across the FULL model panel (not just M) — for selection only
    full_pass = outcomes.frame[evaluated_in_bin].mean(axis=0, skipna=True)  # series indexed by task_id
    bin_mean = float(full_pass.mean())

    informative = [tid for tid in evaluated_in_bin if 0.0 < float(full_pass[tid]) < 1.0]
    candidates = informative if informative else list(evaluated_in_bin)

    # Anchor: most representative of bin mean
    anchor_id = max(candidates, key=lambda t: _representativeness_score(float(full_pass[t]), bin_mean))
    anchor = by_id[anchor_id]

    # Easier tasks: from the easier half of the bin (excluding the anchor)
    others = [tid for tid in evaluated_in_bin if tid != anchor_id]
    others.sort(key=lambda t: by_id[t].fst_minutes)  # ascending FST
    easier_pool = others[: max(1, len(others) // 2)] if len(others) > 1 else others
    easier_pool.sort(
        key=lambda t: (
            -_representativeness_score(float(full_pass[t]), bin_mean),
            by_id[t].fst_minutes,
        )
    )
    easier_ids = easier_pool[:n_easier]
    easier_tasks = [by_id[tid] for tid in easier_ids]

    # Pass rate for M on the evaluated subset of this bin
    pr = outcomes.pass_rate(model, evaluated_in_bin)
    assert pr is not None  # by construction

    anchor_outcome = outcomes.outcome(model, anchor.task_id)
    easier_outcomes = [outcomes.outcome(model, t.task_id) for t in easier_tasks]

    return SourceBinProfile(
        bin_index=bin_idx,
        pass_rate=pr["rate"],
        n_evaluated=pr["n_evaluated"],
        n_in_bin=n_in_bin,
        n_solved=pr["n_solved"],
        anchor=anchor,
        easier_tasks=easier_tasks,
        anchor_outcome=anchor_outcome,
        easier_outcomes=easier_outcomes,
    )


def sample_target_tasks(
    bin_idx_j: int,
    bins: BinAssignment,
    dataset: LyptusDataset,
    *,
    k: int,
    seed: int,
) -> List[LyptusTask]:
    """
    Stratified-by-log-FST sample of K target tasks from bin j.

    Strategy:
      - Sort bin's tasks by log10(FST), divide into K equal-count strata,
        pick one per stratum at random (seeded).
      - If K == 1, pick the median-FST task deterministically.
    """
    by_id = dataset.task_by_id
    in_bin = [tid for tid, b in zip(dataset.task_ids, bins.bin_index_per_task) if b == bin_idx_j]
    if not in_bin:
        return []

    in_bin.sort(key=lambda t: np.log10(by_id[t].fst_minutes))

    if k <= 1:
        return [by_id[in_bin[len(in_bin) // 2]]]

    rng = np.random.default_rng(seed)
    chunks = np.array_split(np.array(in_bin), k)
    return [by_id[str(rng.choice(c))] for c in chunks]


def build_cell_plans(
    *,
    bins: BinAssignment,
    dataset: LyptusDataset,
    forecasted_models: Sequence[str],
    source_bins_to_show: Union[List[int], str],
    n_examples_per_source_bin: int,
    n_target_tasks_per_cell: int,
    target_sampling_seed: int,
    explicit_target_tasks: Optional[Dict[int, List[str]]] = None,
    resample_anchors_per_target: bool = False,
) -> List[CellPlan]:
    """
    Enumerate all admissible cells, with three source-profile modes:

      `source_bins_to_show = []`          -> "single_bin" mode (default).
            Cells iterate over (i, j) for i != j; source profile shows only
            bin i. Cells = n_bins x (n_bins - 1) x n_models x K.

      `source_bins_to_show = "all_except_target"`  -> "all_except_target".
            Cells iterate over j only; source profile shows every bin except j.
            i is collapsed (CellPlan.source_bin_i = None).
            Cells = n_bins x n_models x K.

      `source_bins_to_show = [list of ints]`  -> "custom_subset".
            Cells iterate over (i, j) for i != j; source profile shows the
            supplied list regardless of i. Per-i prompts are identical for
            fixed j (only useful as temperature-stability sanity check).
            Cells = n_bins x (n_bins - 1) x n_models x K.

    Drops a cell when:
      - M has no evaluated tasks in any of the shown bins, OR
      - M has no outcome on the chosen target task t, OR
      - the target task happens to coincide with one of the shown anchor /
        easier tasks (avoids leaking the answer).
    """
    n_bins = bins.n_bins
    by_id = dataset.task_by_id
    plans: List[CellPlan] = []

    # Resolve mode + per-cell shown-bins selector
    if isinstance(source_bins_to_show, str):
        if source_bins_to_show != "all_except_target":
            raise ValueError(f"Unknown source_bins_to_show keyword: {source_bins_to_show!r}")
        mode = "all_except_target"
    elif isinstance(source_bins_to_show, list) and len(source_bins_to_show) == 0:
        mode = "single_bin"
    elif isinstance(source_bins_to_show, list):
        mode = "custom_subset"
        for b in source_bins_to_show:
            if not (0 <= int(b) < n_bins):
                raise ValueError(f"source_bins_to_show contains out-of-range bin {b} for n_bins={n_bins}")
    else:
        raise ValueError(f"source_bins_to_show must be a list or 'all_except_target', got {type(source_bins_to_show)}")

    def shown_bins_for(i: Optional[int], j: int) -> List[int]:
        if mode == "all_except_target":
            return [b for b in range(n_bins) if b != j]
        if mode == "single_bin":
            assert i is not None
            return [i]
        return [int(b) for b in source_bins_to_show]  # type: ignore[arg-type]

    explicit_target_tasks = explicit_target_tasks or {}

    def targets_for_bin(j: int) -> List[LyptusTask]:
        if j in explicit_target_tasks:
            return [by_id[tid] for tid in explicit_target_tasks[j] if tid in by_id]
        return sample_target_tasks(
            j, bins, dataset, k=n_target_tasks_per_cell, seed=target_sampling_seed + j
        )

    targets_cache: Dict[int, List[LyptusTask]] = {j: targets_for_bin(j) for j in range(n_bins)}

    # Iteration: (i, j) for single_bin / custom_subset; (j only) for all_except_target
    if mode == "all_except_target":
        ij_pairs: List[tuple] = [(None, j) for j in range(n_bins)]
    else:
        ij_pairs = [(i, j) for i in range(n_bins) for j in range(n_bins) if i != j]

    n_dropped_no_anchor = 0
    n_dropped_no_target_outcome = 0
    n_dropped_anchor_is_target = 0

    for (i, j) in ij_pairs:
        shown = shown_bins_for(i, j)
        for M in forecasted_models:
            base_profiles: Optional[List[SourceBinProfile]] = None
            if not resample_anchors_per_target:
                base_profiles = []
                bad = False
                for b in shown:
                    prof = select_anchor_and_easier(b, bins, dataset, M, n_examples_per_source_bin)
                    if prof is None:
                        bad = True
                        break
                    base_profiles.append(prof)
                if bad:
                    n_dropped_no_anchor += 1
                    continue

            for t in targets_cache[j]:
                out_t = dataset.outcomes.outcome(M, t.task_id)
                if out_t is None:
                    n_dropped_no_target_outcome += 1
                    continue

                if resample_anchors_per_target:
                    profiles = []
                    bad = False
                    for b in shown:
                        prof = select_anchor_and_easier(b, bins, dataset, M, n_examples_per_source_bin)
                        if prof is None:
                            bad = True
                            break
                        profiles.append(prof)
                    if bad:
                        n_dropped_no_anchor += 1
                        continue
                else:
                    profiles = base_profiles  # type: ignore[assignment]

                shown_task_ids = {p.anchor.task_id for p in profiles}
                for p in profiles:
                    shown_task_ids.update(et.task_id for et in p.easier_tasks)
                if t.task_id in shown_task_ids:
                    n_dropped_anchor_is_target += 1
                    continue

                plans.append(
                    CellPlan(
                        source_bin_i=i,
                        target_bin_j=j,
                        forecasted_model=M,
                        target_task=t,
                        target_outcome=out_t,
                        source_bins_to_show=list(shown),
                        profiles=profiles,
                        source_profile_type=mode,
                    )
                )

    logger.info(
        f"Built {len(plans)} cell plans (mode='{mode}'; "
        f"dropped: no_anchor={n_dropped_no_anchor}, "
        f"no_target_outcome={n_dropped_no_target_outcome}, "
        f"anchor_is_target={n_dropped_anchor_is_target})"
    )
    return plans
