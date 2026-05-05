#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Binning strategies over task FST (best_available_minutes).

Strategies:
  - 'equal_count' (default): equal number of tasks per bin (np.quantile-based edges).
  - 'equal_log_fst': equal-width bins in log10(FST).
  - 'explicit_edges': user-supplied edges in minutes (right-open intervals,
    last bin closed on both ends).

Returned object exposes per-task bin indices, edge values in minutes, and a
human-readable label per bin.

NB: bin edges are ALWAYS reported in minutes (not log-minutes) for clarity, even
when computed in log-space.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BinAssignment:
    """Output of binning."""

    edges_minutes: List[float]  # length n_bins+1, ascending
    bin_index_per_task: List[int]  # parallel to the input task list, in [0, n_bins-1]
    strategy: str
    n_bins: int

    def label(self, bin_idx: int) -> str:
        lo, hi = self.edges_minutes[bin_idx], self.edges_minutes[bin_idx + 1]
        return f"bin {bin_idx}: [{lo:.2f}, {hi:.2f}] min"

    def task_ids_per_bin(self, task_ids: Sequence[str]) -> List[List[str]]:
        out: List[List[str]] = [[] for _ in range(self.n_bins)]
        for tid, b in zip(task_ids, self.bin_index_per_task):
            out[b].append(tid)
        return out


def _assign_from_edges(values: np.ndarray, edges: np.ndarray) -> List[int]:
    """Right-open intervals, except last bin which is closed on the right."""
    n_bins = len(edges) - 1
    # np.searchsorted with side='right' gives idx such that edges[idx-1] <= v < edges[idx]
    raw = np.searchsorted(edges, values, side="right") - 1
    raw = np.clip(raw, 0, n_bins - 1)
    return raw.astype(int).tolist()


def compute_bins(
    fst_values: Sequence[float],
    *,
    n_bins: int,
    strategy: str = "equal_count",
    explicit_edges: Sequence[float] | None = None,
) -> BinAssignment:
    """Compute bin edges and per-task bin assignments."""
    fst = np.asarray(list(fst_values), dtype=float)
    if fst.ndim != 1 or fst.size == 0:
        raise ValueError("fst_values must be a non-empty 1-D sequence")
    if np.any(fst <= 0):
        raise ValueError("FST values must be strictly positive (log binning would fail)")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    strategy = strategy.lower().strip()
    if strategy == "equal_count":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(fst, qs).astype(float)
        # Nudge endpoints to guarantee every task is included
        edges[0] = min(edges[0], fst.min()) * 0.999999
        edges[-1] = max(edges[-1], fst.max()) * 1.000001
    elif strategy == "equal_log_fst":
        lo, hi = np.log10(fst.min()), np.log10(fst.max())
        edges = np.logspace(lo, hi, n_bins + 1).astype(float)
        edges[0] *= 0.999999
        edges[-1] *= 1.000001
    elif strategy == "explicit_edges":
        if not explicit_edges or len(explicit_edges) != n_bins + 1:
            raise ValueError(
                f"explicit_edges must have length n_bins+1 = {n_bins + 1}, "
                f"got {len(explicit_edges) if explicit_edges else 0}"
            )
        edges = np.asarray(list(explicit_edges), dtype=float)
        if not np.all(np.diff(edges) > 0):
            raise ValueError("explicit_edges must be strictly increasing")
    else:
        raise ValueError(
            f"Unknown bin strategy '{strategy}'. "
            "Choose one of: equal_count, equal_log_fst, explicit_edges"
        )

    bin_indices = _assign_from_edges(fst, edges)
    counts = np.bincount(bin_indices, minlength=n_bins)
    logger.info(f"Bin strategy='{strategy}', n_bins={n_bins}")
    logger.info(f"Edges (min): {[round(e, 3) for e in edges]}")
    logger.info(f"Counts per bin: {counts.tolist()}")

    return BinAssignment(
        edges_minutes=edges.tolist(),
        bin_index_per_task=bin_indices,
        strategy=strategy,
        n_bins=n_bins,
    )
