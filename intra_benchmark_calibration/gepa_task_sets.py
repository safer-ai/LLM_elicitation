#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seeded task-set selection for the GEPA forecaster-optimisation experiment.

Builds the four disjoint task sets from the Lyptus data and writes them to a
JSON manifest so a GEPA run is fully reproducible and restartable:

  - train pool   (per bin; targets resampled each GEPA iteration)
  - val set      (per bin, fixed for the whole run)
  - finalist set (per bin, fixed; used only for end-of-run re-ranking)
  - test set     (ALL usable CVEBench+CyberGym tasks; held-out benchmarks)

Design constraints (see the GEPA experiment design summary):
  - Mandatory 5/2 benchmark split: train/val/finalist come only from
    CyBashBench, NL2Bash, InterCode-CTF, NYUCTF, CyBench; the test set is
    CVEBench + CyberGym.
  - Binning uses FIXED explicit log10-minute edges (NOT per-subset
    quantiles), so train and test are binned consistently. Intervals are
    right-closed (`binning.compute_bins_right_closed`): tasks at exactly 60
    or 180 min belong to the lower bin and the 2160-min task lands in bin 4,
    matching the design-table allocation (54/52/40/18/11 train-side).
  - Non-uniform per-bin val/finalist reservations (option a): 5/5/5/3/3,
    leaving a rotatable train pool even in the sparse hard bins.
  - The test count is NOT hard-coded: it is whatever remains after the
    usability filter (`has estimation_instructions`), applied inside
    `load_lyptus_dataset`.

Usage:
    python intra_benchmark_calibration/gepa_task_sets.py \\
        --lyptus-repo ~/gitrepos/cyber-task-horizons-data \\
        --seed 42 --output gepa_task_manifest.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from intra_benchmark_calibration.binning import BinAssignment, compute_bins_right_closed  # noqa: E402
from intra_benchmark_calibration.config import DEFAULT_DROP_MODELS  # noqa: E402
from intra_benchmark_calibration.lyptus_data import LyptusDataset, load_lyptus_dataset  # noqa: E402

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = 1

# 5/2 benchmark split (canonical parquet `task_family` values use underscores).
TRAIN_FAMILIES = ("cybashbench", "nl2bash", "intercode_ctf", "nyuctf", "cybench")
TEST_FAMILIES = ("cvebench", "cybergym")

# Fixed explicit FST bin edges in minutes = 10^[-0.34, 0.45, 1.11, 1.78, 2.26, 3.33].
# Derived from the source-coverage analysis that produced the option-(a)
# allocation table (train-side per-bin availability 54/52/40/18/11).
DEFAULT_EDGES_MINUTES = (0.46, 2.81, 12.82, 60.0, 180.0, 2160.0)

# Option (a): smaller reservations in the sparse hard bins.
DEFAULT_VAL_TASKS_PER_BIN = {0: 5, 1: 5, 2: 5, 3: 3, 4: 3}
DEFAULT_FINALIST_TASKS_PER_BIN = {0: 5, 1: 5, 2: 5, 3: 3, 4: 3}


def canonical_family(task_family: str) -> str:
    return task_family.strip().lower().replace("-", "_")


def build_task_manifest(
    dataset: LyptusDataset,
    *,
    seed: int,
    edges_minutes: Sequence[float] = DEFAULT_EDGES_MINUTES,
    val_tasks_per_bin: Optional[Dict[int, int]] = None,
    finalist_tasks_per_bin: Optional[Dict[int, int]] = None,
    train_families: Sequence[str] = TRAIN_FAMILIES,
    test_families: Sequence[str] = TEST_FAMILIES,
) -> dict:
    """Build the seeded train-pool/val/finalist/test manifest.

    The dataset must be the FULL usable pool (all benchmarks, post
    `estimation_instructions` filter); binning is computed on it with the
    fixed explicit edges before the benchmark split, so bin membership is
    identical on both sides of the split.
    """
    val_tasks_per_bin = dict(val_tasks_per_bin or DEFAULT_VAL_TASKS_PER_BIN)
    finalist_tasks_per_bin = dict(finalist_tasks_per_bin or DEFAULT_FINALIST_TASKS_PER_BIN)
    n_bins = len(edges_minutes) - 1
    if set(val_tasks_per_bin) != set(range(n_bins)) or set(finalist_tasks_per_bin) != set(range(n_bins)):
        raise ValueError(f"val/finalist reservations must cover bins 0..{n_bins - 1}")

    train_set = {canonical_family(f) for f in train_families}
    test_set = {canonical_family(f) for f in test_families}
    families_seen = {canonical_family(t.task_family) for t in dataset.tasks}
    unknown = families_seen - train_set - test_set
    if unknown:
        raise ValueError(f"Dataset contains families outside the 5/2 split: {sorted(unknown)}")

    bins: BinAssignment = compute_bins_right_closed(
        dataset.fst_array(), explicit_edges=list(edges_minutes)
    )

    # Partition task ids per bin and per split side. Sort within each bin so
    # the seeded sampling is independent of dataset load order.
    train_by_bin: List[List[str]] = [[] for _ in range(n_bins)]
    test_by_bin: List[List[str]] = [[] for _ in range(n_bins)]
    for task, b in zip(dataset.tasks, bins.bin_index_per_task):
        fam = canonical_family(task.task_family)
        (train_by_bin if fam in train_set else test_by_bin)[b].append(task.task_id)
    for b in range(n_bins):
        train_by_bin[b].sort()
        test_by_bin[b].sort()

    rng = random.Random(seed)
    bins_out: Dict[str, dict] = {}
    for b in range(n_bins):
        available = list(train_by_bin[b])
        n_val = val_tasks_per_bin[b]
        n_fin = finalist_tasks_per_bin[b]
        if len(available) <= n_val + n_fin:
            raise ValueError(
                f"Bin {b}: only {len(available)} train-benchmark tasks available; "
                f"cannot reserve {n_val} val + {n_fin} finalist and keep a non-empty train pool."
            )
        val_ids = sorted(rng.sample(available, n_val))
        remaining = [t for t in available if t not in set(val_ids)]
        finalist_ids = sorted(rng.sample(remaining, n_fin))
        pool_ids = sorted(t for t in remaining if t not in set(finalist_ids))

        assert not (set(val_ids) & set(finalist_ids))
        assert not (set(val_ids) & set(pool_ids))
        assert not (set(finalist_ids) & set(pool_ids))
        assert val_ids and finalist_ids and pool_ids, f"Bin {b}: empty set after reservation"
        assert len(val_ids) + len(finalist_ids) + len(pool_ids) == len(available)

        bins_out[str(b)] = {
            "n_available": len(available),
            "val": val_ids,
            "finalist": finalist_ids,
            "train_pool": pool_ids,
        }

    test_ids = sorted(tid for ids in test_by_bin for tid in ids)
    test_bin_distribution = {str(b): len(test_by_bin[b]) for b in range(n_bins)}
    # The test benchmarks live almost entirely in bins 2-4; make that visible.
    logger.warning(
        "Test set (%s): %d usable tasks with bin distribution %s — "
        "bins 0-1 are (near-)empty by construction; the transfer claim is joint "
        "domain + hard-difficulty transfer.",
        "+".join(sorted(test_set)),
        len(test_ids),
        test_bin_distribution,
    )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": seed,
        "lyptus_repo_dir": str(dataset.repo_dir),
        "lyptus_commit_sha": dataset.commit_sha,
        "n_usable_tasks": len(dataset.tasks),
        "n_dropped_no_estimation_instructions": dataset.n_dropped_no_estimation_instructions,
        "bin_edges_minutes": list(edges_minutes),
        "bin_strategy": "explicit_edges_right_closed",
        "train_families": sorted(train_set),
        "test_families": sorted(test_set),
        "val_tasks_per_bin": {str(k): v for k, v in val_tasks_per_bin.items()},
        "finalist_tasks_per_bin": {str(k): v for k, v in finalist_tasks_per_bin.items()},
        "bins": bins_out,
        "test": {
            "task_ids": test_ids,
            "bin_distribution": test_bin_distribution,
        },
        "counts": {
            "train_pool": sum(len(v["train_pool"]) for v in bins_out.values()),
            "val": sum(len(v["val"]) for v in bins_out.values()),
            "finalist": sum(len(v["finalist"]) for v in bins_out.values()),
            "test": len(test_ids),
        },
    }
    return manifest


def write_manifest(manifest: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info(f"Wrote task manifest to {path}")


def load_manifest(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Manifest schema version {manifest.get('schema_version')} != {MANIFEST_SCHEMA_VERSION}"
        )
    return manifest


def cli() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lyptus-repo", required=True, help="Path to the cyber-task-horizons-data checkout")
    p.add_argument("--seed", type=int, required=True, help="RNG seed for the reservations")
    p.add_argument("--output", required=True, help="Output manifest JSON path")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    dataset = load_lyptus_dataset(
        Path(args.lyptus_repo).expanduser(), drop_models=DEFAULT_DROP_MODELS
    )
    manifest = build_task_manifest(dataset, seed=args.seed)
    write_manifest(manifest, Path(args.output))

    for b, entry in manifest["bins"].items():
        logger.info(
            f"Bin {b}: available={entry['n_available']} val={len(entry['val'])} "
            f"finalist={len(entry['finalist'])} train_pool={len(entry['train_pool'])}"
        )
    logger.info(f"Totals: {manifest['counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(cli())
