#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generic per-task inter-benchmark ground truth computation.

Computes P(model solves specific target task | model scores in source bin)
using a binary solve matrix for the target benchmark and a leaderboard for
the source benchmark.

This differs from compute_ground_truth.py (which uses overall score thresholds)
because we have per-task binary outcomes.

Usage:
    python compute_ground_truth_per_task.py \
        --source-leaderboard ../difficulty_estimation/benchmark_tasks/livebench_LCB_generation_leaderboard.json \
        --target-solve-matrix ../difficulty_estimation/benchmark_tasks/livebench_coding_completion_solve_matrix.json \
        --target-ordered ../difficulty_estimation/benchmark_tasks/livebench_coding_completion_ordered.yaml \
        --source-bins auto --n-source-bins 4 \
        --target-percentiles 30,40,50,60 \
        --output input_data/ground_truth/LCB_generation_to_coding_completion_gt.json
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_leaderboard(path: str) -> Tuple[str, Dict[str, float]]:
    """Load a leaderboard JSON -> (benchmark_name, {model: score})."""
    with open(path, "r") as f:
        data = json.load(f)
    bm_name = data.get("metadata", {}).get("benchmark_name", Path(path).stem)
    entries = data.get("results", data.get("models", []))
    return bm_name, {e["model"]: e["score"] for e in entries}


def load_solve_matrix(path: str) -> Dict[str, Dict[str, bool]]:
    """Load a solve matrix JSON -> {model: {question_id: bool}}."""
    with open(path, "r") as f:
        return json.load(f)


def load_ordered_task_ids(path: str) -> Tuple[str, List[str]]:
    """Load ordered YAML -> (benchmark_name, [question_id, ...] in order)."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    bm_name = Path(path).stem.replace("_ordered", "")
    task_ids = [t["question_id"] for t in data.get("tasks", [])]
    return bm_name, task_ids


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def compute_source_bins(
    scores: np.ndarray,
    config: Union[List[List[float]], str],
    n_bins: int = 4,
) -> List[List[float]]:
    if isinstance(config, str) and config.lower() == "auto":
        lo, hi = float(scores.min()), float(scores.max()) + 0.01
        edges = np.linspace(lo, hi, n_bins + 1)
        return [[float(edges[i]), float(edges[i + 1])] for i in range(n_bins)]
    return config


def assign_to_bins(
    scores: np.ndarray, bins: List[List[float]]
) -> np.ndarray:
    assignments = np.full(len(scores), -1, dtype=int)
    for bi, (lo, hi) in enumerate(bins):
        mask = (scores >= lo) & (scores < hi)
        assignments[mask] = bi
    # Include max score in last bin
    if bins:
        last_hi = bins[-1][1]
        assignments[np.abs(scores - last_hi) < 0.02] = len(bins) - 1
    return assignments


# ---------------------------------------------------------------------------
# Target percentile -> task mapping
# ---------------------------------------------------------------------------

def get_task_at_percentile(percentile: float, ordered_ids: List[str]) -> Tuple[int, str]:
    """Map a percentile (0-100) to a task index and question_id."""
    n = len(ordered_ids)
    idx = math.floor(n * percentile / 100.0)
    idx = max(0, min(idx, n - 1))
    return idx, ordered_ids[idx]


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_per_task_ground_truth(
    source_leaderboard_path: str,
    target_solve_matrix_path: str,
    target_ordered_path: str,
    source_bins_config: Union[List[List[float]], str] = "auto",
    n_source_bins: int = 4,
    target_percentiles: List[float] = None,
    min_sample_size: int = 3,
) -> Dict[str, Any]:
    """
    Compute inter-benchmark ground truth using per-task solve data.

    P(model solves target task T | model scores in source bin B)
    = (models in bin B that solved T) / (models in bin B)
    """
    # Load data
    src_name, src_scores = load_leaderboard(source_leaderboard_path)
    solve_matrix = load_solve_matrix(target_solve_matrix_path)
    tgt_name, ordered_ids = load_ordered_task_ids(target_ordered_path)

    # Inner join on model names
    common_models = sorted(set(src_scores.keys()) & set(solve_matrix.keys()))
    if not common_models:
        raise ValueError(
            f"No overlapping models between source leaderboard ({len(src_scores)} models) "
            f"and target solve matrix ({len(solve_matrix)} models)"
        )
    logger.info(f"Joined {len(common_models)} models ({src_name} -> {tgt_name})")

    model_names = np.array(common_models)
    primary_scores = np.array([src_scores[m] for m in common_models])

    # Compute source bins
    source_bins = compute_source_bins(primary_scores, source_bins_config, n_source_bins)
    bin_assignments = assign_to_bins(primary_scores, source_bins)

    # Default target percentiles
    if target_percentiles is None:
        target_percentiles = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]

    # Build bin metadata
    bin_metadata = []
    for bi, (lo, hi) in enumerate(source_bins):
        models_in = [model_names[k] for k in range(len(model_names)) if bin_assignments[k] == bi]
        bin_metadata.append({
            "bin_id": bi,
            "range": [lo, hi],
            "range_str": f"[{lo:.3f}, {hi:.3f})",
            "n_models": len(models_in),
            "models": list(models_in),
        })

    # Compute ground truth for each (source_bin, target_percentile)
    gt_results = []
    for bi, (lo, hi) in enumerate(source_bins):
        mask = bin_assignments == bi
        n_in_bin = int(mask.sum())
        models_in_bin = model_names[mask]

        for tp in target_percentiles:
            task_idx, task_id = get_task_at_percentile(tp, ordered_ids)

            if n_in_bin >= min_sample_size:
                n_solving = sum(
                    1 for m in models_in_bin
                    if solve_matrix.get(m, {}).get(task_id, False)
                )
                p_solve = n_solving / n_in_bin
                sufficient = True
            else:
                n_solving = sum(
                    1 for m in models_in_bin
                    if solve_matrix.get(m, {}).get(task_id, False)
                ) if n_in_bin > 0 else 0
                p_solve = float("nan")
                sufficient = False

            gt_results.append({
                "source_bin": bi,
                "source_bin_range": [lo, hi],
                "source_bin_range_str": f"[{lo:.3f}, {hi:.3f})",
                "target_percentile": tp,
                "target_task_index": task_idx,
                "target_task_id": task_id,
                "n_in_source_bin": n_in_bin,
                "n_solving_target": n_solving,
                "p_solve": p_solve,
                "sufficient_sample": sufficient,
            })

    output = {
        "metadata": {
            "source_benchmark": src_name,
            "target_benchmark": tgt_name,
            "n_source_bins": len(source_bins),
            "n_target_percentiles": len(target_percentiles),
            "n_models": len(common_models),
            "min_sample_size": min_sample_size,
            "source_score_range": [float(primary_scores.min()), float(primary_scores.max())],
            "n_target_tasks": len(ordered_ids),
            "total_predictions": len(gt_results),
            "sufficient_sample_count": sum(1 for r in gt_results if r["sufficient_sample"]),
            "method": "per_task_solve_matrix",
        },
        "source_bins": bin_metadata,
        "target_percentiles": target_percentiles,
        "ground_truth": gt_results,
    }

    # Print summary
    print(f"\n{'='*60}")
    print("Per-Task Inter-Benchmark Ground Truth Summary")
    print(f"{'='*60}")
    print(f"Source: {src_name} ({len(src_scores)} models in leaderboard)")
    print(f"Target: {tgt_name} ({len(solve_matrix)} models in solve matrix, {len(ordered_ids)} tasks)")
    print(f"Models after join: {len(common_models)}")
    print(f"Source bins: {len(source_bins)}")
    print(f"Target percentiles: {target_percentiles}")
    print(f"Total predictions: {len(gt_results)}")
    print(f"Sufficient sample (n>={min_sample_size}): {output['metadata']['sufficient_sample_count']}")
    print(f"\nSource bin distribution:")
    for bm in bin_metadata:
        print(f"  Bin {bm['bin_id']} {bm['range_str']}: {bm['n_models']} models")
    print(f"\nGround truth matrix:")
    for bi, (lo, hi) in enumerate(source_bins):
        row = [r for r in gt_results if r["source_bin"] == bi]
        vals = " | ".join(
            f"p{r['target_percentile']:.0f}={r['p_solve']:.3f}" if r["sufficient_sample"]
            else f"p{r['target_percentile']:.0f}=n/a"
            for r in row
        )
        print(f"  Bin {bi} [{lo:.3f},{hi:.3f}): {vals}")
    print(f"{'='*60}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-task inter-benchmark ground truth using solve matrices"
    )
    parser.add_argument(
        "--source-leaderboard", "-s", type=str, required=True,
        help="Path to source benchmark leaderboard JSON"
    )
    parser.add_argument(
        "--target-solve-matrix", "-m", type=str, required=True,
        help="Path to target benchmark solve matrix JSON"
    )
    parser.add_argument(
        "--target-ordered", "-t", type=str, required=True,
        help="Path to target benchmark ordered YAML (maps percentiles to task IDs)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output path for ground truth JSON"
    )
    parser.add_argument(
        "--source-bins", type=str, default="auto",
        help='Source bins: "auto" or JSON list like "[[0.1,0.3],[0.3,0.5]]"'
    )
    parser.add_argument(
        "--n-source-bins", type=int, default=4,
        help="Number of source bins for auto mode (default: 4)"
    )
    parser.add_argument(
        "--target-percentiles", type=str, default="20,30,40,50,60,70,80",
        help="Comma-separated target percentiles (default: 20,30,40,50,60,70,80)"
    )
    parser.add_argument(
        "--min-sample-size", type=int, default=3,
        help="Minimum models per bin for reliable statistics (default: 3)"
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    sb_config: Union[str, list] = "auto"
    if args.source_bins.lower() != "auto":
        sb_config = json.loads(args.source_bins)

    tp_list = [float(x) for x in args.target_percentiles.split(",")]

    result = compute_per_task_ground_truth(
        source_leaderboard_path=args.source_leaderboard,
        target_solve_matrix_path=args.target_solve_matrix,
        target_ordered_path=args.target_ordered,
        source_bins_config=sb_config,
        n_source_bins=args.n_source_bins,
        target_percentiles=tp_list,
        min_sample_size=args.min_sample_size,
    )

    if args.output:
        output_path = Path(args.output)
    else:
        src = result["metadata"]["source_benchmark"]
        tgt = result["metadata"]["target_benchmark"]
        output_path = Path(f"input_data/ground_truth/{src}_to_{tgt}_gt.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved ground truth to: {output_path}")
