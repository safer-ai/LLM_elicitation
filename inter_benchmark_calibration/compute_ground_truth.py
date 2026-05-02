#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute inter-benchmark ground truth probabilities.

Loads per-benchmark score files, joins by model name, bins models by the
primary source benchmark score, and computes P(solve target task at percentile X
| model in source bin Y) for each (source_bin, target_percentile) pair.
"""

import json
import numpy as np
from pathlib import Path
import argparse
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def load_score_file(score_path: str) -> Dict[str, float]:
    """
    Load a per-benchmark score file and return a dict mapping model name -> score.

    Expected format:
    {
      "benchmark_name": "cybench",
      "models": [
        {"model": "GPT-4o", "score": 12.5},
        ...
      ]
    }
    """
    with open(score_path, 'r') as f:
        data = json.load(f)

    models = data.get('models', [])
    return {m['model']: m['score'] for m in models}


def join_scores(
    source_score_files: List[str],
    target_score_file: str
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Join multiple benchmark score files by model name (inner join).

    Returns:
        Tuple of (joined_models, source_benchmark_names) where joined_models
        is a list of dicts with keys: model, source_scores (list), target_score.
    """
    source_scores_list = []
    source_names = []
    for sf in source_score_files:
        with open(sf, 'r') as f:
            data = json.load(f)
        source_names.append(data.get('benchmark_name', Path(sf).stem))
        source_scores_list.append({m['model']: m['score'] for m in data.get('models', [])})

    with open(target_score_file, 'r') as f:
        tgt_data = json.load(f)
    target_scores = {m['model']: m['score'] for m in tgt_data.get('models', [])}

    # Inner join: only models present in ALL files
    common_models = set(source_scores_list[0].keys())
    for ss in source_scores_list[1:]:
        common_models &= set(ss.keys())
    common_models &= set(target_scores.keys())

    if not common_models:
        logger.warning("No models found in common across all score files!")

    joined = []
    for model_name in sorted(common_models):
        entry = {
            'model': model_name,
            'source_scores': [ss[model_name] for ss in source_scores_list],
            'target_score': target_scores[model_name]
        }
        joined.append(entry)

    logger.info(f"Joined {len(joined)} models across {len(source_score_files)} source + 1 target benchmark(s)")
    return joined, source_names


def compute_source_bins(
    primary_scores: np.ndarray,
    source_bins_config: Union[List[List[float]], str],
    n_source_bins: int = 4
) -> List[List[float]]:
    """
    Compute source bin boundaries.

    If source_bins_config is "auto", creates n_source_bins equal-width bins from
    min to max of primary_scores. Otherwise uses the explicit list.
    """
    if isinstance(source_bins_config, str) and source_bins_config.lower() == 'auto':
        lo = float(primary_scores.min())
        hi = float(primary_scores.max()) + 0.01
        edges = np.linspace(lo, hi, n_source_bins + 1)
        bins = [[float(edges[i]), float(edges[i + 1])] for i in range(n_source_bins)]
        logger.info(f"Auto source bins ({n_source_bins}): {bins}")
        return bins
    else:
        logger.info(f"Using explicit source bins: {source_bins_config}")
        return source_bins_config


def compute_target_percentiles(
    target_scores: np.ndarray,
    target_percentiles_config: Union[List[float], str],
    n_target_percentiles: int = 5
) -> List[float]:
    """
    Compute target percentile thresholds.

    If "auto", creates n_target_percentiles evenly spaced points across the
    target score range. Otherwise uses the explicit list.
    """
    if isinstance(target_percentiles_config, str) and target_percentiles_config.lower() == 'auto':
        lo = float(target_scores.min())
        hi = float(target_scores.max())
        pcts = np.linspace(lo, hi, n_target_percentiles + 2)[1:-1]  # exclude endpoints
        pcts = [round(float(p), 1) for p in pcts]
        logger.info(f"Auto target percentiles ({n_target_percentiles}): {pcts}")
        return pcts
    else:
        logger.info(f"Using explicit target percentiles: {target_percentiles_config}")
        return [float(p) for p in target_percentiles_config]


def compute_inter_benchmark_ground_truth(
    source_score_files: List[str],
    target_score_file: str,
    source_bins_config: Union[List[List[float]], str] = "auto",
    n_source_bins: int = 4,
    target_percentiles_config: Union[List[float], str] = "auto",
    n_target_percentiles: int = 5,
    min_sample_size: int = 3
) -> Dict[str, Any]:
    """
    Compute inter-benchmark ground truth: P(target_score >= percentile | primary source bin).

    Models are binned by their score on the PRIMARY (first) source benchmark only.
    For each (source_bin, target_percentile) pair, P(solve) is the fraction of models
    in the source bin whose target score meets or exceeds the target percentile.

    Args:
        source_score_files: List of paths to source benchmark score JSONs (first = primary)
        target_score_file: Path to target benchmark score JSON
        source_bins_config: Explicit [[lo,hi],...] or "auto"
        n_source_bins: Number of bins for auto mode
        target_percentiles_config: Explicit [p1, p2, ...] or "auto"
        n_target_percentiles: Number of percentiles for auto mode
        min_sample_size: Minimum models in a bin for reliable statistics

    Returns:
        Dict with metadata, bin_info, target_percentiles, and ground_truth entries.
    """
    joined, source_names = join_scores(source_score_files, target_score_file)

    if not joined:
        raise ValueError("No overlapping models found between score files")

    primary_scores = np.array([m['source_scores'][0] for m in joined])
    target_scores = np.array([m['target_score'] for m in joined])
    model_names = [m['model'] for m in joined]

    # Compute bins and percentiles
    source_bins = compute_source_bins(primary_scores, source_bins_config, n_source_bins)
    target_pcts = compute_target_percentiles(target_scores, target_percentiles_config, n_target_percentiles)

    # Assign models to source bins (by primary source score)
    bin_assignments = []
    for score in primary_scores:
        assigned = -1
        for bi, (lo, hi) in enumerate(source_bins):
            if lo <= score < hi:
                assigned = bi
                break
        # Include max score in last bin
        if assigned == -1 and len(source_bins) > 0:
            last_lo, last_hi = source_bins[-1]
            if abs(score - last_hi) < 0.02:
                assigned = len(source_bins) - 1
        bin_assignments.append(assigned)

    bin_assignments = np.array(bin_assignments)

    # Build bin metadata
    bin_metadata = []
    for bi, (lo, hi) in enumerate(source_bins):
        models_in_bin = [model_names[k] for k in range(len(model_names)) if bin_assignments[k] == bi]
        bin_metadata.append({
            'bin_id': bi,
            'range': [lo, hi],
            'range_str': f"[{lo:.1f}, {hi:.1f})",
            'n_models': len(models_in_bin),
            'models': models_in_bin
        })

    # Compute ground truth for each (source_bin, target_percentile)
    gt_results = []
    for bi, (lo, hi) in enumerate(source_bins):
        mask = bin_assignments == bi
        n_in_bin = int(mask.sum())

        for tp in target_pcts:
            if n_in_bin >= min_sample_size:
                n_solving = int((target_scores[mask] >= tp).sum())
                p_solve = n_solving / n_in_bin
                sufficient = True
            else:
                n_solving = int((target_scores[mask] >= tp).sum()) if n_in_bin > 0 else 0
                p_solve = float('nan')
                sufficient = False

            gt_results.append({
                'source_bin': bi,
                'source_bin_range': [lo, hi],
                'source_bin_range_str': f"[{lo:.1f}, {hi:.1f})",
                'target_percentile': tp,
                'n_in_source_bin': n_in_bin,
                'n_solving_target': n_solving,
                'p_solve': p_solve,
                'sufficient_sample': sufficient
            })

    with open(target_score_file, 'r') as f:
        tgt_bm_name = json.load(f).get('benchmark_name', Path(target_score_file).stem)

    output = {
        'metadata': {
            'source_benchmarks': source_names,
            'primary_source': source_names[0],
            'target_benchmark': tgt_bm_name,
            'n_source_bins': len(source_bins),
            'n_target_percentiles': len(target_pcts),
            'n_models': len(joined),
            'min_sample_size': min_sample_size,
            'source_score_range': [float(primary_scores.min()), float(primary_scores.max())],
            'target_score_range': [float(target_scores.min()), float(target_scores.max())],
            'total_predictions': len(gt_results),
            'sufficient_sample_count': sum(1 for r in gt_results if r['sufficient_sample']),
            'note': 'Models binned by PRIMARY source benchmark score only. Other source scores are context for the LLM.'
        },
        'source_bins': bin_metadata,
        'target_percentiles': target_pcts,
        'ground_truth': gt_results
    }

    # Print summary
    print(f"\n{'='*60}")
    print("Inter-Benchmark Ground Truth Summary")
    print(f"{'='*60}")
    print(f"Source benchmarks: {source_names}")
    print(f"Target benchmark: {tgt_bm_name}")
    print(f"Models (after join): {len(joined)}")
    print(f"Source bins: {len(source_bins)}")
    print(f"Target percentiles: {target_pcts}")
    print(f"Total predictions: {len(gt_results)}")
    print(f"With sufficient sample (n>={min_sample_size}): {output['metadata']['sufficient_sample_count']}")
    print(f"\nSource bin distribution:")
    for bm in bin_metadata:
        print(f"  Bin {bm['bin_id']} {bm['range_str']}: {bm['n_models']} models -- {bm['models']}")
    print(f"{'='*60}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute inter-benchmark ground truth probabilities')
    parser.add_argument('--source-scores', '-s', type=str, nargs='+', required=True,
                        help='Path(s) to source benchmark score JSON files (first = primary)')
    parser.add_argument('--target-scores', '-t', type=str, required=True,
                        help='Path to target benchmark score JSON file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for ground truth JSON')
    parser.add_argument('--source-bins', type=str, default='auto',
                        help='Source bins: "auto" or JSON list like "[[5,10],[10,15]]"')
    parser.add_argument('--n-source-bins', type=int, default=4,
                        help='Number of source bins for auto mode (default: 4)')
    parser.add_argument('--target-percentiles', type=str, default='auto',
                        help='Target percentiles: "auto" or comma-separated like "30,40,50,60"')
    parser.add_argument('--n-target-percentiles', type=int, default=5,
                        help='Number of target percentiles for auto mode (default: 5)')
    parser.add_argument('--min-sample-size', type=int, default=3,
                        help='Minimum models per bin for reliable statistics (default: 3)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    # Parse source bins
    if args.source_bins.lower() == 'auto':
        sb_config = 'auto'
    else:
        sb_config = json.loads(args.source_bins)

    # Parse target percentiles
    if args.target_percentiles.lower() == 'auto':
        tp_config = 'auto'
    else:
        tp_config = [float(x) for x in args.target_percentiles.split(',')]

    result = compute_inter_benchmark_ground_truth(
        source_score_files=args.source_scores,
        target_score_file=args.target_scores,
        source_bins_config=sb_config,
        n_source_bins=args.n_source_bins,
        target_percentiles_config=tp_config,
        n_target_percentiles=args.n_target_percentiles,
        min_sample_size=args.min_sample_size
    )

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        src_name = result['metadata']['primary_source']
        tgt_name = result['metadata']['target_benchmark']
        output_path = Path(f"input_data/ground_truth/{src_name}_to_{tgt_name}_gt.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved ground truth to: {output_path}")
