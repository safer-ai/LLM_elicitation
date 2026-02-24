#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data loading utilities for intra-benchmark calibration experiments.

This module provides functions to load ground truth data from pre-computed
conditional probability files.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


def load_ground_truth(benchmark_name: str, n_bins: int, ground_truth_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Load ground truth conditional probability data for a benchmark.

    Args:
        benchmark_name: Name of the benchmark (e.g., "cybench")
        n_bins: Number of bins used to compute ground truth (e.g., 4)
        ground_truth_dir: Directory containing ground truth files (if None, uses default path)

    Returns:
        Dictionary containing:
            - metadata: Dict with n_bins, n_models, score_range, etc.
            - bin_info: List of dicts with bin details (range, models, counts)
            - ground_truth: List of dicts with (i,j) pairs and P(j|i) values
                Only includes pairs where sufficient_sample == True
        Returns None if file not found or parsing fails.

    Example usage:
        >>> gt_data = load_ground_truth("cybench", 4, Path("input_data/ground_truth"))
        >>> if gt_data:
        ...     for pair in gt_data['ground_truth']:
        ...         print(f"P({pair['bin_j']}|{pair['bin_i']}) = {pair['p_j_given_i']}")
    """
    # Construct path to ground truth file
    if ground_truth_dir is None:
        ground_truth_dir = Path.cwd() / "input_data" / "ground_truth"
    
    filename = f"{benchmark_name}_ground_truth_n{n_bins}.json"
    filepath = ground_truth_dir / filename

    if not filepath.exists():
        logger.error(f"Ground truth file not found: {filepath}")
        logger.error(f"Expected path: input_data/benchmark/scores/{filename}")
        return None

    logger.info(f"Loading ground truth data from: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file {filepath}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to read file {filepath}: {e}")
        return None

    # Validate structure
    if not isinstance(data, dict):
        logger.error(f"Invalid ground truth file format: expected dict, got {type(data)}")
        return None

    required_keys = ['metadata', 'bin_info', 'ground_truth']
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        logger.error(f"Missing required keys in ground truth file: {missing_keys}")
        return None

    # Extract metadata
    metadata = data.get('metadata', {})
    bin_info = data.get('bin_info', [])
    ground_truth_all = data.get('ground_truth', [])

    # Filter to only include pairs with sufficient_sample == True
    ground_truth_filtered = [
        pair for pair in ground_truth_all
        if pair.get('sufficient_sample', False) is True
    ]

    logger.info("Ground truth loaded successfully:")
    logger.info(f"  - Benchmark: {benchmark_name}")
    logger.info(f"  - Number of bins: {metadata.get('n_bins', 'N/A')}")
    logger.info(f"  - Number of models: {metadata.get('n_models', 'N/A')}")
    logger.info(f"  - Score range: {metadata.get('score_range', 'N/A')}")
    logger.info(f"  - Total (i,j) pairs in file: {len(ground_truth_all)}")
    logger.info(f"  - Pairs with sufficient sample: {len(ground_truth_filtered)}")

    # Validate n_bins matches between config and file
    file_n_bins = metadata.get('n_bins')
    if n_bins != file_n_bins:
        logger.error(f"n_bins MISMATCH: Config specifies n_bins={n_bins} but "
                     f"file '{filename}' contains n_bins={file_n_bins}")
        logger.error("This likely means the file was renamed incorrectly, is corrupted, "
                     "or config points to wrong n_bins value.")
        logger.error(f"Expected file: {benchmark_name}_ground_truth_n{n_bins}.json with internal n_bins={n_bins}")
        logger.error(f"Actual file: {filename} with internal n_bins={file_n_bins}")
        return None

    if not ground_truth_filtered:
        logger.warning("No ground truth pairs with sufficient sample found!")

    # Return structured data with filtered ground truth
    return {
        'metadata': metadata,
        'bin_info': bin_info,
        'ground_truth': ground_truth_filtered  # Only sufficient sample pairs
    }


def get_bin_info_by_id(ground_truth_data: Dict[str, Any], bin_id: int) -> Optional[Dict[str, Any]]:
    """
    Get bin information for a specific bin ID.

    Args:
        ground_truth_data: Ground truth data dict from load_ground_truth()
        bin_id: The bin ID to look up (0-indexed)

    Returns:
        Dict with bin details (range, n_models_in_bin, n_models_reaching_bin, models)
        Returns None if bin_id not found.
    """
    bin_info_list = ground_truth_data.get('bin_info', [])

    for bin_info in bin_info_list:
        if bin_info.get('bin_id') == bin_id:
            return bin_info

    logger.warning(f"Bin ID {bin_id} not found in ground truth data")
    return None


def get_ground_truth_pair(ground_truth_data: Dict[str, Any], bin_i: int, bin_j: int) -> Optional[Dict[str, Any]]:
    """
    Get ground truth data for a specific (i,j) bin pair.

    Args:
        ground_truth_data: Ground truth data dict from load_ground_truth()
        bin_i: Source bin ID
        bin_j: Target bin ID

    Returns:
        Dict with p_j_given_i, n_reaching_i, n_reaching_j, bin ranges, etc.
        Returns None if pair not found or doesn't have sufficient sample.
    """
    ground_truth_list = ground_truth_data.get('ground_truth', [])

    for pair in ground_truth_list:
        if pair.get('bin_i') == bin_i and pair.get('bin_j') == bin_j:
            return pair

    logger.warning(f"Ground truth pair ({bin_i}, {bin_j}) not found or insufficient sample")
    return None


def validate_ground_truth_data(ground_truth_data: Dict[str, Any]) -> bool:
    """
    Validate that ground truth data has expected structure and reasonable values.

    Args:
        ground_truth_data: Ground truth data dict from load_ground_truth()

    Returns:
        True if validation passes, False otherwise
    """
    if not ground_truth_data:
        logger.error("Ground truth data is None or empty")
        return False

    # Check metadata
    metadata = ground_truth_data.get('metadata', {})
    if not metadata:
        logger.error("Missing metadata in ground truth data")
        return False

    n_bins = metadata.get('n_bins')
    if not isinstance(n_bins, int) or n_bins <= 0:
        logger.error(f"Invalid n_bins in metadata: {n_bins}")
        return False

    # Check bin_info
    bin_info = ground_truth_data.get('bin_info', [])
    if len(bin_info) != n_bins:
        logger.error(f"bin_info length ({len(bin_info)}) doesn't match n_bins ({n_bins})")
        return False

    # Check ground_truth pairs
    ground_truth = ground_truth_data.get('ground_truth', [])
    if not ground_truth:
        logger.warning("No ground truth pairs with sufficient sample")
        return False

    # Validate each pair
    for pair in ground_truth:
        required_fields = ['bin_i', 'bin_j', 'p_j_given_i', 'n_reaching_i', 'n_reaching_j']
        missing = [f for f in required_fields if f not in pair]
        if missing:
            logger.error(f"Ground truth pair missing fields: {missing}")
            return False

        # Check probability is in valid range
        p_j_given_i = pair.get('p_j_given_i')
        if not isinstance(p_j_given_i, (int, float)) or not (0 <= p_j_given_i <= 1):
            logger.error(f"Invalid p_j_given_i value: {p_j_given_i}")
            return False

        # Check bin IDs are valid
        bin_i = pair.get('bin_i')
        bin_j = pair.get('bin_j')
        if not isinstance(bin_i, int) or not isinstance(bin_j, int):
            logger.error(f"Invalid bin IDs: bin_i={bin_i}, bin_j={bin_j}")
            return False

        if not (0 <= bin_i < n_bins) or not (0 <= bin_j < n_bins):
            logger.error(f"Bin IDs out of range: bin_i={bin_i}, bin_j={bin_j}, n_bins={n_bins}")
            return False

        if bin_i >= bin_j:
            logger.error(f"Invalid pair: bin_i ({bin_i}) must be < bin_j ({bin_j})")
            return False

    logger.info("Ground truth data validation passed")
    return True


if __name__ == "__main__":
    # Test the data loader
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("Testing ground truth data loader...")
    print("="*60)

    # Test loading
    gt_data = load_ground_truth("cybench", 4)

    if gt_data:
        print("\n✓ Successfully loaded ground truth data")

        # Validate
        if validate_ground_truth_data(gt_data):
            print("✓ Validation passed")
        else:
            print("✗ Validation failed")

        # Print summary
        print("\nGround truth pairs:")
        for pair in gt_data['ground_truth']:
            print(f"  Bin {pair['bin_i']} → {pair['bin_j']}: "
                  f"P(j|i) = {pair['p_j_given_i']:.3f} "
                  f"({pair['n_reaching_j']}/{pair['n_reaching_i']} models)")

        # Test helper functions
        print("\nTesting helper functions:")
        bin_info = get_bin_info_by_id(gt_data, 0)
        if bin_info:
            print(f"  Bin 0 range: {bin_info['range']}")

        pair_info = get_ground_truth_pair(gt_data, 0, 1)
        if pair_info:
            print(f"  P(1|0) = {pair_info['p_j_given_i']:.3f}")
    else:
        print("✗ Failed to load ground truth data")

    print("="*60)
