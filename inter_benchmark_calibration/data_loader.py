#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data loading utilities for inter-benchmark calibration experiments.

Loads ground truth data from pre-computed inter-benchmark ground truth files.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_ground_truth(ground_truth_file: Path) -> Optional[Dict[str, Any]]:
    """
    Load inter-benchmark ground truth data.

    Args:
        ground_truth_file: Path to the ground truth JSON file

    Returns:
        Dictionary with metadata, source_bins, target_percentiles, ground_truth entries.
        ground_truth entries are filtered to only those with sufficient_sample == True.
        Returns None on failure.
    """
    gt_path = Path(ground_truth_file)

    if not gt_path.exists():
        logger.error(f"Ground truth file not found: {gt_path}")
        return None

    logger.info(f"Loading ground truth from: {gt_path}")

    try:
        with open(gt_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        return None

    if not isinstance(data, dict):
        logger.error(f"Invalid format: expected dict, got {type(data)}")
        return None

    required_keys = ['metadata', 'source_bins', 'target_percentiles', 'ground_truth']
    missing = [k for k in required_keys if k not in data]
    if missing:
        logger.error(f"Missing required keys: {missing}")
        return None

    # Filter to sufficient sample only
    gt_all = data.get('ground_truth', [])
    gt_filtered = [e for e in gt_all if e.get('sufficient_sample', False)]

    metadata = data['metadata']
    logger.info("Ground truth loaded:")
    logger.info(f"  - Source: {metadata.get('primary_source', 'N/A')}")
    logger.info(f"  - Target: {metadata.get('target_benchmark', 'N/A')}")
    logger.info(f"  - Models: {metadata.get('n_models', 'N/A')}")
    logger.info(f"  - Total entries: {len(gt_all)}")
    logger.info(f"  - Sufficient sample: {len(gt_filtered)}")

    return {
        'metadata': metadata,
        'source_bins': data['source_bins'],
        'target_percentiles': data['target_percentiles'],
        'ground_truth': gt_filtered
    }


def validate_ground_truth_data(ground_truth_data: Dict[str, Any]) -> bool:
    """Validate that ground truth data has expected structure and reasonable values."""
    if not ground_truth_data:
        logger.error("Ground truth data is None or empty")
        return False

    metadata = ground_truth_data.get('metadata', {})
    if not metadata:
        logger.error("Missing metadata")
        return False

    gt_entries = ground_truth_data.get('ground_truth', [])
    if not gt_entries:
        logger.warning("No ground truth entries with sufficient sample")
        return False

    for entry in gt_entries:
        required = ['source_bin', 'target_percentile', 'p_solve']
        missing = [f for f in required if f not in entry]
        if missing:
            logger.error(f"Ground truth entry missing fields: {missing}")
            return False

        p = entry.get('p_solve')
        if not isinstance(p, (int, float)) or not (0 <= p <= 1):
            logger.error(f"Invalid p_solve value: {p}")
            return False

    logger.info("Ground truth validation passed")
    return True
