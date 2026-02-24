#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Response parsing utilities for intra-benchmark calibration.

Parses LLM responses to extract analysis, probability estimates,
and uncertainty bounds.
"""

import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def parse_probability_response(response_text: str) -> Dict[str, Any]:
    """
    Parses a conditional probability estimation response, looking for:
    - 25th percentile (lower quartile)
    - 50th percentile (median)
    - 75th percentile (upper quartile)
    - Rationale

    These three percentiles characterise the expert's probability distribution
    and will be used for fitting a Beta distribution with support [0, 1].

    Args:
        response_text: The raw text output from the LLM.

    Returns:
        A dictionary containing the parsed values.
    """
    result: Dict[str, Any] = {
        "percentile_25th": None,
        "percentile_50th": None,
        "percentile_75th": None,
        "rationale": "",
        "estimate": None, # primary estimate, e.g. median
    }

    def _extract_float(pattern: str, text: str) -> Optional[float]:
        """Helper to extract a float value using a regex pattern."""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                val_str = match.group(1).strip()
                val = float(val_str)
                # Probabilities must be in [0, 1] range
                if 0.0 <= val <= 1.0:
                    return val
                else:
                    logger.warning(f"Parsed value '{val}' from string '{val_str}' is outside the valid probability range [0, 1]. Ignoring.")
                    return None
            except (ValueError, IndexError):
                logger.warning(f"Could not parse float from match: {match.groups()} with pattern: {pattern}")
        return None

    # Parse the three percentiles
    # Handles cases like: "25th percentile: 0.18", "25th percentile (lower quartile): **0.18**"
    result["percentile_25th"] = _extract_float(r'(?:25th percentile|25th Percentile).*?:\s*\**\s*([0-9]+\.?[0-9]*)\**', response_text)
    result["percentile_50th"] = _extract_float(r'(?:50th percentile|50th Percentile|Median).*?:\s*\**\s*([0-9]+\.?[0-9]*)\**', response_text)
    result["percentile_75th"] = _extract_float(r'(?:75th percentile|75th Percentile).*?:\s*\**\s*([0-9]+\.?[0-9]*)\**', response_text)

    if result["percentile_50th"] is not None:
        result["estimate"] = result["percentile_50th"]
    else:
        logger.warning("Could not parse '50th percentile (median)'. The primary 'estimate' will be null.")
    
    # Extract Rationale, robust to markdown
    rationale_match = re.search(r'\**\s*Rationale\s*\**\s*:(.*?)(?:\Z)', response_text, re.IGNORECASE | re.DOTALL)
    if rationale_match:
        result["rationale"] = rationale_match.group(1).strip()
    else:
        logger.warning("Could not find 'Rationale:' pattern in probability response. Rationale will be empty.")

    return result

