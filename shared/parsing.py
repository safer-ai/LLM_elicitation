#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Response parsing utilities shared across experiments.

Supports XML-first parsing with legacy markdown fallback.
Handles probability responses, quantity responses, and analysis responses.
"""

import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def parse_analysis_response(response_text: str) -> Dict[str, str]:
    """
    Parses the analysis response by capturing the full text.

    Returns:
        A dictionary with 'full_analysis' and 'technical_capabilities' keys.
    """
    full_analysis = response_text.strip()

    if full_analysis:
        logger.info("Captured full analysis response for prompt context.")
    else:
        logger.warning("Analysis response appears to be empty.")

    return {
        "full_analysis": full_analysis,
        "technical_capabilities": full_analysis,
        "llm_impact": "",
        "real_world_translation": ""
    }


def _extract_xml_tag(tag: str, text: str) -> Optional[str]:
    """Helper to extract content from an XML tag."""
    pattern = rf'<{tag}>\s*(.*?)\s*</{tag}>'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _extract_percentile_estimates(text: str) -> Dict[str, Optional[str]]:
    """Extract p25, p50, p75 from <percentile_estimates> block."""
    estimates_block = _extract_xml_tag('percentile_estimates', text)
    if not estimates_block:
        return {"p25": None, "p50": None, "p75": None}

    return {
        "p25": _extract_xml_tag('p25', estimates_block),
        "p50": _extract_xml_tag('p50', estimates_block),
        "p75": _extract_xml_tag('p75', estimates_block),
    }


def _extract_rationale(text: str) -> Optional[str]:
    """Extracts the rationale summary from a structured LLM response.

    Tries, in order:
      1. A properly-closed `<rationale>...</rationale>` block.
      2. An unclosed `<rationale>...` block running to end-of-text. This case
         is the signature of a response truncated by max_tokens /
         max_completion_tokens — we keep the partial content and log a warning.
      3. A legacy `Rationale:` header. This pattern requires the header to
         sit at the start of a line (or right after `**`) so it doesn't latch
         onto inline phrases like `- Highest reasonable: 0.95. Rationale: ...`
         that the model may emit inside its reasoning prose.
    """
    closed = re.search(
        r'<rationale>\s*(.*?)\s*</rationale>',
        text, re.IGNORECASE | re.DOTALL,
    )
    if closed:
        return closed.group(1).strip()

    open_only = re.search(
        r'<rationale>\s*(.*)\Z',
        text, re.IGNORECASE | re.DOTALL,
    )
    if open_only:
        logger.warning(
            "Found <rationale> with no closing tag — the response was likely truncated "
            "by max_tokens / max_completion_tokens. Capturing the partial content."
        )
        return open_only.group(1).strip()

    legacy = re.search(
        r'(?:^|\n)[ \t]*\**\s*Rationale\s*\**\s*:\s*\**\s*(.*)\Z',
        text, re.IGNORECASE | re.DOTALL,
    )
    if legacy:
        return legacy.group(1).strip()

    return None


def parse_probability_response(response_text: str) -> Dict[str, Any]:
    """
    Parses a probability estimation response with percentiles for Beta distribution fitting.
    Tries XML format first, falls back to legacy markdown patterns.

    Returns:
        Dictionary with percentile_25th, percentile_50th, percentile_75th, rationale, estimate.
    """
    result: Dict[str, Any] = {
        "percentile_25th": None,
        "percentile_50th": None,
        "percentile_75th": None,
        "rationale": "",
        "estimate": None,
    }

    def _parse_probability(val_str: Optional[str]) -> Optional[float]:
        if val_str is None:
            return None
        try:
            cleaned = re.sub(r'[\[\]\*\s]', '', val_str)
            val = float(cleaned)
            if 0.0 <= val <= 1.0:
                return val
            else:
                logger.warning(f"Parsed probability '{val}' is outside [0, 1]. Ignoring.")
                return None
        except ValueError:
            logger.warning(f"Could not parse probability from: '{val_str}'")
            return None

    # Try XML format first
    estimates = _extract_percentile_estimates(response_text)
    xml_found = any(v is not None for v in estimates.values())

    if xml_found:
        result["percentile_25th"] = _parse_probability(estimates["p25"])
        result["percentile_50th"] = _parse_probability(estimates["p50"])
        result["percentile_75th"] = _parse_probability(estimates["p75"])
    else:
        logger.info("XML tags not found, falling back to legacy markdown parsing.")

        def _extract_float_legacy(pattern: str, text: str) -> Optional[float]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return _parse_probability(match.group(1))
            return None

        result["percentile_25th"] = _extract_float_legacy(r'[-*\s]*25th\s+percentile[^:]*:\s*\[?\**\s*([0-9]+\.?[0-9]*)\**\]?', response_text)
        result["percentile_50th"] = _extract_float_legacy(r'[-*\s]*50th\s+percentile[^:]*:\s*\[?\**\s*([0-9]+\.?[0-9]*)\**\]?', response_text)
        result["percentile_75th"] = _extract_float_legacy(r'[-*\s]*75th\s+percentile[^:]*:\s*\[?\**\s*([0-9]+\.?[0-9]*)\**\]?', response_text)

    if result["percentile_50th"] is not None:
        result["estimate"] = result["percentile_50th"]
    else:
        logger.warning("Could not parse '50th percentile (median)'. The primary 'estimate' will be null.")

    rationale = _extract_rationale(response_text)
    if rationale:
        result["rationale"] = rationale
    else:
        logger.warning("Could not find rationale in response.")

    return result


def parse_quantity_response(response_text: str) -> Dict[str, Any]:
    """
    Parses a quantity estimation response with percentiles.
    Tries XML format first, falls back to legacy markdown patterns.

    Returns:
        Dictionary with percentile_25th, percentile_50th, percentile_75th, rationale, estimate.
    """
    result: Dict[str, Any] = {
        "percentile_25th": None,
        "percentile_50th": None,
        "percentile_75th": None,
        "rationale": "",
        "estimate": None,
    }

    def _parse_numeric(val_str: Optional[str]) -> Optional[float]:
        if val_str is None:
            return None
        try:
            cleaned = re.sub(r'[\[\]\*\s\$,]', '', val_str)
            val = float(cleaned)
            return val
        except ValueError:
            logger.warning(f"Could not parse numeric value from: '{val_str}'")
            return None

    estimates = _extract_percentile_estimates(response_text)
    xml_found = any(v is not None for v in estimates.values())

    if xml_found:
        result["percentile_25th"] = _parse_numeric(estimates["p25"])
        result["percentile_50th"] = _parse_numeric(estimates["p50"])
        result["percentile_75th"] = _parse_numeric(estimates["p75"])
    else:
        logger.info("XML tags not found, falling back to legacy markdown parsing.")

        def _extract_numeric_legacy(pattern: str, text: str) -> Optional[float]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return _parse_numeric(match.group(1))
            return None

        result["percentile_25th"] = _extract_numeric_legacy(r'[-*\s]*25th\s+percentile[^:]*:\s*\[?\**\s*\$?(-?[0-9,]+\.?[0-9]*)\**\]?', response_text)
        result["percentile_50th"] = _extract_numeric_legacy(r'[-*\s]*50th\s+percentile[^:]*:\s*\[?\**\s*\$?(-?[0-9,]+\.?[0-9]*)\**\]?', response_text)
        result["percentile_75th"] = _extract_numeric_legacy(r'[-*\s]*75th\s+percentile[^:]*:\s*\[?\**\s*\$?(-?[0-9,]+\.?[0-9]*)\**\]?', response_text)

    if result["percentile_50th"] is not None:
        result["estimate"] = result["percentile_50th"]
    else:
        logger.warning("Could not parse '50th percentile (median)'. The primary 'estimate' will be null.")

    rationale = _extract_rationale(response_text)
    if rationale:
        result["rationale"] = rationale
    else:
        logger.warning("Could not find rationale in response.")

    return result
