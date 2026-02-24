# src/parsing.py

import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def parse_analysis_response(response_text: str) -> Dict[str, str]:
    """
    Parses the analysis response by capturing the full text.
    
    Args:
        response_text: The raw text output from the LLM.
    
    Returns:
        A dictionary containing the full analysis text. For backward compatibility
        with the workflow, the 'technical_capabilities' key is also populated
        with the full analysis, ensuring the next prompt stage receives the context.
    """
    full_analysis = response_text.strip()
    
    if full_analysis:
        logger.info("Captured full analysis response for prompt context.")
    else:
        logger.warning("Analysis response appears to be empty.")
    
    sections = {
        "full_analysis": full_analysis,
        # For backward compatibility with workflow.py, which expects this key
        # to populate the {technical_analysis} placeholder in the next prompt.
        "technical_capabilities": full_analysis,
        "llm_impact": "", # No longer used, kept for safety
        "real_world_translation": "" # No longer used, kept for safety
    }
    
    return sections


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


def parse_probability_response(response_text: str) -> Dict[str, Any]:
    """
    Parses a probability estimation response with percentiles for Beta distribution fitting:
    - 25th percentile (lower quartile)
    - 50th percentile (median)
    - 75th percentile (upper quartile)
    - Rationale

    Expects XML format:
    <percentile_estimates>
    <p25>0.xx</p25>
    <p50>0.xx</p50>
    <p75>0.xx</p75>
    </percentile_estimates>
    <rationale>...</rationale>

    Falls back to legacy markdown patterns if XML not found.

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
        "estimate": None,  # Primary estimate (median)
    }

    def _parse_probability(val_str: Optional[str]) -> Optional[float]:
        """Parse a probability value string, ensuring it's in [0, 1]."""
        if val_str is None:
            return None
        try:
            # Remove any surrounding whitespace, brackets, asterisks
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
        # Fallback to legacy markdown patterns
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
    
    # Extract Rationale - try XML first, then legacy
    rationale = _extract_xml_tag('rationale', response_text)
    if rationale:
        result["rationale"] = rationale
    else:
        # Fallback to legacy markdown pattern
        rationale_match = re.search(r'\**\s*Rationale\s*\**\s*:(.*?)(?:\Z)', response_text, re.IGNORECASE | re.DOTALL)
        if rationale_match:
            result["rationale"] = rationale_match.group(1).strip()
        else:
            logger.warning("Could not find rationale in response.")

    return result


def parse_quantity_response(response_text: str) -> Dict[str, Any]:
    """
    Parses a quantity estimation response with percentiles:
    - 25th percentile (lower quartile)
    - 50th percentile (median)
    - 75th percentile (upper quartile)
    - Rationale

    Expects XML format:
    <percentile_estimates>
    <p25>INTEGER</p25>
    <p50>INTEGER</p50>
    <p75>INTEGER</p75>
    </percentile_estimates>
    <rationale>...</rationale>

    Falls back to legacy markdown patterns if XML not found.

    Args:
        response_text: The raw text output from the LLM.

    Returns:
        A dictionary containing the parsed values. 'estimate' is populated with
        the 50th percentile (median) for compatibility with aggregation logic.
    """
    result: Dict[str, Any] = {
        "percentile_25th": None,
        "percentile_50th": None,
        "percentile_75th": None,
        "rationale": "",
        "estimate": None,  # Primary estimate (median)
    }

    def _parse_numeric(val_str: Optional[str]) -> Optional[float]:
        """Parse a numeric value string, handling commas and currency symbols."""
        if val_str is None:
            return None
        try:
            # Remove whitespace, brackets, asterisks, dollar signs, commas
            cleaned = re.sub(r'[\[\]\*\s\$,]', '', val_str)
            val = float(cleaned)
            return val
        except ValueError:
            logger.warning(f"Could not parse numeric value from: '{val_str}'")
            return None

    # Try XML format first
    estimates = _extract_percentile_estimates(response_text)
    xml_found = any(v is not None for v in estimates.values())
    
    if xml_found:
        result["percentile_25th"] = _parse_numeric(estimates["p25"])
        result["percentile_50th"] = _parse_numeric(estimates["p50"])
        result["percentile_75th"] = _parse_numeric(estimates["p75"])
    else:
        # Fallback to legacy markdown patterns
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
    
    # Extract Rationale - try XML first, then legacy
    rationale = _extract_xml_tag('rationale', response_text)
    if rationale:
        result["rationale"] = rationale
    else:
        # Fallback to legacy markdown pattern
        rationale_match = re.search(r'\**\s*Rationale\s*\**\s*:(.*?)(?:\Z)', response_text, re.IGNORECASE | re.DOTALL)
        if rationale_match:
            result["rationale"] = rationale_match.group(1).strip()
        else:
            logger.warning("Could not find rationale in response.")

    return result


# --- Test Execution Block ---
if __name__ == "__main__":
    print("--- Running Parser Tests ---")
    logging.basicConfig(level=logging.INFO)

    # Test Case 1: Analysis Response
    test_analysis_1 = """
    **Task Decomposition**
    To solve this, one must:
    1. Identify the encoding.
    2. Write a script to decode it.

    **Capability Boundaries**
    An LLM that solves this can handle basic data manipulation and scripting.
    It would likely fail at complex algorithmic challenges.
    """
    print("\nTesting Analysis Parser:")
    parsed_analysis = parse_analysis_response(test_analysis_1)
    print(f"Parsed Analysis: {parsed_analysis}")
    assert parsed_analysis["full_analysis"].startswith("**Task Decomposition**")
    assert parsed_analysis["technical_capabilities"] == parsed_analysis["full_analysis"]

    # Test Case 2: XML Probability Response (NEW FORMAT)
    test_prob_xml = """
    **Phase 1 - Range Establishment:**
    Given the baseline of 60% and the LLM's capabilities...

    **Phase 2 - Confidence Check:**
    I may be slightly overconfident...

    **Phase 3 - Reality Check:**
    A ratio of 1.2x seems reasonable...

    <percentile_estimates>
    <p25>0.62</p25>
    <p50>0.72</p50>
    <p75>0.80</p75>
    </percentile_estimates>

    <rationale>
    The LLM provides moderate uplift to the baseline 60% success rate by automating technical bottlenecks.
    </rationale>
    """
    print("\nTesting XML Probability Parser:")
    parsed_prob_xml = parse_probability_response(test_prob_xml)
    print(f"Parsed Probability (XML): {parsed_prob_xml}")
    assert parsed_prob_xml["percentile_25th"] == 0.62
    assert parsed_prob_xml["percentile_50th"] == 0.72
    assert parsed_prob_xml["percentile_75th"] == 0.80
    assert parsed_prob_xml["estimate"] == 0.72
    assert "moderate uplift" in parsed_prob_xml["rationale"]

    # Test Case 3: XML Quantity Response (NEW FORMAT)
    test_quant_xml = """
    Analysis of actor motivation...

    <percentile_estimates>
    <p25>3</p25>
    <p50>5</p50>
    <p75>8</p75>
    </percentile_estimates>

    <rationale>
    The operation requires stealth and coordination, which is more typical of a small group.
    </rationale>
    """
    print("\nTesting XML Quantity Parser:")
    parsed_quant_xml = parse_quantity_response(test_quant_xml)
    print(f"Parsed Quantity (XML): {parsed_quant_xml}")
    assert parsed_quant_xml["percentile_25th"] == 3
    assert parsed_quant_xml["percentile_50th"] == 5
    assert parsed_quant_xml["percentile_75th"] == 8
    assert parsed_quant_xml["estimate"] == 5
    assert "stealth" in parsed_quant_xml["rationale"]

    # Test Case 4: XML Quantity with large numbers (damage estimation)
    test_damage_xml = """
    <percentile_estimates>
    <p25>620000</p25>
    <p50>800000</p50>
    <p75>1050000</p75>
    </percentile_estimates>

    <rationale>
    LLM-assisted execution increases mean economic damage by approximately 45%.
    </rationale>
    """
    print("\nTesting XML Damage Parser:")
    parsed_damage_xml = parse_quantity_response(test_damage_xml)
    print(f"Parsed Damage (XML): {parsed_damage_xml}")
    assert parsed_damage_xml["percentile_25th"] == 620000
    assert parsed_damage_xml["percentile_50th"] == 800000
    assert parsed_damage_xml["percentile_75th"] == 1050000
    assert parsed_damage_xml["estimate"] == 800000

    # Test Case 5: Legacy Probability Response (backward compatibility)
    test_prob_legacy = """
    **Percentile Estimates:**
    - 25th percentile: [0.62]
    - 50th percentile (median): [0.72]
    - 75th percentile: [0.80]

    **Rationale:**
    The LLM provides moderate uplift to the baseline 60% success rate.
    """
    print("\nTesting Legacy Probability Parser:")
    parsed_prob_legacy = parse_probability_response(test_prob_legacy)
    print(f"Parsed Probability (Legacy): {parsed_prob_legacy}")
    assert parsed_prob_legacy["percentile_25th"] == 0.62
    assert parsed_prob_legacy["percentile_50th"] == 0.72
    assert parsed_prob_legacy["percentile_75th"] == 0.80
    assert parsed_prob_legacy["estimate"] == 0.72

    # Test Case 6: Legacy Quantity Response with commas and markdown
    test_quant_legacy = """
    **Percentile Estimates:**
    - 25th percentile: **50,000**
    - 50th percentile (median): **250,000**
    - 75th percentile: **1,500,000**
    
    **Rationale:**
    This is a test case with commas and markdown.
    """
    print("\nTesting Legacy Quantity Parser with commas:")
    parsed_quant_legacy = parse_quantity_response(test_quant_legacy)
    print(f"Parsed Quantity (Legacy): {parsed_quant_legacy}")
    assert parsed_quant_legacy["percentile_25th"] == 50000
    assert parsed_quant_legacy["percentile_50th"] == 250000
    assert parsed_quant_legacy["percentile_75th"] == 1500000
    assert parsed_quant_legacy["estimate"] == 250000

    # Test Case 7: Legacy format without brackets
    test_prob_no_brackets = """
    **Percentile Estimates:**
    - 25th percentile: 0.45
    - 50th percentile (median): 0.55
    - 75th percentile: 0.68

    **Rationale:**
    Simple test without brackets.
    """
    print("\nTesting Legacy Probability Parser (no brackets):")
    parsed_prob_no_brackets = parse_probability_response(test_prob_no_brackets)
    print(f"Parsed Probability (no brackets): {parsed_prob_no_brackets}")
    assert parsed_prob_no_brackets["percentile_25th"] == 0.45
    assert parsed_prob_no_brackets["percentile_50th"] == 0.55
    assert parsed_prob_no_brackets["percentile_75th"] == 0.68
    assert parsed_prob_no_brackets["estimate"] == 0.55

    # Test Case 8: Legacy format with markdown bold
    test_prob_bold = """
    **Percentile Estimates:**
    - 25th percentile: **0.30**
    - 50th percentile (median): **0.42**
    - 75th percentile: **0.58**

    **Rationale:**
    Test with markdown bold formatting.
    """
    print("\nTesting Legacy Probability Parser (bold):")
    parsed_prob_bold = parse_probability_response(test_prob_bold)
    print(f"Parsed Probability (bold): {parsed_prob_bold}")
    assert parsed_prob_bold["percentile_25th"] == 0.30
    assert parsed_prob_bold["percentile_50th"] == 0.42
    assert parsed_prob_bold["percentile_75th"] == 0.58
    assert parsed_prob_bold["estimate"] == 0.42

    print("\n--- Parser Tests Complete ---")
