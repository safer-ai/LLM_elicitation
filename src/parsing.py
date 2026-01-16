# src/parsing.py

import re
import logging
from typing import Dict, Any, Optional, List

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


def parse_probability_response(response_text: str) -> Dict[str, Any]:
    """
    Parses a four-point probability estimation response, looking for:
    - Minimum probability
    - Maximum probability
    - Final probability (most likely)
    - Confidence in range
    - Rationale

    Args:
        response_text: The raw text output from the LLM.

    Returns:
        A dictionary containing the parsed values. 'estimate' is populated with
        the 'most_likely' value for backward compatibility with aggregation logic.
    """
    result: Dict[str, Any] = {
        "minimum": None,
        "maximum": None,
        "most_likely": None,
        "confidence": None,
        "rationale": "",
        "estimate": None, # For backward compatibility
    }

    def _extract_float(pattern: str, text: str) -> Optional[float]:
        """Helper to extract a float value using a regex pattern."""
        match = re.search(pattern, text.replace("**", ""), re.IGNORECASE)
        if match:
            try:
                # Use group 1, which is the captured numeric part
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

    # Use specific, non-overlapping patterns for each value, robust to markdown
    result["minimum"] = _extract_float(r'\**\s*Minimum\s+probability[*\s:]+\s*([0-9]*\.?[0-9]+)', response_text)
    result["maximum"] = _extract_float(r'\**\s*Maximum\s+probability[\s*:]+\s*([0-9]*\.?[0-9]+)', response_text)
    result["most_likely"] = _extract_float(r'\**\s*Final\s+probability[\s*:]+\s*([0-9]*\.?[0-9]+)', response_text)
    result["confidence"] = _extract_float(r'\**\s*Confidence\s+in\s+range[\s*:]+\s*([0-9]*\.?[0-9]+)', response_text)

    # We rename 'most_likely' to 'estimate' for backward compatibility in the workflow
    if result["most_likely"] is not None:
        result["estimate"] = result["most_likely"]
    else:
        logger.warning("Could not parse 'Final probability'. The primary 'estimate' will be null.")
    
    # Extract Rationale, robust to markdown
    rationale_match = re.search(r'\**\s*Rationale\s*\**\s*:(.*?)(?:\Z)', response_text, re.IGNORECASE | re.DOTALL)
    if rationale_match:
        result["rationale"] = rationale_match.group(1).strip()
    else:
        logger.warning("Could not find 'Rationale:' pattern in probability response. Rationale will be empty.")
        # Unlike integer parsing, we don't attempt a fallback here as the format is more structured.

    return result


def parse_quantity_response(response_text: str) -> Dict[str, Any]:
    """
    Parses a four-point quantity estimation response, looking for:
    - Minimum value
    - Maximum value
    - Final Estimated Value (most likely)
    - Confidence in range
    - Rationale

    Args:
        response_text: The raw text output from the LLM.

    Returns:
        A dictionary containing the parsed values. 'estimate' is populated with
        the 'most_likely' value for backward compatibility with aggregation logic.
    """
    result: Dict[str, Any] = {
        "minimum": None,
        "maximum": None,
        "most_likely": None,
        "confidence": None,
        "rationale": "",
        "estimate": None, # For backward compatibility
    }

    def _extract_numeric(pattern: str, text: str) -> Optional[float]:
        """Helper to extract a numeric value (float or int) using a regex pattern."""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                val_str = match.group(1).strip()
                # Remove commas for parsing
                val_str_no_commas = val_str.replace(',', '')
                # Try to convert to float, which handles ints as well
                val = float(val_str_no_commas)
                return val
            except (ValueError, IndexError):
                logger.warning(f"Could not parse numeric value from match: {match.groups()} with pattern: {pattern}")
        return None

    # Patterns for quantity estimation, now robust to markdown and commas
    result["minimum"] = _extract_numeric(r'\**\s*Minimum\s+value[\s*:]+\s*(-?[0-9,]*\.?[0-9,]+)', response_text)
    result["maximum"] = _extract_numeric(r'\**\s*Maximum\s+value[\s*:]+\s*(-?[0-9,]*\.?[0-9,]+)', response_text)
    result["most_likely"] = _extract_numeric(r'\**\s*Final\s+Estimated\s+Value[\s*:]+\s*(-?[0-9,]*\.?[0-9,]+)', response_text)
    
    # Confidence is a probability, so it should be between 0 and 1
    confidence_match = re.search(r'\**\s*Confidence\s+in\s+range[\s*:]+\s*([0-9]*\.?[0-9]+)', response_text, re.IGNORECASE)
    if confidence_match:
        try:
            conf_val = float(confidence_match.group(1))
            if 0.0 <= conf_val <= 1.0:
                result["confidence"] = conf_val
            else:
                logger.warning(f"Parsed confidence value '{conf_val}' is outside the valid range [0, 1]. Ignoring.")
        except (ValueError, IndexError):
            logger.warning(f"Could not parse confidence float from match: {confidence_match.groups()}")

    # We rename 'most_likely' to 'estimate' for backward compatibility in the workflow
    if result["most_likely"] is not None:
        result["estimate"] = result["most_likely"]
    else:
        logger.warning("Could not parse 'Final Estimated Value'. The primary 'estimate' will be null.")
    
    # Extract Rationale, robust to markdown
    rationale_match = re.search(r'\**\s*Rationale\s*\**\s*:(.*?)(?:\Z)', response_text, re.IGNORECASE | re.DOTALL)
    if rationale_match:
        result["rationale"] = rationale_match.group(1).strip()
    else:
        logger.warning("Could not find 'Rationale:' pattern in quantity response. Rationale will be empty.")

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

    # Test Case 2: Four-Point Probability Response
    test_prob_1 = """
    Here is my thinking process...

    **Final probability**: 0.65
    Minimum probability: 0.40
    Maximum probability: 0.85
    **Confidence in range**: 0.90

    **Rationale**:
    The LLM provides a significant uplift by automating the main bottleneck.
    The range reflects uncertainty about real-world friction.
    """
    print("\nTesting Four-Point Probability Parser:")
    parsed_prob = parse_probability_response(test_prob_1)
    print(f"Parsed Probability: {parsed_prob}")
    assert parsed_prob["most_likely"] == 0.65
    assert parsed_prob["estimate"] == 0.65
    assert parsed_prob["minimum"] == 0.40
    assert parsed_prob["maximum"] == 0.85
    assert parsed_prob["confidence"] == 0.90
    assert parsed_prob["rationale"].strip().startswith("The LLM provides")

    # Test Case 3: Four-Point Quantity Response
    test_quant_1 = """
    Analysis of actor motivation and scenario complexity suggests a small, dedicated team.

    Minimum value: 2
    Maximum value: 10
    Final Estimated Value: 5
    Confidence in range: 0.8
    
    Rationale:
    The operation requires stealth and coordination, which is more typical of a small, organized group rather than a large number of disparate actors.
    """
    print("\nTesting Quantity Parser:")
    parsed_quant = parse_quantity_response(test_quant_1)
    print(f"Parsed Quantity: {parsed_quant}")
    assert parsed_quant["estimate"] == 5
    assert parsed_quant["most_likely"] == 5
    assert parsed_quant["minimum"] == 2
    assert parsed_quant["maximum"] == 10
    assert parsed_quant["confidence"] == 0.8
    assert parsed_quant["rationale"].strip().startswith("The operation requires stealth")

    # Test Case 4: Four-Point Quantity Response with commas and markdown
    test_quant_2 = """
    **Minimum value**: 1,000
    **Maximum value**: 1,500,000
    **Final Estimated Value**: 250,000
    **Confidence in range**: 0.75
    
    **Rationale**:
    This is a test case with commas and markdown.
    """
    print("\nTesting Quantity Parser with commas and markdown:")
    parsed_quant_2 = parse_quantity_response(test_quant_2)
    print(f"Parsed Quantity 2: {parsed_quant_2}")
    assert parsed_quant_2["minimum"] == 1000
    assert parsed_quant_2["maximum"] == 1500000
    assert parsed_quant_2["most_likely"] == 250000
    assert parsed_quant_2["estimate"] == 250000
    assert parsed_quant_2["confidence"] == 0.75
    assert parsed_quant_2["rationale"].strip().startswith("This is a test case")

    # Test Case 5: Probability response with markdown colon
    test_prob_2 = """
    **Final probability:** 0.25
    Minimum probability: 0.10
    Maximum probability: 0.40
    **Confidence in range**: 0.95
    **Rationale**:
    This is a test case with a bolded colon.
    """
    print("\nTesting Probability Parser with markdown colon:")
    parsed_prob_2 = parse_probability_response(test_prob_2)
    print(f"Parsed Probability 2: {parsed_prob_2}")
    assert parsed_prob_2["most_likely"] == 0.25
    assert parsed_prob_2["estimate"] == 0.25
    assert parsed_prob_2["minimum"] == 0.10
    assert parsed_prob_2["maximum"] == 0.40
    assert parsed_prob_2["confidence"] == 0.95
    assert parsed_prob_2["rationale"].strip().startswith("This is a test case")

    print("\n--- Parser Tests Complete ---")
