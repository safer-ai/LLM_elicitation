"""
reparse_estimations.py

Re-parses the 'raw_estimation' field in a full_results.json file using a
flexible percentile parser, and overwrites 'parsed_estimation' with the result.

Usage:
    python reparse_estimations.py <path_to_full_results.json> [--percentiles 20 40 60 80]
    python reparse_estimations.py <path_to_full_results.json> --inplace

By default, writes the updated JSON to stdout. Use --inplace to overwrite the
input file, or --output <path> to write to a different file.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Parsing helpers (self-contained, no dependency on src/parsing.py)
# ---------------------------------------------------------------------------

def _extract_xml_tag(tag: str, text: str) -> Optional[str]:
    pattern = rf'<{tag}>\s*(.*?)\s*</{tag}>'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def _parse_probability(val_str: Optional[str]) -> Optional[float]:
    if val_str is None:
        return None
    try:
        cleaned = re.sub(r'[\[\]\*\s]', '', val_str)
        val = float(cleaned)
        if 0.0 <= val <= 1.0:
            return val
        print(f"  Warning: parsed probability {val} is outside [0, 1], ignoring.", file=sys.stderr)
        return None
    except ValueError:
        print(f"  Warning: could not parse probability from '{val_str}'", file=sys.stderr)
        return None


def parse_probability_response(response_text: str, percentiles: List[int]) -> Dict[str, Any]:
    """
    Parse a probability estimation response, extracting the given percentiles.

    Looks for XML tags <p{n}>...</p{n}> inside <percentile_estimates>, then
    falls back to legacy markdown patterns like '- 20th percentile: 0.61'.

    The middle element of `percentiles` is used as the primary 'estimate'.
    """
    result: Dict[str, Any] = {f"percentile_{p}th": None for p in percentiles}
    result["rationale"] = ""
    result["estimate"] = None

    # --- XML path ---
    estimates_block = _extract_xml_tag('percentile_estimates', response_text)
    if estimates_block:
        for p in percentiles:
            result[f"percentile_{p}th"] = _parse_probability(_extract_xml_tag(f'p{p}', estimates_block))
    else:
        # --- Legacy markdown fallback ---
        print("  Info: XML tags not found, falling back to legacy markdown parsing.", file=sys.stderr)
        for p in percentiles:
            pattern = rf'[-*\s]*{p}th\s+percentile[^:]*:\s*\[?\**\s*([0-9]+\.?[0-9]*)\**\]?'
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                result[f"percentile_{p}th"] = _parse_probability(match.group(1))

    # Primary estimate = middle percentile
    mid = percentiles[len(percentiles) // 2]
    result["estimate"] = result[f"percentile_{mid}th"]
    if result["estimate"] is None:
        print(f"  Warning: could not parse {mid}th percentile; 'estimate' will be null.", file=sys.stderr)

    # --- Rationale ---
    rationale = _extract_xml_tag('rationale', response_text)
    if rationale:
        result["rationale"] = rationale
    else:
        m = re.search(r'\**\s*Rationale\s*\**\s*:(.*?)(?:\Z)', response_text, re.IGNORECASE | re.DOTALL)
        if m:
            result["rationale"] = m.group(1).strip()

    return result


# ---------------------------------------------------------------------------
# JSON traversal
# ---------------------------------------------------------------------------

def reparse_results(data: Any, percentiles: List[int]) -> tuple[int, int]:
    """
    Walk the full_results structure and re-parse every 'raw_estimation' field,
    updating 'parsed_estimation' in place.

    Returns (n_updated, n_skipped).
    """
    updated = skipped = 0

    for step in data.get("results_per_step", []):
        for task in step.get("results_per_task", []):
            for round_data in task.get("rounds_data", []):
                for response in round_data.get("responses", []):
                    raw = response.get("raw_estimation")
                    if not raw:
                        skipped += 1
                        continue
                    parsed = parse_probability_response(raw, percentiles)
                    response["parsed_estimation"] = parsed
                    # Mirror top-level convenience keys if they exist
                    for p in percentiles:
                        key = f"percentile_{p}th"
                        if key in response:
                            response[key] = parsed[key]
                    if "estimate" in response:
                        response["estimate"] = parsed["estimate"]
                    updated += 1

    return updated, skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Path to full_results.json")
    parser.add_argument(
        "--percentiles", nargs="+", type=int, default=[20, 40, 60, 80],
        metavar="N",
        help="Percentile values to extract (default: 20 40 60 80)",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--inplace", action="store_true", help="Overwrite the input file")
    output_group.add_argument("--output", type=Path, metavar="PATH", help="Write result to this file")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    print(f"Re-parsing with percentiles: {args.percentiles}", file=sys.stderr)
    updated, skipped = reparse_results(data, args.percentiles)
    print(f"Done: {updated} responses updated, {skipped} skipped (no raw_estimation).", file=sys.stderr)

    output_json = json.dumps(data, indent=2)

    if args.inplace:
        args.input.write_text(output_json)
        print(f"Wrote updated results to {args.input}", file=sys.stderr)
    elif args.output:
        args.output.write_text(output_json)
        print(f"Wrote updated results to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
