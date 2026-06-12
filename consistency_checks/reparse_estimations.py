"""
reparse_estimations.py

Re-parses the 'raw_estimation' field in a full_results.json file, auto-detecting
percentile labels from the response XML, and overwrites 'parsed_estimation'.

Usage:
    python reparse_estimations.py <path_to_full_results.json>
    python reparse_estimations.py <path_to_full_results.json> --inplace
    python reparse_estimations.py <path_to_full_results.json> --output <path>

By default, writes the updated JSON to stdout.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_xml_tag(tag: str, text: str) -> Optional[str]:
    pattern = rf'<{tag}>\s*(.*?)\s*</{tag}>'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def _parse_probability(val_str: Optional[str]) -> Optional[float]:
    if val_str is None:
        return None
    try:
        val = float(re.sub(r'[\[\]\*\s]', '', val_str))
        if 0.0 <= val <= 1.0:
            return val
        print(f"  Warning: parsed probability {val} is outside [0, 1], ignoring.", file=sys.stderr)
        return None
    except ValueError:
        print(f"  Warning: could not parse probability from '{val_str}'", file=sys.stderr)
        return None


def parse_probability_response(response_text: str) -> Dict[str, Any]:
    """
    Parse a probability estimation response, auto-detecting percentile labels.

    Looks for <pN> tags inside <percentile_estimates> and returns:
      {
        "estimates": {20: 0.40, 40: 0.57, ...},   # int keys
        "estimate": 0.57,                           # median percentile value
        "rationale": "...",
      }
    """
    estimates_block = _extract_xml_tag('percentile_estimates', response_text)
    if not estimates_block:
        print("  Warning: no <percentile_estimates> block found.", file=sys.stderr)
        estimates_block = ""

    tag_strs = sorted(re.findall(r'<p(\d+)>', estimates_block), key=int)
    estimates = {int(t): _parse_probability(_extract_xml_tag(f'p{t}', estimates_block)) for t in tag_strs}
    percentiles = list(estimates.keys())

    mid = percentiles[len(percentiles) // 2] if percentiles else None
    estimate = estimates.get(mid) if mid is not None else None
    if estimate is None and mid is not None:
        print(f"  Warning: could not parse p{mid} as primary estimate.", file=sys.stderr)

    rationale = _extract_xml_tag('rationale', response_text) or ""

    return {"estimates": estimates, "estimate": estimate, "rationale": rationale}


# ---------------------------------------------------------------------------
# JSON traversal
# ---------------------------------------------------------------------------

def reparse_results(data: Any) -> tuple[int, int]:
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
                    response["parsed_estimation"] = parse_probability_response(raw)
                    updated += 1
    return updated, skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Path to full_results.json")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--inplace", action="store_true", help="Overwrite the input file")
    output_group.add_argument("--output", type=Path, metavar="PATH", help="Write result to this file")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    updated, skipped = reparse_results(data)
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
