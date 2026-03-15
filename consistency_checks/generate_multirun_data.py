#!/usr/bin/env python
"""Run src/main.py multiple times to generate data for consistency analysis."""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = "consistency_checks/config_multirun_claude.yaml"


def run_once(run_index: int, total: int, config_path: str, debug: bool) -> bool:
    """Run src/main.py once and return True on success."""
    print(f"\n--- Run {run_index}/{total} ---", flush=True)

    cmd = [sys.executable, "src/main.py", "-c", config_path]
    if debug:
        cmd.append("-d")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode == 0:
        print(f"Run {run_index}/{total} completed successfully.")
        return True
    else:
        print(f"Run {run_index}/{total} failed with exit code {result.returncode}.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run src/main.py multiple times for consistency analysis."
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of times to run the pipeline (default: 3)"
    )
    parser.add_argument(
        "-c", "--config", default=DEFAULT_CONFIG_PATH,
        help=f"Path to config file (default: {DEFAULT_CONFIG_PATH})"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true",
        help="Pass --debug flag to each run"
    )
    args = parser.parse_args()

    if args.runs < 1:
        print("Error: --runs must be at least 1.", file=sys.stderr)
        sys.exit(1)

    print(f"Starting {args.runs} run(s) using config: {args.config}")

    successes = 0
    for i in range(1, args.runs + 1):
        if run_once(i, args.runs, args.config, args.debug):
            successes += 1

    print(f"\n=== Done: {successes}/{args.runs} runs succeeded ===")
    if successes < args.runs:
        sys.exit(1)


if __name__ == "__main__":
    main()
