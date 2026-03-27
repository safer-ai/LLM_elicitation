#!/usr/bin/env python3
"""
Run the Baseline Uplift Curve Experiment.

This script runs the LLM elicitation pipeline for each baseline condition
(10%, 20%, ... 90%) multiple times to collect estimates.

Usage:
    python run_experiment.py                    # Run all baselines, 10 runs each
    python run_experiment.py --baseline 50      # Run only baseline=50%
    python run_experiment.py --runs 5           # Run 5 times per baseline
    python run_experiment.py --dry-run          # Show commands without running
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

EXPERIMENT_DIR = Path(__file__).parent
LLM_ELICITATION_DIR = EXPERIMENT_DIR.parent.parent.parent
CONFIGS_DIR = EXPERIMENT_DIR / "configs"
RESULTS_DIR = EXPERIMENT_DIR / "results"

BASELINE_VALUES = [10, 20, 30, 40, 50, 60, 70, 80, 90]
DEFAULT_RUNS_PER_BASELINE = 10


def run_single_experiment(baseline: int, run_number: int, dry_run: bool = False) -> bool:
    """Run a single experiment for a given baseline."""
    config_path = CONFIGS_DIR / f"config_baseline_{baseline}.yaml"

    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return False

    # Use src/main.py (not python -m src.main): the latter does not put src/ on
    # sys.path, so `from config import ...` fails with No module named 'config'.
    main_script = LLM_ELICITATION_DIR / "src" / "main.py"
    cmd = [
        sys.executable,
        str(main_script),
        "-c",
        str(config_path.resolve()),
    ]

    print(f"\n{'='*60}")
    print(f"Baseline: {baseline}% | Run: {run_number}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    if dry_run:
        print("[DRY RUN] Skipping execution")
        return True

    try:
        # Change to LLM_elicitation directory and run
        result = subprocess.run(
            cmd,
            cwd=LLM_ELICITATION_DIR,
            capture_output=False,
            text=True
        )

        if result.returncode != 0:
            print(f"WARNING: Run {run_number} for baseline {baseline}% returned non-zero exit code")
            return False

        return True

    except Exception as e:
        print(f"ERROR running experiment: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Baseline Uplift Curve Experiment")
    parser.add_argument("--baseline", type=int, choices=BASELINE_VALUES,
                       help="Run only a specific baseline (default: all)")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS_PER_BASELINE,
                       help=f"Number of runs per baseline (default: {DEFAULT_RUNS_PER_BASELINE})")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print commands without executing")
    parser.add_argument("--start-run", type=int, default=1,
                       help="Start from a specific run number (for resuming)")
    args = parser.parse_args()

    baselines = [args.baseline] if args.baseline else BASELINE_VALUES
    num_runs = args.runs
    start_run = args.start_run

    print("=" * 70)
    print("BASELINE UPLIFT CURVE EXPERIMENT")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Baselines to test: {baselines}")
    print(f"Runs per baseline: {num_runs}")
    print(f"Starting from run: {start_run}")
    print(f"Total experiments: {len(baselines) * (num_runs - start_run + 1)}")
    print("=" * 70)

    if args.dry_run:
        print("\n*** DRY RUN MODE - No experiments will be executed ***\n")

    # Track results
    successes = 0
    failures = 0

    for baseline in baselines:
        print(f"\n{'#'*70}")
        print(f"# BASELINE: {baseline}%")
        print(f"{'#'*70}")

        for run_num in range(start_run, num_runs + 1):
            success = run_single_experiment(baseline, run_num, args.dry_run)

            if success:
                successes += 1
            else:
                failures += 1

            # Small delay between runs to avoid rate limiting
            if not args.dry_run and run_num < num_runs:
                print("Waiting 5 seconds before next run...")
                time.sleep(5)

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Successful runs: {successes}")
    print(f"Failed runs: {failures}")
    print(f"Total: {successes + failures}")
    print("=" * 70)

    if not args.dry_run:
        print("\nNext step: Run analyze_results.py to generate the baseline uplift curve")


if __name__ == "__main__":
    main()
