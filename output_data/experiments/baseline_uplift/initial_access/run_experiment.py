#!/usr/bin/env python3
"""
Run the Initial Access (TA0001) baseline uplift probability experiment.

Baselines 10%–90%; same benchmark task selection as execution_task (bountybench_ordered,
num_tasks: 1 → first task Imaginairy).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

from _paths import project_root

EXPERIMENT_DIR = Path(__file__).parent.resolve()
CONFIGS_DIR = EXPERIMENT_DIR / "configs"

BASELINE_VALUES = [10, 20, 30, 40, 50, 60, 70, 80, 90]
DEFAULT_RUNS_PER_BASELINE = 5


def run_single_experiment(baseline: int, run_number: int, dry_run: bool = False) -> bool:
    config_path = CONFIGS_DIR / f"config_baseline_{baseline}.yaml"
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return False

    main_script = project_root() / "src" / "main.py"
    cmd = [sys.executable, str(main_script), "-c", str(config_path.resolve())]

    print(f"\n{'='*60}")
    print(f"TA0001 Initial Access | Baseline: {baseline}% | Run: {run_number}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    if dry_run:
        print("[DRY RUN] Skipping execution")
        return True

    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_root()),
            capture_output=False,
            text=True,
        )
        if result.returncode != 0:
            print(f"WARNING: Run {run_number} for baseline {baseline}% returned non-zero exit code")
            return False
        return True
    except Exception as e:
        print(f"ERROR running experiment: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run TA0001 baseline uplift experiment")
    parser.add_argument(
        "--baseline",
        type=int,
        nargs="+",
        choices=BASELINE_VALUES,
        metavar="N",
        help="Baselines to run, e.g. --baseline 60 70 80 90 (default: all 10–90)",
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS_PER_BASELINE,
                        help=f"Runs per baseline (default: {DEFAULT_RUNS_PER_BASELINE})")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-run", type=int, default=1)
    args = parser.parse_args()

    baselines = args.baseline if args.baseline is not None else BASELINE_VALUES

    print("=" * 70)
    print("INITIAL ACCESS (TA0001) BASELINE UPLIFT — PROBABILITY")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Baselines: {baselines}")
    print(f"Runs per baseline: {args.runs}")
    print("=" * 70)

    successes = failures = 0
    for baseline in baselines:
        print(f"\n{'#'*70}\n# BASELINE: {baseline}%\n{'#'*70}")
        for run_num in range(args.start_run, args.runs + 1):
            if run_single_experiment(baseline, run_num, args.dry_run):
                successes += 1
            else:
                failures += 1
            if not args.dry_run and run_num < args.runs:
                print("Waiting 5 seconds before next run...")
                time.sleep(5)

    print("\n" + "=" * 70)
    print(f"Successful: {successes} | Failed: {failures}")
    print("=" * 70)


if __name__ == "__main__":
    main()
