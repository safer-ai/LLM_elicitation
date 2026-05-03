#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fetch the Lyptus offensive-cyber-time-horizons dataset, pinned to a specific
commit, without pulling the 18 GB of `.eval` files in Git LFS.

We only need:
  - analysis/figures/data/task_difficulties.parquet
  - analysis/figures/data/model_runs.parquet
  - data/tasks/<benchmark>/<benchmark>_tasks.jsonl  (7 small JSONLs)

Strategy:
  1. If `--repo-dir` already exists and is a git checkout, verify the HEAD SHA
     matches `LYPTUS_PINNED_SHA`. Warn (not fail) on mismatch — the user may be
     intentionally on a different commit.
  2. Otherwise, do a sparse, blobless, no-LFS clone:
        GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout
        git sparse-checkout init --cone
        git sparse-checkout set analysis/figures/data data/tasks
        git checkout <SHA>

Usage:
    python intra_benchmark_calibration/scripts/fetch_lyptus_data.py \\
        --repo-dir ~/lyptus-data
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

LYPTUS_REPO_URL = "https://github.com/lyptus-research/cyber-task-horizons-data.git"
LYPTUS_PINNED_SHA = "a514c63feb9eb34df3a013e46cc625338d5beeab"  # 2026-04-02 initial release


SPARSE_PATHS = [
    "analysis/figures/data",
    "data/tasks",
]


def _run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print(f"$ {' '.join(cmd)}{f'   (cwd={cwd})' if cwd else ''}")
    subprocess.check_call(cmd, cwd=cwd, env=env)


def _git_head(repo_dir: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def fetch(repo_dir: Path, *, force: bool = False) -> None:
    repo_dir = repo_dir.expanduser().resolve()

    head = _git_head(repo_dir)
    if head is not None:
        if head == LYPTUS_PINNED_SHA:
            print(f"OK: existing checkout at {repo_dir} matches pinned SHA {LYPTUS_PINNED_SHA[:10]}.")
            return
        if not force:
            print(
                f"WARNING: existing checkout at {repo_dir} is at {head[:10]} "
                f"but pinned SHA is {LYPTUS_PINNED_SHA[:10]}. "
                "Pass --force to overwrite, or update LYPTUS_PINNED_SHA in this script."
            )
            return
        print(f"--force: removing existing checkout at {repo_dir}")
        shutil.rmtree(repo_dir)

    if repo_dir.exists() and any(repo_dir.iterdir()):
        if not force:
            raise SystemExit(
                f"{repo_dir} exists and is non-empty but is not a git checkout. "
                "Pass --force to wipe and re-clone."
            )
        shutil.rmtree(repo_dir)

    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "git", "clone", "--filter=blob:none", "--no-checkout",
            LYPTUS_REPO_URL, str(repo_dir),
        ],
        env=env,
    )
    _run(["git", "sparse-checkout", "init", "--cone"], cwd=repo_dir, env=env)
    _run(["git", "sparse-checkout", "set", *SPARSE_PATHS], cwd=repo_dir, env=env)
    _run(["git", "checkout", LYPTUS_PINNED_SHA], cwd=repo_dir, env=env)

    head = _git_head(repo_dir)
    print(f"\nDone. Lyptus data at {repo_dir} (HEAD={head[:10] if head else '?'}).")
    print("Verify the small files are present, e.g.:")
    print(f"  ls {repo_dir / 'analysis/figures/data'}")
    print(f"  ls {repo_dir / 'data/tasks'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--repo-dir",
        type=Path,
        required=True,
        help="Where to place / find the Lyptus data checkout.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Wipe an existing non-matching checkout and re-clone.",
    )
    args = ap.parse_args()
    try:
        fetch(args.repo_dir, force=args.force)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\nFailed: {e}", file=sys.stderr)
        return e.returncode or 1


if __name__ == "__main__":
    sys.exit(main())
