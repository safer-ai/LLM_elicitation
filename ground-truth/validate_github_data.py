#!/usr/bin/env python3
"""
Validate ground-truth/github_data/*.parquet for internal consistency.

Usage (from repo root):
    python ground-truth/validate_github_data.py

Exit code 0 always; read stdout for [OK] vs [WARN] / [CHECK].
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

KEYS_RUN_MODEL = [
    "task_id",
    "task_family",
    "agent",
    "alias",
    "score_binarized",
    "total_tokens",
]


def main() -> int:
    root = Path(__file__).resolve().parent
    base = root / "github_data"
    runs_path = base / "runs.parquet"
    model_path = base / "model_runs.parquet"
    diff_path = base / "task_difficulties.parquet"

    for p in (runs_path, model_path, diff_path):
        if not p.exists():
            print(f"[ERR] Missing file: {p}", file=sys.stderr)
            return 2

    r = pd.read_parquet(runs_path)
    m = pd.read_parquet(model_path)
    t = pd.read_parquet(diff_path)

    print("=== Files ===")
    for name, df in (
        ("runs.parquet", r),
        ("model_runs.parquet", m),
        ("task_difficulties.parquet", t),
    ):
        print(f"  {name}: {len(df):,} rows × {len(df.columns)} cols")

    issues = 0

    # --- Duplicates ---
    print("\n=== Uniqueness ===")
    td_dup = int(t["task_id"].duplicated().sum())
    if td_dup:
        print(f"[WARN] task_difficulties duplicate task_id: {td_dup}")
        issues += 1
    else:
        print("[OK] task_difficulties: task_id is unique (1 row per task)")

    rk = r.duplicated(subset=KEYS_RUN_MODEL).sum()
    mk = m.duplicated(subset=KEYS_RUN_MODEL).sum()
    if rk or mk:
        print(f"[WARN] Duplicate full keys — runs: {rk}, model_runs: {mk}")
        issues += 1
    else:
        print("[OK] No duplicate rows on (task_id, task_family, agent, alias, score, tokens)")

    # --- runs vs model_runs ---
    print("\n=== runs.parquet vs model_runs.parquet ===")
    merged = r.merge(m.assign(_m=1), on=KEYS_RUN_MODEL, how="outer", indicator=True)
    vc = merged["_merge"].value_counts()
    both = int(vc.get("both", 0))
    left = int(vc.get("left_only", 0))
    right = int(vc.get("right_only", 0))
    print(f"  Exact key match: {both:,} | only in runs: {left} | only in model_runs: {right}")

    only_r = merged[merged["_merge"] == "left_only"]
    if len(only_r):
        tok0 = (only_r["total_tokens"] == 0).all()
        print(
            f"  Runs-only rows: all total_tokens==0: {tok0} "
            f"(score 0/1 counts: {only_r['score_binarized'].value_counts().to_dict()})"
        )
        print("  Runs-only by task_family:")
        print(only_r["task_family"].value_counts().to_string(header=False))
        print("  Runs-only by agent:")
        print(only_r["agent"].value_counts().to_string(header=False))
        if not tok0:
            print("[WARN] Some runs-only rows have total_tokens > 0 (unexpected)")
            issues += 1
        else:
            print(
                "[CHECK] Runs-only rows are legacy evals with zeroed token counts; "
                "model_runs likely omits them on purpose."
            )

    only_m = merged[merged["_merge"] == "right_only"]
    if len(only_m):
        print("  model_runs-only row(s):")
        print(only_m[KEYS_RUN_MODEL].to_string(index=False))
        print(
            "[WARN] At least one model×task outcome appears in model_runs but not runs "
            "(export mismatch — pick one table as canonical for model outcomes)."
        )
        issues += 1

    # --- task difficulty coverage for runs ---
    print("\n=== task_difficulties vs runs ===")
    r_ids = set(r["task_id"])
    t_ids = set(t["task_id"])
    missing = r_ids - t_ids
    if missing:
        print(f"[WARN] {len(missing)} task_ids appear in runs but not task_difficulties")
        issues += 1
    else:
        print("[OK] Every task_id in runs has a row in task_difficulties")

    only_t = t_ids - r_ids
    print(
        f"  Tasks in task_difficulties with no run row: {len(only_t):,} "
        f"(expected: difficulty table covers a wider task universe)"
    )

    # --- Human vs model difficulty clarity ---
    print("\n=== Ground-truth difficulty columns (task_difficulties) ===")
    n = len(t)
    has_human = t["best_available_minutes"].notna()
    has_model = t["model_estimate_minutes"].notna()
    print(f"  Rows: {n:,}")
    print(f"  best_available_minutes (human-composited label): {has_human.sum():,} ({has_human.mean():.0%})")
    print(f"  model_estimate_minutes: {has_model.sum():,} ({has_model.mean():.0%})")
    print(
        "  [CHECK] Sparse columns (completion_minutes, firstblood, …) are often null by design; "
        "use best_available_minutes when present for human-side difficulty."
    )

    # --- Optional HF cross-check ---
    print("\n=== Optional: Hugging Face model_estimates ===")
    try:
        from datasets import load_dataset

        me = load_dataset(
            "lyptus-research/cyber-task-horizons", "model_estimates", split="train"
        ).to_pandas()
        hf_dup = int(me["task_id"].duplicated().sum())
        print(f"  HF rows: {len(me):,}; duplicate task_id rows on HF: {hf_dup}")
        if hf_dup:
            print(
                "  [CHECK] Public HF has duplicate task_ids (same task, two estimates); "
                "your Parquet has unique task_id — cleaner for joins."
            )
        if len(me) == len(t):
            print("  [OK] Row count matches task_difficulties.parquet")
    except Exception as exc:
        print(f"  (skipped: {exc})")

    print("\n=== Canonical use (recommendation) ===")
    print("  • Model pass/fail + tokens: prefer model_runs.parquet for frontier coverage;")
    print("    merge runs.parquet if you need the 109 zero-token legacy rows.")
    print("  • Task-level difficulty: task_difficulties.parquet; treat best_available_minutes")
    print("    as human-grounded only when non-null.")
    print(f"\nDone. Issue-style flags: {issues}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
