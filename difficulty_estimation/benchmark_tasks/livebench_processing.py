#!/usr/bin/env python3
"""
Process LiveBench HuggingFace datasets into benchmark YAMLs, leaderboards,
ordered files, and solve matrices for both LCB_generation and coding_completion.

Requires: datasets, pyyaml
Requires HF authentication: `huggingface-cli login`

Outputs (per task type, in the current directory):
  livebench_{task}.yaml                  -- raw benchmark questions
  livebench_{task}_leaderboard.json      -- per-model overall pass rates
  livebench_{task}_ordered.yaml          -- tasks sorted by increasing difficulty
  livebench_{task}_solve_matrix.json     -- binary model x task solve outcomes
"""

import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import yaml
from datasets import load_dataset

OUTPUT_DIR = Path(__file__).parent

EXPECTED_COUNTS = {
    "LCB_generation": 78,
    "coding_completion": 50,
}

BENCHMARK_DESCRIPTIONS = {
    "LCB_generation": (
        "LiveBench LCB Generation is a code generation benchmark comprising 78 competition "
        "programming problems randomly selected from the April 2024 release of LiveCodeBench. "
        "Problems are sourced from LeetCode and AtCoder, released in or after November 2023. "
        "Models must parse a textual problem statement and write a complete, correct Python 3 "
        "solution. Evaluation uses pass@1: a solution is considered correct if and only if it "
        "passes all public and private test cases. Each model has one attempt per question."
    ),
    "coding_completion": (
        "LiveBench Coding Completion is a code completion benchmark comprising 50 competition "
        "programming problems from LeetCode, sourced from LiveCodeBench's April 2024 release. "
        "Unlike full code generation, a partial correct solution is provided in the prompt and "
        "the model must complete it to solve the question. Partial solutions come from GitHub "
        "(kamyu104/LeetCode-Solutions), with the last 15%% of medium/hard solutions and "
        "30-70%% of easy solutions omitted. Evaluation uses pass@1: a solution is considered "
        "correct if and only if it passes all public and private test cases."
    ),
}

CONTENT_COLUMNS_BASE = ["question_id", "question_title", "turns", "public_test_cases"]
CONTENT_COLUMNS_COMPLETION = CONTENT_COLUMNS_BASE + ["partial_solution", "remainder"]


# ---------------------------------------------------------------------------
# Load & filter
# ---------------------------------------------------------------------------

def load_judgment_data():
    """Load livebench/model_judgment, filter to coding category."""
    print("Loading livebench/model_judgment ...")
    ds = load_dataset("livebench/model_judgment")
    # Dataset may come as DatasetDict with a single split
    if hasattr(ds, "keys"):
        split_name = list(ds.keys())[0]
        ds = ds[split_name]

    # Filter to coding category
    ds = ds.filter(lambda row: row["category"] == "coding")
    print(f"  Coding rows after filter: {len(ds)}")

    # Keep relevant columns
    records = []
    for row in ds:
        records.append({
            "question_id": row["question_id"],
            "task": row["task"],
            "model": row["model"],
            "score": row["score"],
        })

    return records


def load_coding_content():
    """Load livebench/coding, return dict keyed by question_id."""
    print("Loading livebench/coding ...")
    ds = load_dataset("livebench/coding")
    if hasattr(ds, "keys"):
        split_name = list(ds.keys())[0]
        ds = ds[split_name]

    content = {}
    for row in ds:
        qid = row["question_id"]
        turns_raw = row.get("turns", [])
        # turns is typically a list of strings; keep as-is for YAML storage
        content[qid] = {
            "question_id": qid,
            "question_title": row.get("question_title", ""),
            "turns": turns_raw,
            "public_test_cases": row.get("public_test_cases", ""),
            "task": row.get("task", ""),
            "partial_solution": row.get("partial_solution", ""),
            "solution": row.get("solution", ""),
            "remainder": row.get("remainder", ""),
        }

    print(f"  Loaded {len(content)} coding questions")
    return content


# ---------------------------------------------------------------------------
# Processing per task type
# ---------------------------------------------------------------------------

def process_task(task_name: str, judgment_rows: list, content_by_qid: dict):
    """
    For a given task (LCB_generation or coding_completion):
      1. raw benchmark YAML
      2. leaderboard JSON
      3. ordered YAML (by increasing difficulty)
      4. solve matrix JSON
    """
    slug = task_name  # e.g. "LCB_generation"
    print(f"\n{'='*60}")
    print(f"Processing task: {slug}")
    print(f"{'='*60}")

    # Filter judgment rows for this task
    rows = [r for r in judgment_rows if r["task"] == task_name]
    print(f"  Judgment rows for {slug}: {len(rows)}")

    # Unique questions and models
    question_ids = sorted(set(r["question_id"] for r in rows))
    models = sorted(set(r["model"] for r in rows))
    print(f"  Unique questions: {len(question_ids)}")
    print(f"  Unique models: {len(models)}")

    expected = EXPECTED_COUNTS[task_name]
    assert len(question_ids) == expected, (
        f"Expected {expected} questions for {task_name}, got {len(question_ids)}"
    )

    # Verify scores are binary (0 or 1) for coding tasks
    scores = [r["score"] for r in rows]
    unique_scores = sorted(set(scores))
    print(f"  Unique score values: {unique_scores}")
    assert all(s in (0, 0.0, 1, 1.0) for s in unique_scores), (
        f"Expected binary scores for coding tasks, got: {unique_scores}"
    )

    # Build solve matrix: {model: {question_id: bool}}
    solve_matrix = defaultdict(dict)
    for r in rows:
        solve_matrix[r["model"]][r["question_id"]] = bool(r["score"] >= 1)

    # Check: some models may not have been evaluated on every question
    models_with_missing = []
    for m in models:
        evaluated = set(solve_matrix[m].keys())
        if evaluated != set(question_ids):
            missing = set(question_ids) - evaluated
            models_with_missing.append((m, len(missing)))
    if models_with_missing:
        print(f"  {len(models_with_missing)} models have partial evaluations "
              f"(treated as 'not solved' for missing questions)")

    # --- 1. Raw benchmark YAML ---
    if task_name == "coding_completion":
        columns = CONTENT_COLUMNS_COMPLETION
    else:
        columns = CONTENT_COLUMNS_BASE

    tasks_list = []
    missing_content = []
    for qid in question_ids:
        if qid not in content_by_qid:
            missing_content.append(qid)
            continue
        c = content_by_qid[qid]
        entry = {col: c[col] for col in columns}
        tasks_list.append(entry)

    if missing_content:
        print(f"  WARNING: {len(missing_content)} questions missing from coding content dataset")
        print(f"    First few: {missing_content[:5]}")

    raw_yaml_path = OUTPUT_DIR / f"livebench_{slug}.yaml"
    raw_output = {
        "benchmark_description": BENCHMARK_DESCRIPTIONS[task_name],
        "tasks": tasks_list,
    }
    with open(raw_yaml_path, "w") as f:
        yaml.safe_dump(raw_output, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    print(f"  Saved raw YAML: {raw_yaml_path} ({len(tasks_list)} tasks)")

    # --- 2. Leaderboard JSON ---
    n_questions = len(question_ids)
    model_scores = []
    for m in sorted(models):
        n_solved = sum(1 for qid in question_ids if solve_matrix[m].get(qid, False))
        pass_rate = n_solved / n_questions
        model_scores.append({"model": m, "score": round(pass_rate, 6)})

    model_scores.sort(key=lambda x: x["score"], reverse=True)

    leaderboard = {
        "metadata": {
            "benchmark_name": f"livebench_{slug}",
            "source": "livebench/model_judgment (HuggingFace)",
            "n_models": len(model_scores),
            "n_questions": n_questions,
            "score_scale": "fraction_0_1",
            "notes": f"Pass@1 rate across {n_questions} {slug} coding tasks from LiveBench.",
        },
        "results": model_scores,
    }

    lb_path = OUTPUT_DIR / f"livebench_{slug}_leaderboard.json"
    with open(lb_path, "w") as f:
        json.dump(leaderboard, f, indent=2)
    print(f"  Saved leaderboard: {lb_path} ({len(model_scores)} models)")

    # Print score distribution summary
    scores_list = [m["score"] for m in model_scores]
    print(f"  Score range: [{min(scores_list):.3f}, {max(scores_list):.3f}]")
    print(f"  Score median: {median(scores_list):.3f}")

    # --- 3. Ordered YAML (increasing difficulty = easiest first) ---
    # Per-task solve rate: fraction of models evaluated on this question that solved it.
    # Only count models that were actually evaluated (have an entry in the solve matrix).
    per_task_stats = []
    for qid in question_ids:
        evaluated_models = [m for m in models if qid in solve_matrix[m]]
        n_evaluated = len(evaluated_models)
        n_solved_by = sum(1 for m in evaluated_models if solve_matrix[m][qid])
        solve_rate = n_solved_by / n_evaluated if n_evaluated > 0 else 0.0
        per_task_stats.append({
            "question_id": qid,
            "solve_rate": solve_rate,
            "n_evaluated": n_evaluated,
        })

    # Sort by solve_rate DESCENDING (highest solve rate = easiest = first)
    per_task_stats.sort(key=lambda x: x["solve_rate"], reverse=True)

    ordered_tasks = []
    for stat in per_task_stats:
        qid = stat["question_id"]
        if qid not in content_by_qid:
            continue
        c = content_by_qid[qid]
        entry = {col: c[col] for col in columns}
        entry["metrics"] = {"solve_rate": round(stat["solve_rate"], 6)}
        ordered_tasks.append(entry)

    ordered_output = {
        "benchmark_description": BENCHMARK_DESCRIPTIONS[task_name],
        "metrics_to_use_for_estimation": ["solve_rate"],
        "tasks": ordered_tasks,
    }

    ordered_path = OUTPUT_DIR / f"livebench_{slug}_ordered.yaml"
    with open(ordered_path, "w") as f:
        yaml.safe_dump(ordered_output, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    print(f"  Saved ordered YAML: {ordered_path} ({len(ordered_tasks)} tasks)")

    # Print solve rate distribution summary
    rates = [s["solve_rate"] for s in per_task_stats]
    print(f"  Solve rate range: [{min(rates):.3f}, {max(rates):.3f}]")
    print(f"  Solve rate median: {median(rates):.3f}")
    saturated = sum(1 for r in rates if r >= 0.95)
    impossible = sum(1 for r in rates if r <= 0.05)
    print(f"  Saturated tasks (>=95% solve rate): {saturated}")
    print(f"  Near-impossible tasks (<=5% solve rate): {impossible}")

    # --- 4. Solve matrix JSON ---
    sm_output = {m: {qid: solve_matrix[m].get(qid, False) for qid in question_ids}
                 for m in sorted(models)}

    sm_path = OUTPUT_DIR / f"livebench_{slug}_solve_matrix.json"
    with open(sm_path, "w") as f:
        json.dump(sm_output, f, indent=2)
    print(f"  Saved solve matrix: {sm_path}")

    return {
        "task_name": task_name,
        "n_questions": n_questions,
        "n_models": len(models),
        "question_ids": question_ids,
        "models": models,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    judgment_rows = load_judgment_data()
    content_by_qid = load_coding_content()

    # Inspect what tasks are available
    tasks_in_data = sorted(set(r["task"] for r in judgment_rows))
    print(f"\nTasks in coding category: {tasks_in_data}")

    results = {}
    for task_name in ["LCB_generation", "coding_completion"]:
        assert task_name in tasks_in_data, (
            f"Task '{task_name}' not found in data. Available: {tasks_in_data}"
        )
        results[task_name] = process_task(task_name, judgment_rows, content_by_qid)

    # Print cross-task model overlap summary
    models_lcb = set(results["LCB_generation"]["models"])
    models_cc = set(results["coding_completion"]["models"])
    overlap = models_lcb & models_cc
    print(f"\n{'='*60}")
    print("Cross-task model overlap:")
    print(f"  LCB_generation models: {len(models_lcb)}")
    print(f"  coding_completion models: {len(models_cc)}")
    print(f"  Overlap: {len(overlap)}")
    print(f"  LCB-only: {len(models_lcb - models_cc)}")
    print(f"  CC-only: {len(models_cc - models_lcb)}")
    print(f"{'='*60}")

    print("\nDone! All files saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
