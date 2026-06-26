#!/usr/bin/env python3
"""
Experiment 0b — Benchmark contamination probe.

Asks the forecaster model two types of questions about the Lyptus benchmark:

  Probe A (numeric recall):
      "What is the pass rate of [forecasted_model] on [task]?"
      One call per (forecasted_model, task) pair. Compares stated rate to
      ground-truth binary outcome -> Brier(recall) / Spearman rho.

  Probe B (task recognition):
      "Have you seen this task in your training data?"
      One call per task (not per model). Counts recognition rates by family.

IMPORTANT — these probe the EXACT cells used in Experiment I, not random tasks.
Experiment I's 300 cells = 25 distinct target tasks (5 per difficulty bin) x 12
forecasted models. The probe loads those exact (target_task_id, forecasted_model,
true_outcome) pairs from the Experiment I `control` run CSV, so Brier(recall) is
directly comparable to the Experiment I control Brier (0.137) on the same pairs.

Usage (from repo root):
    python intra_benchmark_calibration/experiments/0b_contamination_probe/run_probe.py \\
        --lyptus-dir ~/lyptus-data \\
        --output-dir intra_benchmark_calibration/experiments/0b_contamination_probe/results
        # optional: --cells-csv <path to a specific *_intra_estimates.csv>
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from intra_benchmark_calibration.lyptus_data import load_lyptus_dataset, LyptusTask
from shared.api_keys import resolve_api_key
from shared.llm_client import LLMSettings, initialize_client, make_api_call

# ---------------------------------------------------------------------------
# Cell loading — pin to the EXACT (task, model) pairs from Experiment I
# ---------------------------------------------------------------------------

DROP_MODELS = ["GPT-2", "GPT-3", "GPT-3.5"]

DEFAULT_CONTROL_GLOB = (
    "intra_benchmark_calibration/experiments/I_prompt_ablation_brier/"
    "results/control/*/*_intra_estimates.csv"
)


def find_default_cells_csv() -> Path:
    """Auto-discover the latest Experiment I control run CSV (the source of truth
    for which (task, model) pairs were elicited)."""
    matches = sorted(REPO_ROOT.glob(DEFAULT_CONTROL_GLOB))
    if not matches:
        sys.exit(
            "Could not auto-find the Experiment I control CSV. Pass --cells-csv "
            "pointing at a *_intra_estimates.csv (e.g. the control run)."
        )
    return matches[-1]


def load_target_pairs(cells_csv: Path):
    """Read the Experiment I cells CSV and return:
      - pairs: list of (target_task_id, forecasted_model, true_outcome)
      - task_ids: ordered list of distinct target task_ids
    These are the EXACT cells Experiment I scored, so recall Brier is comparable.
    """
    import pandas as pd
    df = pd.read_csv(cells_csv)
    needed = {"target_task_id", "forecasted_model", "outcome"}
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"{cells_csv} missing columns: {missing}")
    # one row per (task, model); dedupe in case of repeats
    df = df.drop_duplicates(subset=["target_task_id", "forecasted_model"])
    pairs = [
        (r["target_task_id"], r["forecasted_model"],
         float(r["outcome"]) if pd.notna(r["outcome"]) else None)
        for _, r in df.iterrows()
    ]
    task_ids = list(dict.fromkeys(df["target_task_id"].tolist()))
    return pairs, task_ids


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROBE_A_SYSTEM = (
    "You are a knowledgeable AI researcher with detailed knowledge of AI capability "
    "evaluation benchmarks, published leaderboards, and benchmark results."
)

PROBE_A_USER_TMPL = """\
I am studying benchmark contamination in AI models. The following task is from the \
Lyptus "Cyber Task Horizons" benchmark:

Task ID: {task_id}
Benchmark family: {task_family}

Task description:
{estimation_instructions}

Question: Based on any published results, benchmark documentation, leaderboard entries, \
or information in your training data, what is the pass rate (fraction of evaluated \
attempts scored as a binary success) for the model "{forecasted_model}" on this exact task?

Please respond in this EXACT format (no extra text before or after):
pass_rate: <a single number 0.0–1.0, or "unknown">
confidence: <high / medium / low / none>
reasoning: <one sentence — what specific source you are drawing on, or why you have no information>
"""

PROBE_B_SYSTEM = (
    "You are a knowledgeable AI researcher. Answer honestly and factually about "
    "what information may appear in your training data."
)

PROBE_B_USER_TMPL = """\
I am studying which AI benchmark tasks appear in public training corpora.

Does the following task description appear in any publicly available benchmark, dataset, \
CTF writeup, competition problem set, research paper, or other document that could be in \
an AI training corpus?

Task ID: {task_id}
Benchmark family: {task_family}

Task description:
{estimation_instructions}

Please respond in this EXACT format (no extra text before or after):
recognized: <yes / no / unsure>
source: <name of benchmark or source, or "N/A">
confidence: <high / medium / low / none>
reasoning: <one sentence>
"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _find(pattern: str, text: str, default: str = "") -> str:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else default


def parse_probe_a(text: str) -> dict:
    raw_rate = _find(r"pass_rate\s*:\s*(.+)", text, "unknown")
    confidence = _find(r"confidence\s*:\s*(\w+)", text, "none")
    reasoning = _find(r"reasoning\s*:\s*(.+)", text, "")
    # parse pass_rate to float
    pass_rate: Optional[float] = None
    try:
        pass_rate = float(raw_rate)
        pass_rate = max(0.0, min(1.0, pass_rate))
    except (ValueError, TypeError):
        pass_rate = None  # "unknown" or unparseable
    return {"raw_pass_rate": raw_rate, "pass_rate": pass_rate,
            "confidence": confidence, "reasoning": reasoning}


def parse_probe_b(text: str) -> dict:
    recognized = _find(r"recognized\s*:\s*(\w+)", text, "unsure")
    source = _find(r"source\s*:\s*(.+)", text, "N/A")
    confidence = _find(r"confidence\s*:\s*(\w+)", text, "none")
    reasoning = _find(r"reasoning\s*:\s*(.+)", text, "")
    return {"recognized": recognized, "source": source,
            "confidence": confidence, "reasoning": reasoning}


# ---------------------------------------------------------------------------
# Async runners
# ---------------------------------------------------------------------------

async def run_probe_a(
    client, semaphore, settings: LLMSettings,
    task: LyptusTask, forecasted_model: str, outcome: Optional[float],
    run_id: str,
) -> dict:
    user_prompt = PROBE_A_USER_TMPL.format(
        task_id=task.task_id,
        task_family=task.task_family,
        estimation_instructions=task.estimation_instructions.strip(),
        forecasted_model=forecasted_model,
    )
    raw = await make_api_call(
        client, semaphore, settings,
        PROBE_A_SYSTEM, user_prompt, max_tokens=512,
    )
    parsed = parse_probe_a(raw)
    return {
        "run_id": run_id,
        "probe": "A",
        "task_id": task.task_id,
        "task_family": task.task_family,
        "fst_minutes": task.fst_minutes,
        "forecasted_model": forecasted_model,
        "true_outcome": outcome,
        "raw_response": raw,
        **parsed,
    }


async def run_probe_b(
    client, semaphore, settings: LLMSettings,
    task: LyptusTask, run_id: str,
) -> dict:
    user_prompt = PROBE_B_USER_TMPL.format(
        task_id=task.task_id,
        task_family=task.task_family,
        estimation_instructions=task.estimation_instructions.strip(),
    )
    raw = await make_api_call(
        client, semaphore, settings,
        PROBE_B_SYSTEM, user_prompt, max_tokens=256,
    )
    parsed = parse_probe_b(raw)
    return {
        "run_id": run_id,
        "probe": "B",
        "task_id": task.task_id,
        "task_family": task.task_family,
        "fst_minutes": task.fst_minutes,
        "raw_response": raw,
        **parsed,
    }


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

PROBE_A_COLS = [
    "run_id", "probe", "task_id", "task_family", "fst_minutes",
    "forecasted_model", "true_outcome",
    "raw_pass_rate", "pass_rate", "confidence", "reasoning", "raw_response",
]

PROBE_B_COLS = [
    "run_id", "probe", "task_id", "task_family", "fst_minutes",
    "recognized", "source", "confidence", "reasoning", "raw_response",
]


def _write_row(writer_a, writer_b, row: dict):
    if row["probe"] == "A":
        writer_a.writerow({c: row.get(c, "") for c in PROBE_A_COLS})
    else:
        writer_b.writerow({c: row.get(c, "") for c in PROBE_B_COLS})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(
    lyptus_dir: Path,
    output_dir: Path,
    forecaster: str,
    max_concurrent: int,
    cells_csv: Optional[Path],
):
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    cells_csv = cells_csv or find_default_cells_csv()
    print(f"Loading Experiment I cells from {cells_csv} …")
    pairs, task_ids = load_target_pairs(cells_csv)

    print(f"Loading Lyptus dataset from {lyptus_dir} …")
    dataset = load_lyptus_dataset(lyptus_dir, drop_models=DROP_MODELS)

    # Map task_id -> LyptusTask for the exact target tasks
    missing_tasks = [tid for tid in task_ids if tid not in dataset.task_by_id]
    if missing_tasks:
        sys.exit(f"{len(missing_tasks)} target task_ids not in dataset: {missing_tasks[:5]}")
    probe_tasks = [dataset.task_by_id[tid] for tid in task_ids]

    n_probe_a = len(pairs)
    n_probe_b = len(task_ids)
    n_models = len({fm for _, fm, _ in pairs})
    print(f"  EXACT Experiment-I cells: {n_probe_a} (task, model) pairs")
    print(f"  Distinct target tasks: {n_probe_b}  |  distinct models: {n_models}")
    print(f"  Families covered: {sorted(set(t.task_family for t in probe_tasks))}")
    print(f"  Probe A calls: {n_probe_a}  |  Probe B calls: {n_probe_b}  "
          f"|  Total: {n_probe_a + n_probe_b}")

    # Resolve API keys. Search order: this experiment dir, the shared experiments
    # dir, then repo root .env, then process env (where the interactive shell
    # typically exports ANTHROPIC_API_KEY).
    config_dir = Path(__file__).parent
    experiments_dir = config_dir.parent
    search = (config_dir, experiments_dir, REPO_ROOT)
    api_key_anthropic = resolve_api_key("ANTHROPIC_API_KEY", *search)
    api_key_openai = resolve_api_key("OPENAI_API_KEY", *search)
    api_key_gemini = resolve_api_key("GEMINI_API_KEY", *search)
    if not (api_key_anthropic or api_key_openai):
        sys.exit(
            "No API key found. Export ANTHROPIC_API_KEY in your shell (the way the "
            "Experiment I runs were launched) or drop it into a .env file in "
            f"{config_dir}, {experiments_dir}, or {REPO_ROOT}."
        )

    client = initialize_client(api_key_anthropic, api_key_openai, forecaster, api_key_gemini)
    settings = LLMSettings(
        model=forecaster,
        temperature=0.0,          # deterministic for recall probe
        max_concurrent_calls=max_concurrent,
        rate_limit_calls=100,
        rate_limit_period=60,
        reasoning_effort="off",
    )
    semaphore = asyncio.Semaphore(max_concurrent)

    # Save metadata
    meta = {
        "run_id": run_id,
        "forecaster": forecaster,
        "cells_csv": str(cells_csv),
        "forecasted_models": sorted({fm for _, fm, _ in pairs}),
        "target_task_ids": task_ids,
        "n_probe_a": n_probe_a,
        "n_probe_b": n_probe_b,
        "drop_models": DROP_MODELS,
        "timestamp": datetime.now().isoformat(),
    }
    meta_path = output_dir / f"{run_id}_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  Metadata → {meta_path}")

    # Open CSVs
    path_a = output_dir / f"{run_id}_probe_a.csv"
    path_b = output_dir / f"{run_id}_probe_b.csv"
    fa = open(path_a, "w", newline="", encoding="utf-8")
    fb = open(path_b, "w", newline="", encoding="utf-8")
    writer_a = csv.DictWriter(fa, fieldnames=PROBE_A_COLS)
    writer_b = csv.DictWriter(fb, fieldnames=PROBE_B_COLS)
    writer_a.writeheader()
    writer_b.writeheader()

    # Build coroutines
    # Probe A: one per exact Experiment-I (task, model) pair.
    # Probe B: one per distinct target task.
    coros = []
    for task_id, fm, outcome in pairs:
        task = dataset.task_by_id[task_id]
        coros.append(run_probe_a(client, semaphore, settings, task, fm, outcome, run_id))
    for task in probe_tasks:
        coros.append(run_probe_b(client, semaphore, settings, task, run_id))

    # Run with progress
    from tqdm.asyncio import tqdm
    completed = 0
    errors = 0
    for coro in tqdm(asyncio.as_completed(coros), total=len(coros), desc="probe calls"):
        row = await coro
        if "Error:" in str(row.get("raw_response", "")):
            errors += 1
        _write_row(writer_a, writer_b, row)
        fa.flush()
        fb.flush()
        completed += 1

    fa.close()
    fb.close()
    print(f"\nDone. {completed} calls ({errors} errors).")
    print(f"  Probe A CSV → {path_a}")
    print(f"  Probe B CSV → {path_b}")
    print(f"\nNext: python …/analyse_probe.py --results-dir {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contamination probe for Lyptus benchmark.")
    parser.add_argument("--lyptus-dir", type=Path, default=Path.home() / "lyptus-data")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent / "results",
    )
    parser.add_argument("--forecaster", default="claude-sonnet-4-6")
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument(
        "--cells-csv", type=Path, default=None,
        help="Experiment I cells CSV (*_intra_estimates.csv). Defaults to the "
             "latest control run, so the probe hits the exact same (task, model) pairs.",
    )
    args = parser.parse_args()
    asyncio.run(main(args.lyptus_dir, args.output_dir, args.forecaster,
                     args.max_concurrent, args.cells_csv))
