#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Progressive results persistence for intra-benchmark calibration.

Two output files per run, in `output_dir/{run_id}/`:

  - `<descriptive>_estimates.csv`: ONE ROW PER ELICITATION (i.e. per
    cell × expert × Delphi round). Written under an asyncio lock by
    `append_elicitation_row()` immediately after each elicitation lands,
    so a mid-run crash loses at most one in-flight call.

  - `<descriptive>_results.json`: full structured record including the
    Lyptus provenance dict, the bin definition, every cell plan, every
    elicitation's full prompts + raw response, and the run metadata.
    Updated incrementally per elicitation as well.

CSV column contract (locked per user request, 2026-05-02):

    condition_id, run_id, timestamp,
    source_bin, target_bin,
    forecasted_model, target_task_id, target_task_family, target_fst_minutes,
    expert_id, delphi_round,
    p25, p50, p75,
    outcome,
    anchor_task_id, easier_task_ids,
    anchor_prompt_chars, easier_prompt_chars, target_prompt_chars,
    prompt_hash, rationale

`response_text` lives in the JSON only.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


CSV_COLUMNS: List[str] = [
    "condition_id",
    "run_id",
    "timestamp",
    "source_profile_type",  # single_bin | all_except_target | custom_subset
    "source_bin",           # int for single_bin/custom_subset; NaN for all_except_target
    "source_bins_shown",    # semicolon-joined list of bin indices actually shown
    "target_bin",
    "forecasted_model",
    "target_task_id",
    "target_task_family",
    "target_fst_minutes",
    "expert_id",
    "delphi_round",
    "p25",
    "p50",
    "p75",
    "outcome",
    "anchor_task_id",
    "easier_task_ids",
    "anchor_prompt_chars",
    "easier_prompt_chars",
    "target_prompt_chars",
    "prompt_hash",
    "rationale",
]


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _generate_run_id() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def prompt_hash(*texts: str) -> str:
    """Stable hash of the concatenated prompt strings, for traceability."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _descriptive_filename(run_id: str, model: str, num_experts: int, delphi_rounds: int,
                          temperature: float, base: str, ext: str) -> str:
    temp_str = f"tmp{temperature:.1f}".replace(".", "")
    safe_model = model.replace("/", "_")
    return f"{run_id}_intra_{safe_model}_nexp{num_experts}_nrnd{delphi_rounds}_{temp_str}_{base}.{ext}"


@dataclass
class RunHandles:
    """Bundle of paths + lock used by the workflow to persist incrementally."""

    run_id: str
    run_dir: Path
    csv_path: Path
    json_path: Path
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def initialize_run(
    *,
    output_base_dir: Path,
    model: str,
    num_experts: int,
    delphi_rounds: int,
    temperature: float,
    config_snapshot: Dict[str, Any],
    dataset_provenance: Dict[str, Any],
    bin_definition: Dict[str, Any],
    n_cells_planned: int,
) -> RunHandles:
    """Create the run directory and seed the CSV header + JSON metadata block."""
    run_id = _generate_run_id()
    run_dir = output_base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created run directory: {run_dir}")

    csv_path = run_dir / _descriptive_filename(
        run_id, model, num_experts, delphi_rounds, temperature, "estimates", "csv"
    )
    json_path = run_dir / _descriptive_filename(
        run_id, model, num_experts, delphi_rounds, temperature, "results", "json"
    )

    pd.DataFrame(columns=CSV_COLUMNS).to_csv(csv_path, index=False)
    logger.info(f"Initialised CSV: {csv_path}")

    initial_json: Dict[str, Any] = {
        "run_metadata": {
            "run_id": run_id,
            "timestamp_start": _now_iso(),
            "timestamp_end": None,
            "mode": "intra_benchmark",
            "model": model,
            "num_experts": num_experts,
            "delphi_rounds_max": delphi_rounds,
            "temperature": temperature,
            "n_cells_planned": n_cells_planned,
            # `n_elicitations_persisted` counts every row written to the
            # CSV (successes + parse failures + API failures). `succeeded`
            # is the subset with a parsed p50. Both updated incrementally;
            # `succeeded` is overwritten by workflow's own count at finalize.
            "n_elicitations_persisted": 0,
            "n_elicitations_succeeded": 0,
            "config_snapshot": config_snapshot,
            "dataset_provenance": dataset_provenance,
            "bin_definition": bin_definition,
        },
        "cells": [],          # per-cell summary (CellPlan + Delphi rounds)
        "elicitations": [],   # full per-elicitation records (prompts + raw responses)
    }
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(initial_json, fh, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Initialised JSON: {json_path}")

    return RunHandles(run_id=run_id, run_dir=run_dir, csv_path=csv_path, json_path=json_path)


async def append_elicitation_row(
    handles: RunHandles,
    *,
    csv_row: Dict[str, Any],
    json_elicitation_record: Dict[str, Any],
) -> None:
    """
    Persist one elicitation: append one row to CSV and append one record to the
    JSON's `elicitations` list. Single asyncio lock serialises both writes so
    the on-disk state is always consistent.
    """
    async with handles.lock:
        df = pd.DataFrame([{c: csv_row.get(c) for c in CSV_COLUMNS}])
        df.to_csv(handles.csv_path, mode="a", header=False, index=False)

        with handles.json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        data["elicitations"].append(json_elicitation_record)
        md = data["run_metadata"]
        md["n_elicitations_persisted"] = len(data["elicitations"])
        # A row is "succeeded" iff parse_probability_response found a p50.
        if json_elicitation_record.get("error") is None and json_elicitation_record.get("p50") is not None:
            md["n_elicitations_succeeded"] = md.get("n_elicitations_succeeded", 0) + 1
        with handles.json_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, default=str)


async def append_cell_summary(handles: RunHandles, cell_summary: Dict[str, Any]) -> None:
    """Append a per-cell summary (Delphi-round means etc.) to the JSON."""
    async with handles.lock:
        with handles.json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        data["cells"].append(cell_summary)
        with handles.json_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, default=str)


def finalize_run(
    handles: RunHandles,
    *,
    n_elicitations_attempted: int,
    n_elicitations_succeeded: Optional[int] = None,
) -> None:
    """
    Mark run end time + total counts.

    `n_elicitations_succeeded`, if supplied, OVERWRITES the live counter that
    was being maintained by `append_elicitation_row` (the workflow has the
    authoritative count of parse-successful elicitations).
    """
    with handles.json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    end_iso = _now_iso()
    md = data["run_metadata"]
    md["timestamp_end"] = end_iso
    md["n_elicitations_attempted"] = n_elicitations_attempted
    if n_elicitations_succeeded is not None:
        md["n_elicitations_succeeded"] = n_elicitations_succeeded

    try:
        start_dt = datetime.datetime.fromisoformat(md["timestamp_start"])
        end_dt = datetime.datetime.fromisoformat(end_iso)
        md["duration_seconds"] = (end_dt - start_dt).total_seconds()
    except Exception:
        pass

    with handles.json_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Finalised run {handles.run_id}")


def update_registry(
    *,
    registry_file: Path,
    run_id: str,
    model: str,
    num_experts: int,
    delphi_rounds: int,
    n_elicitations_attempted: int,
    n_elicitations_completed: int,
    output_path: Path,
    config_file: str,
    timestamp_start: str,
) -> None:
    """Add the run to a top-level run registry (created if missing)."""
    registry: Dict[str, Any] = {"runs": [], "last_updated": None}
    if registry_file.is_file():
        try:
            with registry_file.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict) and isinstance(loaded.get("runs"), list):
                registry = loaded
        except Exception as e:
            logger.error(f"Failed to load registry; reinitialising: {e}")

    if any(r.get("run_id") == run_id for r in registry["runs"]):
        logger.warning(f"Run ID {run_id} already in registry; skipping append.")
        return

    registry["runs"].append({
        "run_id": run_id,
        "mode": "intra_benchmark",
        "timestamp": timestamp_start,
        "model": model,
        "num_experts": num_experts,
        "delphi_rounds": delphi_rounds,
        "n_elicitations_attempted": n_elicitations_attempted,
        "n_elicitations_completed": n_elicitations_completed,
        "output_path": str(output_path),
        "config_file": config_file,
    })
    registry["runs"].sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    registry["last_updated"] = _now_iso()

    registry_file.parent.mkdir(parents=True, exist_ok=True)
    with registry_file.open("w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Updated registry at {registry_file}")


def build_csv_row(
    *,
    handles: RunHandles,
    plan,                       # task_selector.CellPlan (typed loosely to avoid circular import)
    expert_id: str,
    delphi_round: int,
    parsed: Dict[str, Any],
    prompts_for_hash: List[str],
    target_prompt_chars: int,
) -> Dict[str, Any]:
    """Materialise the CSV-row dict for one elicitation.

    Anchor / easier fields are semicolon-joined across all shown source bins
    (in `plan.profiles` order). For single_bin mode that yields exactly one
    anchor and `n_examples_per_source_bin` easier tasks, with no semicolons in
    anchor_task_id / anchor_prompt_chars. For all_except_target / custom_subset
    you get one anchor per shown bin (joined) and all easier tasks across all
    shown bins (joined).

    Use the `source_bins_shown` column + `n_examples_per_source_bin` from the
    config to reconstruct per-bin grouping if you need it.
    """
    profiles = plan.profiles
    anchors_ids = ";".join(p.anchor.task_id for p in profiles)
    anchor_chars = ";".join(str(len(p.anchor.estimation_instructions)) for p in profiles)
    easier_ids = ";".join(t.task_id for p in profiles for t in p.easier_tasks)
    easier_chars = ";".join(
        str(len(t.estimation_instructions)) for p in profiles for t in p.easier_tasks
    )
    bins_shown_str = ";".join(str(b) for b in plan.source_bins_to_show)

    # source_bin: real int for single_bin / custom_subset; None (-> NaN cell) for all_except_target
    source_bin_csv = plan.source_bin_i if plan.source_bin_i is not None else None

    return {
        "condition_id": plan.cell_id,
        "run_id": handles.run_id,
        "timestamp": _now_iso(),
        "source_profile_type": plan.source_profile_type,
        "source_bin": source_bin_csv,
        "source_bins_shown": bins_shown_str,
        "target_bin": plan.target_bin_j,
        "forecasted_model": plan.forecasted_model,
        "target_task_id": plan.target_task.task_id,
        "target_task_family": plan.target_task.task_family,
        "target_fst_minutes": round(plan.target_task.fst_minutes, 4),
        "expert_id": expert_id,
        "delphi_round": delphi_round,
        "p25": parsed.get("percentile_25th"),
        "p50": parsed.get("percentile_50th"),
        "p75": parsed.get("percentile_75th"),
        "outcome": int(plan.target_outcome),
        "anchor_task_id": anchors_ids,
        "easier_task_ids": easier_ids,
        "anchor_prompt_chars": anchor_chars,
        "easier_prompt_chars": easier_chars,
        "target_prompt_chars": target_prompt_chars,
        "prompt_hash": prompt_hash(*prompts_for_hash),
        "rationale": (parsed.get("rationale") or "").strip(),
    }


def build_json_record(
    *,
    csv_row: Dict[str, Any],
    plan,
    expert_id: str,
    delphi_round: int,
    system_prompt: str,
    analysis_user_prompt: Optional[str],
    estimation_user_prompt: str,
    raw_analysis: Optional[str],
    raw_estimation: str,
    error: Optional[str],
) -> Dict[str, Any]:
    """Materialise the full JSON record for one elicitation (prompts + responses).

    Per-shown-bin context lives in `per_bin_profile_M` (parallel list to
    `source_bins_shown`). With single_bin mode this list has length 1.
    """
    per_bin = [
        {
            "bin_index": p.bin_index,
            "pass_rate_M": p.pass_rate,
            "n_evaluated_M": p.n_evaluated,
            "n_in_bin": p.n_in_bin,
            "anchor_task_id": p.anchor.task_id,
            "anchor_outcome_M": p.anchor_outcome,
            "easier_task_ids": [t.task_id for t in p.easier_tasks],
            "easier_outcomes_M": list(p.easier_outcomes or []),
        }
        for p in plan.profiles
    ]
    return {
        **csv_row,
        "system_prompt": system_prompt,
        "analysis_user_prompt": analysis_user_prompt,
        "estimation_user_prompt": estimation_user_prompt,
        "raw_analysis": raw_analysis,
        "raw_estimation": raw_estimation,
        "error": error,
        "per_bin_profile_M": per_bin,
    }
