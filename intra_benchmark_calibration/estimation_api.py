#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Thin per-cell estimation API for external callers (notably the GEPA fork).

The batch pipeline (`run_calibration.py` -> `workflow.py`) is YAML-driven,
persists every elicitation to disk and runs cells sequentially. External
optimisers instead need: given a *candidate estimation template* and a set of
cells, run the forecaster once per cell (single API call, no capability
analysis, no Delphi, no persistence) and return parsed percentiles, the full
rollout and the Brier score -- with independent cells evaluated concurrently.

This module deliberately reuses the existing building blocks rather than
duplicating them:

  - `task_selector.build_cell_plans`  (evidence construction / admissibility)
  - `prompt_builder.assemble_prompts` (template filling)
  - `shared.llm_client.make_api_call` (provider calls, retries, rate limits)
  - `shared.parsing.parse_probability_response` (XML percentile parsing)

Template contract: the candidate template is used as the round-1 estimation
template (`initial_intra_solve_estimation`) and is filled via `str.format`
with the usual placeholder set ({forecasted_model}, {capability_profile},
{target_task_text}, ...). Malformed templates (stray braces, unknown
placeholders) are reported per-cell via `CellResult.error` rather than
raising, so an optimiser can score them as failures.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from shared.api_keys import resolve_api_key
from shared.llm_client import LLMSettings, initialize_client, make_api_call
from shared.parsing import parse_probability_response

from intra_benchmark_calibration.binning import BinAssignment, compute_bins_right_closed
from intra_benchmark_calibration.lyptus_data import LyptusDataset
from intra_benchmark_calibration.prompt_builder import assemble_prompts
from intra_benchmark_calibration.task_selector import CellPlan, build_cell_plans

_DEFAULT_MAX_TOKENS = 20000

# Sentinel analysis template: `assemble_prompts` insists on a round-1 analysis
# template even when the caller never issues the analysis call.
_UNUSED_ANALYSIS_TEMPLATE = "(unused -- single-call estimation)"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellResult:
    """Outcome of one single-call elicitation on one cell."""

    cell_id: str
    target_task_id: str
    target_bin: int
    forecasted_model: str
    outcome: float  # ground-truth binary outcome (0.0 / 1.0)
    p25: Optional[float]
    p50: Optional[float]
    p75: Optional[float]
    brier: Optional[float]  # (p50 - outcome)^2; None when p50 unparsable
    rationale: str
    system_prompt: str
    user_prompt: str  # the fully-assembled estimation prompt
    raw_response: str  # full rollout text from the forecaster
    error: Optional[str]  # None on success
    duration_seconds: float
    # Fresh samples spent recovering an unparsable response (0 = first
    # attempt parsed). Lets audits track how often the retry fires.
    parse_retries_used: int = 0

    @property
    def parsed_ok(self) -> bool:
        return self.error is None and self.p50 is not None


def _failure_result(
    plan: CellPlan,
    *,
    system_prompt: str,
    user_prompt: str,
    raw_response: str,
    error: str,
    t0: float,
    parse_retries_used: int = 0,
) -> CellResult:
    return CellResult(
        cell_id=plan.cell_id,
        target_task_id=plan.target_task.task_id,
        target_bin=plan.target_bin_j,
        forecasted_model=plan.forecasted_model,
        outcome=plan.target_outcome,
        p25=None,
        p50=None,
        p75=None,
        brier=None,
        rationale="",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        raw_response=raw_response,
        error=error,
        duration_seconds=time.monotonic() - t0,
        parse_retries_used=parse_retries_used,
    )


# ---------------------------------------------------------------------------
# Single-cell elicitation (async)
# ---------------------------------------------------------------------------

async def _estimate_single_cell(
    *,
    plan: CellPlan,
    estimation_template: str,
    system_prompt: str,
    client,
    semaphore: asyncio.Semaphore,
    llm_settings: LLMSettings,
    include_bin_rate: bool,
    include_task_outcomes: bool,
    max_tokens: int,
    parse_retries: int,
) -> CellResult:
    """One cell -> one estimation API call -> parsed percentiles + Brier."""
    t0 = time.monotonic()
    templates = {
        "intra_capability_analysis": _UNUSED_ANALYSIS_TEMPLATE,
        "initial_intra_solve_estimation": estimation_template,
    }
    try:
        prep = assemble_prompts(
            plan=plan,
            persona_system_prompt=system_prompt,
            prompt_templates=templates,
            round_num=1,
            prev_round_responses=None,
            technical_analysis=None,
            benchmark_description=None,
            ground_truth_summary=None,
            include_target_solution=False,
            include_bin_rate=include_bin_rate,
            include_task_outcomes=include_task_outcomes,
        )
    except Exception as e:
        # Malformed candidate template (unknown placeholder / stray braces /
        # empty template). Never raise for a bad candidate: report per-cell.
        return _failure_result(
            plan,
            system_prompt=system_prompt,
            user_prompt="",
            raw_response="",
            error=f"template_format_error: {type(e).__name__}: {e}",
            t0=t0,
        )

    user_prompt = prep.estimation_user_prompt
    # An unparsable response is almost always a one-character format typo
    # from temperature sampling (e.g. a mismatched </p75> closing tag), so a
    # fresh sample recovers the cell. Retry ONLY the unparsable case:
    # api_error has already been through the client's own retry machinery,
    # and template_format_error is deterministic (retrying cannot help).
    attempts = 1 + max(0, parse_retries)
    response_text = ""
    parsed: dict = {}
    p25 = p50 = p75 = None
    attempt = 0
    for attempt in range(attempts):
        response_text = await make_api_call(
            client, semaphore, llm_settings, system_prompt, user_prompt, max_tokens
        )
        if response_text.startswith("Error:"):
            return _failure_result(
                plan,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_response=response_text,
                error=f"api_error: {response_text}",
                t0=t0,
                parse_retries_used=attempt,
            )
        parsed = parse_probability_response(response_text)
        p25 = parsed.get("percentile_25th")
        p50 = parsed.get("percentile_50th")
        p75 = parsed.get("percentile_75th")
        if p50 is not None:
            break

    error: Optional[str] = None
    brier: Optional[float] = None
    if p50 is None:
        error = f"parse_failed: no p50 extracted from response (after {attempts} attempt(s))"
    else:
        brier = (float(p50) - float(plan.target_outcome)) ** 2

    return CellResult(
        cell_id=plan.cell_id,
        target_task_id=plan.target_task.task_id,
        target_bin=plan.target_bin_j,
        forecasted_model=plan.forecasted_model,
        outcome=plan.target_outcome,
        p25=p25,
        p50=p50,
        p75=p75,
        brier=brier,
        rationale=parsed.get("rationale") or "",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        raw_response=response_text,
        error=error,
        duration_seconds=time.monotonic() - t0,
        parse_retries_used=attempt,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_llm_settings(
    *,
    model: str,
    temperature: float = 1.0,
    reasoning_effort: str = "off",
    max_concurrent_calls: int = 10,
    rate_limit_calls: int = 200,
    rate_limit_period: int = 60,
) -> LLMSettings:
    """Convenience constructor mirroring the YAML `llm_settings` block."""
    return LLMSettings(
        model=model,
        temperature=temperature,
        max_concurrent_calls=max_concurrent_calls,
        rate_limit_calls=rate_limit_calls,
        rate_limit_period=rate_limit_period,
        reasoning_effort=reasoning_effort,
    )


def resolve_anthropic_api_key() -> Optional[str]:
    """Resolve the Anthropic key the same way the batch pipeline does.

    Precedence: <intra dir>/.env -> <repo root>/.env -> process env.
    """
    here = Path(__file__).resolve().parent
    return resolve_api_key("ANTHROPIC_API_KEY", here, here.parent)


async def estimate_cells_async(
    plans: Sequence[CellPlan],
    estimation_template: str,
    *,
    llm_settings: LLMSettings,
    system_prompt: str = "",
    client=None,
    api_key_anthropic: Optional[str] = None,
    api_key_openai: Optional[str] = None,
    api_key_gemini: Optional[str] = None,
    include_bin_rate: bool = True,
    include_task_outcomes: bool = True,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    parse_retries: int = 1,
) -> List[CellResult]:
    """Evaluate `estimation_template` on every cell in `plans`, concurrently.

    Results align one-to-one with `plans`. Per-cell failures (template
    formatting, API errors, unparsable responses) are reported via
    `CellResult.error`; only systemic misconfiguration raises. An unparsable
    response is re-sampled up to `parse_retries` times before being reported
    as a failure (`CellResult.parse_retries_used` records what was spent).
    """
    owns_client = client is None
    if client is None:
        if api_key_anthropic is None:
            api_key_anthropic = resolve_anthropic_api_key()
        client = initialize_client(
            api_key_anthropic=api_key_anthropic,
            api_key_openai=api_key_openai,
            model=llm_settings.model,
            api_key_gemini=api_key_gemini,
        )
    semaphore = asyncio.Semaphore(llm_settings.max_concurrent_calls)

    try:
        coros = [
            _estimate_single_cell(
                plan=plan,
                estimation_template=estimation_template,
                system_prompt=system_prompt,
                client=client,
                semaphore=semaphore,
                llm_settings=llm_settings,
                include_bin_rate=include_bin_rate,
                include_task_outcomes=include_task_outcomes,
                max_tokens=max_tokens,
                parse_retries=parse_retries,
            )
            for plan in plans
        ]
        return list(await asyncio.gather(*coros))
    finally:
        # Close clients we created ourselves (one per event loop); callers
        # that pass their own client manage its lifecycle.
        if owns_client:
            close = getattr(client, "close", None)
            if close is not None:
                await close()


def estimate_cells(
    plans: Sequence[CellPlan],
    estimation_template: str,
    **kwargs,
) -> List[CellResult]:
    """Synchronous facade over `estimate_cells_async` (fresh event loop)."""
    return asyncio.run(estimate_cells_async(plans, estimation_template, **kwargs))


def estimate_cell(plan: CellPlan, estimation_template: str, **kwargs) -> CellResult:
    """Single-cell convenience wrapper."""
    return estimate_cells([plan], estimation_template, **kwargs)[0]


# ---------------------------------------------------------------------------
# Dataset / plan helpers for external callers
# ---------------------------------------------------------------------------

def canonical_family(task_family: str) -> str:
    """Normalise benchmark family names (parquet uses underscores)."""
    return task_family.strip().lower().replace("-", "_")


def compute_explicit_bins(
    dataset: LyptusDataset, edges_minutes: Sequence[float]
) -> BinAssignment:
    """Fixed explicit-edge binning with RIGHT-CLOSED intervals (a, b].

    Fixed edges keep bin membership consistent across train/test subsets
    (per-subset quantiles would not). Right-closed intervals reproduce the
    design-table allocation (54/52/40/18/11 on the train benchmarks): several
    tasks sit exactly on the 60- and 180-minute edges and belong to the LOWER
    bin, and the 2160-minute task belongs to the top bin.
    """
    return compute_bins_right_closed(dataset.fst_array(), explicit_edges=list(edges_minutes))


def build_plans_for_targets(
    *,
    dataset: LyptusDataset,
    bins: BinAssignment,
    forecasted_models: Sequence[str],
    targets_by_bin: Dict[int, List[str]],
    n_examples_per_source_bin: int = 2,
    evidence_task_ids: Optional[frozenset] = None,
) -> Tuple[Dict[Tuple[str, str], CellPlan], List[Tuple[str, str]]]:
    """Build `all_except_target` cell plans for explicit target tasks.

    `evidence_task_ids` restricts the tasks that may be SHOWN as evidence
    (anchor/easier tasks and pass-rate denominators) -- e.g. the training
    benchmarks only, so held-out benchmark content never enters a prompt.

    Returns
      - plans: {(target_task_id, forecasted_model): CellPlan} for every
        admissible cell,
      - dropped: [(target_task_id, forecasted_model)] cells that
        `build_cell_plans` deemed inadmissible (missing outcome, missing
        evidence, or target colliding with a shown evidence task).
    """
    # Bins absent from `explicit_target_tasks` fall back to auto target
    # sampling inside `build_cell_plans`; pass explicit empty lists so only
    # the requested targets generate plans.
    explicit = {b: list(targets_by_bin.get(b, [])) for b in range(bins.n_bins)}
    plans_list = build_cell_plans(
        bins=bins,
        dataset=dataset,
        forecasted_models=list(forecasted_models),
        source_bins_to_show="all_except_target",
        n_examples_per_source_bin=n_examples_per_source_bin,
        n_target_tasks_per_cell=1,  # ignored: explicit targets given for every bin
        explicit_target_tasks=explicit,
        evidence_task_ids=evidence_task_ids,
    )
    plans = {(p.target_task.task_id, p.forecasted_model): p for p in plans_list}
    expected = [
        (tid, model)
        for ids in targets_by_bin.values()
        for tid in ids
        for model in forecasted_models
    ]
    dropped = [key for key in expected if key not in plans]
    return plans, dropped


def grand_brier(results: Sequence[CellResult], *, failure_brier: float = 1.0) -> float:
    """Mean Brier over cells; unparsable cells count as `failure_brier`."""
    if not results:
        raise ValueError("grand_brier needs at least one CellResult")
    vals = [r.brier if r.brier is not None else failure_brier for r in results]
    return float(sum(vals)) / len(vals)


__all__ = [
    "CellResult",
    "estimate_cell",
    "estimate_cells",
    "estimate_cells_async",
    "make_llm_settings",
    "resolve_anthropic_api_key",
    "canonical_family",
    "compute_explicit_bins",
    "build_plans_for_targets",
    "grand_brier",
]
