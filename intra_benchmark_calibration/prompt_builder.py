#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prompt assembly for the intra-benchmark elicitation.

Produces the formatted source-side capability profile and target-side block
that get plugged into the three prompt templates in `prompts/`:

  - intra_capability_analysis.txt           (round 1, stage 1)
  - initial_intra_solve_estimation.txt       (round 1, stage 2)
  - subsequent_intra_solve_estimation.txt    (rounds 2+)

Templates use Python `str.format(**dict)` substitution. Field names exposed:

  - benchmark_description: short blurb about the unified Lyptus dataset
  - forecasted_model:      the model alias whose capability we condition on
  - capability_profile:    the per-bin pass-rate + anchor + easier-task block
  - target_task_text:      the target task's estimation_instructions
                           (optionally extended with solution_walkthrough)
  - technical_analysis:    output of stage 1 (only used in stage 2)
  - context:               other experts' previous-round responses (rounds 2+)

The target task block is INTENTIONALLY MINIMAL: no FST, no bin label, no
task_family tag. The source benchmark identity may leak through the
estimation_instructions text itself; that is by design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from intra_benchmark_calibration.task_selector import CellPlan, SourceBinProfile


def build_default_benchmark_description(n_tasks: int) -> str:
    """The default `<benchmark_context>` text, parameterised by the task count actually used."""
    return (
        f"A unified set of {n_tasks} offensive-cybersecurity tasks drawn from the Lyptus\n"
        f"\"Cyber Task Horizons\" study (Lyptus's 'headline' subset that has both human-derived\n"
        "difficulty estimates and full model-evaluation coverage). Each task is a hands-on\n"
        "cyber capability evaluation (CTF reverse engineering, exploit development, command\n"
        "translation, vulnerability exploitation, etc.) graded by a binary pass/fail outcome.\n"
        "Tasks span ~4 orders of magnitude in human-expert solving time, from <1 minute to\n"
        ">36 hours. For this experiment we treat them as one homogeneous benchmark."
    )


def format_ground_truth_summary(summary: Dict[str, float]) -> str:
    """Render the LyptusOutcomes.ground_truth_summary() dict as a prompt block."""
    n_models = int(summary["n_models"])
    n_tasks = int(summary["n_tasks"])
    return (
        f"Across the {n_models}-model panel actually used in this experiment, evaluated on "
        f"{n_tasks} headline tasks: mean per-task pass rate is "
        f"{summary['mean_per_task_pass_rate']:.3f}, median is "
        f"{summary['median_per_task_pass_rate']:.3f}; "
        f"{summary['frac_tasks_zero_pass'] * 100:.1f}% of tasks ({int(summary['n_tasks_zero_pass'])}) "
        f"are passed by 0/{n_models} models, and "
        f"{summary['frac_tasks_full_pass'] * 100:.1f}% ({int(summary['n_tasks_full_pass'])}) "
        f"are passed by all {n_models}/{n_models}."
    )


@dataclass(frozen=True)
class AssembledPrompts:
    """The set of prompts for one elicitation in one round."""

    system_prompt: str
    analysis_user_prompt: Optional[str]  # only present in round 1
    estimation_user_prompt: str          # round 1 stage 2 OR round >= 2
    template_data: Dict[str, str]        # the dict passed to .format(...)


def _format_pass_rate(profile: SourceBinProfile, model: str) -> str:
    pct = profile.pass_rate * 100.0
    return (
        f"  Empirical pass rate for {model} on this bin: {profile.n_solved}/{profile.n_evaluated} "
        f"= {pct:.1f}% (bin contains {profile.n_in_bin} headline tasks total)"
    )


def _outcome_tag(outcome: Optional[float], model: str) -> str:
    if outcome is None:
        return f"[{model} outcome on this task: not evaluated]"
    if outcome >= 0.5:
        return f"[{model} outcome on this task: SOLVED (1)]"
    return f"[{model} outcome on this task: FAILED (0)]"


def _format_task_block(label: str, task, *, outcome_tag: str, indent: str = "    ") -> str:
    body = task.estimation_instructions.strip()
    body_indented = body.replace(chr(10), chr(10) + indent)
    return (
        f"{indent}--- {label} (task_id={task.task_id}) ---\n"
        f"{indent}{outcome_tag}\n"
        f"{indent}{body_indented}"
    )


def format_capability_profile(profiles: List[SourceBinProfile], model: str) -> str:
    """Render the source capability profile as plain text for the prompt.

    For each shown bin we report the model's empirical pass rate on the bin
    (over the evaluated subset) and then list the anchor + easier tasks. Each
    individual task is annotated with the model's binary outcome on that exact
    task — this is more discriminative information than the bin-level rate
    alone, e.g. it tells the forecaster whether the anchor is one of the
    solved or failed tasks.
    """
    chunks: List[str] = []
    for p in profiles:
        chunks.append(f"=== Source bin {p.bin_index} ===")
        chunks.append(_format_pass_rate(p, model))
        chunks.append("")
        chunks.append(
            _format_task_block(
                f"ANCHOR (representative of bin {p.bin_index})",
                p.anchor,
                outcome_tag=_outcome_tag(p.anchor_outcome, model),
            )
        )
        easier_outcomes = p.easier_outcomes or [None] * len(p.easier_tasks)
        for k, (et, eo) in enumerate(zip(p.easier_tasks, easier_outcomes), 1):
            chunks.append("")
            chunks.append(
                _format_task_block(
                    f"Easier example #{k}",
                    et,
                    outcome_tag=_outcome_tag(eo, model),
                )
            )
        chunks.append("")
    return "\n".join(chunks).rstrip()


def format_target_task(plan: CellPlan, *, include_solution: bool = False) -> str:
    body = plan.target_task.estimation_instructions.strip()
    if include_solution and plan.target_task.solution_walkthrough:
        body += "\n\n--- Solution walkthrough (provided to forecaster) ---\n"
        body += plan.target_task.solution_walkthrough.strip()
    return body


def _format_previous_round_context(prev_responses: List[Dict]) -> str:
    """Render the 'other experts' from the previous round' block."""
    lines: List[str] = ["", "---", "Other experts' estimates from the previous round:", "---", ""]
    others = [r for r in prev_responses if "error" not in r]
    if not others:
        lines.append("(No other valid responses from the previous round)")
    else:
        for r in others:
            p25 = r.get("percentile_25th")
            p50 = r.get("percentile_50th")
            p75 = r.get("percentile_75th")

            def f(v):
                return f"{v:.3f}" if isinstance(v, float) else "N/A"

            rationale = (r.get("rationale") or "").strip()
            if len(rationale) > 300:
                rationale = rationale[:300] + "..."
            lines.append(
                f"Expert {r.get('expert', '?')}: 25th={f(p25)}, 50th={f(p50)}, 75th={f(p75)}\n"
                f"  Rationale: {rationale}"
            )
            lines.append("")
    lines.append("---")
    return "\n".join(lines)


def assemble_prompts(
    *,
    plan: CellPlan,
    persona_system_prompt: str,
    prompt_templates: Dict[str, str],
    round_num: int,
    prev_round_responses: Optional[List[Dict]] = None,
    technical_analysis: Optional[str] = None,
    benchmark_description: Optional[str] = None,
    ground_truth_summary: Optional[Dict[str, float]] = None,
    include_target_solution: bool = False,
) -> AssembledPrompts:
    """
    Build all prompt strings for one expert × round.

    Templates required:
      - intra_capability_analysis            (round 1, stage 1)
      - initial_intra_solve_estimation        (round 1, stage 2)
      - subsequent_intra_solve_estimation     (rounds 2+)

    Args:
        benchmark_description: free-form text shown in `<benchmark_context>`. If
            None, a default is built from `ground_truth_summary['n_tasks']` (or
            falls back to a generic phrasing if the summary isn't supplied).
        ground_truth_summary: output of `LyptusOutcomes.ground_truth_summary()`.
            Used to fill the `{ground_truth_summary}` template placeholder; if
            not supplied, the placeholder is replaced with an empty string and
            the prompt loses its base-rate sanity-check footnote.
    """
    if benchmark_description is None:
        if ground_truth_summary is not None:
            benchmark_description = build_default_benchmark_description(int(ground_truth_summary["n_tasks"]))
        else:
            benchmark_description = build_default_benchmark_description(0).replace(
                "set of 0 offensive", "set of offensive"
            )

    gt_summary_text = (
        format_ground_truth_summary(ground_truth_summary) if ground_truth_summary else ""
    )

    template_data: Dict[str, str] = {
        "benchmark_description": benchmark_description,
        "forecasted_model": plan.forecasted_model,
        "capability_profile": format_capability_profile(plan.profiles, plan.forecasted_model),
        "target_task_text": format_target_task(plan, include_solution=include_target_solution),
        "technical_analysis": technical_analysis or "",
        "context": _format_previous_round_context(prev_round_responses or []),
        "ground_truth_summary": gt_summary_text,
    }

    if round_num == 1:
        analysis_tpl = prompt_templates.get("intra_capability_analysis")
        estimation_tpl = prompt_templates.get("initial_intra_solve_estimation")
        if not analysis_tpl or not estimation_tpl:
            raise ValueError(
                "Missing required round-1 templates: 'intra_capability_analysis' "
                "and/or 'initial_intra_solve_estimation'"
            )
        return AssembledPrompts(
            system_prompt=persona_system_prompt,
            analysis_user_prompt=analysis_tpl.format(**template_data),
            estimation_user_prompt=estimation_tpl.format(**template_data),
            template_data=template_data,
        )

    subsequent_tpl = prompt_templates.get("subsequent_intra_solve_estimation")
    if not subsequent_tpl:
        raise ValueError("Missing 'subsequent_intra_solve_estimation' template")
    return AssembledPrompts(
        system_prompt=persona_system_prompt,
        analysis_user_prompt=None,
        estimation_user_prompt=subsequent_tpl.format(**template_data),
        template_data=template_data,
    )
