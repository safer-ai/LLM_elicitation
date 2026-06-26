# Forecasting results data

Scored predictions from an LLM-forecasting study. **The task:** an LLM ("the
**forecaster**") is shown how various AI models performed on some coding tasks,
then asked to predict the probability that a given model ("the **forecasted
model**") will solve a new, held-out task. We then check the prediction against
the real outcome and score it.

- **Benchmark (ground truth):** Lyptus "Cyber Task Horizons" — per-task pass/fail
  records for many AI models on cybersecurity coding tasks.
- **Each row = one prediction** for one (forecaster, forecasted-model, task) combo.
- **Lower Brier / CRPS = better.** Brier chance baseline ≈ 0.25.

Two questions were studied, one folder each:

| Folder | Question | What varies between files | What is held fixed |
|---|---|---|---|
| `forecaster_model_comparison/` | Which LLM is the best forecaster? | the **forecaster** LLM | the prompt (full prompt) |
| `prompt_variant_comparison/` | Which parts of the prompt actually matter? | the **prompt** | the forecaster (Claude Sonnet 4.6) |

---

## Column schema (identical in every CSV)

| Column | Meaning |
|---|---|
| `forecaster_model` | The LLM making the prediction |
| `forecasted_model` | The model whose success is being predicted |
| `target_task_id` | The held-out task being predicted |
| `target_task_family` | Task category |
| `target_fst_minutes` | Human "first solve time" for the task (difficulty proxy) |
| `target_bin` | Difficulty bin of the target task (1 = easiest … 5 = hardest) |
| `p25`, `p50`, `p75` | Forecaster's 25th / 50th / 75th percentile solve-probability estimate |
| `outcome` | **Ground truth: 1 = model solved the task, 0 = did not** |
| `brier` | Per-row Brier score = (p50 − outcome)² — lower is better |
| `crps` | Per-row CRPS — scores the whole p25/p50/p75 spread, not just the median |
| `repeat_index` | Repeat number (most files = 1; `minimal` has 3 repeats) |
| `prompt_hash` | Hash of the exact prompt text used (provenance) |

Other columns (`condition_id`, `run_id`, `timestamp`, `source_*`, `expert_id`,
`delphi_round`, `rationale`, `beta_alpha/beta_beta`, `*_prompt_chars`) are run
metadata and provenance — safe to ignore for analysis.

---

## `forecaster_model_comparison/`

Same full prompt, different forecaster LLM. ~300 predictions each (one per
forecasted-model × held-out task). Filename = the forecaster.

| File | Forecaster LLM | Rows |
|---|---|---|
| `forecaster_claude-sonnet-4.6.csv` | Claude Sonnet 4.6 | 299 |
| `forecaster_claude-opus-4.7.csv` | Claude Opus 4.7 | 290 |
| `forecaster_claude-haiku-4.5.csv` | Claude Haiku 4.5 | 299 |
| `forecaster_gemini-2.5-flash.csv` | Gemini 2.5 Flash | 300 |
| `forecaster_gpt-5.5_PARTIAL-190rows.csv` | GPT-5.5 — **incomplete run (190/300 rows)** | 190 |

---

## `prompt_variant_comparison/`

Same forecaster (Claude Sonnet 4.6), different prompt. Each file removes one
piece of the full prompt, so you can measure what that piece is worth. All have
300 matched predictions (same tasks across files → directly comparable), except
`minimal` which has 3 repeats (893 rows).

**Filename convention:** `<plain-English description>__<short code>.csv`.
The short code after `__` matches the condition names used in the analysis
summary table (`intra_benchmark_calibration/experiments/I_prompt_ablation_brier/summary/FINAL_RESULTS.csv`).

| File | What this prompt removed (vs the full prompt) | Brier | Worse than full prompt? |
|---|---|---|---|
| `full_prompt_baseline__control.csv` | nothing — the full prompt (the baseline) | 0.1377 | reference |
| `removed_dataset_base_rate_footnote__no_ground_truth_summary.csv` | the dataset-wide base-rate hint | 0.1362 | no |
| `removed_per_task_solved_failed_tags__no_task_outcomes.csv` | the SOLVED/FAILED label on each example task | 0.1367 | no |
| `removed_per_bin_pass_rate__no_bin_rate.csv` | the per-difficulty-bin pass-rate line | 0.1417 | no |
| `removed_reasoning_checklist__trim_reasoning.csv` | the step-by-step reasoning checklist | 0.1427 | no |
| `analysis_merged_into_single_api_call__single_call_analysis.csv` | merged the 2 API calls into 1 (same content) | 0.1428 | slightly |
| `removed_analysis_call_and_reasoning__trim_all.csv` | the analysis call **and** the reasoning checklist | 0.1447 | yes |
| `removed_analysis_api_call__skip_analysis.csv` | the separate stage-1 analysis API call | 0.1482 | yes |
| `minimal_prompt__minimal.csv` | almost everything — only example tasks + the target remain | 0.1502 | yes |
| `removed_all_source_tasks__no_source_context.csv` | all the example tasks (the worst thing to remove) | 0.1554 | yes (most) |

**Headline:** only two things actually help — running the analysis as its own
API call, and showing the example task descriptions. Everything else (base-rate
hint, pass-rate numbers, SOLVED/FAILED tags, reasoning checklist) makes no
measurable difference on its own.
