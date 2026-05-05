# Intra-Benchmark Calibration on the Lyptus Cyber Task Horizons

Predicts **P(model M solves task t in difficulty bin j | M's empirical pass rate
on tasks in difficulty bin i)** using LLM-based forecasting, then scores those
forecasts against the true binary outcome from `model_runs.parquet`.

Operates over the [Lyptus offensive cyber task horizons
dataset](https://github.com/lyptus-research/cyber-task-horizons-data) (291 headline
tasks across 7 sub-benchmarks). Treats Lyptus as one homogeneous benchmark — bin
labels are an experimental-design tool, never shown to the forecaster.

## Pipeline

```
Lyptus parquets + per-family JSONLs        (lyptus_data.py)
       │
       ▼
binning by best_available_minutes           (binning.py: equal_count default)
       │
       ▼
cell plans (i, j, M, t)                     (task_selector.py: anchor heuristic
   = 5 source bins × 4 target bins (i≠j)    + drop-M-if-not-evaluated logic)
   × 12 forecasted models × K target tasks
       │
       ▼
async Delphi loop                            (workflow.py)
   for each cell:
     for each Delphi round:
       for each expert in parallel:
         API call(s) → parse percentiles → CSV row + JSON record
                                          (per-elicitation, lock-protected flush)
       │
       ▼
post-hoc scoring + plots                    (analyse_results.py)
   Brier on p50, ECE, CRPS via Beta fit, calibration scatter, reliability,
   per-cell heatmap, per-model bar, METR-style log-FST plot.
```

## Data

| Quantity | Value |
|---|---|
| Headline tasks (in `task_difficulties.parquet`) | 291 |
| Headline tasks usable (have `estimation_instructions`) | **269** |
| Sub-benchmarks treated as one | 7 (cybashbench, nl2bash, intercode-ctf, nyuctf, cybench, cvebench, cybergym) |
| Forecasted models in default panel | **12** (all 15 minus GPT-2/3/3.5; sparse coverage) |
| Difficulty metric | `best_available_minutes` from `task_difficulties.parquet` |
| Outcome | `score_binarized ∈ {0, 1}` from `model_runs.parquet` (one row per (model, task)) |
| Lyptus commit pinned by fetch script | recorded in run JSON metadata |

The 22 dropped tasks are cybergym ARVO tasks lacking the `estimation_instructions`
field. Their IDs are recorded in `run_metadata.dataset_provenance.dropped_task_ids_no_estimation_instructions`.

## What the forecaster sees and doesn't see

Per cell, the prompt presents M's capability profile on **one shown source bin
i** (default; configurable via `source_bins_to_show`), built from:

  - **Empirical pass rate** for M on bin i (over the evaluated subset).
  - **Anchor task** (representative of bin i's mid-range pass rate) with M's
    binary outcome on it, full `estimation_instructions` text.
  - **N easier tasks** (default N=2) from within bin i, also with M's binary
    outcomes and full text.

Then the **target task** is presented as raw `estimation_instructions` only.

**Hidden from the forecaster**: target task FST, target bin label, source benchmark
identity (sometimes leaks through `estimation_instructions` text — by design,
matches what a human user would see).

The forecaster is asked to output three percentiles (p25, p50, p75) of its
subjective probability distribution over P(M solves t). Headline metric is
Brier on p50; CRPS via Beta fit is the v2 stub.

## Configuration knobs (`config_full.yaml`)

| Knob | Default | Description |
|---|---|---|
| `binning.n_bins` | 5 | Number of difficulty bins |
| `binning.strategy` | `equal_count` | `equal_count` / `equal_log_fst` / `explicit_edges` |
| `forecasted_models` | `null` (= all 12 in panel) | List of model aliases |
| `drop_models` | `["GPT-2", "GPT-3", "GPT-3.5"]` | Sparse-coverage models to exclude |
| `target_selection.n_target_tasks_per_cell` | 1 | K target tasks per cell |
| `target_selection.sampling_seed` | 42 | RNG seed for stratified-by-log-FST sampling |
| `source_profile.source_bins_to_show` | `[]` (= `[i]`) | Which bins to include in capability profile |
| `source_profile.n_examples_per_source_bin` | 2 | Easier-tasks count per shown bin (1 anchor + N) |
| `include_target_solution` | `false` | Show `solution_walkthrough` to forecaster |
| `llm_settings.model` | `claude-sonnet-4-6` | Anthropic alias or OpenAI model name |
| `llm_settings.max_concurrent_calls` | 5 | Concurrent in-flight API calls |
| `llm_settings.rate_limit_calls` | 45 | Soft local cap per `rate_limit_period` seconds |
| `workflow_settings.num_experts` | 2 | Distinct expert personas per cell |
| `workflow_settings.delphi_rounds` | 1 | Total rounds (1 = no Delphi deliberation) |

Cell budget with defaults: `5 × 4 × 12 × 1 = 240` cells × `2` experts × (round 1 = 2 API calls/expert) = **960 API calls**.

## Output structure

Per-run directory `results/{run_id}/`:

```
results/20260502_171824/
├── 20260502_171824_intra_<model>_nexp<N>_nrnd<R>_<temp>_estimates.csv
├── 20260502_171824_intra_<model>_nexp<N>_nrnd<R>_<temp>_results.json
└── plots/                                    # written by analyse_results.py
    ├── statistics.txt
    ├── scored_with_crps.csv
    ├── calibration_scatter.png
    ├── brier_heatmap.png
    ├── per_model_brier.png
    ├── reliability_diagram.png
    ├── metr_style_logfst.png
    └── per_cell_distributions.png
```

CSV columns (one row per cell × expert × Delphi round, **flushed per row**):

```
condition_id, run_id, timestamp,
source_bin, target_bin,
forecasted_model, target_task_id, target_task_family, target_fst_minutes,
expert_id, delphi_round,
p25, p50, p75,
outcome,
anchor_task_id, easier_task_ids,
anchor_prompt_chars, easier_prompt_chars, target_prompt_chars,
prompt_hash, rationale
```

Full system + user prompts and raw API responses live only in the JSON.

JSON has three top-level keys:

  - `run_metadata`: timestamps, model, hyperparameters, full config snapshot,
    dataset provenance dict (Lyptus commit SHA, dropped task IDs, models in
    panel, design observations), bin definition.
  - `cells`: one summary per cell with Delphi-round means/stds.
  - `elicitations`: full per-elicitation records (system prompt, user prompt,
    raw response, parsed values, error if any).

## Design observations (recorded in every run)

  - **Per-task outcomes are visible to the forecaster** alongside each anchor
    and easier task — more discriminative than the bin-level pass rate.
  - **Anchor selection is M-independent in practice**: with 10/12 models having
    full coverage, the heuristic picks the same anchor across forecasted models
    within a bin. Verified in `production_anchors.csv`.
  - **Anchor distribution skews to NYUCTF** (3 of 5 bins under default
    `n_bins=5` / `equal_count`): NYUCTF tasks tend to have tight per-task
    pass-rate distributions that match bin means well.

## Files

| Path | Purpose |
|---|---|
| `lyptus_data.py` | Parquet + JSONL loader, NaN-safe outcome matrix, provenance |
| `binning.py` | FST-binning strategies |
| `task_selector.py` | Anchor + easier-task heuristic, target sampling, cell admissibility |
| `prompt_builder.py` | Assemble system + user prompts from cell plan |
| `config.py` | YAML loader + `.env`-aware API key resolution |
| `workflow.py` | Async Delphi loop with per-elicitation lock-protected flush |
| `results_handler.py` | Progressive CSV + JSON writer, run registry |
| `run_calibration.py` | Entry point for the elicitation experiment |
| `analyse_results.py` | Post-hoc stats + plots + CRPS via Beta fit |
| `prompts/` | 3 templates (analysis, initial, subsequent) |
| `scripts/fetch_lyptus_data.py` | Sparse no-LFS git clone, SHA-pinned |
| `scripts/dump_anchors.py` | Inspect anchor selection without elicitations |

## See also

- [`QUICK_START.md`](QUICK_START.md) — step-by-step setup
- `intra_benchmark_cc_prompt.md` (repo root) — original spec
- `inter_benchmark_calibration/` — sibling experiment (cross-benchmark variant)
