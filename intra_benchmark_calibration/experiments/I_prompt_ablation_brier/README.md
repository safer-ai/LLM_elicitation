# Experiment I — Prompt Ablation (Brier): the Level-0 bracket

## Results — COMPLETE (Sonnet 4.6, no thinking, 300 matched cells/condition, 1 run + bootstrap)

Headline table is auto-written to `summary/ablation_brier.{md,csv}`.

| Condition | Removed vs control | Brier | Paired Δ vs control | 95% CI | Wilcoxon p | Verdict |
|---|---|---|---|---|---|---|
| `control` | — (full prompt) | 0.1377 | — | — | — | reference |
| `no_ground_truth_summary` | base-rate "leak" footnote | 0.1362 | −0.0015 | [−0.009, +0.006] | 0.93 | no effect |
| `no_task_outcomes` | per-task SOLVED/FAILED tags | 0.1367 | −0.0012 | [−0.011, +0.008] | 0.85 | **no effect** |
| `no_bin_rate` | per-bin empirical pass-rate line | 0.1417 | +0.0040 | [−0.006, +0.014] | 0.55 | **no effect** |
| `trim_reasoning` | reasoning checklist | 0.1427 | +0.0050 | [−0.002, +0.012] | 0.43 | no effect |
| `trim_all` | analysis + reasoning | 0.1447 | +0.0070 | [−0.001, +0.015] | 0.002 | sig (bundle) |
| `skip_analysis` | stage-1 analysis call | 0.1482 | **+0.0105** | [+0.003, +0.018] | 2.5e-05 | **sig** |
| `minimal` | analysis + reasoning + footnote + framing | 0.1502 | **+0.0136** | [+0.005, +0.022] | 7.2e-05 | sig |
| `no_source_context` | **entire capability profile** | **0.1554** | **+0.0178** | [−0.003, +0.038] | **1.2e-07** | **sig (evidence floor)** |

**Conclusions (scaffold):**
- **The analysis stage is the only individually-significant scaffold component** (+0.0105, ~77% of the control→minimal gap). It's the one piece earning its tokens.
- **The reasoning checklist does ~nothing** (+0.0050, p=0.43).
- **The `{ground_truth_summary}` "leak" is inert** (−0.0015, p=0.93) → **validity gate PASSED**.

**Conclusions (evidence):**
- **Neither the bin pass-rate nor the per-task SOLVED/FAILED tags individually hurt when removed** (`no_bin_rate` +0.0040/p=0.55, `no_task_outcomes` −0.0012/p=0.85 — both non-significant). The numeric annotations are redundant — the **task description texts** are load-bearing.
- **Removing the entire profile is the worst single intervention** (`no_source_context` +0.0178, p=1.2e-07), even worse than `minimal` which at least retains the task texts in the profile. This confirms: the value comes from showing the model *what the tasks look like*, not from the pass-rate numbers or outcome labels attached to them.
- The bootstrap CI for `no_source_context` just clips zero [−0.003, +0.038] while Wilcoxon is highly significant (p=1.2e-07) — the asymmetry is real: 68% of cells are worse. The CI width reflects task-sampling noise, not a weak effect.

**Overall picture:** The entire capability evidence is worth ~0.018 Brier; of that, ~0.011 comes from the analysis stage processing it. The prose scaffold (reasoning checklist, framing) contributes nothing individually. The numeric annotations (pass rates, outcome tags) are free riders on the task text.

**Caveats:** `no_task_outcomes` and `minimal` have 299/300 parsed (1-cell p50 parse failure each — not a design issue). `trim_all` (+0.0070) vs `skip_analysis` (+0.0105) ordering is within overlapping CIs. Run-to-run noise negligible (SD≈0.0013 from minimal's 3 repeats). `control` is repeat-1 only (cancelled repeat-2 partial was trimmed).

---

The first rung of the strategic ladder in repo-root `PLAN.md` §8. It brackets the
value of prompt complexity and checks a validity concern, using **Brier-on-p50**
against Lyptus ground truth as the metric. Three matched conditions, all sharing
**identical inputs** (same 300 cells, same source profiles) so that only the
prompt scaffold changes:

| Condition | Prompt | Role |
|---|---|---|
| `control` | full canonical prompt (analysis stage + reasoning checklist + base-rate footnote) | ceiling / reference |
| `minimal` | *"here are tasks with pass/fail, predict the next one"* — single call, no scaffold | floor |
| `no_ground_truth_summary` | full prompt minus the `{ground_truth_summary}` base-rate footnote | leak / validity probe |

**Why these run first (see PLAN.md §8):**
- `minimal` vs `control` brackets *how much the elaborate scaffold buys us*. If minimal ≈ or beats control, the scaffold is not earning its tokens — a major finding. (This frames, but does **not** skip, the Level-1 leave-one-out decomposition.)
- `no_ground_truth_summary` is a validity gate: the footnote injects a dataset-wide base rate computed from the full outcome matrix — a "gray-zone" leak. This measures how much accuracy depends on it.

## The three conditions in detail

### `control`
Verbatim copy of the canonical prompts in `intra_benchmark_calibration/prompts/`. Two API calls per cell (stage-1 analysis + stage-2 estimation).

### `no_ground_truth_summary`
Identical to `control` except line 39 of `initial_intra_solve_estimation.txt` — the footnote `- Sanity-check against the underlying ground-truth distribution: {ground_truth_summary}` — is removed. `prompt_builder` uses `str.format(**data)`, so dropping the placeholder needs no code change. Two API calls per cell.

### `minimal`
- `skip_analysis: true` → the stage-1 capability-analysis call is skipped entirely (one API call per cell instead of two). Implemented via a new `workflow_settings.skip_analysis` flag in `config.py` / `workflow.py`.
- A stripped `initial_intra_solve_estimation.txt`: no reasoning checklist, no IQR guidance, no `{ground_truth_summary}` footnote, no `<benchmark_context>` blurb — just the capability evidence + target + the p25/p50/p75 ask.
- **Crucially, the capability evidence (`{capability_profile}`) is identical to the other conditions.** Minimal strips the *scaffold*, not the *evidence* — so the comparison isolates "scaffold vs no scaffold," and the same source examples/outcomes are shown.
- The `intra_capability_analysis.txt` and `subsequent_intra_solve_estimation.txt` files in this variant are placeholders: the analysis one is never sent (skip_analysis), and the subsequent one is unused at `delphi_rounds: 1`. They exist only so the round-1 prompt assembler has a complete template set.

## Level 1 — evidence ablations (the high-effect frontier)

The scaffold decomposition above showed the prose scaffold is worth only ~0.014 Brier,
so the **per-model capability evidence** must be doing the work. These three conditions
leave-one-out the evidence itself. All keep the rest of the control prompt fixed.

| Condition | Removed vs control | Calls/cell | How |
|---|---|---|---|
| `no_bin_rate` | the per-bin empirical pass-rate header line | 2 | `source_profile.include_bin_rate: false` (data toggle) |
| `no_task_outcomes` | the per-task `SOLVED/FAILED` tags | 2 | `source_profile.include_task_outcomes: false` (data toggle) |
| `no_source_context` | the **entire** capability profile (bins + tasks + outcomes) | 1 | dedicated prompt variant + `skip_analysis: true` |

- `no_bin_rate` and `no_task_outcomes` use **byte-identical templates to `control`** — the ablation is data-driven via the new `source_profile` toggles in `config.py` → `prompt_builder.format_capability_profile(...)`. The assembled (filled) prompt still differs, so `prompt_hash` stays disjoint from control and the validity check passes. Anchor/easier task *texts* are retained in both; only the named component is stripped.
- `no_task_outcomes` isolates the value of fine-grained per-task labels **over and above** the aggregate bin pass rate (which summarises the same outcomes), so a near-zero delta is the expected/interesting result.
- `no_source_context` is the **evidence floor**: the forecaster sees only the model name + target task + base-rate footnote. There is no profile to analyse, so the stage-1 analysis call is skipped (like `minimal`). Expect this to be the largest single regression if the evidence is truly load-bearing. Its `initial_intra_solve_estimation.txt` drops the `<capability_profile>` and `<your_prior_analysis>` blocks and rewrites the anchor instruction (no source-bin to anchor on).

## Run provenance / traceability
Every run is self-identifying so you never have to guess which run belongs to which condition:
- **Filenames**: outputs are `<run_id>__<condition>__<model>_intra_estimates.csv` / `..._intra_results.json` (e.g. `20260614_003327__control__claude-sonnet-4-6_intra_estimates.csv`).
- **CSV**: a leading `experiment_label` column carries the condition on every row, alongside `forecaster_model` / `forecasted_model` / `run_id` / `prompt_hash`. (Note: the `condition_id` column is the design *cell* id, **not** the ablation condition — use `experiment_label` for that.)
- **JSON**: `run_metadata.experiment_label` + `models_run` + full `config_snapshot`.
- **Registry**: `results/<condition>/run_registry.json` records each run's `run_id`, `experiment_label`, `models_run`, `output_path`, and `config_file`.
- `summarize_ablation.py` cross-checks that each run's recorded `experiment_label` matches the folder it was found in (a `FAIL` flags a run dropped into the wrong condition dir).

## Controlled-variable guarantee
Cell plans are deterministic functions of the design params (`binning`, `source_profile`, `target_selection`, `forecasted_models`), which are identical across all three configs. So every condition evaluates the **exact same 300 target tasks** with the same source profiles. `summarize_ablation.py` asserts this (identical cell-key set vs control) and that the prompts actually differ (disjoint `prompt_hash` sets).

## Design (identical to Experiment G's Sonnet sweep, except where noted)

| Setting | Value |
|---|---|
| Forecaster model | `claude-sonnet-4-6` |
| `reasoning_effort` | `off` (no extended thinking) |
| Forecasted models | all 12 (drop GPT-2/3/3.5) |
| Bins | 5, `equal_count` |
| Source design | `all_except_target`, 2 examples/bin |
| Target tasks per cell (K) | 5 |
| Experts / Delphi rounds | 1 / 1 |
| Repeats | 3 per condition (matched) |

Per run: 12 models × 5 target-bins × 5 tasks (K) = **300 cells** (= 60 coarse cells × K=5).
> **Naming note:** The config param `n_target_tasks_per_cell` uses the coarser definition where a "cell" is `(model, source_bins, target_bin)` → 60 coarse cells. Everywhere else in this README "cell" means a `CellPlan` — i.e. one specific `(model, source_bins, target_bin, task)` tuple, one CSV row, one API call.
- `control`, `no_ground_truth_summary`: 2 calls/cell → 600 calls/run → ~$8.2/run on Sonnet no-thinking.
- `minimal`: 1 call/cell → 300 calls/run → ~$4/run.
- Full 3-arm Level 0 at 3 repeats each ≈ **$74**.

> Note: the published Sonnet Brier (0.124) used `reasoning_effort: high`. This round is no-thinking, so absolute Brier may shift; the matched **deltas** between conditions are the scientific claim and are valid because all sides share the no-thinking setting.

## Folder layout

```
I_prompt_ablation_brier/
  config_control.yaml
  config_no_ground_truth_summary.yaml
  config_minimal.yaml
  prompt_variants/
    control/                  # 3 canonical prompts, verbatim
    no_ground_truth_summary/  # same 3, footnote line removed from initial_...
    minimal/                  # stripped initial_... + placeholder analysis/subsequent
  results/
    control/  no_ground_truth_summary/  minimal/    # run dirs land here
  summary/                    # ablation_brier.{md,csv} (committed deliverable)
  summarize_ablation.py
```

## Prerequisite: Lyptus data
The configs point `lyptus_repo_dir: ~/lyptus-data`, the full `cyber-task-horizons-data`
checkout (loader expects `analysis/figures/data/*.parquet` and `data/tasks/`). If absent:

```bash
git submodule update --init external/cyber-task-horizons-data
# then point lyptus_repo_dir at the checkout, or symlink it to ~/lyptus-data
```

API keys: `intra_benchmark_calibration/experiments/.env` → repo-root `.env` → env (`ANTHROPIC_API_KEY`).

## Run (from repo root)

```bash
python intra_benchmark_calibration/run_calibration.py \
  -c intra_benchmark_calibration/experiments/I_prompt_ablation_brier/config_control.yaml
python intra_benchmark_calibration/run_calibration.py \
  -c intra_benchmark_calibration/experiments/I_prompt_ablation_brier/config_no_ground_truth_summary.yaml
python intra_benchmark_calibration/run_calibration.py \
  -c intra_benchmark_calibration/experiments/I_prompt_ablation_brier/config_minimal.yaml

# Level 1 — evidence ablations
python intra_benchmark_calibration/run_calibration.py \
  -c intra_benchmark_calibration/experiments/I_prompt_ablation_brier/config_no_bin_rate.yaml
python intra_benchmark_calibration/run_calibration.py \
  -c intra_benchmark_calibration/experiments/I_prompt_ablation_brier/config_no_task_outcomes.yaml
python intra_benchmark_calibration/run_calibration.py \
  -c intra_benchmark_calibration/experiments/I_prompt_ablation_brier/config_no_source_context.yaml
```

Each invocation writes one timestamped run dir under `results/<condition>/<run_id>/`
(`num_repeats: 1` now — run-to-run noise is negligible; uncertainty comes from the
bootstrap CI in `summarize_ablation.py`).

## Score

```bash
# Per condition (writes plots/statistics.txt + scored_with_crps.csv into the run dir)
python intra_benchmark_calibration/analyse_results.py -r .../results/control/<run_id>
python intra_benchmark_calibration/analyse_results.py -r .../results/no_ground_truth_summary/<run_id>
python intra_benchmark_calibration/analyse_results.py -r .../results/minimal/<run_id>

# Aggregate all conditions into the headline table + paired deltas
python intra_benchmark_calibration/experiments/I_prompt_ablation_brier/summarize_ablation.py
```

`summarize_ablation.py` auto-discovers the latest run per condition, computes
Brier-on-p50 per `(condition, repeat)`, and for each non-control condition reports
the **paired per-cell Brier delta vs control** with a bootstrap 95% CI and a
Wilcoxon p-value — the matched-design significance test. Writes
`summary/ablation_brier.{md,csv}`.
