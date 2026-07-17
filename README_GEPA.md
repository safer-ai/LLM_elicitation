# GEPA integration: the `llm-estimator` package

This repo now doubles as an installable package (`**llm-estimator**`) that the
GEPA fork (`gepa`, branch `feat/gepa_on_LLM_estimator`) uses as its evaluation
backend for the forecaster-optimisation experiment. Design background:
`GEPA_experiment_design_summary.md` (in the gepa repo).  
**The GEPA experiment is driven entirely from the gepa repo** —
see `FORECASTER_GEPA_README.md` there for the run instructions; this document
covers what changed on this side and the packaging contract.

Branch: all of this lives on `**feat/ss2026_intrabenchmark_package`**. The
gepa fork consumes this repo as an *editable path dependency*
(`../LLM_elicitation`), so whatever is checked out here is what runs — keep
this branch checked out while running GEPA.

## What was added

- `**pyproject.toml`** — makes the repo `pip install -e .`-able. Only the two
packages external consumers need are installed: `intra_benchmark_calibration`
and `shared`. Import paths are unchanged (`from intra_benchmark_calibration... import ...`); nothing was restructured.
- `**intra_benchmark_calibration/estimation_api.py`** — the per-cell
estimation callable GEPA needs (the batch pipeline in `run_calibration.py`
is YAML-driven, persists everything and runs cells sequentially; this API
does not). Key entry points:
  - `estimate_cells(plans, template, llm_settings=..., system_prompt='')` →
  one **single-call** elicitation per cell (no stage-1 capability analysis,
  no Delphi, no experts, no persistence), run **concurrently across cells**
  via `asyncio.gather` under a semaphore. Returns a `CellResult` per cell:
  `p25/p50/p75`, full rollout text, the assembled prompts, `brier`
  (on p50 vs the binary ground truth) and a per-cell `error` field —
  malformed templates or unparsable responses never raise, they are
  reported per cell so an optimiser can score them as failures.
  - `estimate_cell` (single cell), `estimate_cells_async` (bring your own
  event loop), `make_llm_settings`, `resolve_anthropic_api_key` (same
  `.env` chain as the batch pipeline), `build_plans_for_targets`
  (explicit-target `all_except_target` cell plans), `compute_explicit_bins`,
  `grand_brier`.
  - It reuses `assemble_prompts`, `make_api_call` (retries, rate limiting,
  `reasoning_effort`) and `parse_probability_response` — no duplicated
  prompt or API logic.
- `**intra_benchmark_calibration/gepa_task_sets.py`** — the seeded task-split
script (spec §5). Builds the train-pool / val / finalist / test manifest
with the mandatory 5/2 benchmark split (train: CyBashBench, NL2Bash,
InterCode-CTF, NYUCTF, CyBench; test: CVEBench+CyberGym), 
per-bin reservations (5/5/5/3/3 val and finalist), and
disjointness/non-emptiness assertions. The test count is derived from the
data (the `estimation_instructions` usability filter), not hard-coded.
Standalone usage:
  ```bash
  python intra_benchmark_calibration/gepa_task_sets.py \
      --lyptus-repo ~/gitrepos/cyber-task-horizons-data \
      --seed 42 --output gepa_task_manifest.json
  ```
  (The GEPA harness calls this automatically on first run and reuses the
  manifest thereafter, so you normally don't run it by hand.)

## What was modified

- `**intra_benchmark_calibration/binning.py**` — added
`compute_bins_right_closed(...)`: fixed explicit FST edges
`[0.46, 2.81, 12.82, 60, 180, 2160]` minutes (= 10^[−0.34…3.33]) with
**right-closed** intervals (a, b]. This is what reproduces the design-table
allocation (train-side 54/52/40/18/11): 13 tasks sit at exactly 180 min and
2 at 60 min (lower bin), and one task at exactly 2160 min (top bin). The
existing `compute_bins` (right-open, used by the batch pipeline) is
untouched. Fixed edges — never per-subset quantiles — keep train/test bin
membership consistent.
- `**intra_benchmark_calibration/task_selector.py`** — `build_cell_plans` /
`select_anchor_and_easier` gained an optional `evidence_task_ids` filter: a
whitelist of tasks that may be SHOWN as evidence in the prompt (anchors,
easier examples, pass-rate denominators). The GEPA harness sets this
whitelist to the training-benchmark tasks in **every** phase — including the
held-out test evaluation — so the evidence block never contains
CVEBench/CyberGym content. (In the test phase the *target* task is of course
a CVEBench/CyberGym task — that is what is being forecast — but everything
shown around it comes from the training benchmarks; deliberate, since we
test generalisation to fully unseen contexts.) Default `None` preserves the
old behaviour for the batch pipeline: evidence drawn from the whole loaded
dataset.

Note on the seed prompt: the **GEPA seed template** has a small modification to
what was used before: the `<rationale>` block moved above `<percentile_estimates>`.
This is because we want the LLM to reason about the task first, then answer.

## Single API call per cell

Two equivalent switches, depending on which entry point you use:

- **Batch pipeline** (`run_calibration.py`): set
`workflow_settings.skip_analysis: true` in the YAML — skips the
stage-1 capability-analysis call entirely.
- `**estimation_api`** (what GEPA uses): single-call by construction; there is
no analysis stage to disable. The system prompt defaults to the empty
string (no expert persona) and no benchmark description or ground-truth
summary is injected — the forecaster sees only the instantiated template.

## Setup for collaborators

Nothing to run in this repo. Clone it as a **sibling of the gepa fork, named
`LLM_elicitation`**, check out `feat/ss2026_intrabenchmark_package`, put
`ANTHROPIC_API_KEY=...` in `.env` here (or export it), and do the `uv sync`
in the gepa repo — it installs this package editable. The template contract
for candidate prompts: they are filled with `str.format`, must contain
`{forecasted_model}`, `{capability_profile}` and `{target_task_text}`, and
any literal braces must be doubled.