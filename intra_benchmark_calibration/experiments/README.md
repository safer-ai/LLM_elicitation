# Experiments index

Each subfolder is one self-contained experiment on the **intra-benchmark
forecasting** task: an LLM ("forecaster") is shown how various models performed
on Lyptus cyber tasks, then predicts the probability a given model solves a new
task. Predictions are scored against ground truth with **Brier** (median) and
**CRPS** (full p25/p50/p75). Lower = better; Brier chance ≈ 0.25.

> Folder prefixes (`0b`, `G`, `H`, `I`, `II`, `III`) are just chronological
> labels from the project ladder in the repo-root `PLAN.md`. The plain-English
> name after the prefix is what matters.

## Where the data lives

- **Raw + scored per-run outputs** land in each experiment's `results/<run_id>/`
  (gitignored — regenerate by rerunning). The canonical scored file per run is
  `results/<run_id>/plots/scored_with_crps.csv`.
- **Curated, shareable copies** of the key scored CSVs are committed under the
  repo-root `forecasting_results/` with plain-English names — start there if you
  just want the data.
- **Summaries / figures** committed per experiment under `<experiment>/summary/`.

## Setup (once)

```bash
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY
git submodule update --init --recursive   # Lyptus ground-truth data
```
All commands below are run **from the repo root**.

---

## The experiments

| Folder | Question it answers | Status | Key result |
|---|---|---|---|
| `I_prompt_ablation_brier/` | Which parts of the prompt actually improve accuracy? | ✅ complete | Only the stage-1 analysis call + the source-task texts matter; everything else is inert |
| `G_model_sweep/` | Which LLM is the best forecaster (prompt fixed)? | ✅ complete | per-model Brier/CRPS in `forecasting_results/forecaster_model_comparison/` |
| `0b_contamination_probe/` | Has the forecaster memorized Lyptus answers? | ✅ complete | No memorization → Exp I scores are genuine (validity gate passed) |
| `II_recalibration_decomposition/` | Is the error fixable by recalibration, or is it real difficulty? | ✅ complete | Already well-calibrated (ECE 0.064); error is irreducible, not a calibration offset |
| `H_task_variance_bin1/` | How much do forecasts vary across tasks within one difficulty bin? | ✅ complete | `summary/bin1_estimates_combined.csv` |
| `III_gepa_optimization/` | Can automated prompt search (GEPA) beat the hand-written prompt? | 🔶 pilots + measurement study complete | Yes, modestly — two validated wins on sealed cells: cand 20 (+0.017, replicated ×3) and cand 12 run at temperature 0 (+0.026, 5 paired passes). Key methodology finding: a single 84-cell val pass has re-measurement noise as large as the gains (sd 0.0065; can't even detect a sabotage prompt) and parse failures scored 1.0 masked real winners — multi-pass sealed evals are the reliable instrument. CVEBench/CyberGym transfer test pending (group decision). See `III_gepa_optimization/summary/measurement_study_2026-08.md` |

---

## How to reproduce each

### `I_prompt_ablation_brier/` — prompt ablation
One config per prompt variant. Run, score, then aggregate into the headline table.
```bash
# 1. run each condition (writes results/<condition>/<run_id>/)
python intra_benchmark_calibration/run_calibration.py \
  -c intra_benchmark_calibration/experiments/I_prompt_ablation_brier/config_control.yaml
#    ... repeat for config_minimal.yaml, config_skip_analysis.yaml, config_no_source_context.yaml,
#        config_no_bin_rate.yaml, config_no_task_outcomes.yaml, config_no_ground_truth_summary.yaml,
#        config_trim_reasoning.yaml, config_trim_all.yaml, config_single_call_analysis.yaml

# 2. score one run (Brier/CRPS + scored_with_crps.csv)
python intra_benchmark_calibration/analyse_results.py -r <path to a results/<condition>/<run_id>>

# 3. aggregate all conditions → summary/ablation_brier.{md,csv} + paired deltas
python intra_benchmark_calibration/experiments/I_prompt_ablation_brier/summarize_ablation.py
```
Curated final table: `I_prompt_ablation_brier/summary/FINAL_RESULTS.md`.

### `G_model_sweep/` — forecaster model comparison
One config per forecaster model; same prompt across all.
```bash
python intra_benchmark_calibration/run_calibration.py \
  -c intra_benchmark_calibration/experiments/G_model_sweep/config_sonnet46.yaml
#    ... repeat for config_opus47.yaml, config_haiku45.yaml, config_gpt55.yaml, config_gemini25flash.yaml
python intra_benchmark_calibration/analyse_results.py -r <path to each run dir>
```

### `0b_contamination_probe/` — memorization check
```bash
python intra_benchmark_calibration/experiments/0b_contamination_probe/run_probe.py \
  --lyptus-dir ~/lyptus-data \
  --output-dir intra_benchmark_calibration/experiments/0b_contamination_probe/results
python intra_benchmark_calibration/experiments/0b_contamination_probe/analyse_probe.py \
  --results-dir intra_benchmark_calibration/experiments/0b_contamination_probe/results
```
Output: `results/contamination_report.md`.

### `II_recalibration_decomposition/` — recalibration + Brier decomposition
Post-hoc analysis of an existing Exp I run; no new API calls.
```bash
python intra_benchmark_calibration/experiments/II_recalibration_decomposition/recalibrate_decompose.py
```
Output: `summary/recal_decomposition.md` + `summary/reliability_diagram.png`.

### `H_task_variance_bin1/` — within-bin task variance
```bash
python intra_benchmark_calibration/experiments/H_task_variance_bin1/plot_task_variance.py
```

### `III_gepa_optimization/` — pilots + measurement study complete
Runs live in the sibling `gepa` repo (currently Madhav's fork
`actionproject-madhav/gepa`, branch `feat/gepa_on_LLM_estimator` — PR to
upstream `kryjak-sai/gepa` pending; this repo is the estimation backend —
see repo-root `README_GEPA.md`):
```bash
# from the gepa repo, after its one-time setup (see FORECASTER_GEPA_README.md there)
uv run python -m forecaster_gepa.run --config configs/pilot_baseline.yaml      --phase optimize   # July baseline
uv run python -m forecaster_gepa.run --config configs/pilot_baseline.yaml      --phase finalist   # E1 sealed-cell check
uv run python -m forecaster_gepa.run --config configs/pilot_gate100.yaml       --phase optimize   # E3 gate-size test
uv run python -m forecaster_gepa.run --config configs/pilot_reflection_v2.yaml --phase optimize   # E4 reflection-instruction test
uv run python scripts/val_noise_study.py --prompts seed --repeats-val 5 --repeats-sealed 0 --tag seed5x   # measurement-noise protocol
```
Design plan and pre-run gates: `III_gepa_optimization/PLAN.md`. Verified
results + raw-data pointers: `III_gepa_optimization/summary/ladder_2026-08-02.md`
and `III_gepa_optimization/summary/measurement_study_2026-08.md`.
