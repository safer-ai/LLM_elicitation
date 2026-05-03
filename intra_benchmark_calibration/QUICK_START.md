# Quick Start: Intra-Benchmark Calibration

## Prerequisites

```bash
# From repo root
pip install -r requirements.txt
```

(`scipy` is needed for the analysis script's Beta fit + CRPS — already in
`requirements.txt`.)

## 1. Get the Lyptus data

Either reuse an existing checkout or fetch a fresh one (sparse, no LFS, SHA-pinned):

```bash
# Fresh fetch (recommended for reproducibility):
python intra_benchmark_calibration/scripts/fetch_lyptus_data.py \
    --repo-dir ~/lyptus-data
```

This pulls only `analysis/figures/data/*.parquet` (~10 MB) and `data/tasks/*/*.jsonl`
(~30 MB), skipping the 18 GB of `.eval` files in Git LFS. Pinned to commit
`a514c63` (2026-04-02 initial release).

If you already have a checkout at `/home/you/cyber-task-horizons-data`, point
the config at it directly (next step).

## 2. Set the API key

```bash
cp intra_benchmark_calibration/.env.example intra_benchmark_calibration/.env
# edit .env, paste your ANTHROPIC_API_KEY
```

`.env` is gitignored. Resolution order: YAML config → project `.env` →
repo-root `.env` → process env. The first non-empty source wins; the source
and key prefix are logged at run start.

## 3. Production run (~1–3 hours, ~960 API calls, ~$10–20 on Sonnet 4.6)

```bash
python intra_benchmark_calibration/run_calibration.py \
    -c intra_benchmark_calibration/config_full.yaml -d
```

Defaults: `n_bins=5`, all 12 forecasted models, K=1 target task, 2 expert
personas, 1 Delphi round = `5 × 4 × 12 × 1 = 240` cells × 2 experts × 2 calls
= 960 API calls.

## 4. Analyse

```bash
# Auto-pick the most recent run:
python intra_benchmark_calibration/analyse_results.py --latest

# Or specify:
python intra_benchmark_calibration/analyse_results.py \
    -r intra_benchmark_calibration/results/20260502_171824
```

Writes 6 PNGs + `statistics.txt` + `scored_with_crps.csv` to
`{run_dir}/plots/`. Open the PNGs in your IDE.

Headline metric: Brier-on-p50 (chance baseline = 0.25).

## Common issues

**Bin has no admissible anchor for some model** — model has no evaluated
tasks in that bin (e.g. dropped models). The cell is skipped; row counts
in the CSV will be lower than `n_bins × (n_bins-1) × n_models × K`.

**`No admissible cell plans were produced`** — usually means `forecasted_models`
contains a model not in the outcomes matrix, or all bins are empty. Check
the warning logs.

## Resuming a partial run

There is no `--resume` flag yet. Re-running with the same config produces a
new `run_id` directory; merge the partial CSVs by hand if you want to combine
them. (TODO: add `scripts/resume.py` if/when the elicitation budget gets large
enough that interrupted runs become a real concern.)
