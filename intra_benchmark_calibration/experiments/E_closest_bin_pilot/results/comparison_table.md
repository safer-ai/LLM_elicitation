# Experiment Comparison: Intra-Benchmark Calibration

All runs use claude-sonnet-4-6, 2 experts, 1 Delphi round, 5 bins, K=1 target task per cell.
Metrics computed by `analyse_results.py` (groups by condition_id, keeps final Delphi round per cell).

| Metric | A — all_bins, no think | B — all_bins, think | C — single_bin (all 20 pairs) | **E — closest_bin (5 pairs)** |
|---|---|---|---|---|
| N analyzed | 60 | 60 | 240 | **60** |
| **Brier ↓** | 0.2336 | 0.2255 | 0.1736 | **0.1030** |
| **CRPS ↓** | 0.3113 | 0.3014 | 0.2601 | **0.1908** |
| ECE ↓ | 0.1732 | 0.1755 | **0.0888** | 0.1428 |
| **Spearman rho ↑** | 0.31 | 0.35 | 0.54 | **0.77** |
| **Kendall tau ↑** | 0.26 | 0.29 | 0.45 | **0.64** |
| **MAE ↓** | 0.3892 | 0.3765 | 0.3334 | **0.2555** |
| **RMSE ↓** | 0.4833 | 0.4749 | 0.4167 | **0.3209** |
| Bias | −0.074 | −0.075 | −0.032 | **+0.010** |

## Run details

| Run | Folder | Source profile | Thinking | Cells | API calls |
|---|---|---|---|---|---|
| A | `results/A_all_bins_no_thinking/` | all_except_target | off | 5 | 120 |
| B | `results/B_all_bins_thinking/` | all_except_target | on | 5 | 120 |
| C | `results/C_single_bin_k1/` | single_bin (all 20 i→j pairs) | on | 240 | 480 |
| E | `experiments/E_closest_bin_pilot/results/20260503_142252/` | closest_bin (5 i→j pairs) | on | 60 | 120 |

## What each condition means

Shared across all runs: model=claude-sonnet-4-6, T=1.0, 12 forecasted models, 5 bins (equal-count FST),
2 expert personas, 1 Delphi round, K=1 target task per cell, 1 anchor + 2 easier tasks per source bin.

**What varies is what source context the forecaster sees:**

**A — all_except_target, thinking OFF**
- For each target bin j, show all 4 other bins as source context (i.e. 4 bins × 3 tasks = 12 source tasks).
- Extended thinking disabled (temperature=0.8 based on config filename, but thinking off).
- 5 cells per model (one per target bin). 60 total cells, 120 API calls.
- The forecaster sees a broad capability profile spanning the whole benchmark.

**B — all_except_target, thinking ON**
- Identical to A but with extended thinking enabled (budget=10k tokens).
- Same 5 cells per model, 60 total, 120 API calls.
- Tests whether chain-of-thought reasoning improves calibration vs. no-thinking.

**C — single_bin, all 20 pairs, thinking ON**
- For each (source bin i, target bin j) pair where i≠j, show only source bin i as context.
- All 20 ordered pairs are run: 5×4=20 pairs × 12 models = 240 cells, 480 API calls.
- The forecaster sees a focused, single-bin capability profile rather than the whole benchmark.
- Much more expensive but tests every possible source→target bin direction.

**E — closest_bin, 5 pairs only, thinking ON**
- For each target bin j, show only the single source bin whose mean pass rate is closest to bin j's.
- This gives exactly 5 (i,j) pairs (one per target bin), same cost as A/B.
- The forecaster sees only the most similar source bin — maximally relevant context.
- Key idea: showing a closely matched reference bin gives the forecaster the best analogical anchor.

## Closest-bin mapping used in E

Computed at runtime from panel-wide mean pass rates (see `_compute_closest_source_bins` in `experiments/E_closest_bin_pilot/task_selector.py`):

| Target bin j | Source bin i | Pass rate j | Pass rate i | |Δ| |
|---|---|---|---|---|
| 0 | 1 | 0.931 | 0.869 | 0.062 |
| 1 | 0 | 0.869 | 0.931 | 0.062 |
| 2 | 1 | 0.577 | 0.869 | 0.292 |
| 3 | 4 | 0.272 | 0.197 | 0.075 |
| 4 | 3 | 0.197 | 0.272 | 0.075 |

## Key findings

- E dominates A, B, C on Brier, CRPS, MAE, RMSE, Spearman rho, Kendall tau, and Bias.
- C still has the best ECE (0.0888 vs 0.1428), meaning the full 20-pair loop produces better-calibrated interval widths, but E is far more accurate pointwise.
- E ran 4× faster than C (120 vs 480 API calls) while achieving nearly 2× lower Brier (0.1030 vs 0.1736).
- Bias is essentially zero in E (+0.010), vs systematic underconfidence in all prior runs.
- All metrics verified manually to 4 decimal places against `analyse_results.py` output.

## Note on expert aggregation

`analyse_results.py` uses `groupby("condition_id").idxmax("delphi_round")` to select one row per cell.
With 1 Delphi round, this picks whichever expert row has the higher pandas index (semi-arbitrary).
The expert split in E was 35 / 25 across 60 cells. If both experts are averaged per cell instead:
Brier = 0.1016, Spearman rho = 0.78 — marginally better, consistent with reported numbers.
This behaviour is identical across all four runs, so comparisons are apples-to-apples.
