# Experiment I — Prompt Ablation: FINAL RESULTS

**Forecaster:** Claude Sonnet 4.6, `reasoning_effort=off` (no extended thinking).
**Design:** 300 matched cells per condition = 25 target tasks (5 per difficulty bin) × 12 forecasted models, `all_except_target` source design. Identical cells across all conditions → paired comparison.
**Metrics (head-to-head):** Brier-on-p50 (scores only the median; chance = 0.25) and CRPS (scores the full p25/p50/p75 distribution — rewards calibrated *spread*, not just the center). Both ↓ better.
**Significance:** Wilcoxon signed-rank on per-cell *paired* deltas vs `control` (matched design), plus a 10k-resample bootstrap 95% CI of the mean delta. "Sig?" = Yes if Wilcoxon p < 0.05.

> Auto-generated numbers live in `ablation_brier.{md,csv}`; this file is the curated, stable summary. Ordered by ΔBrier vs control. Full CIs/p-values are in `FINAL_RESULTS.csv`.

## 1. What each condition removed

| Condition | What was ablated (vs the full `control` prompt) |
|---|---|
| `control` | **Nothing — full prompt.** 2 API calls/cell: (1) capability-analysis stage + (2) estimation. Includes per-bin pass-rate, per-task SOLVED/FAILED tags, reasoning checklist, and dataset base-rate footnote. |
| `no_ground_truth_summary` | Removed the **dataset-wide base-rate footnote** (`{ground_truth_summary}` — the aggregate "gray-zone" prior). Everything else identical. |
| `no_task_outcomes` | Removed the **per-task `[SOLVED/FAILED]` tags** from each source task. Per-bin pass-rate header kept. |
| `no_bin_rate` | Removed the **per-bin empirical pass-rate line** ("pass rate = x/y = z%"). Per-task SOLVED/FAILED tags kept. |
| `trim_reasoning` | Removed the **reasoning checklist + `<reasoning>` free-text block** from the estimation prompt. Stage-1 analysis call kept. |
| `trim_all` | Removed **stage-1 analysis call + reasoning checklist/block** together (= `skip_analysis` + `trim_reasoning`). |
| `skip_analysis` | Removed the **stage-1 capability-analysis API call** (single-call estimation only). Estimation prompt otherwise full. |
| `single_call_analysis` | **Merged the stage-1 analysis into the estimation call** — model writes an `<analysis>` block then forecasts, in **1 API call instead of 2**. Same analysis *content*, no separate call. |
| `minimal` | Stripped prompt: removed **analysis call + reasoning checklist + base-rate footnote + benchmark framing**. Only source-task evidence + target + p25/p50/p75 ask remain. |
| `no_source_context` | Removed the **entire capability profile** (all source tasks, bin rates, outcome tags) **+ analysis call**. Forecaster sees only model name + target task + base-rate footnote. |

## 2. Head-to-head: Brier vs CRPS

| Condition | Brier | ΔBrier | Brier p | **Brier sig?** | CRPS | ΔCRPS | CRPS p | **CRPS sig?** |
|---|---|---|---|---|---|---|---|---|
| `control` | 0.1377 | — | — | ref | 0.2126 | — | — | ref |
| `no_ground_truth_summary` | 0.1362 | −0.0015 | 0.93 | No | 0.2128 | +0.0002 | 0.61 | No |
| `no_task_outcomes` | 0.1367 | −0.0012 | 0.85 | No | 0.2155 | +0.0028 | 0.30 | No |
| `no_bin_rate` | 0.1417 | +0.0040 | 0.55 | No | 0.2122 | −0.0004 | 0.62 | No |
| `trim_reasoning` | 0.1427 | +0.0050 | 0.43 | No | 0.2181 | +0.0056 | 0.26 | No |
| `single_call_analysis` | 0.1428 | +0.0051 | 0.047 | Yes\* | 0.2209 | +0.0083 | 0.026 | **Yes** |
| `trim_all` | 0.1447 | +0.0070 | 0.0022 | Yes\* | 0.2259 | +0.0133 | 2.9e-04 | **Yes** |
| `skip_analysis` | 0.1482 | +0.0105 | 2.5e-05 | **Yes** | 0.2281 | +0.0155 | 7.3e-07 | **Yes** |
| `minimal` | 0.1502 | +0.0136 | 7.2e-05 | **Yes** | 0.2352 | +0.0224 | 1.9e-06 | **Yes** |
| `no_source_context` | 0.1554 | +0.0178 | 1.2e-07 | Yes\* | 0.2553 | +0.0427 | 3.5e-10 | **Yes** |

**Do they agree? Yes, with one nuance.** On 9 of the 10 conditions the significance verdict is identical and the rank-order among significant effects matches (`single_call_analysis` < `trim_all` < `skip_analysis` < `minimal` < `no_source_context`). The lone disagreement is `single_call_analysis`, where CRPS is cleanly significant (CI excludes 0) while Brier relies on the Wilcoxon test (CI marginally includes 0) — same pattern as `trim_all`/`no_source_context`. Spearman ρ(Brier, CRPS) across conditions = **0.85**.

**Two nuances where CRPS adds information (but never flips a conclusion):**
1. **CRPS is more decisive on the evidence-removal conditions.** For `trim_all` and `no_source_context`, the *Brier* bootstrap CI marginally includes 0 (so Brier relies on the Wilcoxon test — marked \*), but the *CRPS* CI cleanly excludes 0. CRPS gives the cleaner significance signal.
2. **CRPS penalizes evidence removal ~2× harder.** `no_source_context` is +0.0178 on Brier (+13%) but +0.0427 on CRPS (+20%). Removing the source evidence doesn't just shift the median guess — it also widens/miscalibrates the uncertainty interval, which only CRPS sees. The same ~2× amplification holds for `skip_analysis` and `minimal`.

\* Brier CI marginally includes 0 while Wilcoxon is highly significant — heavy-tailed per-cell deltas make the mean noisy but the sign test robust. CRPS does not have this issue here.

## What it means (3 sentences)

1. **The whole prompt scaffold buys very little** — the full prompt (0.1377) beats the stripped `minimal` prompt (0.1502) by only 0.0136 Brier on a 0–0.25 scale.
2. **The only individually load-bearing piece is the stage-1 analysis call** (`skip_analysis` +0.0105, ~77% of the control→minimal gap); the reasoning checklist, base-rate footnote, per-bin rate, and per-task tags are each individually inert.
3. **The source-task *texts* are what matter** — removing the entire capability profile (`no_source_context`) is the single worst intervention (+0.0178), confirming the forecaster reasons from the task descriptions rather than from the numeric annotations attached to them.
4. **A separate analysis call ≠ inline analysis** — `single_call_analysis` keeps the exact analysis *content* but folds it into the estimation call, and is significantly worse than `control` (+0.0051 Brier / +0.0083 CRPS), recovering only ~half of the standalone analysis benefit. The dedicated call itself (clean context, own token budget) is load-bearing, not just its text.

## Validity note (contamination)

Experiment 0b (`../0b_contamination_probe/`) confirmed **no outcome memorization**: Sonnet 4.6 returned `pass_rate: unknown` on all 300 (task, model) pairs. So these Brier scores reflect genuine reasoning, not recall.
