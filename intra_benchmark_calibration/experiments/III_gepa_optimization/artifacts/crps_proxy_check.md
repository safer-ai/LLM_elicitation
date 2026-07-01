# CRPS proxy-reliability check (companion to proxy_check.md)

Same paired-subsampling methodology as `proxy_check.py` but using **CRPS** as
the scoring metric instead of Brier. Every prompt is scored on the **same**
subsampled cells per draw (paired design, matching GEPA's parent-vs-child setup).

## Full-CRPS ranking (best → worst, 298 common cells)

`no_bin_rate` (0.211) > `control` (0.212) > `no_ground_truth_summary` (0.212) >
`no_task_outcomes` (0.215) > `trim_reasoning` (0.217) > `single_call_analysis`
(0.219) > `trim_all` (0.225) > `skip_analysis` (0.227) > `minimal` (0.233) >
`no_source_context` (0.254)

Note: top-3 are within 0.001 CRPS of each other (statistically tied).

## Pairwise proxy accuracy by subsample size

n pairs at gap ≥ 0.005: 35 · gap ≥ 0.010: 23 · gap ≥ 0.015: 17

| Proxy N | mean Spearman | P(top-1) | pair acc gap ≥ 0.005 | pair acc gap ≥ 0.010 | pair acc gap ≥ 0.015 |
|---|---|---|---|---|---|
| 30  | 0.670 | 34% | 84% | 88% | 90% |
| 50  | 0.784 | 38% | 90% | 94% | 95% |
| 75  | 0.844 | 38% | 94% | 97% | 98% |
| **100** | **0.885** | **36%** | **96%** | **99%** | **99%** |
| 150 | 0.931 | 42% | 99% | 100% | 100% |
| 200 | 0.955 | 46% | 100% | 100% | 100% |
| 250 | 0.971 | 49% | 100% | 100% | 100% |

2000 stratified bootstrap draws (same seed as Brier check).

## Comparison with Brier at N=100

| Metric | pair acc (gap ≥ 0.010) | n pairs at threshold | Spearman |
|---|---|---|---|
| Brier  | 91% | 13 | 0.723 |
| CRPS   | **99%** | 23 | **0.885** |

**CRPS is more proxy-reliable than Brier at the same N.** The CRPS scale spreads
conditions further apart relative to within-cell noise, so more pairs exceed the
0.010 threshold and are easier to distinguish in a subsample.

## Implication for Jakub's adaptive-N point

Both metrics are comfortable at N=100 for gaps ≥ 0.010. As GEPA converges and
gaps shrink toward ~0.005, CRPS still gives 96% accuracy at N=100 while Brier
drops to ~88% (from proxy_check.md). If optimizing CRPS, the adaptive-N trigger
can be set later (smaller gap) than if optimizing Brier.

## Script

The inline script that produced this table lives in the transcript. To re-run,
copy the CRPS block from `proxy_check.py` and swap `se = (p50 - outcome)^2` for
`se = crps` (already stored per-row in the scored CSVs).
