# GEPA Gate B — proxy-ranking validation (BRIER)

**Question:** can a cheap stratified subsample of cells stand in for the full-N BRIER when comparing prompts? **$0** — reuses the 10 already-scored prompt variants in `forecasting_results/prompt_variant_comparison/`. Every prompt is scored on the SAME subsampled cells per draw (paired), exactly as GEPA compares a candidate to its parent.

- Conditions tested: **10** prompt variants · common paired cells: **298** · bootstrap draws per N: 2000 (stratified by difficulty bin)

## TL;DR

- **Strict 10-way ranking** (incl. tied prompts): needs **N = 150** for mean Spearman >= 0.8. Pessimistic — penalizes shuffling of tied prompts.
- **Pairwise decision on a real gap (>=0.01 BRIER):** >=90% correct already at **N = 100** — this is the GEPA-relevant number (parent-vs-child comparisons).

## Results by proxy size

| Proxy N | mean Spearman (10-way) | P(picks true best) | pairwise acc, gap>=0.005 | pairwise acc, gap>=0.010 |
|---|---|---|---|---|
| 30 | 0.436 | 16% | 72% | 77% |
| 50 | 0.541 | 21% | 78% | 82% |
| 75 | 0.642 | 24% | 83% | 88% |
| 100 | 0.723 | 28% | 88% | 91% |
| 150 | 0.831 | 36% | 94% | 96% |
| 200 | 0.907 | 42% | 98% | 99% |
| 250 | 0.954 | 52% | 100% | 100% |

_"pairwise acc, gap>=g" = among prompt pairs whose **true** BRIER differs by at least g, how often the proxy gets the better one right (30 pairs at g=0.005), (13 pairs at g=0.010)._

## Absolute closeness: is the proxy BRIER value near the full-N BRIER?

| Proxy N | avg |error| in BRIER | within +/-0.01 | within +/-0.02 |
|---|---|---|---|
| 30 | 0.0281 | 22% | 43% |
| 50 | 0.0213 | 30% | 55% |
| 75 | 0.0162 | 38% | 68% |
| 100 | 0.0130 | 47% | 78% |
| 150 | 0.0093 | 61% | 90% |
| 200 | 0.0066 | 78% | 98% |
| 250 | 0.0042 | 94% | 100% |

## Recommendation for GEPA

- **Use the proxy only for coarse filtering** (keep/kill on gaps >= ~0.01 BRIER). Don't trust it to split hairs between near-tied candidates.
- **Re-validate surviving Pareto finalists on the full cell set** before declaring a winner.
- Prefer a **proxy of ~100–150 cells** over the 50 originally assumed.

## Reference: full-BRIER ranking on the common cells (best -> worst)

| Rank | Condition | full BRIER |
|---|---|---|
| 1 | `no_ground_truth_summary` | 0.1358 |
| 2 | `no_task_outcomes` | 0.1363 |
| 3 | `control` | 0.1372 |
| 4 | `no_bin_rate` | 0.1408 |
| 5 | `single_call_analysis` | 0.1414 |
| 6 | `trim_reasoning` | 0.1423 |
| 7 | `trim_all` | 0.1440 |
| 8 | `skip_analysis` | 0.1478 |
| 9 | `minimal` | 0.1500 |
| 10 | `no_source_context` | 0.1543 |

> Caveat: validates ranking on 10 hand-built prompts only. GEPA's candidates may cluster even more tightly — treat the proxy as a coarse filter and always confirm finalists on the full set.

