# GEPA Gate B — proxy-ranking validation (CRPS)

**Question:** can a cheap stratified subsample of cells stand in for the full-N CRPS when comparing prompts? **$0** — reuses the 10 already-scored prompt variants in `forecasting_results/prompt_variant_comparison/`. Every prompt is scored on the SAME subsampled cells per draw (paired), exactly as GEPA compares a candidate to its parent.

- Conditions tested: **10** prompt variants · common paired cells: **298** · bootstrap draws per N: 2000 (stratified by difficulty bin)

## TL;DR

- **Strict 10-way ranking** (incl. tied prompts): needs **N = 75** for mean Spearman >= 0.8. Pessimistic — penalizes shuffling of tied prompts.
- **Pairwise decision on a real gap (>=0.01 CRPS):** >=90% correct already at **N = 50** — this is the GEPA-relevant number (parent-vs-child comparisons).

## Results by proxy size

| Proxy N | mean Spearman (10-way) | P(picks true best) | pairwise acc, gap>=0.005 | pairwise acc, gap>=0.010 |
|---|---|---|---|---|
| 30 | 0.670 | 34% | 84% | 88% |
| 50 | 0.784 | 38% | 90% | 94% |
| 75 | 0.844 | 38% | 94% | 97% |
| 100 | 0.885 | 36% | 96% | 99% |
| 150 | 0.931 | 42% | 99% | 100% |
| 200 | 0.955 | 46% | 100% | 100% |
| 250 | 0.971 | 49% | 100% | 100% |

_"pairwise acc, gap>=g" = among prompt pairs whose **true** CRPS differs by at least g, how often the proxy gets the better one right (35 pairs at g=0.005), (23 pairs at g=0.010)._

## Absolute closeness: is the proxy CRPS value near the full-N CRPS?

| Proxy N | avg |error| in CRPS | within +/-0.01 | within +/-0.02 |
|---|---|---|---|
| 30 | 0.0282 | 22% | 43% |
| 50 | 0.0214 | 30% | 55% |
| 75 | 0.0162 | 38% | 68% |
| 100 | 0.0130 | 47% | 78% |
| 150 | 0.0093 | 62% | 91% |
| 200 | 0.0066 | 78% | 98% |
| 250 | 0.0042 | 94% | 100% |

## Recommendation for GEPA

- **Use the proxy only for coarse filtering** (keep/kill on gaps >= ~0.01 CRPS). Don't trust it to split hairs between near-tied candidates.
- **Re-validate surviving Pareto finalists on the full cell set** before declaring a winner.
- Prefer a **proxy of ~100–150 cells** over the 50 originally assumed.

## Reference: full-CRPS ranking on the common cells (best -> worst)

| Rank | Condition | full CRPS |
|---|---|---|
| 1 | `no_bin_rate` | 0.2112 |
| 2 | `control` | 0.2118 |
| 3 | `no_ground_truth_summary` | 0.2119 |
| 4 | `no_task_outcomes` | 0.2149 |
| 5 | `trim_reasoning` | 0.2173 |
| 6 | `single_call_analysis` | 0.2193 |
| 7 | `trim_all` | 0.2249 |
| 8 | `skip_analysis` | 0.2273 |
| 9 | `minimal` | 0.2334 |
| 10 | `no_source_context` | 0.2542 |

> Caveat: validates ranking on 10 hand-built prompts only. GEPA's candidates may cluster even more tightly — treat the proxy as a coarse filter and always confirm finalists on the full set.

