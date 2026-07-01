# GEPA Gate B — proxy-ranking validation

**Question:** can a cheap stratified subsample of cells stand in for the full 300-cell Brier when comparing prompts? If yes, GEPA can score candidates on the subsample (the key cost saving) instead of the full set. **$0** — reuses the 10 already-scored prompt variants in `forecasting_results/prompt_variant_comparison/`. Every prompt is scored on the SAME subsampled cells per draw (paired), exactly as GEPA compares a candidate to its parent.

- Conditions tested: **10** prompt variants · common paired cells: **298** · bootstrap draws per N: 2000 (stratified by difficulty bin)

## TL;DR — the naive small proxy is too noisy for *fine* ranking, but fine for *coarse* decisions

These 10 prompts sit in a very narrow Brier band (0.136–0.155), and the best 4 are statistically tied (Exp I). Distinguishing near-identical prompts on a small subsample is genuinely hard — but that is **not** what GEPA needs. GEPA needs to tell a *clearly* better prompt from a worse one. Two readings:

- **Strict 10-way ranking** (incl. tied prompts): needs **N = 150** for mean Spearman >= 0.8. Pessimistic — penalizes shuffling of tied prompts.
- **Pairwise decision on a real gap (>=0.01 Brier):** >=90% correct already at **N = 100** — this is the GEPA-relevant number (parent-vs-child comparisons).

## Results by proxy size

| Proxy N | mean Spearman (10-way) | P(picks true best) | pairwise acc, gap>=0.005 | pairwise acc, gap>=0.01 |
|---|---|---|---|---|
| 30 | 0.436 | 16% | 72% | 77% |
| 50 | 0.541 | 21% | 78% | 82% |
| 75 | 0.642 | 24% | 83% | 88% |
| 100 | 0.723 | 28% | 88% | 91% |
| 150 | 0.831 | 36% | 94% | 96% |
| 200 | 0.907 | 42% | 98% | 99% |
| 250 | 0.954 | 52% | 100% | 100% |

_"pairwise acc, gap>=g" = among prompt pairs whose **true** Brier differs by at least g, how often the proxy gets the better one right (30 pairs at g=0.005), (13 pairs at g=0.01)._

## Absolute closeness: is the proxy Brier *value* near the full Brier?

Different question from ranking: here we ask how far a single N-cell Brier lands from the true 300-cell Brier (`|proxy - full|`, over all prompts x draws).

| Proxy N | avg |error| in Brier | within +/-0.01 | within +/-0.02 |
|---|---|---|---|
| 30 | 0.0281 | 22% | 43% |
| 50 | 0.0213 | 30% | 55% |
| 75 | 0.0162 | 38% | 68% |
| 100 | 0.0130 | 47% | 78% |
| 150 | 0.0093 | 61% | 90% |
| 200 | 0.0066 | 78% | 98% |
| 250 | 0.0042 | 94% | 100% |

**Key insight:** the prompt differences we care about (0.001–0.018 Brier) are *smaller* than the absolute wobble of a 50-cell Brier (~0.021). To pin the absolute Brier to +/-0.01 you need ~250 of the 300 cells — subsampling barely helps for the *absolute* number. Yet pairwise *comparisons* are reliable at ~100 cells, because comparing two prompts on the **same** cells cancels the shared task-difficulty noise. GEPA needs comparisons, not absolute values — so ~100 is the operative number, not ~250.

## Recommendation for GEPA

- **Use the proxy only for coarse filtering**, where it is reliable: keep/kill candidates that move Brier by >=~0.01. Don't trust it to split hairs between near-tied candidates.
- **Re-validate the surviving Pareto finalists on the full cell set** before declaring a winner (already in the plan).
- Given how tightly prompts cluster here, prefer a **larger proxy (~100–150)** than the 50 originally assumed. Recompute cost with this N.

## Reference: full-Brier ranking on the common cells (best -> worst)

| Rank | Condition | full Brier |
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

> Caveat: validates ranking on 10 hand-built prompts only. GEPA's candidates may cluster even more tightly, so treat the proxy as a coarse filter, re-draw the subsample periodically, and always confirm finalists on the full set.

