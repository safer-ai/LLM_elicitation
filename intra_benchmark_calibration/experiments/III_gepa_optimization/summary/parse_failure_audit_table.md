# Parse-failure audit: which experiments are affected, which need re-running

**The bug.** When the forecaster's answer does not contain a parseable
`<p50>`, the pipeline scores that cell **Brier 1.0** — the worst possible
value — instead of retrying or scoring it neutrally. A typical cell scores
~0.10, so one failure costs about **nine ordinary cells**. Impact of a single
failure on a mean: **20-cell gate batch +0.045**, **84-cell val pass +0.011**,
**230-cell held-out pass +0.004**. Our real prompt effects are ~0.02, so a
handful of failures can erase or invent a result.

Failures are logged explicitly (sentinel score −1.0), so their effect on any
**score** can be recomputed exactly with **no new API calls**. Reproduce every
number below with:

```bash
uv run python scripts/parse_failure_audit.py     # in the gepa repo
```

**What can and cannot be recovered**

| | Recoverable from logs? |
|---|---|
| Any **score** (val, held-out, gate batch) | **Yes, exactly** — recompute with failures dropped |
| Which **decisions** a failure could have flipped | **Yes** — compare the accept/reject margin against the ~0.9 penalty |
| The **trajectory** after a flipped decision | **No.** GEPA is sequential: a flipped accept changes the pool, hence parent selection, hence every later iteration. Re-running does not recover it either — at temperature 1.0 a re-run is a different sample, so it cannot isolate the fix from run-to-run variance |

**Validation that "parsed-only" recomputation is trustworthy:** for cands 12
and 20, the parsed-only estimate from the temp-1 held-out data
(+0.0267/+0.0268 and +0.0151/+0.0173) matched their later temp-0 *measured*
edges (+0.0262 and +0.0172) to within ~0.001. Recomputation and
re-measurement agree.

---

## Master table

"OPT" = the optimization stage (expensive: gate + val cells inside the GEPA
loop). "EVAL" = post-hoc scoring of fixed prompts (cheap, repeatable).

| # | Experiment | Saved at | OPT failures | OPT decisions flipped | EVAL failures | EVAL verdicts changed | Re-run OPT? | Re-run EVAL? |
|---|---|---|---|---|---|---|---|---|
| R1 | **`pilot_baseline`** — the July GEPA run (40 iters, 26 candidates) | `runs/pilot_baseline/`, branch `results/ladder-2026-08-02` | gate **5/1600**; val **7/2184** (incl. **the seed**) | **3** (iters 5, 11, 21 — all rejections that would become accepts) | see R1-E | see R1-E | **No** — a re-run is a new sample, not a correction; its outputs (cands 12/15/20) are already validated by re-measurement. Perturbation is a documented caveat | — |
| R1-E | **E1: held-out re-ranking of R1's top-5 + seed** | `runs/pilot_baseline/finalist_*.{json,jsonl}` | — | — | **17/2530** (cand 12:4, 14:5, 15:5, 9:2, 20:1; **seed: 0**) | **Yes, 3 of them** — cand 12 tie→**+0.027**; **cand 15 tie→+0.012/+0.013**; cand 14 "much worse"→mildly worse | — | **cand 15: yes (~$10)** to convert it from recomputed to measured. cands 9/14 optional. cand 12/20 already re-measured |
| R2 | **`pilot_gate100`** — E3, 100-cell acceptance gate | `runs/pilot_gate100/`, `results/ladder-2026-08-02` | gate **1/8000**; val **2/924** (cands 1, 3) | **0** | no eval phase run | — | **No** — verified: even parsed-only, cands 1/3 score 0.116/0.114 vs seed 0.0995, so "seed never beaten" holds | — |
| R3 | **`pilot_reflection_v2`** — E4, softened reflection instruction | `runs/pilot_reflection_v2/`, `results/ladder-2026-08-02` | gate **0/1600**; val **0/1932** | **0** | — | — | **No** | — |
| R3-E | **E4 held-out check of R3's top-5 + seed** | `runs/pilot_reflection_v2/finalist_*` | — | — | **0/1150** | none | — | **No** |
| M1 | **Seed self-consistency**: same prompt × 5 val passes | `runs/noise_study/noise_*_seed_local_5x.*`, branch `results/priority-2026-08-10` | — | — | **0/420** | none | — | **No** |
| M2 | **cand 20 vs seed**, 3 paired held-out passes (temp 1) | `runs/noise_study/noise_*_sealed_check.*` | — | — | **0/1380** | none | — | **No** |
| M3 | **Sabotage control** ("always 0.99") × 3 val passes | `runs/noise_study/noise_*_control.*` | — | — | **0/252** | none | — | **No** |
| M4 | **temp-0 arm**: seed/cand12/cand20, val+held-out | `runs/noise_study/noise_*_temp0.*` | — | — | **2/2136** (both cand 12) | none — failures are **included** in the reported +0.026, so it is understated | — | **No** |
| M5 | **temp-0 confirmation**: cand 12 vs seed, 3 paired held-out passes | `runs/noise_study/noise_*_temp0_confirm.*` | — | — | **1/1380** (cand 12) | none — included, so conservative | — | **No** |
| V | **verify-sign** sanity checks (N=8 fail, N=24 pass) | `runs/verify_sign_n8/`, `runs/verify_sign_n24/` | — | — | 0 | none — the N=8 failure was sampling noise, not a parse failure | — | **No** |

**Totals: 32 failed cells out of ~25,500 scored cells (0.13%).** All of them
sit in R1 and its eval (R1-E) except 3 in R2 and 3 in the temp-0 arms.

---

## The single most consequential failure

**The seed prompt failed one val cell in the July run**, inflating its val
Brier from **0.1038 → 0.1144**. The five top candidates had **zero** val
failures, so every "improvement over the seed on val" is overstated by
exactly **+0.0107**:

| Candidate | val edge as first reported | val edge, corrected |
|---|---|---|
| 12 | +0.0207 | **+0.0100** |
| 14 | +0.0181 | +0.0074 |
| 20 | +0.0130 | +0.0023 |
| 9 | +0.0112 | +0.0005 |
| 15 | +0.0107 | +0.0001 |

So the widely-quoted July headline **"val 0.114 → 0.094 (+0.021)"** is really
**"0.104 → 0.094 (+0.010)"**. Independent confirmation: the seed's five clean
re-measurements average **0.1036**, matching the parsed value.

This is a **reporting** correction, not a re-run: no decision inside the run
depended on the seed's aggregate val score (parent selection uses per-cell
Pareto standings, where the single failed cell costs the seed exactly one
cell).

## Standing fix

Retry once on parse failure (or score unparseable cells at chance, 0.25,
instead of 1.0) before any further runs — otherwise every future experiment,
including replicate runs, inherits the same bias: the short seed prompt
essentially never fails (~1 in 1,400 cells) while longer evolved prompts fail
~1%, so the penalty lands almost entirely on the candidate side of every
comparison.
