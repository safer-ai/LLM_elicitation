# Clean rerun of the baseline search — 2026-08-29

`configs/pilot_baseline_clean.yaml` in the gepa repo: the July search
unchanged except the fixed measurement pipeline (forecaster at temperature
0, one retry on an unparsable response, run halts if a cell still fails).
Raw data: fork branch `results/clean-rerun-2026-08-29`. Cost ~$175
(optimize + finalist + 3-pass sealed comparison).

**Zero parse failures in ~8,300 calls.** The retry+temp-0 combination fully
removed the artifact; the halt tripwire never fired.

## The de-noised search looks sober

26 candidates, 25/40 accepted. On val only 2 candidates beat the seed at
all (winner cand 3: 0.0972 vs seed 0.1031, an edge of ~1 noise sd). July's
"24 of 26 look better than the seed" was the seed's inflated single draw,
not the search.

## The verdict table (3 paired sealed passes, temp 0, 230 cells)

| prompt | sealed mean | paired edge per pass | verdict |
|---|---|---|---|
| July cand 12 | **0.1040** | +0.0268 / +0.0260 / +0.0270 | **champion re-confirmed** — now better than the seed in 8/8 lifetime paired passes |
| clean cand 7 | 0.1145 | +0.0114 / +0.0155 / +0.0212 | **the clean run's real winner** (+0.016, 3/3) |
| July cand 15 | 0.1163 | +0.0151 / +0.0140 / +0.0137 | **third July winner confirmed** — matches the parsed-only prediction (+0.0140) exactly |
| clean cand 3 (val winner) | 0.1279 | +0.0003 / +0.0015 / +0.0062 | ≈ null |
| seed | 0.1306 | — | consistent with every prior measurement |

## What this establishes

1. **GEPA's effect replicates on a bug-free pipeline.** Two independent
   runs, two genuine sealed winners (July: cands 12/15/20; clean: cand 7).
   The July result was not an artifact of the broken objective.
2. **Cand 12 remains the deployment choice** (+0.027; nothing has come
   close). The clean run did not beat it.
3. **Val-based final selection is still the weak link, even de-noised**:
   the run's own val winner is ≈ null on sealed while val-rank-4 holds the
   real gain. Protocol implication: let GEPA search, but pick the final
   prompt by multi-pass sealed evaluation of the top-k, never by val alone.
4. Cand 15's confirmation closes the last open item from the parse-failure
   audit (predicted +0.0140 from log recomputation; measured +0.0143).
