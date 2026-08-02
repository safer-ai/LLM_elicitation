# Experiment III — GEPA prompt optimization

**Question:** can reflective prompt evolution (GEPA) beat the hand-written
forecaster prompt on grand Brier — and does it find *general, transferable*
prompt patterns rather than metric-overfit rule lists?

The experiment runs across two repos: the GEPA fork
([`gepa`](https://github.com/kryjak-sai/gepa), branch
`feat/gepa_on_LLM_estimator`) drives the optimization; this repo (branch
`feat/ss2026_intrabenchmark_package`) is its estimation backend (see
repo-root `README_GEPA.md`). Full plan: `PLAN.md` (gates A–C were run before
any paid experiment). Per-run configs, the runs index
(`FORECASTER_GEPA_RUNS.md`) and analysis tooling live in the gepa repo; raw
run outputs are archived on branch `results/ladder-2026-08-02` of
[`actionproject-madhav/gepa`](https://github.com/actionproject-madhav/gepa).

## Runs (all share one seeded task split, `task_manifest_seed42.json`)

| Run | One change vs baseline | Question | Key result |
|---|---|---|---|
| `pilot_baseline` (2026-07-19) | — (native GEPA, 20-cell gate) | does the pipeline improve Brier at all? | val Brier 0.114→0.094 over 26 candidates; **but** winning prompts were overfit rule-lists, and the accept-gate agreed with val only 60% of the time |
| E1 `--phase finalist` (2026-08-02) | none — re-scores July's top-5 + seed on 21 sealed tasks × 11 models | was the July gain real or val-selection luck? | **mostly luck**: ranking scrambled (ρ=0.10), 3/5 finalists ≤ seed; a modest real edge (~0.013–0.017) survives for 2 candidates |
| E3 `pilot_gate100` (2026-08-02) | acceptance gate 20→100 cells | is the coin-flip gate information-starved? | **no — falsified**: agreement *fell* to 0.10 (1/10 accepts genuinely better), seed never beaten. Gate size is not the bottleneck |
| E4 `pilot_reflection_v2` (2026-08-02) | reflection instruction only (transferable principles, no numeric bands, decisiveness earned not forced) | is the overfit prompt texture caused by our own instruction? | **yes, largely**: winner (cand 13) improved on seed (0.103 vs 0.114 val) with **zero** numeric bands / platform rules / forced anti-hedging |

Details + verified numbers: `summary/ladder_2026-08-02.md`.

## Main methodological finding

The same seed prompt scored **0.114** (July run) and **0.0995** (gate-100
run) on the *identical* 84 val cells — a ~0.015 swing from forecaster
sampling noise alone (temperature 1.0). That is as large as the effect sizes
being chased, which explains the low gate↔val agreement in every run and the
E1 ranking scramble: **the single-draw val measurement, not the acceptance
gate, is the binding noise source.** Any follow-up design needs repeated
draws per cell, a larger val set, or sealed-set re-ranking as standard
practice.

## Status / next steps

- Finalist re-ranking of the E4 (reflection-v2) winner on sealed cells
  (~$12) — pending; required before its 0.103 can be believed (same
  winner's-curse correction E1 applied to July).
- Gate thresholds (Jakub's `acceptance_criterion` knobs) to be chosen from
  the two runs' retrospective curves, not live sweeps.
- Held-out CVEBench+CyberGym test set remains untouched.
