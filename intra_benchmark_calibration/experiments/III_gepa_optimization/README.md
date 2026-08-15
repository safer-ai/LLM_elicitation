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
(`FORECASTER_GEPA_RUNS.md`), analysis tooling (`scripts/`) and cloud-run
protocols (`CLOUD_RUN.md`) currently live on
[`actionproject-madhav/gepa`](https://github.com/actionproject-madhav/gepa)
branch `feat/gepa_on_LLM_estimator` (Madhav's fork — no write access to
upstream; PR to `kryjak-sai/gepa` pending). Raw run outputs are archived on
that fork's branches `results/ladder-2026-08-02` and
`results/priority-2026-08-10`; curated results live here in `summary/`.

## Runs (all share one seeded task split, `task_manifest_seed42.json`)

| Run | One change vs baseline | Question | Key result |
|---|---|---|---|
| `pilot_baseline` (2026-07-19) | — (native GEPA, 20-cell gate) | does the pipeline improve Brier at all? | val Brier 0.114→0.094 over 26 candidates; **but** winning prompts were overfit rule-lists, and the accept-gate agreed with val only 60% of the time |
| E1 `--phase finalist` (2026-08-02) | none — re-scores July's top-5 + seed on 21 sealed tasks × 11 models | was the July gain real or val-selection luck? | **mostly luck**: ranking scrambled (ρ=0.10); paired bootstrap: only cand 20 is CI-solid (+0.017 [+0.008, +0.027]), cand 12 suggestive, one candidate significantly worse than seed; on the val set itself **no candidate's edge is distinguishable from zero** |
| E3 `pilot_gate100` (2026-08-02) | acceptance gate 20→100 cells | is the coin-flip gate information-starved? | **no — falsified**: agreement *fell* to 0.10 (1/10 accepts genuinely better), seed never beaten. Gate size is not the bottleneck |
| E4 `pilot_reflection_v2` + its finalist check (2026-08-02) | reflection instruction only (transferable principles, no numeric bands, decisiveness earned not forced) | is the overfit prompt texture caused by our own instruction — and does softening keep the gain? | **texture: yes, gain: no.** Winner (cand 13) has zero bands/platform rules/anti-hedging, but its sealed edge over seed is −0.0013 [−0.0083, +0.0053] — no improvement; the run's sealed winner is the seed itself |
| Measurement study (2026-08-03 → 08-10, local + cloud) | no optimization — repeated evaluation of fixed prompts | how noisy is one evaluation pass; is the instrument valid; does cand 20's win replicate; does temp 0 help? | seed × 5 same-hour val passes: **sd 0.0065, range 0.018** (as large as the gains); a sabotage prompt ("always 0.99") landed one val pass **inside the seed's own range**; **cand 20 replicated** (better in all 3 paired sealed repeats); temp 0 not deterministic (serving noise, sd 0.005) but kills parse failures |
| temp-0 confirmation (2026-08-15, local) | none — 3 more paired sealed passes of cand 12 @ temp 0 vs seed | is cand 12's temp-0 edge real? | **confirmed: +0.026** (edges +0.0258/+0.0289/+0.0256; pooled 5 passes: 0.1045±0.003 vs seed 0.1307±0.001, non-overlapping) — **the project's best validated result** |

Details + verified numbers: `summary/ladder_2026-08-02.md` (the pilot ladder)
and `summary/measurement_study_2026-08.md` (noise, control, replications).
The two validated winning prompts: `summary/july_cand20_prompt.txt`,
`summary/july_cand12_prompt.txt`.

## Main methodological finding

A single 84-cell val pass has re-measurement noise as large as the effects
being optimized — seed × 5 fresh passes, same cells, same hour: sd 0.0065,
range 0.018 — and cannot even reliably detect a deliberately sabotaged
prompt. Two pipeline choices amplified this: parse failures scored as Brier
1.0 (one failed call shifts a val mean by +0.011, and ~2% failure rates
erased two genuinely better prompts' wins at temp 1.0), and evaluation at
temperature 1.0. **Multi-pass sealed-set evaluation (rerun sd ~0.002) is the
reliable instrument**; best-so-far val curves are running minimums over
noisy draws and must not be read as loss curves.

## Status / next steps

1. **Group decision pending:** spend the reserved one-shot CVEBench+CyberGym
   test set on the final winner — cand 12 @ temperature 0 vs the seed,
   ≥2 passes each (~$100). This answers the project's actual transfer
   question and is the headline either way.
2. Before any further optimization: retry-once-on-parse-failure patch in
   the estimation API (failures scored 1.0 distort every metric).
3. Jakub's replicate plan (4× default + 4× v2 + extend blue run) if still
   wanted after 1–2: true cost ≈ $1,000–1,100 including the multi-pass
   sealed check per run that makes it interpretable.
4. Gate thresholds (`acceptance_criterion` knobs): choose offline from the
   two runs' retrospective curves, not live sweeps.
