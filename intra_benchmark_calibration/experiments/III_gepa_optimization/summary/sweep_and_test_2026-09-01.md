# Sweep arms A/B and the reserved test — 2026-09-01

Record of the three paid experiments run 2026-09-01, all pre-declared in
`FORECASTER_GEPA_RUNS.md` (gepa repo) and screened offline first (figs 12–16).
Raw data: gepa branch `results/sweep-arms-2026-09-01`. Total spend ≈ $290.

## 1. Sweep arm A — joint acceptance gate (`aggregate_sum_and_min_task_wins`, k = 8)

One change vs the clean baseline: a proposed prompt must beat its parent on the
20-question total AND win ≥ 8 of the 20. Full run (40 proposals) + finalist +
3-pass paired sealed study (`tag accept_joint`).

- 18/40 accepted (native rule: 25/40 in both prior runs) — the gate is visibly stricter; zero cell failures.
- Val-based selection misled for the third straight run: the val winner (cand 16) ranked worst of the top-5 on the finalist set.
- **Sealed verdict: joint cand 5 +0.0248 vs seed (3/3 passes) — champion-level**, indistinguishable from july cand 12 (+0.0254 same session); joint cand 16 +0.0165 (3/3).

## 2. Sweep arm B — (model × bin) Pareto instances (`pareto_instance: model_bin`)

One change vs the clean baseline: parents compete on 20 (model × difficulty)
group means instead of 84 single questions. One Anthropic server-500 tripped
the halt tripwire mid-run (its designed purpose); resumed cleanly from the
per-iteration checkpoint.

- 22/40 accepted, zero parse failures. Group-val and finalist winners agreed (cand 18, ρ = 0.6) — selection tracked better under aggregation, but the field is weaker (best finalist 0.1144 vs arm A's 0.1026; the seed itself sat in the top-5 by group-val).
- **Sealed verdict: modelbin cand 18 +0.0217 (3/3); modelbin cand 10 +0.0115 (3/3)**; july cand 12 re-confirmed +0.0256 (now 14/14 lifetime paired passes).

**Sweep conclusion (one run per arm, preliminary):** every setting — native,
joint k = 8, model_bin — produces real sealed winners (+0.016 to +0.026), and
the differences between arms sit within the search's own run-to-run spread
(the two native runs drew +0.026 and +0.016). No acceptance/frontier setting
is clearly better; the binding constraints are final selection and transfer.

## 3. The reserved test — CVEBench + CyberGym, first and only use

Pre-registered one-shot transfer test, spent by group go-ahead: seed,
july cand 12, clean cand 7; 2 paired repeats × 1,033 questions (94 frozen
tasks × 11-model panel, all usable tasks of both benchmarks — no difficulty
filter; the families are intrinsically hard, bins 2–4). Zero failures in
6,198 calls. Fig 17.

| arm | test Brier | paired vs seed |
|---|---|---|
| GT LOO bin-mean table (no LLM) | 0.1488 | — |
| seed | 0.1634 | — |
| clean cand 7 | 0.1872 | +0.0238 worse (t = 6.9) |
| july cand 12 | 0.1995 | +0.0362 worse (t = 9.3) |

**The in-distribution ordering inverts: the optimized prompts do not
transfer.** Robustness (all recomputed from logged cells): survives
task-level clustering (t = 5.2/4.2, n = 94; seed better on 70 %/65 % of
tasks), model-level clustering (worse for 9/11 models), both repeats;
carried by cybergym (869 cells) while cvebench alone (164 cells) is a wash.

Mechanism, consistent with the causal feature ablation: the winners' edge is
numeric anchors encoding the training families' bin → solve-rate mapping.
That mapping does not transfer (same difficulty bin, share solved: 0.96 → 0.50
in bin 1, 0.84 → 0.40 in bin 2), and on unsolved test questions the winners
forecast too high (mean p50 0.41/0.38 vs seed 0.30 at base rate 0.24). It is
not a hardness effect — in-distribution the winners' largest gains are on the
hardest bins (+0.05 on bin 4) — and not a design artifact: all arms see
identical evidence (training-family evidence only, the deployment condition).

## 4. Test extension — generality and mechanism (same day, ~$100)

Four more arms, 2 repeats each on the same 1,033 cells, paired against the
seed's test measurements (zero failures):

| arm | test Brier | paired vs seed |
|---|---|---|
| v2 cand 13 (evolved, no-numbers rewriter; in-dist null) | 0.1634 | +0.0001 (t = 0.0) |
| bands-only probe (in-dist +0.020) | 0.1686 | +0.0053 worse (t = 2.3) |
| procedure-only probe (in-dist +0.006) | 0.1831 | +0.0197 worse (t = 7.0) |
| joint cand 5 (arm A winner; in-dist +0.0248) | 0.2300 | +0.0666 worse (t = 11.5) |

- **Generality: confirmed and amplified.** A third confirmed winner (cand 5)
  transfers worst of all — the strictest-gate champion overfits hardest.
- **Mechanism, refined:** both halves of the winning recipe fail out of
  domain, and their order flips — the aggressive number-free *procedure*
  (anchoring, nearest-example weighting, task-type rules, anti-hedging)
  hurts more (+0.0197) than the numeric bands (+0.0053), the reverse of
  in-distribution where the bands carried the gain. So the failure is not
  only wrong memorized numbers; the whole tuned recipe is domain-specific.
- **The number-free evolved prompt (v2 cand 13) is exactly neutral** — no
  in-distribution gain, no out-of-distribution damage. The v2 run's null
  result gains a reframe: soft-constrained evolution bought nothing but
  also broke nothing.
- Across all six measured evolved/probe prompts, in-distribution gain
  roughly predicts out-of-distribution harm.

## 5. Pre-registered slice analysis — bounding the two axes ($0, offline)

The split-design doc pre-registered separate scores on
CVEBench ∪ {CyberGym | FST < 2 h} (domain shift with capped difficulty
shift) vs {CyberGym | FST ≥ 2 h} (the full joint shift). Recomputed from the
logged test cells, paired vs seed:

| arm | EASY slice (48 tasks, 527 cells) | HARD slice (46 tasks, 506 cells) |
|---|---|---|
| july cand 12 | +0.0298 worse (t = 5.3) | +0.0428 worse (t = 7.9) |
| clean cand 7 | +0.0190 worse (t = 4.1) | +0.0289 worse (t = 5.6) |
| joint cand 5 | +0.0565 worse (t = 6.8) | +0.0771 worse (t = 9.7) |
| v2 cand 13 | −0.0019 (n.s.) | +0.0021 (n.s.) |

Task-level clustered (conservative), split further by family:

| arm | CVEBench (15 tasks) | CyberGym < 2 h (33) | CyberGym ≥ 2 h (46) |
|---|---|---|---|
| july cand 12 | +0.0009 (t = 0.1) | +0.0428 (t = 3.4) | +0.0428 (t = 4.4) |
| clean cand 7 | +0.0104 (t = 1.1) | +0.0229 (t = 2.4) | +0.0289 (t = 3.3) |
| joint cand 5 | +0.0063 (t = 0.5) | +0.0793 (t = 4.4) | +0.0771 (t = 5.0) |

- **The deficit is a domain effect and is flat in difficulty once the domain
  is fixed** — within CyberGym, < 2 h and ≥ 2 h tasks show the same deficit
  (cand 12: identical +0.0428 on both). An earlier pooled-slice reading
  ("difficulty amplifies by ~40 %") was a composition artifact — the easy
  slice mixed in CVEBench, which shows no deficit — and is retracted here.
- **Domain distance shows a dose–response:** CVEBench (modest shift,
  median ~67 min, programmatic validation) — no significant deficit for any
  winner, CIs excluding a CyberGym-sized effect; CyberGym (memory-safety
  PoC, the far shift) — full deficit at every difficulty.
- Combined with the in-distribution result (winners best on the hardest
  training bins), both frozen-axis comparisons now exist: freeze domain,
  vary difficulty → deficit unchanged (or absent); freeze the FST band,
  vary domain → the sign flips. Under the FST definition of difficulty the
  team's design uses, the failing axis is domain.
- Definitional caveat: under model-experienced difficulty (solve rate), the
  axes are inherently entangled — the same FST band being less solvable IS
  part of the domain shift — so the separation claim is stated under the
  FST definition only; the design doc's joint-shift framing remains the
  headline.

**Standing caveats:** one reserved test set; the two axes remain correlated
in this dataset by construction (the design doc's own resolution: the joint
shift is the deployment-relevant target distribution); one measurement day;
preliminary insights, not conclusions.

## Reproduce

- Runs: `uv run python -m forecaster_gepa.run --config configs/pilot_accept_joint.yaml --phase optimize` (then `--phase finalist`); same for `pilot_pareto_modelbin.yaml`.
- Sealed studies: `scripts/val_noise_study.py --config <arm yaml> --prompts-dir configs/noise_study_prompts_<arm> --repeats-val 0 --repeats-sealed 3 --tag <arm>`.
- Test: `scripts/val_noise_study.py --config configs/pilot_baseline_clean.yaml --prompts seed,july_cand12,clean_cand7 --repeats-val 0 --repeats-sealed 0 --repeats-test 2 --tag test_set`.
- Figures: `make_param_sweep_figs.py` (figs 12–16, pre-run screen), `make_fig17.py` (test result); data `sweep_report_data.json`.
