# GEPA measurement study — verified results (2026-08-03 → 2026-08-15)

Follow-up to `ladder_2026-08-02.md`. The ladder ended with "the val
measurement is the binding noise source"; this study measures that claim
directly instead of inferring it from accidental draws, re-checks the
ladder's one positive result, and validates the instrument with a sabotage
control. Runs: local (2026-08-03 seed re-measurement, 2026-08-15 temp-0
confirmation) + one cloud session (2026-08-10; container restarts on one
arm — stitched from complete 230-cell blocks only, see
`noise_provenance_sealed_check.md`). Every number below independently
re-verified from per-cell data. Raw data: branch `results/priority-2026-08-10`
of `actionproject-madhav/gepa`. Cost ≈ $55 total.

## 1. Self-consistency of a single val pass (the noise floor)

Seed prompt, 5 fresh passes over the same 84 val cells, same settings, same
hour, zero parse failures (`noise_summary_seed_local_5x.txt`):

    0.1027  0.1021  0.0956  0.1041  0.1137   →  sd 0.0065, range 0.0181

A single val pass therefore cannot resolve the ~0.01–0.02 effects being
optimized. This also retro-explains the ladder: best-so-far curves are
running minimums over such draws (they descend under zero real improvement),
and the July "0.1144 vs 0.0995 seed swing" additionally contained a parse
failure — the July seed pass had 1 unparseable cell scored Brier 1.0
(83 parsed cells averaged 0.1038; one failure adds +0.0106 to an 84-cell mean).

## 2. Sabotage control (instrument validity)

A deliberately broken prompt (seed + "always report p50 = 0.99") measured
3× on val (`noise_summary_control.txt`): 0.1334 / **0.1116** / 0.1561.
One of three passes landed inside the seed's own re-measurement range —
**a single val pass cannot reliably detect deliberate sabotage.** At the
multi-pass level the instrument is valid (control mean 0.134 ≫ seed 0.104).

## 3. Candidate 20's win replicates

Three fresh paired sealed passes (seed and cand 20 measured in the same
repeat, `noise_summary_sealed_check.txt` + provenance): cand 20 better in
**all three** — paired edges **+0.0153 / +0.0201 / +0.0162**. With bonus
blocks from the interrupted cloud attempts, 8 total cand-20 sealed passes
vs 4 seed passes: ≈ 0.116 vs ≈ 0.131. The ladder's +0.017 finding is solid.
Sealed-set re-measurement noise: seed worst gap across 3 passes = 0.0029 —
~6× tighter than a val pass (230 cells > 84, consistent with √n scaling).

## 4. Temperature 0: not deterministic, but the best regime found

- Repeated temp-0 passes still differ (seed val sd 0.0048 across 3 passes;
  Anthropic serving nondeterminism) — **temp 0 does not make measurement
  free**, multi-pass evals remain necessary.
- But parse failures nearly vanish at temp 0, and scores improve slightly.
- **Candidate 12 at temperature 0** (`noise_summary_temp0.txt` +
  `noise_summary_temp0_confirm.txt`): sealed, 5 paired passes vs seed —
  edges +0.0204 / +0.0304 / +0.0258 / +0.0289 / +0.0256, pooled
  **cand12 0.1045 ± 0.0031 vs seed 0.1307 ± 0.0010 → +0.026,
  distributions non-overlapping. The project's best validated result.**
  (At temperature 1.0, cand 12's ~2% parse-failure rate — each failure
  scored 1.0 — had erased this edge; parsed-only re-analysis of the ladder
  data had already hinted at it: cands 12/15 have real skill edges of
  +0.027/+0.014 on parsed cells.)

## 5. Bootstrap vs true rerun variance (Jeff's question)

Bootstrapping cells within one pass **over**-estimates rerun variance:
bootstrap sd 0.0159 (val) / 0.0123 (sealed) vs true measured rerun sd 0.0065 /
0.0015–0.002 — 2.5×/7× too wide, because resampling cells mixes in
which-cells variance that does not vary between reruns of the same set.

## Conclusions

1. **Two validated prompt improvements exist**: cand 20 (+0.017, temp 1.0,
   replicated ×3) and cand 12 @ temp 0 (+0.026, 5 paired passes). Both
   texts: `july_cand20_prompt.txt`, `july_cand12_prompt.txt`.
2. **The instrument rules**: single val passes are unusable at these effect
   sizes (can't even detect sabotage); multi-pass sealed evals (rerun sd
   ~0.002) are the reliable comparison and should end every future run.
3. **Two pipeline choices amplified noise** and should be fixed before more
   optimization: parse failures scored as Brier 1.0 (retry-once patch
   recommended) and temperature 1.0 for evaluation passes.
4. **Recommended next step** (pending group decision): spend the reserved
   CVEBench+CyberGym test set — once, as designed — on cand 12 @ temp 0 vs
   the seed, ≥2 passes each (~$100). Then Jakub's replicate plan (4+4+
   extension) if still wanted: true cost ≈ $1,000–1,100 with multi-pass
   sealed checks included.

## Figures

| File | Shows |
| --- | --- |
| `fig6_winner_replication.png` | **The headline figure**: seed vs each winning prompt across every paired re-measurement pass — the winners are below the seed every single time |
| `fig3_improvements_vs_noise.png` | The two validated improvements vs the instrument's own re-measurement noise, by set |
| `fig5_best_val_trajectory.png` | Best-so-far val curves of all three runs inside the seed's empirical re-measurement band |
