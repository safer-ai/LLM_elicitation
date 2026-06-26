# Is the forecaster's error fixable by recalibration? — control run

`intra_benchmark_calibration/experiments/I_prompt_ablation_brier/results/control/20260614_021706/plots/scored_with_crps.csv` · N = 300 cells · base rate = 0.53

## TL;DR

The forecaster is **already well-calibrated** (ECE = 0.064), so post-hoc recalibration removes essentially nothing from the Brier (+0.7%). Its error is dominated by *irreducible task difficulty*, not a fixable calibration offset. **Implication for GEPA: there is no free calibration win to bank — the only way to improve is to make the forecaster genuinely better at separating solvable from unsolvable tasks (raise "slope"/resolution), which is a harder bar.**

---

## 1. Does recalibration help? (the test)

Recalibration = fit a correction curve on (forecast, outcome) pairs and rewrite the forecasts. Fit with 5-fold cross-validation so the number isn't cheating.

| Forecasts | Brier | change |
|---|---|---|
| raw p50 | **0.1377** | — |
| Platt (logistic) recal | 0.1407 | +0.0030 |
| isotonic recal | 0.1367 | -0.0010 |

Recalibration moves the Brier by < 0.003 (Platt makes it slightly *worse*). **No meaningful free lunch.**

## 2. Why? — splitting the Brier into calibration vs discrimination

Two standard decompositions of the *same* Brier number (Murphy = ML language, Yates = plain language). Neither is a calibration step — they just show where the error lives.

**Murphy:** `Brier = reliability − resolution + uncertainty`

| term | value | meaning | can recalibration fix it? |
|---|---|---|---|
| reliability | 0.0082 | calibration error | yes — but it's already tiny |
| resolution | 0.1156 | genuine discrimination | no |
| uncertainty | 0.2491 | irreducible (base rate) | no |

**Yates:** bias / slope / scatter

| term | value | meaning |
|---|---|---|
| bias | -0.0017 | systematic over/under-forecast → ~0, nothing to correct |
| slope | 0.4560 | gap between forecasts on solved vs unsolved tasks (discrimination) |
| scatter | 0.0639 | within-outcome noise |

Reliability (0.008) and bias (≈0) are both tiny → the forecaster's numbers are already the right magnitude, which is exactly why recalibration can't help. The score is mostly irreducible uncertainty (0.25) plus the resolution the forecaster already earns.

## 3. What this means for prompt optimization / GEPA

- Report future Brier **against 0.1367** (post-recalibration), not 0.1377 — though here they're nearly identical.
- The metric to beat is **slope = 0.456** (equivalently resolution = 0.116). A prompt only genuinely helps if it raises this.
- GEPA cannot win here by fixing calibration (already fixed); it must extract *more discrimination signal*. Whether such signal exists is the open question.

> Caveats: ECE/decomposition use 10 equal-width bins (the Murphy reconstruction sits ~0.004 above the exact Brier — a standard binning artifact; relative term sizes are what matter). Recalibration is a 5-fold CV estimate on one run; it will be redone on the real train/test benchmark split.
