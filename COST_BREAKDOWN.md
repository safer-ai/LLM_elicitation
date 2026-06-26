# SPAR Funding Request — Forecaster Prompt-Optimization

**Model assumption: optimization runs on Sonnet 4.6** (~$8 / 300-forecast eval), with pilots on Haiku 4.5 (~$3) and final validation on GPT-5.5 (~$15, our best performer).
_If we optimize directly on GPT-5.5 instead, costs roughly double for items 1–4._

| # | Experiment | Forecaster model | What it does | Cost |
|---|-----------|-----------------|--------------|------|
| 0 | Pipeline setup | — | Code only: train/test split + prompt-variant harness + Brier scoring | $0 |
| 1 | Prompt-component ablation | Haiku 4.5 pilot → Sonnet 4.6 full | Turn each prompt piece on/off, measure accuracy (Brier) | ~$150 |
| 2 | Minimal-prompt test | Sonnet 4.6 | Strip prompt to "5 examples → predict the 6th" | folded into #1 |
| 3 | Automated optimization (GEPA) | Sonnet 4.6 | LLM reads wrong forecasts and rewrites the prompt to fix them | ~$150 |
| 4 | Optimization on a 2nd model | GPT-5 | Repeat #3 on another forecaster model | ~$130 |
| 5 | Held-out validation | Sonnet 4.6 + GPT-5.5 | Score the optimized prompt on unseen tasks | ~$60 |
| 6 | Contamination check | GPT-5.5, Sonnet 4.6 | Test if models already "know" benchmark outcomes | ~$10 |
| 7 | Cross-model transfer | Opus 4.7, GPT-5.5 | Run the tuned prompt on our best models | ~$60 |
| 8 | Extended-thinking re-runs | Sonnet 4.6 + GPT-5.5 | Re-run best prompts with deep-reasoning mode on | ~$130 |
| | **Subtotal** | | | **~$690** |
| | Buffer (failed runs, reruns) ~30% | | | ~$210 |
| | **Total request** | | | **~$900** |

---

## Notes & assumptions

**Goal:** lower the forecaster's **Brier score** (forecast accuracy vs. real model
pass/fail outcomes on the Lyptus cyber-benchmark), then prove the improvement
generalizes to unseen tasks and other models.

**Cost basis:** 1 forecast = 2 API calls ≈ 4,600 input + 900 output tokens
(measured from our existing model-sweep run). One evaluation pass = 300 forecasts.
Dev/optimization done on Sonnet 4.6; pilots on Haiku 4.5 to keep exploration cheap.

**Per-300-forecast evaluation cost by model:** Haiku 4.5 ~$2.7 · GPT-5 ~$4.4 ·
Sonnet 4.6 ~$8.2 · GPT-5.5 ~$15 · Sonnet 4.6 with deep reasoning ~$44.
(Prices per 1M tokens, June 2026: Haiku $1/$5, GPT-5 $1.25/$10, Sonnet $3/$15, GPT-5.5 $5/$30.)

**Minimum-viable ($220):** items 0, 1, 3, 5 only (no 2nd model, no transfer, no thinking).

**Glossary:** *Brier score* = forecast accuracy, lower is better (0.25 = guessing,
0 = perfect). *GEPA* = published method (arXiv:2507.19457) where an LLM rewrites
prompts from failure traces. *Forecaster* = the LLM predicting; *evaluated model* =
the LLM whose pass/fail we predict (12 in Lyptus).
