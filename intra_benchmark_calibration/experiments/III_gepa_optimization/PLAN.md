# Experiment III — GEPA prompt optimization

**Question:** can reflective prompt evolution (GEPA) find a forecasting prompt that
beats our best hand-built prompt on grand Brier, and does that gain transfer to
*harder* tasks and *unseen* benchmark domains?

**Seed prompt:** `single_call_analysis` (one API call instead of two). It is a hair
worse than `control` on grand Brier but statistically tied with the top cluster
(Exp I), and it is the cheapest / simplest object to optimize — one prompt, one
call. We optimize the seed; we are not bound to its current score.

---

## 0. Inputs (all already produced — $0 to reuse)

- **Scored prompt variants** — `forecasting_results/prompt_variant_comparison/`
  (10 conditions). Used by the two offline gate scripts below.
- **Scored model sweep** — `forecasting_results/forecaster_model_comparison/`.
- **Exp I** (`I_prompt_ablation_brier/`) — headroom between prompts (~0.018 Brier
  best-to-worst; top 4 tied).
- **Exp II** (`II_recalibration_decomposition/`) — the forecaster is already
  well-calibrated; most Brier loss is *resolution*, not *reliability*. So GEPA
  should chase sharper, better-discriminating forecasts, not just recalibration.

---

## 1. Gates — cheap checks to run BEFORE spending on GEPA

### Gate A — is there headroom? (done, Exp I)
Best vs worst hand-built prompt ≈ **0.018** Brier; the top 4 are statistically
tied. There is *some* headroom but it is small, so the proxy used to rank
candidates must be trustworthy at that scale → Gate B.

### Gate B — is a cheap cell-subsample a valid proxy for full Brier? (done)
`proxy_check.py` ($0, reuses the 10 scored prompts; paired subsampling — every
prompt scored on the SAME cells per draw, exactly like GEPA parent-vs-child).

Result (`artifacts/proxy_check.md`):

| Proxy N (cells) | pairwise decision acc, true gap ≥ 0.01 | mean Spearman (strict 10-way) |
|---|---|---|
| 50  | 82% | 0.54 |
| 100 | **91%** | 0.72 |
| 150 | 96% | 0.83 |

- The naive 50-cell proxy is **too noisy** for these tightly clustered prompts.
- **~100 cells** gives ≥90% accuracy on the GEPA-relevant unit (pairwise
  keep/kill on a real ≥0.01 gap). Pin the *absolute* Brier to ±0.01 needs ~250
  cells, but GEPA only needs *comparisons*, which cancel shared task noise.
- **Decision: proxy = 100–150 stratified cells per candidate**, not 50.

### Gate C — sanity: do smaller-N Briers preserve known conclusions? (done)
`subsample_recovery.py` confirms the model-sweep winner (GPT-5.5) is recovered
≥94% of the time by N=75, while near-tied prompts need much larger N — consistent
with Gate B.

---

## 2. Data split (fixed before any GEPA run)

Grand-Brier grid = 5 target bins × 5 target tasks/bin × 12 forecasted models = **300 cells**.

- **Train pool** — used by GEPA's proxy. ~100 tasks (20/bin) × 12 models = 1200
  cells; GEPA samples a fresh **100–150-cell** stratified minibatch per candidate
  evaluation from this pool.
- **Validation (full)** — the standard 300-cell grand-Brier grid, held out from
  GEPA. Only the surviving Pareto finalists are scored here.
- **Transfer tests** — held-out *harder tasks* and *unseen benchmark domains*
  (CVEBench + CyberGym) to test generalization, not memorization. Domain and
  difficulty are confounded in Lyptus, so transfer is reported jointly first, with
  pure-difficulty / pure-domain diagnostic slices run only if the main transfer
  shows a gain.

---

## 3. GEPA loop (high-level algorithm)

```
seed_prompt        = single_call_analysis
candidates         = [seed_prompt]
pareto             = {seed_prompt}                     # by proxy Brier
budget             = B rollouts                         # cost cap (see §5)

while budget not exhausted:
    parent  = sample_from(pareto)                       # favor Pareto frontier
    batch   = stratified_sample(train_pool, n=100-150)  # fresh proxy minibatch
    traces  = run(parent, batch)                        # forecasts + rationales
    reflection = LLM_reflect(parent, traces, errors)    # propose a targeted edit
    child   = apply_edit(parent, reflection)
    score_p = brier(parent, batch); score_c = brier(child, batch)  # SAME cells
    if score_c < score_p - eps:                         # coarse keep/kill (Gate B)
        candidates.append(child); update_pareto(child)
    budget -= len(batch)

finalists = pareto_frontier(candidates)
revalidate(finalists, full 300-cell grid)               # no proxy here
winner    = argmin grand_Brier(finalists)
report_transfer(winner, harder_tasks, unseen_domains)
```

Notes:
- Reflection is driven by the **resolution** gap from Exp II: prompt the reflector
  to make forecasts sharper / better-separated, not merely well-calibrated.
- `eps` ≈ proxy noise floor; only act on gaps the proxy can see (≥~0.01).
- Re-draw the proxy minibatch periodically to avoid overfitting to fixed cells.

---

## 4. Runs and when they fire

- **Run C (baseline, always):** GEPA on train pool → revalidate finalists on the
  full 300-cell grid. Decides: does GEPA beat the seed at all?
- **Run A (pure-difficulty transfer):** only if Run C shows a real gain. Test the
  winner on held-out *harder* tasks within the same domain.
- **Run B (pure-domain transfer):** only if Run C shows a real gain. Test the
  winner on held-out *unseen* benchmark domains (CVEBench + CyberGym).

A and B isolate the two axes that are confounded in the main (combined) transfer
test; run them only when there is a gain worth explaining.

---

## 5. Cost

- Proxy eval per candidate: ~100–150 cells × (1 call, `single_call_analysis`).
- Budget cap B set so total rollouts stay within target $ (recompute with the
  chosen proxy N — note 100–150, not the originally assumed 50).
- Full-grid revalidation: 300 cells × #finalists only.

---

## 6. Success criteria

1. **Beats seed on grand Brier** on the held-out 300-cell grid (not just proxy),
   by more than proxy/validation noise.
2. **Transfers**: non-negative (ideally positive) Brier delta on harder tasks
   and/or unseen domains. A prompt that only wins in-distribution is reported as
   such.
3. **No calibration regression**: reliability term (Exp II decomposition) does not
   worsen while resolution improves.

---

## Reproduce the gates

```bash
# Gate B — proxy validity (writes artifacts/proxy_check.{md,png})
python3 intra_benchmark_calibration/experiments/III_gepa_optimization/proxy_check.py

# Gate C — subsample recovery (writes artifacts/subsample_recovery.md + hists)
python3 intra_benchmark_calibration/experiments/III_gepa_optimization/subsample_recovery.py
```
