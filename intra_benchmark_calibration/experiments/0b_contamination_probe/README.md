# Experiment 0b — Benchmark Contamination Probe

**Purpose:** Check whether the forecaster model (Sonnet 4.6) has memorized ground-truth
solve rates for specific (forecasted-model, task) pairs from the Lyptus benchmark. If it
has, the Brier scores in Experiment I may be inflated by recall rather than genuine
reasoning from the provided source tasks.

---

## Motivation

Experiment I (Level 0 + Level 1 ablations) showed:

- `control` (full prompt) Brier = 0.137
- `no_source_context` (no capability profile at all) Brier = 0.155

The fact that *removing all source context hurts* is already weak evidence against strong
memorization — a model recalling answers from memory would perform *better* without the
distracting (possibly mismatching) source tasks in the prompt, not worse. But "weak
evidence" is not a validity proof. This experiment quantifies it directly.

---

## Sampling design — the EXACT Experiment I cells (not random tasks)

This is the key design point. Experiment I's 300 cells are **25 distinct target tasks
(5 per difficulty bin) × 12 forecasted models**. The same 25 tasks are reused across all
12 models (`build_cell_plans` computes the per-bin target set once). The probe loads
these *exact* `(target_task_id, forecasted_model, true_outcome)` triples straight from the
Experiment I `control` run CSV — so `Brier(recall)` is computed on the **same 300 pairs**
that Experiment I scored, and is directly comparable to the control Brier of 0.137.

(The script auto-discovers the latest control run; override with `--cells-csv`.)

## Two probe types

### Probe A — Direct numeric recall
Ask Sonnet 4.6 directly: *"What is the pass rate of [model] on [task]?"*

- One call per **exact Experiment I (task, model) pair** = **300 Probe A calls**
- Parse the stated pass rate; compare to the ground-truth binary outcome
- Key metric: **Brier(recall)** — if it ≈ the Experiment I Brier (0.137), the model may
  be recalling rather than reasoning. The analysis also computes the **matched** Exp I
  Brier on exactly the pairs the model answered numerically (apples-to-apples).
- Also report: Spearman rho between stated rate and true outcome; binary accuracy;
  by-model, by-family, and by-confidence breakdowns; and the stated-"unknown" rate.

### Probe B — Task recognition
Ask Sonnet 4.6: *"Have you seen this task in your training data?"*

- One call per **distinct target task** (not per model) = **25 Probe B calls**
- Count recognition rates by task family and difficulty
- High recognition rate ≠ outcome memorization, but it is a prerequisite for it

**Total: 325 calls** — cheap (simple prompts, ~500 input + 150 output tokens each).
Estimated cost at Sonnet no-thinking: **~$1–2**.

---

## What we expect

| Scenario | Probe A Brier | Spearman rho | Interpretation |
|---|---|---|---|
| No memorization | ~0.25 (chance) | ~0 | forecaster is guessing; Exp I signal is genuine |
| Partial recall | 0.15–0.20 | 0.1–0.3 | weak contamination, minor inflation |
| Strong contamination | ≤0.14 (≈ Exp I) | ≥0.4 | Exp I accuracy is largely recall-driven |

The prior expectation is "no memorization": Lyptus is a recent private evaluation dataset
(not a public leaderboard), and `no_source_context` already being *worse* than `control`
points away from recall. The probe makes this rigorous.

---

## Run

```bash
# From repo root
python intra_benchmark_calibration/experiments/0b_contamination_probe/run_probe.py \
  --lyptus-dir ~/lyptus-data \
  --output-dir intra_benchmark_calibration/experiments/0b_contamination_probe/results \
  [--forecaster claude-sonnet-4-6]
```

API keys resolved from `intra_benchmark_calibration/experiments/.env` → repo-root `.env`
→ environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).

## Analyse

```bash
python intra_benchmark_calibration/experiments/0b_contamination_probe/analyse_probe.py \
  --results-dir intra_benchmark_calibration/experiments/0b_contamination_probe/results
```

Writes `results/contamination_report.md` and `results/probe_a_rows.csv`.

---

## Results (Sonnet 4.6, run 20260616_145931, 300 Probe-A + 25 Probe-B calls)

### Probe A — direct numeric recall: **NO contamination**
**All 300/300 (task, model) pairs → `pass_rate: unknown`, `confidence: none`.** Not a
single pair produced a numeric recall. The model explicitly and consistently states it has
no information about the "Lyptus Cyber Task Horizons" benchmark or any model's pass rate on
these tasks. Representative response:

> `pass_rate: unknown` / `confidence: none` / "I have no information in my training data
> about a benchmark called 'Lyptus Cyber Task Horizons' or the specific task 'arvo:59243'
> with results for Claude 3 Opus."

→ **The forecaster cannot recall outcomes. Experiment I's Brier (0.137) is therefore NOT
recall-driven — it is genuine reasoning from the provided source tasks.** This is the
decisive validity result.

### Probe B — task recognition: tasks partially public, but that's harmless
Of the 25 distinct target tasks: **11 (44%) recognized, 12 (48%) unsure, 2 (8%) not**.

| Family | Recognized (yes) |
|---|---|
| cybench, nyuctf, intercode_ctf, cvebench, nl2bash | ~100% |
| cybergym | 0% (all "unsure") |
| cybashbench | 17% yes, rest unsure/no |

The publicly-sourced CTF/CVE families (cybench, nyuctf, intercode, cvebench) are recognized
as *task content* — unsurprising, they come from public competitions. But recognizing the
task text is **not** memorizing the solve outcomes (Probe A rules that out), and the task
text is the *legitimate* forecasting signal anyway (it's literally shown in the prompt).
The synthetic/recent families (cybergym arvo tasks, cybashbench) are largely unknown.

### Overall conclusion → **validity gate PASSED**
The two probes together give the cleanest possible answer:
- The model knows *what some tasks are* (public CTF problems) ...
- ... but has **zero recall of the per-model pass/fail outcomes** that constitute the
  ground truth.

Combined with the Experiment I finding that `no_source_context` is the *worst* condition
(removing the evidence hurts — the opposite of what a memorizing model would show), there
is **no evidence of outcome contamination**. The forecaster's accuracy is earned by
reasoning over the provided source-task evidence, not by recalling answers.
