# Poster Context: LLM Forecasters for Quantitative Cyber Risk Modelling
## SPAR Spring 2026 — Demo Day

This file consolidates everything read from the poster folder. It is the canonical reference for making the poster. Do not assume anything not stated here.

---

## 1. Project Identity

**Title:** Measuring LLM Forecasters for Quantitative Risk Modelling of Cyber Misuse

**Authors:** Jeff T. Mohl · Madhav Khanal · Jakub Kryś · Matt Smith

**Affiliation:** SaferAI · Supervised Program for Alignment Research, Spring 2026

---

## 2. Core Problem & Motivation

SaferAI has developed **9 cybersecurity risk models** that map AI benchmark scores to forecasts of cyberattack capability uplift from frontier models (Barrett et al. 2025, arXiv:2512.08864). These models use the MITRE ATT&CK framework to quantify how much a frontier AI model raises the probability of an adversary successfully completing a given attack step.

The mapping between benchmark performance and risk forecast has historically been built by **human expert elicitation** — slow, expensive, and hard to scale. The goal of this project is to test whether **automated LLM "expert" forecasters** can replace humans, and to develop a methodology for evaluating and improving them.

**Three project objectives:**
1. Develop a metric capturing desirable characteristics of elicited forecasts.
2. Use that metric to investigate the impact of experimental manipulations on the forecasters.
3. Develop a scalable **ground-truth accuracy measure** that does not rely on scarce human-labeled data.

Objectives 1 and 2 were substantially completed by midterm (Weeks 1–8). Objective 3 — the intra-benchmark calibration pipeline — was designed and executed in Weeks 8–13.

---

## 3. File-by-File Summary (what each file actually contains)

| File | What it actually contains |
|---|---|
| `mid_term_report_forecasting_llm.pdf` | Full report of Weeks 1–8: forecast elicitation methodology, Wasserstein consistency metrics, self-consistency results, elicitation invariance, Fréchet ANOVA (model vs persona), prompt sensitivity, baseline anchoring, challenges/next steps |
| `latest_reports.md` | **Week 8** (Jeff): exploration and design plan for using the Lyptus cyber-time-horizons dataset as intra-benchmark ground truth; **+ Week 9** (Madhav): repeated content identical to week9.md |
| `week9.md` | **Week 9** (Madhav): design debates (per-LLM vs averaged ground truth, single task variance, K=3 idea) + metric analysis (CRPS recommendation, Interval Score, Brier, log score, MAE); **+ Week 10** (Madhav): repeat/refinement of the same CRPS writeup; link to Lyptus prompt methodology |
| `week 11 and 12.md` | **Week 11** (Madhav): intra-benchmark experimental conditions A/B/C/E/F, full results table, closest-bin sanity check table, key findings; **Week 12** (Madhav): per-expert breakdown tables, model sweep mention, ground truth visualization clarification, target task variance analysis |
| `week13.md` | **Model sweep results table only** (Madhav): 5 models, N valid, Brier, CRPS; note about GPT-5 refusals |
| `discussion.md` | **May 12, 2026 meeting notes** (Jakub, Jeff, Madhav): clarification of K interpretation; formal A/B/C experiment typology (DIFFERENT from conditions A–F); non-stationary problem; single LLM vs generic frontier LLM debate; action items |
| `intra_benchmark_notes.md` | Technical spec (Jakub/Jeff): dataset details, elicitation prompt shape, scoring options, ground-truth derivation, outer loop structure, locked decisions, open items |
| `metrics.md` | Standalone writeup (Madhav): detailed analysis of CRPS, Brier, Interval Score, Log Score, MAE; baselines; recommendation |
| `project_plan.md` | **Empty** (1 line, no usable content) |
| `SPAR_LLM_Forecasting_poster (2).pdf` | Template: 4-column landscape layout with header strip |
| `sample poster winner of previous year.pdf` | Stuxbench poster (SPAR S25 winner): dense, bold emphasis, large flow diagram, bar chart for results |
| `sample poster second prize.pdf` | Deception Probes poster (SPAR S25 second prize): colored section boxes, SPAR logo top-left, 3-column layout |

---

## 4. Full Methodology

### 4.1 Forecast Elicitation — MITRE Phase (Weeks 1–8)

Forecasters estimate **P(success on a MITRE ATT&CK step | observed benchmark performance)**, following the approach in Barrett et al. 2025. Benchmarks used: Cybench and BountyBench.

Forecasters return the **25th, 50th (modal), and 75th percentiles** of their probability estimate. These are fit to a **Beta(α, β) distribution** via least-squares (rejecting fits where |CDF(q) − q| > 0.05).

Key prompt components experimentally manipulated:
- **Persona descriptions:** 10 custom expert personas.
- **Baseline risk estimates:** P(success) + CI from human experts for each MITRE step.
- **Reasoning prompts:** multi-phase chain guiding capability analysis then probability estimation.

### 4.2 Consistency Metrics (Weeks 1–8)

**Wasserstein distances W1 and W2** (Panaretos & Zemel 2019):
- W1 (earthmover's distance): average magnitude of probability mass shift between two distributions.
- W2: more sensitive to large deviations.
- 95% CIs via bootstrapping.
- Two computation modes: **linear interpolation** on quantiles (default), or **Beta-fit CDF** numeric integration (used when number of elicited points differs across conditions).
- Supplementary statistics: modal discrepancy, IQR discrepancy.

**Fréchet ANOVA** (Dubey & Müller 2019): attributes variance to factors (model, persona, temperature, prompt) using ICC_F = between-group Fréchet variance / total Fréchet variance.

### 4.3 Intra-Benchmark Pipeline — Ground Truth Phase (Weeks 8–13)

**Conceived in Week 8 (Jeff):** Jeff identified the Lyptus Offensive Cyber Time Horizons dataset as a clean source for objective ground truth — it establishes difficulty ranking via human time-to-completion, analogous to what was done in the inter-benchmark experiment but using a single unified dataset.

Jeff's week 8 proposal:
- Provide anchoring context: representative tasks from each time bin + pass rate for that difficulty tier
- Select test task from a target difficulty bin; elicit expected pass rate
- Compare elicited pass rates to actual pass rates
- The dataset doesn't extend to new model releases automatically (would need to run those models on the task set) — an alternative using overall benchmark-level pass rates from model cards is noted as a design choice

**Dataset details (from intra_benchmark_notes.md):**
- `model_runs.parquet`: per (agent, task_id) binary pass/fail. 15 models across 630 tasks, 8,963 rows, 100% non-null.
- `task_difficulties.parquet`: `best_available_minutes` and `best_available_source` per task.
- `runs.parquet`: pre-joined convenience table.
- Filter to **291 "headline" tasks** (best_available_source in {completion, first_blood, expert_estimate}).
- Difficulty ordering: `best_available_minutes` (hybrid of actual completion time, first-blood time, expert estimate). Do NOT use LLM time estimates (circular) or empirical solve rates (tautological).
- **5 difficulty bins** using Lyptus native time bins.
- 12 models used in analysis (after filtering from 15 in the raw data).

**Elicitation prompt shape (Jeff's primary plan):**
```
Bin 1 (0-4 min): model passed 90% of tasks in this bin.
  Representative tasks: A, B, C (descriptions only, no pass/fail)

Bin 2 (4-16 min): model passed 65%.
  Representative tasks: D, E, F

Bin 3 (16-64 min): model passed 40%.
  Representative tasks: G, H, I

Bin 4 (64-256 min): model passed 15%.
  Representative tasks: J, K, L

Target (Bin 5, 256+ min):
  Task Z: <description>

What is P(model passes Task Z)? Give 25/50/75 percentiles.
```

Source bins: show aggregate pass rate + 3 representative task descriptions (no individual pass/fail labels). Target bin: show 1 task description only. No pass rate for the target bin in the prompt.

**K (n_target_tasks_per_cell) — confirmed interpretation:**
From discussion.md (May 12): K=1 means one target task per elicitation, one CSV row per elicitation. The correct interpretation is **Madhav's**: K separate elicitations, each with one target task. Jeff's interpretation (all K tasks in a single prompt) was also discussed. Interpretation (3) — K tasks as examples — does NOT apply to target bin; it does apply to source bins via `n_examples_per_source_bin`.

**Ground truth derivation (per intra_benchmark_notes.md):**
```
ground_truth(M, j) = (headline tasks in bin j solved by M) / (headline tasks in bin j)
```
Per model, per bin. No averaging across models. 15 models × 5 bins = 75 ground-truth cells. With K=1, each experimental condition produces 75 elicited predictions.

**Two scoring approaches** (same elicitation, computed together for free):
1. **Primary:** compare Beta mean to model's bin-level pass rate on the target bin. Assumes the target task is representative of the bin.
2. **Secondary:** compare Beta mean to 0 or 1 (did the model actually pass Task Z?). Drops representativeness assumption; noisier per-datapoint.
3. Aggregate variant (NOT chosen for MVP): show several target tasks, ask for aggregate pass rate, compare to bin pass rate. Different elicitation, deferred.

**Variance-scaled Brier is explicitly NOT a proper scoring rule:** wider CIs artificially improve the score. Not used as primary metric.

---

## 5. Scoring Metrics

Primary metric: **CRPS**. Secondary: Brier (on Beta mean), ECE, Spearman ρ, Kendall τ, MAE, Bias.

### CRPS (recommended primary metric)
Integrates Brier score over all thresholds:
```
CRPS(F, o) = ∫(F(y) − 1{y ≥ o})² dy
           = E_F|X − o| − (1/2) E_F|X − X'|
```
- Uses the full Beta distribution (not just median).
- Strictly proper: LLM's best strategy is to report true belief.
- Robust: unlike log score, doesn't blow up when outcome is in low-probability region.
- Reduces to MAE when forecast is a point mass.
- Units = probability points (CRPS of 0.05 means ~5 pp from truth).
- Computational note: no trivial closed form for arbitrary distributions, but Beta can be computed exactly or via 10k-sample Monte Carlo.
- Source: Gneiting & Raftery 2007.

### Brier Score
(Beta mean − observed)². Discards distributional information. "Median 50% tight" and "median 50% wide" get the same score.

### ECE (Expected Calibration Error)
Measures calibration of interval widths. Condition C had best ECE despite weaker pointwise accuracy.

### Interval Score
Evaluates [L, U] interval: IS_α = (U−L) + (2/α)(L−o)·1{o<L} + (2/α)(o−U)·1{o>U}. Proper but uses only 2 of 3 elicited percentiles (loses skewness info from median).

### Log Score
log f(o) where f is Beta PDF. Strictly proper but fragile near 0/1. Can blow up if outcome is in low-density region.

### MAE
|median(Beta) − o|. Discards all distributional information. Not a proper scoring rule for distributional forecasts.

### Baselines
- **Uniform Beta(1,1):** maximally uncertain. Brier ≈ 0.25, CRPS ≈ 0.33. Any method worse than this is subtracting information.
- **Empirical baseline:** use source bin pass rate as point estimate with no LLM involved. No-elicitation benchmark.

---

## 6. Results

### 6.1 Midterm Results (Weeks 1–8, from mid_term_report_forecasting_llm.pdf)

#### Baseline Self-Consistency
Models tested: Claude Sonnet 4.6, GPT-5 mini. 3 MITRE steps (Disc, Exec, IA) from single cyberattack scenario. Single expert persona. 20 runs per step → 190 pairs.

- **Claude Sonnet 4.6:** W1 ≈ 0.005 (Disc/Exec), ≈ 0.010 (IA). Very consistent.
- **GPT-5 mini:** W1 ≈ 0.025–0.030 range. Higher but still solid.
- Reference: W1 of 0.05 = probability mass shifted an average of 5% across runs.
- Conclusion: not a large degree of inherent stochasticity in either model.

#### Elicitation Invariance
5 variations tested: negation (P(failure) instead of success), deciles (10%–90%), pseudo-random quantiles (15%, 60%, 77%), and two others varying point count/location. W1 computed after Beta fitting to handle variable point counts.

- Cross-approach W1 comparable to within-approach variance for most conditions.
- Models are consistent across diverse elicitation formats.
- One exception: eliciting 5%/50%/95% points in Claude Sonnet 4.6 showed meaningful divergence across task steps.

#### Model Identity vs. Expert Persona (Fréchet ANOVA)
Three models: Claude Sonnet 4.5, GPT-4o, Gemini 2.5 Pro. 10 personas, 10 runs/persona. Three attack steps: TA0002 (50% baseline), TA0007 (85% baseline), T1657 (30% baseline).

- **Within-model persona effects:** Claude and GPT-4o — no significant effects (ICC_F 5.7–13.9%, p>0.05). Gemini — two significant effects (ICC_F 18.3–24.9%, p<0.05) but still smaller than cross-model.
- **Cross-model: model identity explains 48–69% of distributional variance (p<0.001), 3.5–12× more than persona.**
- Integer quantity estimates (# actors): model ICC_F=24% vs persona ICC_F=9%.
- Figure 4 in report: persona distributions for Claude Sonnet 4.5 overlap tightly within each attack step.
- Figure 5 in report: cross-model distributions show clear separation.

#### Prompt Sensitivity (7 conditions, Claude Sonnet 4.6)
Control prompt ≈ 615 words. W1 from control, and within-condition Fréchet variance.

| Condition | W1 from control | Within-cond. variance (× control) |
|---|---|---|
| No CI from baseline | 0.013 | 0.4× |
| **No baseline** | **0.154** | **18.1×** |
| **No baseline + no CI** | **0.239** | **18.1×** |
| Skip analysis section | 0.041 | 12.3× |
| Trim reasoning scaffold | 0.011 | 0.3× |
| Trim all (100-word prompt) | 0.016 | 0.07× |

**Key: removing the baseline value is by far the largest single manipulation (W1=0.154). Trimming all reasoning/analysis is negligible (W1=0.016, lower within-variance than control).**

#### Baseline Anchoring
Setup: Claude Sonnet 4.6, single persona (AI/ML Security Researcher), TA0002 (Execution step), imaginary benchmark. Swept baseline probability 10%→90% (10% increments, 10 runs per point). Baseline value and CI removed from prompt to isolate the anchoring effect.

- Elicited median follows a **near-linear trend with the provided baseline** (ceiling effect attenuates uplift near 100%).
- For unbounded quantity estimates (# actors), the linear relationship persists without ceiling effect.
- Pattern confirmed on at least one other attack step (Initial Access).
- Conclusion: strong baseline anchoring for both bounded and unbounded estimates.

### 6.2 Week 8 — Intra-Benchmark Pipeline Design (Jeff, from latest_reports.md)

Jeff proposed using the Lyptus Offensive Cyber Time Horizons paper dataset as ground truth because:
- Establishes objective difficulty ranking via human time-to-completion.
- 291 headline tasks with human-validated difficulty estimates.
- Prompts provided in a usable format.
- Per (model × task) pass/fail values available for 15 models.

Jeff's framing of the goal: *"eliciting p(success on task of unknown difficulty | performance on tasks of known difficulty)"* — somewhat different from P(MITRE step | benchmark performance), but a useful testbed.

Design concerns Jeff raised:
- Dataset doesn't extend to new model releases automatically (need to run the model on this task set).
- Alternative: elicit using overall benchmark-level pass rate (could be taken from model card/leaderboard).
- Need to ensure representative tasks are selected (tasks with 0% pass rate in an easy bin would be misleading context).

Jeff's original plan:
- 5× difficulty bins as target bins × 15 models = 75 datapoints.
- Y axis: estimated pass rate (model × bin); X axis: actual pass rate for that model in that bin.
- Sensitivity tests: does selection of context or target tasks matter? Does changing pass rates impact elicited probabilities?

### 6.3 Weeks 9–10 — Metric and Design Debates (Madhav, from week9.md)

**Per-LLM vs. averaged ground truth debate:**
- Single target task for a specific model resembles P(MITRE | benchmark performance) most closely.
- But single task is very noisy; K=3 might reduce within-bin variance.
- Averaging ground truth across many LLMs including very different generations (GPT-2 vs Opus 4.7) defeats the purpose.
- Averaging across same-generation frontier models may be acceptable.

**CRPS recommended** over Brier/log score/MAE for distributional forecasts. See Section 5 above for full reasoning.

### 6.4 Weeks 11–12 — Intra-Benchmark Experimental Results (Madhav, from week 11 and 12.md)

**Setup for all conditions:** claude-sonnet-4-6, 2 experts, 1 Delphi round, 5 bins, K=1 target task per cell. Metrics from `analyse_results.py` (groups by condition_id, keeps final Delphi round per cell).

**Who ran which condition:**
- Conditions A, B, C: run by Jakub.
- Condition E: Madhav's pilot run.
- Condition F: partial, appears to be a separate run.

**Condition descriptions:**

**A — all_except_target, thinking OFF:**
- Source context: all 4 other bins (4 bins × 3 tasks = 12 source tasks shown).
- Extended thinking disabled (temperature=0.8 per config filename).
- 5 cells per model × 12 models = 60 total cells, 240 API calls.
- Forecaster sees full benchmark capability profile.

**B — all_except_target, thinking ON:**
- Identical to A but extended thinking enabled (budget=10k tokens).
- 60 total cells, 240 API calls.
- Tests whether chain-of-thought reasoning improves calibration.

**C — single_bin, all 20 pairs, thinking ON:**
- For each (source bin i, target bin j) pair where i≠j, show only bin i as context.
- All 20 ordered pairs (5×4): 20 pairs × 12 models = 240 cells, 960 API calls.
- Forecaster sees a focused, single-bin profile.
- Much more expensive; tests every possible source→target direction.

**E — closest_bin, 5 pairs only, thinking ON (Madhav's pilot):**
- For each target bin j, show only the single source bin whose mean pass rate is closest to bin j's.
- Gives exactly 5 (i,j) pairs (one per target bin), same cell count as A/B.
- Key idea: showing a closely matched reference bin = best analogical anchor. Aligns with earlier findings that too much context can confuse LLMs.
- 120 API calls (E ran 4× faster than C's 480 API calls per Delphi round).

**F — full loop on 20 source/target pairs, K=3 (partial):**
- Same as C in source/target structure but K=3 target tasks per cell.
- Partial run: N=1389 analyzed (not complete).

**Main results table:**

| Metric | A — all_bins, no think | B — all_bins, think | C — single_bin, K=1 | **E — closest_bin, K=1** | F — full loop, K=3 (partial) |
|---|---|---|---|---|---|
| N analyzed | 60 | 60 | 240 | **60** | 1389 |
| **Brier ↓** | 0.2336 | 0.2255 | 0.1736 | **0.1030** | 0.1925 |
| **CRPS ↓** | 0.3113 | 0.3014 | 0.2601 | **0.1908** | 0.2821 |
| ECE ↓ | 0.1732 | 0.1755 | **0.0888** | 0.1428 | **0.0701** |
| **Spearman ρ ↑** | 0.31 | 0.35 | 0.54 | **0.77** | 0.49 |
| **Kendall τ ↑** | 0.26 | 0.29 | 0.45 | **0.64** | 0.41 |
| MAE ↓ | 0.3892 | 0.3765 | 0.3334 | **0.2555** | 0.3587 |
| Bias | −0.074 | −0.075 | −0.032 | **+0.010** | +0.058 |

**Key findings from week 11:**
- E dominates A, B, C on Brier, CRPS, MAE, Spearman ρ, Kendall τ, and Bias.
- C still has the best ECE (0.0888 vs 0.1428 for E): the full 20-pair loop produces better-calibrated interval widths, but E is far more accurate pointwise.
- E ran **4× faster than C (120 vs 480 API calls)** while achieving nearly 2× lower Brier (0.1030 vs 0.1736).
- Bias is essentially zero for E (+0.010), vs systematic underestimation in A, B, C.
- Thinking ON (B vs A) gives only marginal improvement; not worth cost.

**Madhav's key caveat on E:** For the real-world use case of P(MITRE step | benchmark performance), we don't know upfront where a MITRE attack step falls on the difficulty spectrum. So the closest-bin strategy (condition E) requires knowing the target difficulty to pick the closest source bin — which we may not have in practice. Madhav notes: *"if we somehow justify 'based on the target task description, bin X is the closest', and use that as anchor instead of providing all four source bins, the forecasts seem to be better."*

**Closest-bin sanity check (week 11):**
As a validation of E's mechanism, Madhav sliced condition C's results by whether the source bin was the "closest" to the target:

| Slice | N | Brier | Spearman ρ | Bias |
|---|---|---|---|---|
| C — all 20 pairs | 240 | 0.1736 | 0.54 | −0.032 |
| **C — closest 5 pairs only** | 60 | **0.1046** | **0.79** | −0.002 |
| C — other 15 pairs | 180 | 0.1966 | 0.45 | −0.042 |
| E — closest_bin run | 60 | 0.1030 | 0.77 | +0.010 |

This confirms the mechanism: E's gains come from the close-bin analogy, not an artifact of the experimental setup. E and "C-closest" perform nearly identically.

**Per-expert breakdown (week 12):**

Dr Capability:

| Metric | A | B | C | E | F |
|---|---|---|---|---|---|
| N | 60 | 60 | 240 | 60 | 718 |
| Brier | 0.2275 | 0.2245 | 0.1743 | 0.1050 | 0.1903 |
| CRPS | 0.3061 | 0.3035 | 0.2617 | 0.1927 | 0.2785 |
| ECE | 0.1768 | 0.1687 | 0.0881 | 0.1465 | 0.0630 |
| Spearman ρ | 0.32 | 0.34 | 0.54 | 0.76 | 0.50 |
| Kendall τ | 0.27 | 0.29 | 0.44 | 0.63 | 0.41 |
| MAE | 0.3865 | 0.3793 | 0.3353 | 0.2575 | 0.3544 |
| Bias | −0.062 | −0.080 | −0.033 | +0.001 | +0.057 |

Prof. Psychometrics & Test Design:

| Metric | A | B | C | E | F |
|---|---|---|---|---|---|
| N | 59 | 60 | 240 | 59 | 720 |
| Brier | 0.2249 | 0.2245 | 0.1774 | 0.1018 | 0.1888 |
| CRPS | 0.3041 | 0.3029 | 0.2641 | 0.1912 | 0.2790 |
| ECE | 0.1790 | 0.1830 | 0.0907 | 0.1220 | 0.0765 |
| Spearman ρ | 0.37 | 0.33 | 0.53 | 0.78 | 0.50 |
| Kendall τ | 0.31 | 0.28 | 0.43 | 0.65 | 0.41 |
| MAE | 0.3810 | 0.3797 | 0.3378 | 0.2566 | 0.3550 |
| Bias | −0.076 | −0.073 | −0.039 | +0.006 | +0.060 |

**Conclusion (week 12):** "There seems to be a tiny difference between elicitation results between two experts. We probably mightn't need two experts."

**Ground truth visualization clarification (week 12):**
Earlier plots showed bin-level average pass rate as ground truth while elicitation was per-task — a misleading mismatch (Jeff noted this on Slack too). Analysis scripts compute Brier correctly at row level (per task, 0/1 outcome). The bin-level plots just made the visualization simpler but were not the basis of the reported metrics. Row-level ground truth is the correct approach.

**Target task variance analysis (week 12):**
Keeping bin 1 (or bin 3) as target with bins 2–5 as anchor: large variation in elicited outcomes across the 20 possible target tasks in a bin. The choice of a specific target task strongly impacts the elicited outcome. Source of variance: task-specific LLM miscalibration (K=1 is noisy). Madhav simulated K=3 and K=10 by resampling from 20 real Brier scores per bin (5000 iterations): K>1 reduces variance, as expected statistically. This simulation doesn't require new API calls but is a simple statistical fact.

### 6.5 Week 13 — Model Sweep (Madhav, from week13.md)

**Action item from May 12 discussion:** "Madhav: Estimate the cost for model sweep and run the model sweep experiments with setup A."

**Clarification:** "Setup A" here refers to **discussion.md's Setup A** — the single model, single target task framing (elicit P(s(t_TB, l) | i(l, t_SB)) with per-task 0/1 ground truth). This is the framing typology from the May 12 meeting, NOT intra-benchmark experimental condition A (all_except_target, thinking OFF). The model sweep tests which LLM forecaster performs best under this single-model/single-task framing.

**Model sweep results** (discussion.md Setup A framing):

| Model | N valid | Brier ↓ | CRPS ↓ |
|---|---|---|---|
| `gpt55` | 190 | **0.1008** | **0.1712** |
| `opus47` | 290 | 0.1370 | 0.2094 |
| `sonnet46` | 299 | 0.1387 | 0.2119 |
| `haiku45` | 299 | 0.1712 | 0.2594 |
| `gemini25flash` | 300 | 0.1822 | 0.2450 |

**Notes:**
- GPT-5.5 (gpt55): some runs did not succeed — flagged by GPT-5 as cyberattack risks. A prompt change could bypass this but would create inconsistency. N=190 vs ~300 for others. Results should be treated cautiously.
- All models beat uniform baseline (Brier 0.25, CRPS 0.33). Meaningful signal from LLM elicitation across all models.
- Opus 4.7 and Sonnet 4.6 perform nearly identically. This is good news: most experiments were run on Sonnet (Opus is ~2× more expensive).

### 6.6 May 12 Discussion — Conceptual Clarifications (from discussion.md)

**IMPORTANT: The A/B/C labels in discussion.md refer to a DIFFERENT typology than conditions A/B/C/E/F in the intra-benchmark experiments. Do not confuse them.**

**Discussion.md's A/B/C are experimental framing types:**
- **Discussion A:** Single model, single target task — elicit P(s(t_TB, l) | i(l, t_SB)). Ground truth = 0/1 per task per model. Aggregated via Brier. Avoids non-stationary problem. Expensive.
- **Discussion B:** Average model, single target task — elicit for "typical LLM" using averaged pass rates. Ground truth = average pass rate across all LLMs on target task. Suffers from non-stationary problem.
- **Discussion C:** Single model, average target task — elicit P(s(t̄_TB, l) | ...). Ground truth = bin pass rate per model. Like C but from a different angle.

Jakub noted that earlier plots mixed elicitation type A with ground truth type C (displaying bin-average pass rate against per-task elicitation), which was the source of the visualization confusion.

**Non-stationary problem:**
Averaging across LLMs from different generations (e.g., GPT-4 and GPT-5.5) loses the individual capability profile. Older models have jagged frontiers; newer ones are more balanced. Practical solution: be selective and only include LLMs from the same generation (~last 12 months). Averaging within same-generation is acceptable and may be desired.

**Single model vs generic frontier LLM debate:**
- Jeff: prefers single specific LLM.
- Jakub: prefers generic frontier LLM (for policy/regulation purposes — you don't know in advance which model will be used in an attack).
- Counter-argument for single-model: regulation thresholds for individual releases.
- Jakub's nuance: P(MITRE step | scores on multiple benchmarks) implicitly averages over many unobserved capabilities anyway. Two models with identical BountyBench+CyBench scores could differ on other dimensions.

---

## 7. Key Takeaways (factual, directly from sources)

1. **Model identity is the dominant driver of forecast variance** (48–69% Fréchet variance in MITRE experiments) vs. persona (6–25%) and temperature (≤12%).
2. **Baselines anchor forecasts strongly.** Removing the baseline value is the single most disruptive prompt manipulation (W1=0.154, 18× variance increase). All other prompt components — including full multi-phase reasoning scaffolding — have negligible effect.
3. **Closest-bin anchoring dramatically improves intra-benchmark calibration.** Condition E: Brier 0.1030, Spearman ρ=0.77. Condition A (all bins): Brier 0.2336, ρ=0.31. Both beat uniform baseline (0.25 Brier).
4. **The mechanism is confirmed:** Within condition C, the closest-5-pairs slice achieves Brier 0.1046 / ρ=0.79, nearly identical to E's 0.1030 / ρ=0.77.
5. **All 5 tested models (model sweep) beat the uniform baseline.** Sonnet 4.6 and Opus 4.7 perform nearly identically, validating the use of Sonnet for cost efficiency.
6. **Extended thinking (condition B) gives only marginal improvement** over no-thinking (condition A). Not worth the cost.
7. **Per-expert variance is negligible.** The two expert personas ("Dr Capability" and "Prof. Psychometrics") produce nearly identical results across all conditions.
8. **K=1 (single target task) is noisy** due to task-specific LLM miscalibration. K=3 reduces variance, but this requires more API calls (or is approximated via resampling from existing runs).
9. **CRPS is the right primary metric.** Uses full Beta distribution, strictly proper, robust to tail outcomes, interpretable in probability-point units.
10. **GPT-5.5 results are ambiguous** — some tasks refused as cyberattack risk, N=190 vs 300 for others. Cannot make a clean comparison.

---

## 8. Open Questions / Tensions (from the sources)

- **Closest-bin strategy in practice:** We don't know a MITRE step's difficulty tier upfront. How to pick the right source bin for production use of P(MITRE | benchmark performance)?
- **Single model vs frontier average:** Unresolved design debate between Jeff and Jakub (see Section 6.6).
- **Ground truth representativeness:** The primary comparison is elicited P(task) vs bin-level pass rate. This assumes the sampled target task is representative of the bin. The secondary comparison (vs actual 0/1 task outcome) is noisier but more rigorous.
- **GPT-5.5 model sweep:** Can't make clean inference due to task refusals (N=190).
- **Condition F:** Partial run (N=1389). Full results not in the notes.

---

## 9. Poster Format Requirements

### Template Layout (from SPAR_LLM_Forecasting_poster (2).pdf)
One landscape page. Full-width header strip, then 4 columns below:
- **Col 1:** Abstract (top) + Introduction (bottom)
- **Col 2:** Methods (top) + one Results block (bottom)
- **Col 3:** Results (full height or top-heavy)
- **Col 4:** Discussion (top-heavy) + References (bottom)

Target: **380–420 total body words**. Per-section cap: ~80 words. Reader should finish a section in <10 seconds.

### Key Style Observations from Sample Posters

**Stuxbench (winner):**
- Bold for every key claim; reader can skim bolds and get the story.
- One large flow diagram as centerpiece of Methods.
- Results uses a simple bar chart — no dense tables.
- Introduction uses numbered "levels of success" list — scannable.
- Discussion uses numbered bullet observations.

**Deception Probes (second prize):**
- SPAR logo prominently in top-left of header.
- Colored section header boxes (dark purple/red/blue) for visual separation.
- Dense but organized by color coding.
- Figures appear only in Results blocks.
- 3-column layout (narrower than Stuxbench).

### Pre-flight Checks (from poster_spec.md, retained as valid checklist)
1. All bold numbers match source files.
2. The baseline-anchoring theme visibly connects Results 2 and Results 3 (same accent color or arrow).
3. SPAR submission form filled: title, link, "Whole Team", all four collaborators.

---

## 10. Figure Assets (Confirmed Existing in Repo per poster_spec.md)

1. `output_data/experiments/frechet_anova/beta_cross_model_distributions_percentile.png` — cross-model Beta distributions for 3 attack steps
2. `prompt_sensitivity/output/wasserstein_combined_all_conditions.png` — W1/W2 from each prompt ablation to control
3. `inter_benchmark_calibration/results/comparison_no_anchor_vs_few_shot.png` — earlier inter-benchmark calibration result (may or may not be used for final poster)

Also available but secondary:
- `output_data/experiments/frechet_anova/beta_persona_distributions_percentile.png` (persona overlap)
- `output_data/experiments/frechet_anova/beta_temperature_means_grid.png` (temperature grid)
- `inter_benchmark_calibration/results/comparison_all_conditions_final.png` (all conditions bar)

**Note:** The figures for the intra-benchmark results (conditions A–F, closest-bin, model sweep) are not listed in poster_spec.md (that spec predates those experiments). Figures for those results need to be identified or generated from the repo.
