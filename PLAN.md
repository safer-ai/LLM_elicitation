Rough Next Steps Meeting Note:

Gemini notes: [https://docs.google.com/document/d/1PblHb5wjXP3l0onbU2Rl7kEQAtjhEVz2Zi9Ga6cjNcM/edit?usp=sharing](https://docs.google.com/document/d/1PblHb5wjXP3l0onbU2Rl7kEQAtjhEVz2Zi9Ga6cjNcM/edit?usp=sharing)

Two things are missing at the moment in order to turn this into a workshop-level project:

1. testing on 1-2 more benchmarks, ideally from different domains, not just coding. We should use 1 benchmark as the ‘train set’, i.e. optimise our prompt structure on it, and then, use the other benchmark(s) as the ‘test set’, i.e. hopefully observed an improved prediction quality on them.
2. the actual optimisation procedure. So far:
  1. we’ve done the groundwork for how we would *measure* an improvement in prediction quality
  2. we also found some elements of hte prompts that do not have a significant effect on this quality (e.g. expert persona)
    t we haven’t really found a principled way of making these improvements.

Things to look into:

- textual gradient — automated LLM improvements
  - similar to best-of-N jailbreaking? similar to automated red-teaming?
  - GEPA
  - [https://arxiv.org/abs/2305.03495](https://arxiv.org/abs/2305.03495)
- DSPy
- [https://www.metaculus.com/notebooks/38421/automated-prompt-engineering-for-forecasting/](https://www.metaculus.com/notebooks/38421/automated-prompt-engineering-for-forecasting/)
- is there persona?
- temperature
  - will have to be deprioritised as T is not available for many currnet models
- does the prompt include baselines?
  - I don’t mean the baseline in the context of our cyber risk models, i.e. the ‘human-only baseline of P(MITRE step)’. Instead, we could take as the baseline the fits that Madhav+Jeff produced to calculating the baseline Brier score (e.g. a linear fit).
- what is the transferability of prompt optimisations between models? I.e. say we devise a method to optimise forecaster performance on Sonnet 4.6 — what is the transfer of this method to Opus 4.8?
- another option is to just define the ‘features’ in the prompts ourselves (baselines or not, reasoning or not, analysis or not, multi-stage Delphi or not, …) and do a sweep over all combinations to see what works best
- run an experiment with an extremely minimal prompt — ‘here are 5 questions from a benchmark, predict performance on the 6th one’

## Plan

1. re-run the same checks as were done for the variance, but now for the Brier score on the Lyptus dataset (same thing on the Lyptus dataset as the initial prompt)
2. do the ‘minimal prompt’ elicitation
3. let’s look into the textual gradient method
4. test benchmark contamination with the current LLM forecasters

Jeff: little capacity over the next few weeks

Madhav: ≤5hrs/week

## Benchmarks

- Lyptus has multiple benchmarks within it, so if we don’t find anything apart from Lyptus, we can take the subgroups of Lyprus and use them as the train/test set
  - two approaches for subdividing lyptus:
  1. hold 1-2 sub-benchmarks as the test set
  2. don’t care about which sub-benchmark a task belongs to, just do an 80/20 split
  - it might be useful to hold out the hardest tasks (in terms of FST) because that allows fits our objective of making predictions about future levels of risk
- 
- requirements:
  - task-level pass rates
  - cybersecurity relevant
  - a good spread of scores, so that it’s not around 0% and 100%
  - multiple LLMs evaluated
  - the Epoch dataset did not have FST — we used that as a comparison point
- 

- [ ] do cost estimation and then apply for more SPAR money
- [ ] ask LLMs directly for their knowledge of benchmarks/ task-level solve rates to see if there is benchmark contamination
- [ ] Jakub: talk to Mateusz about automated red-teaming with LLMs and whether the methods there could be applied to imrpoving prompts for LLM forecasters

# Next Steps: LLM Forecaster Prompt Optimization + Benchmark Generalization

*Owner: Madhav (≤5 hrs/week). Jeff: limited capacity. Last updated: 2026-06-12.*

This plan turns the SPAR project into a workshop-level contribution. It is scoped
to the **intra-benchmark calibration** pipeline (`intra_benchmark_calibration/`),
which already has ground truth and a Brier/CRPS scoring harness.

---

## 1. Context: what exists, what's missing

**Done (final report):**

- Consistency experiments measuring **W1/W2** under prompt manipulations (no ground truth) → forecaster is internally stable; baseline anchor matters most, persona/reasoning barely matter.
- Intra-benchmark experiment measuring **Brier/CRPS** vs Lyptus ground truth → all LLMs beat uninformed; best design = Condition E (closest bin); best model = GPT-5.5.

**The two gaps (from team notes):**

1. **Generalization** — everything was measured on one benchmark (Lyptus). Need a train/test split across benchmarks/domains to show prompt improvements *transfer*.
2. **A principled optimization procedure** — we know how to *measure* quality (Brier) and which features are inert (persona), but we have **no method that actually improves the forecaster**. This is the core missing piece.

**Key realization that defines the next step:** the prompt-sensitivity work used **W1** (a shift/consistency measure, not accuracy). We never checked whether those same prompt changes help or hurt **Brier** (accuracy vs real outcomes). Re-running the ablations under Brier is step 1.

---

## 2. Objective

> Devise and validate a **principled prompt-optimization method** that measurably lowers the forecaster's **Brier score** on a held-out benchmark, and characterize its **transferability** across forecaster models.

Brier-on-p50 is the optimization target (per the paper, §3.2). CRPS is reported as a secondary check.

---

## 3. Reference numbers (already in repo — our comparison bar)

From `report_analyses/.../model_sweep_baseline/comparison_table.csv` (common 186-task set, `all_except_target` design):


| Source                  | Brier     | CRPS  | Role                            |
| ----------------------- | --------- | ----- | ------------------------------- |
| uninformed (flat 0.5)   | 0.250     | 0.333 | floor                           |
| model_pass_rate         | 0.207     | 0.415 | naive baseline                  |
| **model_bin_pass_rate** | **0.098** | 0.188 | strong baseline                 |
| **irt_logistic_fit**    | **0.093** | 0.187 | near-oracle (uses FST directly) |
| GPT-5.5 (forecaster)    | 0.102     | 0.172 | best LLM                        |
| Sonnet 4.6 (forecaster) | 0.124     | 0.190 | our dev model                   |
| Haiku 4.5 (forecaster)  | 0.179     | 0.267 | cheap pilot model               |


**The bar to beat:** the LLM forecaster currently sits *between* `model_bin_pass_rate` (0.098) and the IRT oracle (0.093) for the best model. The optimization goal is to push the LLM Brier toward / past the IRT oracle **without** giving it FST — and crucially to show that gain holds on a held-out benchmark.

---

## 4. Benchmark / train-test strategy

Per team notes, **Lyptus-internal splitting is the primary mechanism** (guaranteed task-level pass rates, FST, 12 models). External benchmarks are a stretch goal.

### Score spread per Lyptus sub-benchmark (computed from `model_runs.parquet`, 12-model panel)


| Sub-benchmark | Panel mean pass rate | Tasks w/ FST label | Usable for split?    |
| ------------- | -------------------- | ------------------ | -------------------- |
| cybergym      | 0.23                 | 102                | ✅ good spread, large |
| cvebench      | 0.33                 | 14                 | ✅ good spread, small |
| cybench       | 0.45                 | 37                 | ✅ best spread        |
| nyuctf        | 0.46                 | 33                 | ✅ best spread        |
| intercode_ctf | 0.91                 | 45                 | ⚠️ saturated         |
| nl2bash       | 0.85                 | 9                  | ⚠️ saturated, tiny   |
| cybashbench   | 0.95                 | 51                 | ⚠️ saturated         |


### Three split designs (in priority order)

- **Split A — hardest-task holdout (recommended primary).** Train on bins 0–3 (easier/mid FST), test on bin 4 (hardest tasks). Directly mirrors the real goal: predicting capability on *harder-than-seen* tasks. This is the most defensible scientific framing.
- **Split B — sub-benchmark holdout.** Train on {cybergym, cybench, intercode_ctf, nl2bash, cybashbench}, hold out **{nyuctf, cvebench}** as test (both have good spread, different task styles). Tests cross-distribution transfer.
- **Split C — 80/20 random** over all tasks (ignores family). Simplest; weakest scientific claim. Use only as a sanity baseline.

**External stretch benchmark:** **AutoPenBench** (33 pen-test tasks, 8 LLMs, ~21% SR — cyber-relevant, good spread, but small and no FST) is the closest external match. CyberSecEval 3 is *not* suitable (measures refusal/injection rates, not per-task capability solve). Decision deferred until after WS1–WS3 show signal.

---

## 5. Prompt feature space (what we optimize over)

The intra-benchmark prompt has discrete, separable components (verified in `prompt_builder.py` + `prompts/`). These are our optimization "knobs":


| Feature                             | Where                                                                     | On/off mechanism                                                   |
| ----------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **Bin-level pass rate**             | `capability_profile` header                                               | format toggle in `format_capability_profile`                       |
| **Per-task binary outcomes**        | anchor/easier task tags                                                   | `_outcome_tag` toggle                                              |
| **Analysis stage** (4-axis)         | `intra_capability_analysis.txt` (separate API call)                       | workflow flag to skip stage-1 call                                 |
| **Reasoning checklist**             | `initial_intra_solve_estimation.txt`                                      | template variant                                                   |
| **Ground-truth base-rate footnote** | `{ground_truth_summary}` in `initial_intra_solve_estimation.txt` line 39  | template variant — present in **all 5** sweep runs; see note below |
| **Source design**                   | `source_bins_to_show`                                                     | config: `all_except_target` vs `closest_bin`                       |
| **# source examples**               | `n_examples_per_source_bin`                                               | config int                                                         |
| **Delphi rounds / experts**         | workflow                                                                  | config int (persona shown inert — keep at 1)                       |
| **NEW: baseline-as-prior**          | inject `model_bin_pass_rate` or `irt_logistic_fit` prediction into prompt | new template field                                                 |


**Two ground-truth-derived fields — do not confuse them:**

- `{capability_profile}` (the source-bin pass rates + per-task SOLVED/FAILED tags) is the **legitimate forecasting signal** — it *is* the experiment. Only the extreme `no_source_context` condition removes it.
- `{ground_truth_summary}` is a separate **aggregate base-rate prior** (dataset-wide mean pass rate ≈0.566, % all-pass/zero-pass). It does **not** reveal the target's outcome, but it is computed from the full outcome matrix — a global prior a real forecaster wouldn't have. It was live in all 5 sweep runs, so it stays in `control`; we ablate it as `no_ground_truth_summary`.

The "baseline-as-prior" knob is the team's idea: feed the *statistical* baseline (e.g. logistic fit value) into the prompt as an explicit prior the LLM can anchor on or override.

---

## 6. Workstreams (mapped to team's 4-step plan)

> **The `WSn` numbers are a catalogue, not an execution order.** Execution order is the strategic ladder in §8, which deliberately front-loads the cheapest probes that can prune the most downstream work (e.g. the minimal-prompt test runs *before* the heavy ablations).

### WS0 — Infrastructure prerequisites (must come first)

1. **Train/test split support.** Add a task-id / family / bin filter to `load_lyptus_dataset` or `build_cell_plans` (currently enumerates over the whole dataset; no holdout exists). ~1 day.
2. **Prompt-condition harness.** A `--prompts-dir` + feature-flag layer so each condition is a config, not a code edit. Skip-analysis needs a small `workflow.py` flag. ~1 day.
3. **Brier feedback function.** Wrap `analyse_results.py` scoring into a callable `evaluate(prompt, split) -> (brier, per_task_feedback)` that returns *both* the scalar and textual misprediction traces (needed for GEPA). ~1 day.
4. **Baseline-as-prior plumbing.** Expose `model_bin_pass_rate` / `irt_logistic_fit` predictions (already in `calculate_baselines.py`) as an optional prompt field. ~0.5 day.

### WS1 — Brier replication of the variance/prompt-sensitivity checks *(team step 1)*

Re-run the manual prompt ablations, but score **Brier on the train split** instead of W1.

- Conditions: control, no_bin_rate, no_task_outcomes, no_ground_truth_summary, no_source_context, skip_analysis, trim_reasoning, trim_all, closest_bin design, baseline-as-prior. (~10 conditions.)
- Output: a Brier-by-condition table (analogue of `compare_all_conditions.py`, but accuracy not shift). **This is also a feature sweep** — the team's "define the features ourselves and sweep combinations" idea.

**Executing now — Level 0 bracket (`experiments/I_prompt_ablation_brier/`).** See §8 for *why this runs first*: it is the cheapest set of probes that can prune the rest of the tree. Three matched conditions — same 300 cells, Sonnet `reasoning_effort: off`, 3 repeats each:

- `control` (full prompt) — the reference. **Built.**
- `no_ground_truth_summary` (gray-zone base-rate leak removed — line 39 of `initial_intra_solve_estimation.txt`). **Built.**
- `minimal` (*"here are 5 tasks with pass/fail; predict the 6th"* — no analysis stage, no reasoning checklist, no footnote). **To add**; depends on the small skip-analysis flag (WS0-infra), since a truly minimal prompt is single-call.
- **Setup (Option A, self-contained):** all conditions fresh in one folder, byte-identical inputs, cells deterministic, so every condition sees the same 300 target tasks. Each condition = a copy of the canonical prompts in `prompt_variants/<condition>/` + a config cloned from `config_sonnet46.yaml`.
- **Cost:** design = G sweep (12 models × 5 bins `all_except_target` × K=5 = 300 cells = 600 calls/run); ≈ $8.2/condition-run on Sonnet no-thinking. 2-arm leak check ≈ $49; full 3-arm Level 0 ≈ $74.
- **Caveat:** published Sonnet Brier 0.124 used thinking=high; this round is no-thinking, so absolute Brier may shift, but matched **deltas** are valid. Thinking re-runs of winners stay in a later level.
- **Metric:** Brier-on-p50 (primary), CRPS secondary, via `analyse_results.py` → `summarize_ablation.py`, which aggregates a `(condition, repeat)` table and the per-cell **paired** delta + bootstrap CI (the real significance test; to be added before run).

### WS2 — Minimal-prompt elicitation *(team step 2 — PROMOTED to run first)*

One extreme condition: *"Here are 5 tasks from a benchmark with pass/fail; predict pass on the 6th."* No analysis, no reasoning scaffold, no base-rate footnote. Tests how much the elaborate prompt is actually buying us. **This is deliberately moved to Level 0 (see §8): if the minimal prompt matches or beats `control`, it invalidates the need for the heavy scaffold and most of the WS1 ablations, and saves their token cost — so it must be tested before, not after, the expensive conditions.**

### WS3 — Principled optimization via textual gradient *(team step 3 — core deliverable)*

Use **GEPA** (Genetic-Pareto reflective prompt evolution, `arXiv:2507.19457`, pip `gepa`):

- **Why GEPA over DSPy/MIPRO:** GEPA consumes the *textual misprediction feedback* (our per-task rationale + "predicted 0.8, actual 0") as an "actionable side-information" gradient, not just a scalar — a natural fit since our forecaster already emits rationales. Sample-efficient (~35× fewer rollouts than RL).
- **Setup:** seed = current control prompt; metric = Brier on a train minibatch; feedback = per-task (p50, outcome, rationale); reflection LM = a strong model (GPT-5.5 or Sonnet 4.6); evolve the analysis + estimation templates.
- **Validation:** evaluate the GEPA-optimized prompt on the **held-out test split** and compare Brier to the seed control. Success = test-set Brier drop with no train-only overfit.
- **Fallback if GEPA integration is heavy:** the WS1 feature sweep already provides a discrete-search baseline; GEPA is the continuous/automated upgrade.

### WS4 — Benchmark generalization *(team gap 1)*

- Take the WS3-optimized prompt and the WS1 best discrete config; evaluate **both** on the held-out test split (Split A primary, Split B secondary).
- Headline figure: "Brier on held-out hard tasks: control vs optimized," with the `model_bin_pass_rate` and `irt_logistic_fit` baselines drawn in.

### WS5 — Benchmark contamination test *(team step 4)*

- **Probe 1 (direct recall):** ask each forecaster model directly, "What is the pass rate of `<model>` on `<task_id>` in `<benchmark>`?" / "Do you recognize this task?" Measure whether it can reproduce ground-truth solve rates from memory.
- **Probe 2 (base-rate audit):** the `{ground_truth_summary}` footnote feeds a dataset-wide base rate into the prompt. The WS1 `no_ground_truth_summary` ablation quantifies how much of current accuracy depends on it.
- Cheap (~200 direct Q&A calls). Important for the validity of all accuracy claims.

### WS6 — Transferability across forecaster models *(team open question)*

- Optimize on Sonnet 4.6, then apply the *frozen* optimized prompt to **Opus 4.8 / GPT-5.5** on the test split. Does the gain transfer, or is each prompt model-specific?
- This is the cheapest high-value experiment once WS3 produces an optimized prompt.

---

## 7. Cost breakdown

**Token basis** (measured from the committed Sonnet 4.6 G-run `scored_with_crps.csv`): per cell = 2 API calls ≈ **4,600 input + 900 output tokens**. Pricing per 1M tok (June 2026): Haiku 4.5 $1/$5, GPT-5 $1.25/$10, Sonnet 4.6 $3/$15, GPT-5.5 $5/$30. Thinking tokens billed as output.

**Cost per cell and per 300-cell evaluation pass:**


| Model                              | $/cell  | $/300-cell eval |
| ---------------------------------- | ------- | --------------- |
| Haiku 4.5                          | $0.009  | ~$2.7           |
| GPT-5                              | $0.015  | ~$4.4           |
| Sonnet 4.6 (no thinking)           | $0.027  | ~$8.2           |
| GPT-5.5                            | $0.050  | ~$15.0          |
| Sonnet 4.6 (thinking ~4k tok/call) | ~$0.147 | ~$44            |


**Per-workstream estimate** (dev model = Sonnet 4.6 no-thinking unless noted; train split ≈ 300 cells):


| Workstream                                     | What runs                                                                                 | API cost  |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------- | --------- |
| WS1 pilot                                      | 9 conditions × 1 run on **Haiku** (find signal cheaply)                                   | ~$24      |
| WS1 full                                       | top ~5 conditions × 3 runs, Sonnet                                                        | ~$123     |
| WS2                                            | folded into WS1 (1 extra condition)                                                       | ~$0       |
| WS3 GEPA                                       | ~1 optimization run ≈ 40 candidate minibatch evals + 10 full evals + ~50 reflection calls | ~$150     |
| WS3 (2nd model)                                | GEPA on GPT-5                                                                             | ~$130     |
| WS4                                            | control + optimized on test split (~150 cells) × 3 models                                 | ~$60      |
| WS5                                            | contamination probes (~200 direct calls)                                                  | ~$10      |
| WS6                                            | frozen prompt on Opus 4.8 + GPT-5.5, test split                                           | ~$60      |
| Thinking re-runs                               | winners re-run with extended thinking (Sonnet)                                            | ~$130     |
| **Subtotal**                                   |                                                                                           | **~$690** |
| Buffer (failed runs, reruns, exploration) ~30% |                                                                                           | ~$210     |
| **Total budget ask**                           |                                                                                           | **~$900** |


**Tiered options:**

- **Minimum viable (WS0–WS4, no thinking, single GEPA run):** ~$220.
- **Recommended (everything, 1 model deep + transfer):** ~$690.
- **SPAR funding ask (with buffer):** **~$900–1,000.**

> Cost is dominated by GEPA rollouts and thinking re-runs, not by the manual sweep. Haiku-first piloting keeps exploration cheap.

---

## 8. Sequencing — a strategic escalation ladder (given ≤5 hrs/week)

**Guiding principle:** at each step, run the *cheapest experiment that could most decisively prune everything below it*. We do not run a blind batch of conditions; we run the one probe whose outcome could make the others unnecessary. Every level yields a standalone insight and a **gate** that decides what (if anything) runs next. Cost rises as we climb; the cheap lower rungs (Levels 0-2, <~$150 total) exist precisely to avoid wasting the expensive upper rungs.

### Level 0 — Bracket the design space + validity *(cheapest, most decisive)* — **Experiment I**

Three matched conditions (Sonnet no-thinking, 3 repeats, identical 300 cells):

- `control` (full prompt), `minimal` (5 examples → predict 6th), `no_ground_truth_summary` (leak removed).
- **Why first:**
  - `minimal` vs `control` **brackets the entire value of prompt complexity**. *If `minimal` ≈ or > `control`, the elaborate scaffold isn't buying accuracy — a major finding that reshapes what we optimize (we'd optimize the cheap minimal prompt). This does not skip Level 1; the leave-one-out decomposition still tells us which components mattered (or that none did).*
  - `no_ground_truth_summary` is a **validity gate on** 
  -  **accuracy number we report**. *If removing the leak tanks Brier, our headline numbers are partly leak-driven and the honest baseline drops.*
- **GATE →** how big is the control-minus-minimal gap, and is the baseline honest? This *frames* Level 1 (how much total accuracy the scaffold is responsible for) — it does **not** skip it. We do not assume minimal works; Level 1 runs regardless. Cost ≈ $74.

### Level 0b — Contamination direct-recall probe *(cheap, parallelizable)*

Ask each forecaster directly for task-level solve rates / "do you recognize this task" (~200 calls, ~$10). Validity sidecar; can run alongside Level 0. (The base-rate-audit half of contamination *is* the `no_ground_truth_summary` result above.)

- **GATE →** are accuracy claims confounded by memorization?

### Level 1 — Systematic leave-one-out decomposition *(committed; the prompt_sensitivity replication, now under Brier)*

This is the heart of "team step 1" and runs **regardless of Level 0** — we do not assume the minimal prompt works. From the **same `control` baseline**, each condition removes exactly **one** component (leave-one-out), so every Brier delta is attributable to that component. `control` (ceiling) and `minimal` (floor, from Level 0) bracket the range; these conditions explain the gap between them. Each is one matched run (same 300 cells, Sonnet no-thinking, 3 repeats).


| Intra condition           | Removes (one thing)                                                   | prompt_sensitivity ancestor                                             |
| ------------------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `control`                 | nothing (full prompt)                                                 | control                                                                 |
| `no_ground_truth_summary` | dataset-wide base-rate footnote                                       | `no_baseline` (the dominant anchor in the W1 study) — *done at Level 0* |
| `no_bin_rate`             | empirical source-bin pass-rate header                                 | (intra-specific)                                                        |
| `no_task_outcomes`        | per-task SOLVED/FAILED tags                                           | (intra-specific)                                                        |
| `skip_analysis`           | the stage-1 analysis call                                             | `skip_analysis`                                                         |
| `trim_reasoning`          | the reasoning checklist                                               | `trim_reasoning`                                                        |
| `trim_all`                | analysis + reasoning + technical-analysis block                       | `trim_all`                                                              |
| `no_source_context`       | the entire `capability_profile` (extreme; ~the floor below `minimal`) | `no_baseline_no_ci` (extreme)                                           |


(`persona` is already known inert from the W1 work → stays off; not re-tested.)

Two reads of the same table: top-down (each row = marginal cost of *removing* that piece from the full prompt) and, for any piece that looks important, a confirmatory bottom-up *add-it-back-to-minimal* check. Optionally also a cumulative staircase `control → … → minimal` if we want the monotone "each successive removal costs X" narrative.

- **GATE →** the leanest prompt that *retains* accuracy = the optimization seed for Level 3. Cost ≈ $8.2 × (#conditions × 3 repeats); ~6 new conditions ≈ $150.

### Level 2 — Information knobs on the lean skeleton

`closest_bin` design (prior-best Condition E) and `baseline-as-prior` (inject `model_bin_pass_rate` / `irt_logistic_fit`). These change the *evidence* shown, not the scaffold.

- **GATE →** best hand-built config.

### Level 3 — Principled optimization (GEPA)

Seed = the Level 1-2 winner; optimize Brier on the **train** split.

- **GATE →** does automated optimization beat the best hand-built prompt on train?

### Level 4 — Generalization

Best prompt(s) on the held-out **test** split (Split A primary). Headline figure: control vs optimized vs statistical baselines.

- **GATE →** does the gain hold out-of-sample?

### Level 5 — Transferability

Freeze the optimized prompt; apply to Opus 4.8 / GPT-5.5 on test. Does the gain transfer or is it model-specific?

### Level 6 — External benchmark (AutoPenBench)

Stretch / workshop polish.


| Level | What runs                           | Decisive gate                            | Could kill                                       |
| ----- | ----------------------------------- | ---------------------------------------- | ------------------------------------------------ |
| 0     | control / minimal / no_gts          | how big is control−minimal? leak honest? | frames Level 1 scope; flags an inflated baseline |
| 0b    | direct-recall contamination         | memorization confound?                   | validity of all accuracy claims                  |
| 1     | leave-one-out ablations (committed) | which single parts earn their tokens?    | inert components (dropped from the seed)         |
| 2     | closest_bin, baseline-as-prior      | best hand-built config                   | weaker designs                                   |
| 3     | GEPA optimization (train)           | beats hand-built on train?               | the manual-only story                            |
| 4     | test-split eval                     | holds out-of-sample?                     | overfit prompts                                  |
| 5     | cross-model transfer                | model-specific?                          | one-model claims                                 |
| 6     | AutoPenBench                        | external transfer                        | —                                                |


Each level is independently publishable-incremental, so limited hours still produce results.

---

## 9. Decisions needed before coding

1. **Primary split:** confirm **Split A (hardest-bin holdout)** as the headline, or prefer sub-benchmark holdout (Split B)? *(Recommend A — matches the real forecasting goal.)*
2. **Dev model:** confirm **Sonnet 4.6 no-thinking** for cheap iteration, with thinking re-runs only on winners? Or optimize directly with thinking (5× cost)?
3. `**{ground_truth_summary}` footnote:** confirm we **keep it in `control*`* (it ran in all 5 sweeps) and measure its effect via the `no_ground_truth_summary` ablation, rather than silently removing it? *(Recommend keep + ablate.)*
4. **GEPA scope:** full `gepa` library integration, or a lighter hand-rolled reflect-and-mutate loop for v1? *(Recommend the library to cite it cleanly in the workshop paper.)*
5. **External benchmark:** in scope for this round, or defer entirely to Lyptus-internal splits? *(Recommend defer until WS4 shows signal.)*
6. **Funding:** is the ~$900 ask the right size to request from SPAR, or should I produce a leaner ~$220 minimum-viable proposal?

