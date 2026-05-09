# Intra-benchmark experiment: clarification notes

## Core setup

Dataset: `ground-truth/github_data/` directly. It is the pre-processed Lyptus data, no need for the HuggingFace loader.

- `model_runs.parquet`: per (agent, task_id) binary pass/fail. 15 models across 630 tasks, 8,963 rows with 100 percent non-null pass/fail.
- `task_difficulties.parquet`: per task, `best_available_minutes` and `best_available_source`.
- `runs.parquet`: pre-joined convenience table (model runs merged with `human_minutes`).

Filter to the 291 headline tasks, defined as those with `best_available_source` in {completion, first_blood, expert_estimate}.

Difficulty ordering: `best_available_minutes` (hybrid of actual completion time, first-blood time, and expert estimate). Do not use LLM time estimates as the primary ordering (circular), and do not use empirical per-task solve rates as the primary ordering (tautological with the calibration target). IRT and solve-rate orderings are optional ablations.

Bins: use Lyptus's own time bins (5 bins) as the default.

## What is elicited vs. what is scored

### Elicitation (what the LLM is asked, and what it returns)

Default framing (Jeff's primary plan):

- Prompt contains source context for 4 of 5 bins. For each source bin, the prompt states the model's aggregate pass rate on that bin (a single number like 0.65) plus a handful of representative task descriptions. The representative tasks are shown as descriptions only, without individual pass/fail labels.
- Prompt then shows ONE task description sampled from the target (5th) bin. No other information about the 5th bin is shown, in particular no pass rate on the 5th bin.
- The LLM returns an estimate of P(model passes this one target task), expressed as 25th, 50th, 75th percentiles. These are fit to a Beta distribution, and the mean is used as the point estimate p_i.

So the answer to "do we ask for one task or overall score on the 5th bin?" is: one task. The 5th bin's overall score is never referenced in the prompt, because that score is the ground truth we are trying to compare against.

### Concrete prompt shape

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

### Scoring (three possible comparisons, same elicitation)

1. Primary (Jeff's plan). Compare p_i to the model's bin-level pass rate on bin 5. Assumes Task Z is representative of bin 5.
2. Post-meeting Brier alternative. Compare p_i to 0 or 1, depending on whether the model actually passed Task Z. Drops the representativeness assumption, noisier per-datapoint.
3. Aggregate variant from the "wrinkles" section (NOT chosen). Different elicitation: show several target tasks, ask for aggregate pass rate, compare to bin pass rate. Matches elicitation to ground truth more cleanly but diverges from the long-term P(MITRE step) goal.

Default is scoring option 1. Options 1 and 2 use the same elicitation, so both can be computed from one run at no extra cost. Option 3 is a different elicitation and is deferred.

## Source vs. target bin structure (resolving the apparent contradiction)

There is no contradiction. The two statements in the meeting notes refer to different bins.

- Source bins appear in the prompt as context. Each source bin contriblutes multiple representative tasks and the model's empirical pass rate on that bin. Number of source bins shown and number of representative tasks per bin are both configurable.
- Target bin is the one being predicted. One target task is sampled from it. The LLM sees its prompt or description only and estimates P(solve) via three percentiles (25th, 50th, 75th), which are fit to a Beta distribution.

## Ground-truth derivation

Per model, per bin. No averaging across models.

```
ground_truth(M, j) = (number of headline tasks in bin j solved by M) / (number of headline tasks in bin j)
```

With 15 models and 5 bins, this gives 75 ground-truth cells. This matches the "75 datapoints" figure in the meeting notes.

With K = 1 sampled target task per cell (Jeff's original spec), each experimental condition produces 75 elicited predictions and one scalar Brier score.

## Direction of prediction

Mechanically, source bins are whichever subset of bins you show in the prompt, and the target bin is the one you predict. There is no built-in ordering constraint.

First implementation pass: forward direction only (easier source bins, harder target bin). Reverse direction (harder to easier) and arbitrary subsets are straightforward extensions once source and target are independent config lists.

## The outer loops

The most likely intended structure is nested:

```
for condition in experimental_conditions:       # outer loop
    # e.g. {n_source_bins: 4, prompt_variant: "with_analysis", direction: "forward"}
    predictions = []
    for model M in 15 models:                   # inner loop
        for target_bin j in 5 bins:             # inner loop
            source_bins = choose_source_bins(condition, j)
            sample 1 target task t from bin j
            p = elicit(M, source_bins, t)
            o_bin = pass_rate(M, j)
            o_task = did_M_pass(t)
            predictions.append((p, o_bin, o_task))
    record: Brier_bin[condition], Brier_task[condition], log_score[condition], CRPS[condition]
```

Inner loop: the 75 prediction cells that feed one Brier score.
Outer loop: experimental conditions you sweep to compare, such as number of source bins, prompt variant, anchor strategy, direction.

This is the interpretation to proceed with unless Matt clarifies otherwise.

## Scoring metrics

Compute all four from the same Beta fit. Computation cost is trivial once the Beta is in hand.

1. Plain Brier on the point estimate. `(1/N) * sum((mean(Beta_i) - o_i)^2)`. Standard.
2. Task-level Brier using 0/1 outcomes. Same formula, with `o_i` in {0, 1}. Strictly proper scoring rule.
3. Log score. `-(1/N) * sum(log Beta_pdf(o_i; alpha_i, beta_i))`. Strictly proper. Penalises confident-wrong predictions heavily.
4. CRPS (Continuous Ranked Probability Score). `integral of (F_i(x) - indicator(x >= o_i))^2 dx` where F_i is the Beta CDF. Strictly proper and more robust than log score to tail misspecification.

Variance-scaled Brier (for example `(p - o)^2 / sigma^2`) is NOT a proper scoring rule: wider CIs artificially improve the score. Do not use as a primary metric. It is acceptable as a diagnostic.

## Relation to existing code

- `inter_benchmark_calibration/`. Predicts P(target task on benchmark B | source tasks on benchmark A) via Delphi. Reuse scaffolding: Delphi loop, expert profile loading, results handler, config pattern.
- `intra_benchmark_calibration/`. Already implements a P(j|i) framing. Closest structural match. Decision on whether to extend in place or start a new module is deferred.
- `task_elicitation/`. Elicits P(specific task | anchor tasks with known pass/fail), scores against 0/1. Partial precedent for framing 2 above, but its source context format differs from Jeff's plan (anchor tasks, not bin-level pass rates). Scoring code is reusable.
- `ground-truth/github_data/`. Primary input.
- `ground-truth/data.py`. Not needed; `github_data/` already contains everything required.

## Locked decisions

1. Source bins contribute multiple representative tasks plus the model's bin pass rate. Target bin contributes one sampled task per (model, bin) cell.
2. K = 1 target task per (model, bin) cell, giving 75 datapoints per condition.
3. Forward direction only for the first pass.
4. Data source: `ground-truth/github_data/` parquet files.
5. Difficulty: `best_available_minutes` on the 291 headline tasks, 5 bins.
6. Ground truth: per-model bin pass rate (primary), per-task 0/1 (free secondary).
7. Scoring: plain Brier, log score, CRPS, task-level Brier.
8. Outer-loop interpretation: experimental conditions sweep in the outer, (model, target_bin) grid in the inner.

## Open items

- Bin count and bin edges. Lyptus native bins versus a custom choice. Decide before the first run.
- Number of source bins to show in prompt. Sensitivity variable. Sweep for example {1, 2, 3, 4} in the outer loop.
- Number of representative source tasks per bin. Default 3 (matches inter-benchmark). Sensitivity variable.
- Pass rate presentation format: numerical, verbal ("the model solved about 40 percent"), or both. Prompt-design ablation.
- Delphi convergence behaviour on this framing may need retuning relative to inter-benchmark findings.
- Reverse direction and arbitrary-subset-of-bins-as-source. Post-MVP.
- Matt S's exact meaning of "outer loops". Proceed with the nested interpretation unless clarified.

## Relation to the prior ground-truth folder experiment

The concern that "we were eliciting P(particular task | other bin performance) but here we are talking about P(other bin | easier bins performance)" is partially correct:

- The elicitation target in both framings is P(particular task). This is the same.
- The difference is what the source context looks like. `task_elicitation/` uses anchor tasks with pass/fail labels. Jeff's plan uses bin-level pass rates plus representative tasks.
- The ground-truth comparison also differs: `task_elicitation/` scores against 0/1 task outcomes, Jeff's primary plan scores against bin pass rates.

Both are defensible framings. Jeff's primary plan is preferred because it matches the long-term goal of P(MITRE step | benchmark-level capability signals) more directly.