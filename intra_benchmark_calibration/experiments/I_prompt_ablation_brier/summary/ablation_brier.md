# Experiment I — Level-0 prompt ablation: Brier-on-p50

- `control` run dir: `20260614_021706`
- `no_ground_truth_summary` run dir: `20260614_151919`
- `trim_reasoning` run dir: `20260614_210535`
- `skip_analysis` run dir: `20260614_155809`
- `trim_all` run dir: `20260614_210840`
- `no_bin_rate` run dir: `20260615_234522`
- `no_task_outcomes` run dir: `20260615_234547`
- `no_source_context` run dir: `20260616_105520`
- `minimal` run dir: `20260614_004058`
- `single_call_analysis` run dir: `20260621_220921`

## Brier-on-p50 by condition

| Condition | repeats | cells | Brier mean | Brier sd | Brier pooled | CRPS mean |
|---|---|---|---|---|---|---|
| control | 1 | 300 | 0.1377 | 0.0000 | 0.1377 | 0.2126 |
| no_ground_truth_summary | 1 | 300 | 0.1362 | 0.0000 | 0.1362 | 0.2128 |
| trim_reasoning | 1 | 300 | 0.1427 | 0.0000 | 0.1427 | 0.2181 |
| skip_analysis | 1 | 300 | 0.1482 | 0.0000 | 0.1482 | 0.2281 |
| trim_all | 1 | 300 | 0.1447 | 0.0000 | 0.1447 | 0.2259 |
| no_bin_rate | 1 | 300 | 0.1417 | 0.0000 | 0.1417 | 0.2122 |
| no_task_outcomes | 1 | 299 | 0.1367 | 0.0000 | 0.1367 | 0.2155 |
| no_source_context | 1 | 300 | 0.1554 | 0.0000 | 0.1554 | 0.2553 |
| minimal | 3 | 893 | 0.1502 | 0.0013 | 0.1502 | 0.2352 |
| single_call_analysis | 1 | 300 | 0.1428 | 0.0000 | 0.1428 | 0.2209 |

## Paired per-cell delta vs control
_delta = Brier(condition) − Brier(control), averaged over matched cells. Negative = condition better (lower Brier) than control._

| Condition | paired cells | mean delta | 95% CI (bootstrap) | cells worse | Wilcoxon p |
|---|---|---|---|---|---|
| no_ground_truth_summary | 300 | -0.0015 | [-0.0093, +0.0055] | 38% | 0.926 |
| trim_reasoning | 300 | +0.0050 | [-0.0015, +0.0120] | 38% | 0.432 |
| skip_analysis | 300 | +0.0105 | [+0.0028, +0.0181] | 50% | 2.5e-05 |
| trim_all | 300 | +0.0070 | [-0.0012, +0.0150] | 48% | 0.00217 |
| no_bin_rate | 300 | +0.0040 | [-0.0060, +0.0144] | 47% | 0.546 |
| no_task_outcomes | 299 | -0.0012 | [-0.0108, +0.0079] | 41% | 0.854 |
| no_source_context | 300 | +0.0178 | [-0.0030, +0.0376] | 68% | 1.2e-07 |
| minimal | 299 | +0.0136 | [+0.0051, +0.0224] | 53% | 7.2e-05 |
| single_call_analysis | 300 | +0.0051 | [-0.0028, +0.0129] | 43% | 0.0466 |

_Interpretation: if a condition's 95% CI excludes 0, the prompt change has a statistically reliable effect on accuracy across the 300 matched cells._

## Per-repeat Brier

| Condition | repeat | Brier | n |
|---|---|---|---|
| control | 1 | 0.1377 | 300 |
| no_ground_truth_summary | 1 | 0.1362 | 300 |
| trim_reasoning | 1 | 0.1427 | 300 |
| skip_analysis | 1 | 0.1482 | 300 |
| trim_all | 1 | 0.1447 | 300 |
| no_bin_rate | 1 | 0.1417 | 300 |
| no_task_outcomes | 1 | 0.1367 | 299 |
| no_source_context | 1 | 0.1554 | 300 |
| minimal | 1 | 0.1516 | 299 |
| minimal | 2 | 0.1500 | 296 |
| minimal | 3 | 0.1490 | 298 |
| single_call_analysis | 1 | 0.1428 | 300 |

## Controlled-variable checks

- [PASS] control: run's recorded experiment_label matches its folder — recorded=['control'], folder='control'
- [PASS] no_ground_truth_summary: run's recorded experiment_label matches its folder — recorded=['no_ground_truth_summary'], folder='no_ground_truth_summary'
- [PASS] trim_reasoning: run's recorded experiment_label matches its folder — recorded=['trim_reasoning'], folder='trim_reasoning'
- [PASS] skip_analysis: run's recorded experiment_label matches its folder — recorded=['skip_analysis'], folder='skip_analysis'
- [PASS] trim_all: run's recorded experiment_label matches its folder — recorded=['trim_all'], folder='trim_all'
- [PASS] no_bin_rate: run's recorded experiment_label matches its folder — recorded=['no_bin_rate'], folder='no_bin_rate'
- [PASS] no_task_outcomes: run's recorded experiment_label matches its folder — recorded=['no_task_outcomes'], folder='no_task_outcomes'
- [PASS] no_source_context: run's recorded experiment_label matches its folder — recorded=['no_source_context'], folder='no_source_context'
- [PASS] minimal: run's recorded experiment_label matches its folder — recorded=['minimal'], folder='minimal'
- [PASS] single_call_analysis: run's recorded experiment_label matches its folder — recorded=['single_call_analysis'], folder='single_call_analysis'
- [PASS] no_ground_truth_summary: identical cell set vs control — control=300, no_ground_truth_summary=300, shared=300
- [PASS] no_ground_truth_summary: prompt differs from control (disjoint prompt_hash) — overlap=0 (expected 0)
- [PASS] trim_reasoning: identical cell set vs control — control=300, trim_reasoning=300, shared=300
- [PASS] trim_reasoning: prompt differs from control (disjoint prompt_hash) — overlap=0 (expected 0)
- [PASS] skip_analysis: identical cell set vs control — control=300, skip_analysis=300, shared=300
- [PASS] skip_analysis: prompt differs from control (disjoint prompt_hash) — overlap=0 (expected 0)
- [PASS] trim_all: identical cell set vs control — control=300, trim_all=300, shared=300
- [PASS] trim_all: prompt differs from control (disjoint prompt_hash) — overlap=0 (expected 0)
- [PASS] no_bin_rate: identical cell set vs control — control=300, no_bin_rate=300, shared=300
- [PASS] no_bin_rate: prompt differs from control (disjoint prompt_hash) — overlap=0 (expected 0)
- [FAIL] no_task_outcomes: identical cell set vs control — control=300, no_task_outcomes=299, shared=299
- [PASS] no_task_outcomes: prompt differs from control (disjoint prompt_hash) — overlap=0 (expected 0)
- [PASS] no_source_context: identical cell set vs control — control=300, no_source_context=300, shared=300
- [PASS] no_source_context: prompt differs from control (disjoint prompt_hash) — overlap=0 (expected 0)
- [FAIL] minimal: identical cell set vs control — control=300, minimal=299, shared=299
- [PASS] minimal: prompt differs from control (disjoint prompt_hash) — overlap=0 (expected 0)
- [PASS] single_call_analysis: identical cell set vs control — control=300, single_call_analysis=300, shared=300
- [PASS] single_call_analysis: prompt differs from control (disjoint prompt_hash) — overlap=0 (expected 0)
