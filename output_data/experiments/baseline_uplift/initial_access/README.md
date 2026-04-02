# Initial Access (TA0001) — baseline uplift bundle

Mirror of [`execution_task/`](../execution_task/) for the **TA0001 - Initial Access** step instead of **TA0002 - Execution**.

## What matches the execution experiment

| Piece | Value |
|--------|--------|
| Scenario narrative | OC3 + ransomware + large enterprise (same text block as execution) |
| Expert | `single_expert.yaml` (AI/ML Security Researcher) |
| Model / workflow flags | Same as execution configs (`num_tasks: 1`, `include_easier_tasks`, etc.) |
| Benchmark | `input_data/benchmark/bountybench_ordered.yaml` |
| **Primary elicitation task** | First task in that file — **Imaginairy** (not chosen per-step; same as execution) |
| Easier examples in prompt | `num_example_tasks: 2` from the same benchmark |

## What differs from execution

- **Step:** `TA0001 - Initial Access` in scenario `steps:` and `scenario_steps`.
- **Assumptions:** Previous step is **Resource Development**; baseline line is swept 10%–90%; sourcing text matches initial-access framing (WAF/IDS, exploit choice).

## Layout

- `setup_experiment.py` — regenerates all `scenarios/`, `configs/`, and `numactors/{scenarios,configs}/`.
- `run_experiment.py` — probability baseline sweep (5 runs × 9 baselines by default).
- `analyze_results.py` — stats + plots from `results/baseline_*/runs/*/detailed_estimates.csv`.
- `numactors/` — same **scenario-level** num-actors uplift design as `execution_task/numactors` (orthogonal to TA0001 vs TA0002; `steps: []`). Default **3 runs** per actor-count baseline (3, 5, 10, 20, 50).

## Commands

```bash
cd LLM_elicitation

# Regenerate YAMLs if you edit templates
python3 output_data/experiments/baseline_uplift/initial_access/setup_experiment.py

# Fill in API keys in configs/*.yaml and numactors/configs/*.yaml

python3 output_data/experiments/baseline_uplift/initial_access/run_experiment.py --dry-run
python3 output_data/experiments/baseline_uplift/initial_access/numactors/run_experiment.py --dry-run
```

Outputs go under `initial_access/results/` and `initial_access/numactors/results/` only (paths are self-contained under this folder).
