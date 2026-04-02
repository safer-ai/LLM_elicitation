# NumActors baseline uplift (`initial_access` bundle)

Same experiment as `execution_task/numactors/`: **scenario-level** estimate of how many actor groups fit the OC3 ransomware profile, with only the stated baseline count (3, 5, 10, 20, 50) changing.

- **Not** tied to TA0001 vs TA0002 — `steps: []` and Phase 2 metric only.
- Uses `bountybench_ordered.yaml` with `num_tasks: 1` for the metric elicitation context (first task: **Imaginairy**), same as the execution bundle.

Run from repo root:

Default is **3 runs** per baseline (3, 5, 10, 20, 50 actors).

```bash
cd LLM_elicitation
python3 output_data/experiments/baseline_uplift/initial_access/numactors/run_experiment.py
python3 output_data/experiments/baseline_uplift/initial_access/numactors/run_experiment.py --baseline 10 --runs 5
```

Results: `../numactors/results/baseline_<n>/` (under `initial_access`).
