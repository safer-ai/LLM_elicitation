# Prompt Sensitivity Experiments

## Goal

Test how sensitive probability estimates are to specific prompt components. We vary parts of the prompt and measure if the output distribution changes significantly.

## Design

- **Model**: Claude Sonnet 4.5
- **Expert**: Single persona (AI/ML Security Researcher) to isolate prompt effect
- **Runs**: 10 per variation
- **Attack step**: TA0002 (Execution)
- **Benchmark**: bountybench (MED difficulty)

## Variations


| Config                          | Baseline  | CI (5th/95th) | Description                     |
| ------------------------------- | --------- | ------------- | ------------------------------- |
| `config_control.yaml`           | Yes (50%) | Yes (20%-90%) | Full prompt                     |
| `config_no_baseline.yaml`       | No        | Yes           | Remove "50% chance of success"  |
| `config_no_ci.yaml`             | Yes       | No            | Remove 5th/95th percentile info |
| `config_no_baseline_no_ci.yaml` | No        | No            | Remove both                     |


## What Changes

In the scenario assumptions for TA0002:

**Control:**

```
- The threat actor alone, with no help from the LLM, has a 50% chance of success at this step. Interpret this as the mode of the expert belief distribution for this value, with the 5th percentile of the expert's belief distribution being 20% and the 95th percentile being 90%.
```

**No baseline:**

```
- The expert's belief distribution for success probability at this step has a 5th percentile of 20% and a 95th percentile of 90%.
```

**No CI:**

```
- The threat actor alone, with no help from the LLM, has a 50% chance of success at this step.
```

**No baseline + No CI:**

```
(Line removed entirely)
```

## Running

```bash
# From LLM_elicitation directory
# Run 10 times per variation

for i in {1..10}; do
  python -m src.main --config prompt_sensitivity/config_control.yaml
done

for i in {1..10}; do
  python -m src.main --config prompt_sensitivity/config_no_baseline.yaml
done

# Or run all at once:
for config in control no_baseline no_ci no_baseline_no_ci; do
  for i in {1..10}; do
    python -m src.main --config prompt_sensitivity/config_${config}.yaml
  done
done
```

## Analysis

Compare groups using:

1. **Fréchet ANOVA** on fitted Beta distributions (Wasserstein distance)
2. Measure ICC_F to see if prompt variation explains variance
3. p-value from permutation test

If p < 0.05, the prompt component significantly affects estimates.

## Output

Results go to `output_data/runs/` with timestamps. Move relevant runs to `prompt_sensitivity/output/` for analysis.