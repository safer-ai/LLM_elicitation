# NumActors Baseline Uplift Experiment

## Goal

Test how LLM quantity estimates (# of actors) change as a function of the **baseline number of actors** provided in the prompt. This complements the probability baseline experiment by testing an **unbounded** quantity (no ceiling effect expected).

## Research Questions

1. Does the linear anchoring pattern from probability estimates also appear for quantities?
2. Without a ceiling (unlike probabilities bounded at 100%), does uplift remain constant?
3. Is the anchoring behavior consistent across different types of estimates?

## Experimental Design

| Variable | Value |
|----------|-------|
| Model | Claude Sonnet 4.6 |
| Expert | 1 (AI/ML Security Researcher) |
| Metric | Number of threat actor groups (ScenarioLevelMetric_NumActors) |
| Baselines | 3, 5, 10, 20, 50 actors |
| Runs per baseline | 5 |
| Total experiments | 25 |

### What Changes Between Conditions

ONLY this line changes in the scenario:
```
- The baseline estimation of the number of active actor groups fitting this profile is X.
```
Where X = {3, 5, 10, 20, 50}

Everything else is identical.

## Expected Results

**If anchoring holds (like probability experiment):**
- Output ≈ Baseline + constant uplift
- Linear relationship between baseline and output

**Key difference from probability:**
- No ceiling effect (# actors is unbounded)
- If we still see linear anchoring, it confirms the behavior is not due to ceiling effects

## Running the Experiment

```bash
cd LLM_elicitation

# Run all baselines (5 runs each = 25 total)
python3 output_data/experiments/baseline_uplift/numactors/run_experiment.py

# Run specific baseline
python3 output_data/experiments/baseline_uplift/numactors/run_experiment.py --baseline 10

# Dry run
python3 output_data/experiments/baseline_uplift/numactors/run_experiment.py --dry-run
```
