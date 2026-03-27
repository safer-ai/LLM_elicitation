# Baseline Uplift Curve Experiment

## Goal

Test how LLM probability estimates change as a function of the **baseline probability** provided in the prompt. This helps understand anchoring effects and ceiling effects in LLM probability elicitation.

## Research Questions

1. Do LLMs anchor heavily to the provided baseline?
2. Is the uplift additive (constant) or multiplicative (percentage)?
3. Is there a ceiling effect at high baselines (bounded by 100%)?
4. What is the shape of the baseline-output curve?

## Experimental Design (from meeting discussion)

Key design decisions based on meeting notes:

1. **"Cut out the CIs"** - No confidence interval values included
2. **"Vary the baseline"** - 10%, 20%, 30%... 90%
3. **"Rest of prompt stays fixed"** - Only baseline % changes
4. **"Upper triangular curve"** - Expected shape due to ceiling effects
5. **"Find one without too many other numbers"** - TA0002 section is clean

### Variables

| Variable | Value |
|----------|-------|
| Model | Claude Sonnet 4.6 |
| Expert | 1 (AI/ML Security Researcher) |
| Step | TA0002 - Execution |
| Baselines | 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90% |
| CI values | None (removed to isolate baseline effect) |
| Runs per baseline | 10 |
| Total experiments | 90 |

### What Changes Between Conditions

ONLY this line changes:
```
- The threat actor alone, with no help from the LLM, has a X% chance of success at this step.
```
Where X = {10, 20, 30, 40, 50, 60, 70, 80, 90}

Everything else is identical.

## Expected Results

**Upper triangular curve:**
- At low baselines (10-30%): LLM can predict significant uplift
- At high baselines (70-90%): Ceiling effect - can't exceed 100%
- The curve should asymptotically flatten approaching 100%

## Future Extensions (from meeting discussion)

1. **SWEBench variant**: Repeat with SWEBench benchmark
2. **Quantity nodes**: Test # of actors (unbounded, no ceiling effect expected)
3. **Multiple models**: Compare GPT-4o, Gemini, Claude

## File Structure

```
baseline_uplift/
├── README.md                    # This file
├── setup_experiment.py          # Generates scenario/config files
├── run_experiment.py            # Runs experiments
├── analyze_results.py           # Analyzes results, generates plots
├── single_expert.yaml           # Expert profile
├── scenarios/                   # 9 scenario files (baseline varies)
│   ├── scenario_baseline_10.yaml
│   └── ...
├── configs/                     # 9 config files
│   ├── config_baseline_10.yaml
│   └── ...
└── results/                     # Output from experiments
```

## Running the Experiment

### 1. Add API Keys

Edit config files in `configs/` to add your API key, OR set environment variable:
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### 2. Run Experiments

From the `LLM_elicitation` directory:

**Single run (manual):** use `src/main.py` so imports resolve (`python -m src.main` will fail with `No module named 'config'`):

```bash
cd LLM_elicitation
python3 src/main.py -c output_data/experiments/baseline_uplift/configs/config_baseline_10.yaml
```

**Batch runner:**

```bash
# Run all experiments (90 total)
python3 output_data/experiments/baseline_uplift/run_experiment.py

# Run only a specific baseline
python3 output_data/experiments/baseline_uplift/run_experiment.py --baseline 50

# Dry run (see commands without executing)
python3 output_data/experiments/baseline_uplift/run_experiment.py --dry-run
```

### 3. Analyze Results

```bash
python3 output_data/experiments/baseline_uplift/analyze_results.py
```

Generates:
- `baseline_uplift_results.csv` - Summary statistics
- `baseline_uplift_curve.png` - Visualization
- `baseline_uplift_curve.pdf` - Publication-ready figure

## Connection to Previous Work

Extends the `prompt_sensitivity/` experiments which tested presence/absence of baseline.
This experiment asks: **What happens as we vary the baseline value?**
