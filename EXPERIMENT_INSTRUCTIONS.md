# Fréchet ANOVA Experiments - Percentile Format

## Overview

This directory contains all files needed to run 9 LLM elicitation experiments for Fréchet ANOVA analysis.

**Experimental Design:**
- **9 experiments** = 3 models × 3 attack steps
- **Per experiment:** 10 runs × 10 expert personas = 100 estimates
- **Total:** 900 estimates across all experiments

## Files Created

### Config Files (9 total)
- `config_claude_TA0002.yaml` - Claude Sonnet 4.5, TA0002 Execution (50%)
- `config_claude_TA0007.yaml` - Claude Sonnet 4.5, TA0007 Discovery (85%)
- `config_claude_T1657.yaml` - Claude Sonnet 4.5, T1657 Financial Theft (30%)
- `config_gpt4o_TA0002.yaml` - GPT-4o, TA0002 Execution (50%)
- `config_gpt4o_TA0007.yaml` - GPT-4o, TA0007 Discovery (85%)
- `config_gpt4o_T1657.yaml` - GPT-4o, T1657 Financial Theft (30%)
- `config_gemini_TA0002.yaml` - Gemini 2.5 Pro, TA0002 Execution (50%)
- `config_gemini_TA0007.yaml` - Gemini 2.5 Pro, TA0007 Discovery (85%)
- `config_gemini_T1657.yaml` - Gemini 2.5 Pro, T1657 Financial Theft (30%)

### Automation Scripts
- `run_single_experiment.sh` - Run one experiment 10 times
- `run_all_experiments.sh` - Run all 9 experiments automatically

## Running Experiments

### Option 1: Run All Experiments (Recommended)

```bash
./run_all_experiments.sh
```

This will:
- Run all 9 experiments sequentially
- Each experiment runs 10 times
- Automatically organize results into `output_data/experiments/percentile_*/`
- Takes several hours to complete

### Option 2: Run Individual Experiment

```bash
./run_single_experiment.sh <config_file> <experiment_name>
```

Example:
```bash
./run_single_experiment.sh config_claude_TA0002.yaml percentile_claude_TA0002_50pct
```

### Option 3: Manual Single Run (for testing)

```bash
python3 src/main.py -c config_claude_TA0002.yaml
```

Results will be in `output_data/runs/<timestamp>/`

## Output Structure

After running experiments, the directory structure will be:

```
output_data/experiments/
├── percentile_claude_TA0002_50pct/
│   ├── run_1_20260309_HHMMSS/
│   │   ├── detailed_estimates.csv  (10 experts × p25/p50/p75)
│   │   └── full_results.json
│   ├── run_2_20260309_HHMMSS/
│   ├── ...
│   └── run_10_20260309_HHMMSS/
├── percentile_claude_TA0007_85pct/
├── percentile_claude_T1657_30pct/
├── percentile_gpt4o_TA0002_50pct/
├── percentile_gpt4o_TA0007_85pct/
├── percentile_gpt4o_T1657_30pct/
├── percentile_gemini_TA0002_50pct/
├── percentile_gemini_TA0007_85pct/
└── percentile_gemini_T1657_30pct/
```

## CSV Output Format

Each `detailed_estimates.csv` contains:

```csv
run_id,timestamp_start,model,temperature,step_name,task_name,round,expert_name,
percentile_25th,percentile_50th,percentile_75th,rationale,has_error,error_message,...
```

**Key fields for Fréchet ANOVA:**
- `percentile_25th` (p25): Lower quartile
- `percentile_50th` (p50): Median
- `percentile_75th` (p75): Upper quartile

## Experiment Parameters

**Common settings across all experiments:**
- `temperature: 1.0` - High temperature for diversity
- `num_experts: 10` - All 10 expert personas
- `num_tasks: 1` - Only "Imaginairy" benchmark task
- `delphi_rounds: 1` - No deliberation (single round)
- `thinking: false` - Extended thinking disabled

**Models:**
- Claude Sonnet 4.5: `claude-sonnet-4-5`
- GPT-4o: `gpt-4o`
- Gemini 2.5 Pro: `gemini-2.5-pro`

**Attack Steps:**
- TA0002 - Execution (50% baseline)
- TA0007 - Discovery (85% baseline)
- T1657 - Impact: Financial Theft / Extortion (30% baseline)

## Troubleshooting

### API Key Issues
- Check that all three API keys are valid in config files
- Verify API credits/quotas are sufficient

### Rate Limiting
- Scripts include delays between runs (2s) and experiments (5s)
- Adjust `rate_limit_calls` in config if needed

### Failed Runs
- Check `output_data/runs/` for error logs
- Review `has_error` column in CSV files
- Re-run specific experiment with `run_single_experiment.sh`

## Next Steps After Data Collection

1. **Verify Data Quality**
   ```bash
   # Check all experiments have 10 runs
   find output_data/experiments/percentile_* -type d -name "run_*" | wc -l
   # Should output: 90 (9 experiments × 10 runs)

   # Check each CSV has 10 expert rows (excluding header)
   for csv in output_data/experiments/percentile_*/run_*/detailed_estimates.csv; do
       echo "$csv: $(tail -n +2 "$csv" | wc -l) rows"
   done
   # Each should output: 10 rows
   ```

2. **Run Fréchet ANOVA Analysis**
   - Implement `fit_beta_from_percentiles(p25, p50, p75)` function
   - Adapt `frechet_anova/frechet_anova.py` for new format
   - Generate results and plots

3. **Compare with Old Format**
   - Compare Fréchet ICC between percentile vs min/max/confidence
   - Validate that conclusions remain consistent

## Estimated Costs & Runtime

**Per experiment (approximate):**
- 10 runs × 10 experts = 100 API calls
- ~2 prompts per expert (analysis + estimation)
- ~200 API calls per experiment

**Total across all 9 experiments:**
- ~1,800 API calls
- Runtime: 6-10 hours (with rate limiting)
- Cost: ~$50-150 depending on model pricing

## References

See `FRECHET_ANOVA_IMPLEMENTATION.md` for:
- Complete experimental design rationale
- Statistical methodology
- Beta distribution fitting details
- Wasserstein distance computation
