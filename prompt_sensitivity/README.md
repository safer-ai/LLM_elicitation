# Prompt Sensitivity Experiments

## Conditions

| Condition | What Was Removed |
|-----------|------------------|
| **control** | Nothing (full prompt) |
| **no_ci** | Removed uncertainty range (5th/95th percentile: "20%-90%") from baseline |
| **no_baseline** | Removed point estimate ("50% chance of success") from baseline |
| **no_baseline_no_ci** | Removed entire baseline statement |
| **skip_analysis** | Removed detailed capability analysis instructions (4-section breakdown of task difficulty, capability correlation, boundaries) |
| **trim_reasoning** | Removed 3-phase reasoning scaffold (Phase 1: establish ranges, Phase 2: check confidence, Phase 3: reality check) |
| **trim_all** | Removed capability analysis + reasoning scaffold + technical analysis output |

## Results

| Condition | W1 Distance | W² Variance (×Control) |
|-----------|-------------|------------------------|
| no_ci | 0.013 | 0.36× |
| no_baseline | 0.154 | 18.1× |
| no_baseline_no_ci | 0.230 | 12.3× |
| skip_analysis | 0.041 | 0.78× |
| trim_reasoning | 0.011 | 0.30× |
| trim_all | 0.016 | 0.07× |

## Findings

1. **Baseline anchoring**: Removing "50%" baseline causes large distribution shift (W1=0.15-0.23) and explodes variance (12-18×)
2. **CI has minimal effect**: Removing uncertainty range alone has negligible impact (W1=0.013, variance 0.36×)
3. **Reasoning scaffold reduces variance**: trim_reasoning and trim_all have lowest variance (0.30× and 0.07×) - model converges to consistent answers
4. **trim_all extreme convergence**: All 10 runs produced nearly identical outputs (0.07× variance)
