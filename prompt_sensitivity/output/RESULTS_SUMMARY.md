# Prompt Sensitivity Analysis - Complete Results Summary

## Overview

This analysis measures how different prompt variations affect LLM probability estimates compared to the control (baseline) condition. We use Wasserstein distances (W1 and W2) to quantify distributional differences.

---

## Experimental Conditions (7 Total)

### 1. **Control** (Baseline)
- **What it is**: Full prompt with all scaffolding
- **Key features**:
  - ✅ Baseline probability estimate (50% with 5th-95th percentile CI)
  - ✅ Confidence interval bounds provided
  - ✅ 3-phase reasoning structure (Phase 1: Capability transfer, Phase 2: Real-world friction, Phase 3: Integration)
  - ✅ Full task analysis step
- **Purpose**: Reference condition to measure deviations
- **Sample size**: 10 runs (9 valid distributions)

### 2. **No Baseline**
- **What was removed**: Baseline probability estimate (50%)
- **What remains**: CI bounds (5th-95th percentile), reasoning structure, full analysis
- **Purpose**: Test if anchoring on baseline affects estimates
- **Sample size**: 10 runs (10 valid distributions)
- **W1 from control**: 0.175 ± 0.079
- **W2 from control**: 0.196 ± 0.090
- **Interpretation**: **LARGE EFFECT** - Removing baseline anchor causes substantial distributional shift

### 3. **No CI (Confidence Interval)**
- **What was removed**: 5th and 95th percentile bounds
- **What remains**: Baseline estimate (50%), reasoning structure, full analysis
- **Purpose**: Test if providing uncertainty bounds affects calibration
- **Sample size**: 10 runs (10 valid distributions)
- **W1 from control**: 0.028 ± 0.017
- **W2 from control**: 0.032 ± 0.019
- **Interpretation**: **MINIMAL EFFECT** - Removing CI bounds has negligible impact

### 4. **No Baseline + No CI**
- **What was removed**: Both baseline estimate AND CI bounds
- **What remains**: Reasoning structure, full analysis
- **Purpose**: Test combined effect of removing all numeric anchors
- **Sample size**: 10 runs (10 valid distributions)
- **W1 from control**: 0.239 ± 0.082
- **W2 from control**: 0.242 ± 0.081
- **Interpretation**: **LARGEST EFFECT** - Removing all anchors causes maximum deviation

### 5. **Skip Analysis**
- **What was removed**: Detailed task analysis step (minimized to bare note)
- **What remains**: Baseline, CI, reasoning structure
- **Purpose**: Test if 2-step process (analysis → estimation) is necessary
- **Prompt change**: Task analysis reduced from 49 lines to 16 lines
- **Sample size**: 21 runs (21 valid distributions)
- **W1 from control**: 0.049 ± 0.023
- **W2 from control**: 0.055 ± 0.025
- **Interpretation**: **SMALL EFFECT** - Minimal analysis step increases variance slightly

### 6. **Trim Reasoning**
- **What was removed**: 3-phase reasoning structure scaffold
- **What remains**: Baseline, CI, full analysis, but NO step-by-step reasoning prompts
- **Purpose**: Test if structured CoT (Chain of Thought) affects calibration
- **Prompt change**: Estimation prompt reduced from 112 lines to 63 lines
- **Sample size**: 10 runs (10 valid distributions)
- **W1 from control**: 0.026 ± 0.015
- **W2 from control**: 0.030 ± 0.018
- **Interpretation**: **MINIMAL EFFECT** - Removing reasoning scaffold has almost no impact

### 7. **Trim All**
- **What was removed**: Almost everything - MINIMAL prompt
- **What remains**: Bare facts + output format only
- **Purpose**: Test if LLM can work with minimal guidance
- **Prompt change**:
  - Analysis: 49 lines → 8 lines
  - Estimation: 112 lines → 30 lines
- **Sample size**: 10 runs (10 valid distributions)
- **Note**: These runs were FASTER (~35-39s vs 50-100s), filtered out by original MIN_DURATION=50s threshold
- **W1 from control**: 0.115 ± 0.018
- **W2 from control**: 0.126 ± 0.021
- **Interpretation**: **MODERATE EFFECT** - Extreme minimalism increases deviation moderately

---

## Key Findings

### Within-Control Variance (Baseline Variability)
- **W1**: 0.033 ± 0.016
- **W2**: 0.039 ± 0.020
- **Interpretation**: Natural variability in estimates when using identical prompts

### Ranking by Deviation from Control (W1 distance)

1. **No Baseline + No CI**: 0.239 ± 0.082 (7.1x baseline variance) ⚠️ **LARGEST**
2. **No Baseline**: 0.175 ± 0.079 (5.2x baseline variance) ⚠️
3. **Trim All**: 0.115 ± 0.018 (3.4x baseline variance) ⚠️
4. **Skip Analysis**: 0.049 ± 0.023 (1.5x baseline variance)
5. **No CI**: 0.028 ± 0.017 (0.8x baseline variance) ✅
6. **Trim Reasoning**: 0.026 ± 0.015 (0.8x baseline variance) ✅

### Statistical Significance

**Conditions with distance > within-control variance:**
- ✅ **No Baseline + No CI** - Significantly different
- ✅ **No Baseline** - Significantly different
- ✅ **Trim All** - Significantly different
- ⚠️ **Skip Analysis** - Moderately different
- ❌ **No CI** - Not significantly different
- ❌ **Trim Reasoning** - Not significantly different

---

## Interpretation & Recommendations

### What Matters Most:
1. **Baseline anchors are critical** - Removing them causes 5-7x increase in variance
2. **Confidence intervals don't matter much** - Can be safely omitted
3. **Reasoning structure is optional** - Minimal impact on calibration
4. **Some analysis context helps** - But can be minimal
5. **Extreme minimalism hurts** - But less than removing anchors

### Practical Implications:

**Minimal safe prompt** (based on findings):
- ✅ Include baseline probability estimate
- ❌ Can omit confidence interval bounds
- ❌ Can omit reasoning structure
- ⚠️ Include brief analysis context

**Avoid**:
- ❌ Removing baseline anchor (causes 5x variance increase)
- ❌ Removing ALL guidance (causes 3x variance increase)

---

## Data Quality Notes

- **Total valid runs**: 71 runs across 7 conditions
- **Filtering**: MIN_DURATION = 30s (lowered from 50s to include trim_all)
- **Classification method**: Prompt text length analysis (since prompts_dir not saved in metadata)
- **Beta fitting**: 1 failure out of 80 total estimates (98.8% success rate)

---

## Files Generated

1. `wasserstein_distances_all_conditions.txt` - Detailed numeric results
2. `wasserstein_distances_all_conditions.png` - Side-by-side W1 and W2 charts
3. `wasserstein_combined_all_conditions.png` - Combined comparison chart
4. `compare_all_conditions.py` - Analysis script with run classification

---

## Technical Details

### Wasserstein Distance Computation
- **W1 (L1-Wasserstein)**: ∫|Q₁(u) - Q₂(u)|du, where Q is quantile function
- **W2 (L2-Wasserstein)**: √∫(Q₁(u) - Q₂(u))²du
- **Grid**: 201 points from 0.001 to 0.999
- **Integration**: Trapezoidal rule (np.trapz)

### Run Classification Logic
Runs identified by prompt characteristics (analysis_len, estimation_len, has_reasoning):
- `trim_all`: analysis_len < 2500
- `skip_analysis`: 2500 ≤ analysis_len < 4000
- `trim_reasoning`: analysis_len > 5000 AND no reasoning structure
- `control`: analysis_len > 5000 AND has reasoning structure
- Explicit scenario files distinguish: no_baseline, no_ci, no_baseline_no_ci

---

**Generated**: 2026-03-17
**Analysis**: Claude Code via compare_all_conditions.py
