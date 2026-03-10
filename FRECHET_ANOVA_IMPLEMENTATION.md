# Fréchet ANOVA Implementation Documentation

**Date:** 2026-03-09
**Branch:** spar_spring_2026_variance (OLD min/max/confidence elicitation format)
**Status:** COMPLETE - Results generated, needs re-implementation for percentile format

---

## Executive Summary

This document captures the complete Fréchet ANOVA implementation for distributional variance analysis in LLM-elicited expert panels. The current implementation uses the OLD elicitation format (min/max/confidence). Future work will adapt this for the NEW percentile format (p25/p50/p75).

---

## 1. Experimental Design

### 1.1 Research Questions
1. Does persona assignment systematically shift probability estimates (distributional analysis)?
2. Does model identity generate significant distributional variance?

### 1.2 Scenario & Attack Steps
- **Scenario:** OC3 ransomware targeting large enterprise ($250M-$1B revenue, MIL2/MIL3 maturity)
- **Benchmark:** Imaginairy DoS vulnerability detection (capability ceiling)

**Attack Steps Tested:**
- **TA0002 - Execution** (50% baseline)
- **TA0007 - Discovery** (85% baseline)
- **T1657 - Financial Theft** (30% baseline)

### 1.3 Personas (n=10)
1. Defensive Security Specialist
2. Malware Reverse Engineer
3. AI/ML Security Researcher
4. Threat Intelligence Analyst
5. Security Software Developer
6. Incident Response Specialist
7. Academic Security Researcher
8. Red Team Operator
9. CTF Competition Expert
10. Security Compliance Officer

### 1.4 Models (n=3)
1. Claude Sonnet 4.5
2. GPT-4o
3. Gemini 2.5 Pro

All models use temperature=1.0, single Delphi round (no deliberation).

### 1.5 Sample Size
- **Persona experiments:** 10 personas × 10 runs = 100 estimates per (model, step) combination
- **Cross-model experiments:** 3 models × ~100 estimates = ~300 per step
- **Total:** 6 persona experiments + 3 cross-model experiments = 9 total experiments

---

## 2. Elicitation Format (CURRENT - OLD)

### 2.1 LLM Output Format
```
Final probability: 0.XX (most likely estimate - NOT USED in Fréchet ANOVA)
Minimum probability: 0.XX
Maximum probability: 0.XX
Confidence in range: 0.XX
```

### 2.2 CSV Schema
```csv
run_id,timestamp_start,model,temperature,step_name,task_name,round,expert_name,
most_likely_estimate,minimum_estimate,maximum_estimate,confidence_in_range,
rationale,has_error,error_message,task_metric,...
```

**Key fields:**
- `minimum_estimate`: Lower bound of confidence interval
- `maximum_estimate`: Upper bound of confidence interval
- `confidence_in_range`: Confidence level (typically 0.75 or 0.85)
- `most_likely_estimate`: Point estimate (NOT used in distributional analysis)

### 2.3 Data Locations
```
output_data/experiments/
├── anova_probability/              # Claude, TA0002 (50%)
├── pilot_experiments/
│   ├── cross_step_TA0007_85pct/   # Claude, TA0007 (85%)
│   ├── cross_step_T1657_30pct/    # Claude, T1657 (30%)
│   ├── cross_model_gpt4o/         # GPT-4o, TA0002 (50%)
│   ├── cross_model_gpt4o_TA0007_85pct/
│   ├── cross_model_gpt4o_T1657_30pct/
│   ├── cross_model_gemini_TA0002_50pct/
│   ├── cross_model_gemini_TA0007_85pct/
│   └── cross_model_gemini_T1657_30pct/
└── frechet_anova/                 # Analysis code + results
```

---

## 3. Beta Distribution Fitting

### 3.1 Fitting Method
**Approach:** Fit Beta(α, β) on [0,1] using **symmetric confidence intervals** with equal tail probabilities.

**Constraints:**
- `CDF(min) = (1 - confidence) / 2`
- `CDF(max) = (1 + confidence) / 2`

**Example:**
- min=0.40, max=0.60, confidence=0.80
- CDF(0.40) = 0.10 (10th percentile)
- CDF(0.60) = 0.90 (90th percentile)

### 3.2 Solver Details
**File:** `frechet_anova/frechet_anova.py::fit_beta_from_elicitation()`

**Method:** Numerical root-finding with `scipy.optimize.fsolve`

**System of equations:**
```python
def equations(params):
    a, b = params
    eq1 = sp_stats.beta.cdf(lo, a, b) - target_lo_cdf
    eq2 = sp_stats.beta.cdf(hi, a, b) - target_hi_cdf
    return [eq1, eq2]
```

**Initial guesses:** 16 different (α, β) pairs to ensure convergence:
```python
init_guesses = [
    (2.0, 2.0), (5.0, 5.0), (1.5, 1.5), (10.0, 10.0),
    (3.0, 8.0), (8.0, 3.0), (1.2, 0.8), (0.8, 1.2),
    (1.5, 3.0), (3.0, 1.5), (15.0, 5.0), (5.0, 15.0),
    (0.5, 0.5), (1.0, 2.0), (2.0, 1.0), (20.0, 20.0),
]
```

**Boundary handling:**
- Clamp bounds: `lo = max(lo, 0.005)`, `hi = min(hi, 0.995)`
- Avoid exact 0 and 1 (CDF undefined at boundaries)

**Convergence criterion:** `residual < 1e-6`

### 3.3 Default Confidence
If `confidence_in_range` is missing or invalid: `DEFAULT_CONFIDENCE = 0.85`

---

## 4. Wasserstein Distance

### 4.1 Why W2 (not W1)?

**Source:** Dubey & Müller (2019), "Fréchet Analysis of Variance for Random Objects"
- arXiv: https://arxiv.org/pdf/1710.02761

**Mathematical Reason:**
Fréchet ANOVA is built on Fréchet variance:
```
Var_F(Y) = E[d²(μ, Y)]
```

For W2 (2-Wasserstein distance):
```
W2²(P, Q) = ∫₀¹ [F_P⁻¹(u) - F_Q⁻¹(u)]² du
```

**Key property:** W2² equals the **L2 squared norm of quantile function differences**.

This makes W2² the natural choice because:
1. Fréchet variance requires d² (squared distance)
2. W2² provides this via quantile function L2 norm
3. Enables exact variance decomposition: `V_total = V_within + V_between`
4. W1 would NOT satisfy this (no Hilbert space structure)

### 4.2 W1 vs W2 Usage

**In the code:**
- **W2² (L2-Wasserstein squared):** Used for Fréchet ANOVA test statistic (lines 134-136, 148-154)
- **W1 (L1-Wasserstein):** Used only for descriptive statistics in results (lines 139-141, 330-348)

**Why both?**
- W2: Theoretically required for ANOVA
- W1: More interpretable for reporting (average difference in quantiles)

### 4.3 Implementation

**Quantile grid:** 201 points from 0.001 to 0.999
```python
QUANTILE_GRID_SIZE = 201
Q_GRID = np.linspace(0.001, 0.999, QUANTILE_GRID_SIZE)
```

**Why 201?**
- Numerical integration accuracy vs. computational cost trade-off
- Avoids exact 0 and 1 where Beta quantile function diverges
- Could use 101 (faster, less accurate) or 501 (slower, more accurate)

**W2² computation:**
```python
def w2_squared(qf1: np.ndarray, qf2: np.ndarray) -> float:
    """L2-Wasserstein squared distance between two distributions."""
    return np.trapz((qf1 - qf2) ** 2, Q_GRID)
```

**W1 computation:**
```python
def w1_distance(qf1: np.ndarray, qf2: np.ndarray) -> float:
    """L1-Wasserstein distance between two distributions."""
    return np.trapz(np.abs(qf1 - qf2), Q_GRID)
```

**Quantile function:**
```python
def quantile_function(alpha: float, beta: float, grid: np.ndarray = Q_GRID) -> np.ndarray:
    """Compute the quantile function of Beta(alpha, beta) on the grid."""
    return sp_stats.beta.ppf(grid, alpha, beta)
```

---

## 5. Fréchet ANOVA Test Statistic

### 5.1 Decomposition

**Pooled Fréchet variance:**
```
V_pooled = (1/n) Σᵢ W2²(Qᵢ, μ_pooled)
```

**Within-group variance:**
```
V_within = Σⱼ λⱼ · [(1/nⱼ) Σᵢ∈Gⱼ W2²(Qᵢ, μⱼ)]
```
where λⱼ = nⱼ/n

**Between-group variance component:**
```
F_n = V_pooled - V_within
```

**Heterogeneity term:**
```
U_n = Σᵢ<ⱼ λᵢ λⱼ (V_within,i - V_within,j)²
```

**Test statistic:**
```
T_n = n · F_n + n · U_n
```

### 5.2 Fréchet ICC

**Intraclass correlation coefficient (distributional):**
```
ICC_F = F_n / V_pooled
```

**Interpretation:**
- ICC_F = 0: No between-group variance (null hypothesis)
- ICC_F = 1: All variance is between groups
- ICC_F ∈ [0, 1]: Proportion of total distributional variance explained by grouping factor

**Comparison to scalar ICC:**
Scalar ICC uses point estimates only; Fréchet ICC captures full distributional differences.

### 5.3 Statistical Tests

**Permutation test (PRIMARY):**
- 5000 permutations
- Randomly shuffle group labels
- p-value = (# permutations with T_perm ≥ T_obs + 1) / (5000 + 1)
- **This is the valid test at small sample sizes**

**Asymptotic test (REFERENCE ONLY):**
- T_n ~ χ²(k-1) under H₀ (large n)
- Unreliable at n ~ 100
- Shown for comparison but NOT used for inference

---

## 6. Key Results (OLD Format)

### 6.1 Persona ANOVA (NOT Significant)

| Model | Step | N | T_n | p (perm) | ICC_F | Mean W1 (within) | Mean W1 (between) |
|-------|------|---|-----|----------|-------|------------------|-------------------|
| Claude | TA0002 (50%) | 100 | 0.1039 | 0.1204 | 0.133 | 0.0932 | 0.0950 |
| Claude | TA0007 (85%) | 90 | 0.0058 | 0.6499 | 0.085 | 0.0296 | 0.0294 |
| Claude | T1657 (30%) | 97 | 0.0070 | 0.7722 | 0.068 | 0.0328 | 0.0325 |
| GPT-4o | TA0002 (50%) | 100 | 0.0628 | 0.0628 | 0.139 | 0.0722 | 0.0747 |
| GPT-4o | TA0007 (85%) | 100 | 0.0041 | 0.6913 | 0.075 | 0.0248 | 0.0248 |
| GPT-4o | T1657 (30%) | 100 | 0.0205 | 0.2466 | 0.111 | 0.0473 | 0.0476 |

**Interpretation:**
- 5/6 experiments: p > 0.05 (not significant)
- Persona explains 7-14% of distributional variance
- Within-persona W1 ≈ Between-persona W1 (personas don't differ)

### 6.2 Cross-Model ANOVA (SIGNIFICANT)

| Step | N | T_n | p (perm) | ICC_F | Mean W1 (between) |
|------|---|-----|----------|-------|-------------------|
| TA0002 (50%) | 300 | 0.7991 | 0.0002 | 0.271 | 0.1164 |
| TA0007 (85%) | 290 | 0.1959 | 0.0002 | 0.417 | 0.0462 |
| T1657 (30%) | 297 | 0.2721 | 0.0002 | 0.361 | 0.0640 |

**Interpretation:**
- All 3 experiments: p < 0.001 (highly significant)
- Model explains 27-42% of distributional variance
- Model identity matters far more than persona assignment

### 6.3 Ceiling/Floor Effects (IMPORTANT FINDING)

**Observation:** Variance compresses near boundaries.

At **85% baseline (TA0007):**
- Ceiling effect: Can't go much above 1.0
- Max variance = 0.85 × (1 - 0.85) = 0.127

At **30% baseline (T1657):**
- Floor effect: Can't go much below 0.0
- Max variance = 0.30 × (1 - 0.30) = 0.21

At **50% baseline (TA0002):**
- Maximum variance capacity = 0.50 × (1 - 0.50) = 0.25

**Mathematical property of bounded [0,1] distributions:**
```
Var_max = mean × (1 - mean)
```

**Interpretation for manuscript:**
> "At TA0007 (85%) and T1657 (30%), all 10 persona distributions collapse into nearly identical shapes due to **ceiling and floor effects** respectively: distributions compress near the boundaries of [0,1]. At TA0002 (50% baseline), personas show the widest spread, consistent with maximum variance capacity at mid-range probabilities."

---

## 7. Implementation Files

### 7.1 Core Files

**`frechet_anova/frechet_anova.py`** (553 lines)
- Beta distribution fitting from (min, max, confidence)
- Wasserstein distance computation (W1 and W2)
- Fréchet ANOVA test statistic
- Permutation test (5000 permutations)
- Data loading from CSV
- Results formatting

**`frechet_anova/plot_beta_distributions.py`** (195 lines)
- Persona distributions (3-panel: TA0002, TA0007, T1657)
- Cross-model distributions (3-panel)
- Full 3×3 grid (3 models × 3 steps)
- Uses mean Beta PDF across runs per persona

**`frechet_anova/plot_individual_runs_beta.py`** (not yet documented)
- Individual run visualization (optional)

### 7.2 Output Files

**Results:**
- `frechet_anova/frechet_anova_results.txt` - Full text report

**Figures:**
- `frechet_anova/beta_persona_distributions.png`
- `frechet_anova/beta_cross_model_distributions.png`
- `frechet_anova/beta_full_distribution_grid.png`

---

## 8. Adaptation Plan for Percentile Format

### 8.1 New Elicitation Format (from spar_spring_2026)

**LLM Output:**
```xml
<percentile_estimates>
<p25>0.xx</p25>
<p50>0.xx</p50>
<p75>0.xx</p75>
</percentile_estimates>
```

**CSV Schema (expected):**
```csv
...,p25,p50,p75,...
```

### 8.2 Required Code Changes

**1. Beta fitting function** (`fit_beta_from_elicitation`):
```python
# OLD: fit_beta_from_elicitation(lo, hi, confidence)
# NEW: fit_beta_from_percentiles(p25, p50, p75)
```

**New constraints:**
- `CDF(p25) = 0.25`
- `CDF(p50) = 0.50`
- `CDF(p75) = 0.75`

**Challenge:** 3 constraints, 2 parameters (α, β) → Overdetermined system
- Need to use 2 of 3 percentiles for fitting
- Recommended: Use (p25, p75) for fitting, validate with p50
- Alternative: Least-squares fit to all 3 percentiles

**2. Data loading** (`load_and_fit`):
```python
# Change CSV field names from:
lo_val = float(row["minimum_estimate"])
hi_val = float(row["maximum_estimate"])
conf_val = float(row["confidence_in_range"])

# To:
p25_val = float(row["p25"])
p50_val = float(row["p50"])
p75_val = float(row["p75"])
```

**3. Default handling:**
- Remove `DEFAULT_CONFIDENCE = 0.85`
- Add validation: p25 < p50 < p75

**4. Data paths:**
- Update `DATA_DIRS` dictionary to point to new experiment runs

### 8.3 Validation Strategy

After implementing new fitting:
1. **Sanity check:** Fit Beta to synthetic (p25, p50, p75), verify CDF values
2. **Goodness-of-fit:** For each fitted Beta, check if p50_obs ≈ p50_fitted
3. **Compare results:** Do conclusions change vs. old format?

---

## 9. References

### 9.1 Papers

**Fréchet ANOVA:**
- Dubey, P., & Müller, H. G. (2019). Fréchet Analysis of Variance for Random Objects. *Biometrika*, 106(4), 803-821.
- arXiv: https://arxiv.org/pdf/1710.02761

**Wasserstein Distance:**
- Panaretos, V. M., & Zemel, Y. (2019). Statistical Aspects of Wasserstein Distances. *Annual Review of Statistics and Its Application*, 6, 405-431.

**Beta Distribution Elicitation:**
- O'Hagan, A., et al. (2006). *Uncertain Judgements: Eliciting Experts' Probabilities*. Wiley.

### 9.2 Related Experiments

**Scalar ANOVA (point estimates only):**
- File: `output_data/experiments/experiment_report.tex`
- Uses `most_likely_estimate` field
- Results: Persona ICC = 0-14%, Model ICC = 28-65%

**Comparison:**
- Scalar ANOVA: Tests if group means differ
- Fréchet ANOVA: Tests if group **distributions** differ
- Fréchet is more powerful (uses full uncertainty information)

---

## 10. Known Issues & Future Work

### 10.1 Current Limitations

1. **Sample size:** n=100 per group is small for asymptotic theory
   - **Solution:** Use permutation test (already implemented)

2. **Missing data:** Some runs failed Beta fitting (residual > 1e-6)
   - **Reported:** "X/Y rows failed Beta fitting" warnings
   - **Impact:** Minimal (~5-10% failure rate)

3. **Quantile grid resolution:** 201 points may be overkill for smooth Beta distributions
   - **Future:** Benchmark 101 vs 201 vs 501 points

4. **W1 sampling:** For large groups, we sample max 500 pairs for computational speed
   - **Impact:** W1 descriptive stats are approximate (not used in inference)

### 10.2 Future Extensions

1. **Post-hoc tests:** Which persona pairs differ significantly? (currently not implemented)

2. **Effect size:** Standardize T_n for interpretability across experiments

3. **Visualization:** Add confidence bands around mean Beta PDFs

4. **Alternative distributions:** Try other families (Logit-Normal, Johnson SU) if Beta fit is poor

5. **Multi-factor ANOVA:** Test persona × model interaction (requires more data)

---

## 11. Quick Start (After Percentile Migration)

### 11.1 Rerun Experiments

```bash
# 1. Merge from spar_spring_2026 to get new prompts
git merge origin/spar_spring_2026

# 2. Run elicitation experiments (pseudocode)
python run_experiments.py --format percentile --personas 10 --runs 10

# 3. Verify CSV format
head -2 output_data/experiments/[NEW_DIR]/run_1_*/detailed_estimates.csv
# Should show: ...,p25,p50,p75,...
```

### 11.2 Update Fréchet ANOVA Code

```python
# In frechet_anova.py, implement:
def fit_beta_from_percentiles(p25, p50, p75):
    """Fit Beta(α, β) from three percentiles."""
    # Method 1: Use p25 and p75 (symmetric, standard approach)
    target_lo_cdf = 0.25
    target_hi_cdf = 0.75

    def equations(params):
        a, b = params
        eq1 = sp_stats.beta.cdf(p25, a, b) - 0.25
        eq2 = sp_stats.beta.cdf(p75, a, b) - 0.75
        return [eq1, eq2]

    # [Use same solver logic as fit_beta_from_elicitation]

    # Validate: Check if fitted CDF(p50) ≈ 0.50
    a_fit, b_fit = solution
    p50_fitted = sp_stats.beta.cdf(p50, a_fit, b_fit)
    if abs(p50_fitted - 0.50) > 0.05:
        print(f"Warning: p50 validation failed (CDF={p50_fitted:.3f})")

    return (a_fit, b_fit)
```

### 11.3 Run Analysis

```bash
# Run Fréchet ANOVA
python frechet_anova/frechet_anova.py

# Generate plots
python frechet_anova/plot_beta_distributions.py

# Check results
cat frechet_anova/frechet_anova_results.txt
```

### 11.4 Expected Changes

**Hypothesis:** Results should be qualitatively similar to old format:
- Persona: Still not significant (p > 0.05)
- Model: Still highly significant (p < 0.001)
- ICC magnitudes may differ slightly due to different fitting method

**If results diverge substantially:** Investigate whether (p25, p50, p75) provides tighter or looser uncertainty than (min, max, conf).

---

## 12. Contact & Maintenance

**Author:** Madhav (with Claude Code assistance)
**Date Created:** 2026-03-09
**Last Updated:** 2026-03-09
**Branch:** spar_spring_2026_variance

**Questions/Issues:**
- Consult this document first
- Check `frechet_anova_results.txt` for current results
- Review Dubey & Müller (2019) paper for theoretical foundation

---

## Appendix: Mathematical Details

### A.1 Beta Distribution CDF

For Beta(α, β) on [0,1]:
```
F(x; α, β) = I_x(α, β) = ∫₀ˣ tᵅ⁻¹(1-t)ᵝ⁻¹ dt / B(α, β)
```
where `B(α, β) = Γ(α)Γ(β)/Γ(α+β)` is the Beta function.

**Implemented via:** `scipy.stats.beta.cdf(x, a, b)`

### A.2 Beta Distribution Quantile Function

For Beta(α, β) on [0,1]:
```
F⁻¹(p; α, β) = Q_Beta(p; α, β)
```
Computed by inverting the CDF numerically.

**Implemented via:** `scipy.stats.beta.ppf(p, a, b)`

### A.3 Wasserstein Distance Formula

For two probability distributions P and Q with quantile functions Q_P and Q_Q:

**L1-Wasserstein (W1):**
```
W1(P, Q) = ∫₀¹ |Q_P(u) - Q_Q(u)| du
```

**L2-Wasserstein squared (W2²):**
```
W2²(P, Q) = ∫₀¹ [Q_P(u) - Q_Q(u)]² du
```

**Numerical approximation:** Trapezoidal rule over quantile grid
```python
W2² ≈ Σᵢ [Q_P(uᵢ) - Q_Q(uᵢ)]² · Δu
```
where u ∈ {0.001, 0.002, ..., 0.999} (201 points), Δu ≈ 0.005

### A.4 Fréchet Mean

For distributions {Q₁, ..., Qₙ} with Wasserstein distance:
```
μ_F = argmin_Q Σᵢ W2²(Qᵢ, Q)
```

**For Beta distributions on [0,1]:**
The Fréchet mean is approximated as the pointwise mean of quantile functions:
```
μ_F(u) ≈ (1/n) Σᵢ Qᵢ(u)
```

**Implemented via:** `qf_matrix.mean(axis=0)` (line 164 in frechet_anova.py)

---

**End of Document**
