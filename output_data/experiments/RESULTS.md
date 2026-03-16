# LLM Elicitation Expert Variance Analysis

## Research Questions

Testing whether LLM expert personas replicate human expert variance patterns:

1. **H1**: Do quantity estimates (num_actors) show more expert variance than probability estimates?
2. **H2**: Does expert variance for quantities increase with benchmark task difficulty?
3. **H3**: For the same attack step, does expert variance on probability estimates remain stable across benchmark task difficulty?

---

## Summary

| Hypothesis | Human Experts | GPT-4o | Claude | Gemini |
| ---------- | ------------- | ------ | ------ | ------ |
| **H1**: Quantities > Probabilities variance | ✓ | ✓ | ✓ | ✓ |
| **H2**: Quantity variance ↑ with difficulty | ✓ | ✗ | ✗ | ✗ |
| **H3**: Probability variance stable across difficulty | ✓ | ✓ | ✓ | — |

---

## Method

**Metric**: CV (Coefficient of Variation) of expert-mean p50 values

```
1. Each expert persona gives ~10 estimates across runs
2. Compute mean p50 per expert → 10 values
3. CV = std(expert means) / mean(expert means) × 100%
```

- **Expert variance** = disagreement between personas (not within-persona sampling noise)
- Each model evaluated independently (no mixing of outputs between models)
- 10 expert personas, ~10 runs per experiment

---

## Experimental Design

### Benchmark Tasks (by difficulty)

| Difficulty | Task | CVSS Score |
| ---------- | ---- | ---------- |
| LOW | cURL | 5.3 |
| MED | Imaginairy | 7.5 |
| HIGH | MLFlow0 | 10.0 |

### H1 & H2: Quantity Elicitation
- **Target**: Number of actors capable of executing attack scenario
- **Experiments**: 3 models × 3 difficulty levels = 9 experiments

### H3: Probability Elicitation
- **Target**: Probability of successful attack step execution
- **Attack step**: TA0002 (Execution) — held constant
- **Baseline**: 50% — held constant
- **Varying**: Benchmark task difficulty (LOW/MED/HIGH)
- **Experiments**: 3 models × 3 difficulty levels = 9 experiments

---

## Results

### H1: Quantities > Probabilities

| Model | Quantity CV | Probability CV | Ratio |
| ----- | ----------- | -------------- | ----- |
| GPT-4o | 4.7% | 1.9% | 2.5× |
| Gemini | 6.0% | 2.2% | 2.7× |
| Claude | 6.2% | 1.7% | 3.7× |

All three models show ~3× higher expert variance for quantities than probabilities, matching human pattern.

---

### H2: Quantity Variance vs Difficulty

| Model | LOW (CVSS 5.3) | MED (CVSS 7.5) | HIGH (CVSS 10.0) | Trend |
| ----- | -------------- | -------------- | ---------------- | ----- |
| GPT-4o | 4.8% | 4.8% | 4.6% | Flat |
| Gemini | 6.5% | 5.5% | 6.0% | Non-monotonic |
| Claude | 8.6% | 4.4% | 5.6% | Non-monotonic |

Unlike humans, LLM expert variance does not increase with task difficulty.

---

### H3: Probability Variance vs Difficulty

| Model | LOW (CVSS 5.3) | MED (CVSS 7.5) | HIGH (CVSS 10.0) | Trend |
| ----- | -------------- | -------------- | ---------------- | ----- |
| GPT-4o | 1.96% | 1.62% | 1.89% | Stable |
| Claude | 1.07% | 0.69% | 1.18% | Stable |
| Gemini | 2.21% | 3.53% | [PLACEHOLDER] | — |

CV ranges: GPT-4o = 0.34%, Claude = 0.49%. Expert variance on probability estimates remains stable across difficulty, matching human pattern.

---

## Notes

- **Why CV?** Paper states "greater variance" (absolute spread), not proportion. CV is scale-independent.
- **H3 design choice**: Fixed baseline (50%) and attack step (TA0002) isolates the effect of benchmark task difficulty. Varying baseline would confound results due to ceiling/floor effects.
- **Gemini H3 HIGH**: Runs failed, placeholder for later.
