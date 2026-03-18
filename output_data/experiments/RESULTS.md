# LLM Elicitation Expert Variance Analysis

## Model Choice vs Expert Persona Variance

**Question**: Does model choice (GPT-4o vs Claude vs Gemini) create more variance than expert persona choice?

**Method**: Fréchet ANOVA with Wasserstein distance on fitted distributions. ICC_F = proportion of variance explained by grouping factor.

| Estimate Type | Persona ICC_F | Model ICC_F | Ratio | Model p-value |
| ------------- | ------------- | ----------- | ----- | ------------- |
| Probability | 0.12 | 0.58 | ~5× | <0.001 |
| Quantity (num_actors) | 0.09 | 0.24 | ~2.6× | 0.005 |

**Conclusion**: Model choice creates significantly more variance than persona choice for both estimate types.

**Limitation**: Quantity analysis uses PERT fitting (bounded), which may not suit unbounded integers.

---

## Summary

| Hypothesis | Human | GPT-4o | Claude | Gemini |
| ---------- | ----- | ------ | ------ | ------ |
| **H1**: Quantities > Probabilities variance | ✓ | ✓ | ✓ | ✓ |
| **H2**: Quantity variance ↑ with difficulty | ✓ | ✗ | ✗ | ✗ |
| **H3**: Probability variance stable across difficulty | ✓ | ✓ | ✓ | — |

**Metric**: CV (Coefficient of Variation) of expert-mean p50 values. Measures disagreement between 10 expert personas. Each model evaluated independently.

---

## H1: Quantities Show More Expert Variance Than Probabilities

**Question**: Do quantity estimates (num_actors) show more expert variance than probability estimates?

**Design**:
- Quantity CV: averaged across 3 difficulty levels (LOW/MED/HIGH)
- Probability CV: averaged across 3 attack steps with different baselines (T1657 30%, TA0002 50%, TA0007 85%)
- Benchmark tasks: cURL (CVSS 5.3), Imaginairy (CVSS 7.5), MLFlow0 (CVSS 10.0)

**Results**:

| Model | Quantity CV | Probability CV | Ratio |
| ----- | ----------- | -------------- | ----- |
| GPT-4o | 4.7% | 1.9% | 2.5× |
| Gemini | 6.0% | 2.2% | 2.7× |
| Claude | 6.2% | 1.7% | 3.6× |

**Conclusion**: All three models show higher variance for quantities. Matches human pattern.

**Limitation**: Probability CV averages across different baselines (30%, 50%, 85%), which may not be directly comparable due to ceiling/floor effects. However, the ~3× ratio is large enough that this is unlikely to change the conclusion.

---

## H2: Quantity Variance Increases With Difficulty

**Question**: Does expert variance for quantity estimates increase with benchmark task difficulty?

**Design**:
- Same elicitation target: num_actors
- Varying: benchmark task difficulty (LOW/MED/HIGH)
- 10 runs per experiment, 10 experts per model

**Results**:

| Model | LOW (CVSS 5.3) | MED (CVSS 7.5) | HIGH (CVSS 10.0) | Trend |
| ----- | -------------- | -------------- | ---------------- | ----- |
| GPT-4o | 4.8% | 4.8% | 4.6% | Flat |
| Gemini | 6.5% | 5.5% | 6.0% | Non-monotonic |
| Claude | 8.6% | 4.4% | 5.6% | Non-monotonic |

**Conclusion**: Unlike humans, LLM expert variance does not increase with task difficulty.

**Data quality**:
- GPT-4o: 10 values/expert (complete)
- Claude: 9-10 values/expert (near-complete)
- Gemini: 5-10 values/expert (some API failures)

---

## H3: Probability Variance Stable Across Difficulty

**Question**: For the same attack step, does expert variance on probability estimates remain stable when we vary benchmark task difficulty?

**Design**:
- Attack step: TA0002 (Execution) — held constant
- Baseline: 50% — held constant
- Varying: benchmark task difficulty only (LOW/MED/HIGH)
- This isolates the difficulty effect by controlling for baseline and step

**Results**:

| Model | LOW (CVSS 5.3) | MED (CVSS 7.5) | HIGH (CVSS 10.0) | Trend |
| ----- | -------------- | -------------- | ---------------- | ----- |
| GPT-4o | 1.96% | 1.62% | 1.89% | Stable |
| Claude | 1.07% | 0.69% | 1.18% | Stable |
| Gemini | 2.21% | 3.53% | [PLACEHOLDER] | — |

CV ranges: GPT-4o = 0.34%, Claude = 0.49%.

**Conclusion**: Expert variance remains stable across difficulty for GPT-4o and Claude. Matches human pattern.

**Data quality**:
- GPT-4o: 10 values/expert (complete)
- Claude: 9-10 values/expert (near-complete)
- Gemini: 5-9 values/expert (significant data loss); HIGH missing entirely

**Limitation**: Gemini incomplete. Cannot draw conclusion for Gemini on H3.
