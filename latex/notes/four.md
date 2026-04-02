# Week 4

## Jeff

- Spent a lot of time updating the consistency metrics. I also updated it to work with arbitrary percentile elicitation to support that consistency check. Current script includes:
    - w1, w2, iqr and p50 divergence
    - bootstrapped CIs for w1 and w2
    - function to run comparison within or across results dataframes. These match a specific format I’ve been using so probably need to generalize or add that to the file.
    - plotting helpers
    - Todos
        - CI for iqr and p50
        - a better implementation of w1 and w2 with beta assumption. The old approach is too slow for bootstrapping.
        - frechet anova as an option for statistical testing
        - add data parsing to file to match with metrics function?

**Consistency under different percentile elicitations**

Most of the work was just getting this set up. I only ran a single experiment as a proof of concept. Can run more with more variations, or can run them on the new framework once built.

I compared elicitation with the default 25 50 75 elicitations vs a minimally altered version that elicits quintiles instead (20, 40, 60, 80). This combines two degrees of freedom because the elicitation values and number elicited are both changed, but I wanted to test both forms of flexibility for code validation purposes.

I generated 10 iterations across 3 steps (single persona), using sonnet 4.6 for all runs. This gives 100 across approach comparisons and 45 within approach. Confidence intervals are bootstrapped 1000x.

**Results**

Cl here is the default approach, and quin is when eliciting 4 points at quintiles instead. 

![image.png](attachment:746455fe-dc4c-49b7-ba04-3469ff379ee0:image.png)

![image.png](attachment:5dda9b4a-668e-46b5-a282-7e1bf2a81ed0:image.png)

=== Cross (cl vs quin) ===
                                                w1 (95% CI)          w2 (95% CI)
TA0001 - Initial Access  0.022 (0.020–0.024)  0.031 (0.029–0.033)
TA0002 - Execution       0.030 (0.025–0.036)  0.038 (0.032–0.044)
TA0007 - Discovery       0.024 (0.023–0.026)  0.045 (0.043–0.048)

=== Within cl ===
                                                w1 (95% CI)          w2 (95% CI)
TA0001 - Initial Access  0.008 (0.005–0.010)  0.010 (0.006–0.011)
TA0002 - Execution       0.013 (0.008–0.015)  0.015 (0.010–0.017)
TA0007 - Discovery       0.006 (0.004–0.007)  0.008 (0.005–0.009)

=== Within quin ===
                                                 w1 (95% CI)          w2 (95% CI)
TA0001 - Initial Access  0.019 (0.011–0.021)  0.024 (0.013–0.027)
TA0002 - Execution       0.022 (0.013–0.024)  0.024 (0.015–0.027)
TA0007 - Discovery       0.006 (0.003–0.007)  0.007 (0.004–0.009)

I didn’t compute the confidence intervals for the beta assumption due to the aforementioned slowness and not having time to fix it. But the point estimates were broadly similar but with the difference between cross and within somewhat attenuated. Note that fitting a beta adds yet another complication here because one is fit on 3 points and one is fit on 4.

=== Cross (cl vs quin) ===
                                           w1_beta   w2_beta
TA0001 - Initial Access  0.031000  0.037850
TA0002 - Execution       0.029956  0.034235
TA0007 - Discovery       0.011604  0.014469

=== Within cl ===
                                          w1_beta   w2_beta
TA0001 - Initial Access  0.013268  0.015427
TA0002 - Execution       0.021259  0.024356
TA0007 - Discovery       0.012382  0.015339

=== Within quin ===
                                            w1_beta   w2_beta
TA0001 - Initial Access  0.032212  0.039259
TA0002 - Execution       0.028186  0.030762
TA0007 - Discovery       0.010184  0.012729

**Notes on these results**

- Meaningful variation across elicitation approaches
    - either violates consistency or a helpful knob to get more diversity?
- Seems like less consistency within quin than within the standard approach, though need more data to see if that persists
    - And probably many more variations on this theme to make any real conclusions.

![image.png](attachment:da70fd68-117b-46de-8478-005a730ba3a8:image.png)

![image.png](attachment:c62f0e05-8cc6-4008-a6d9-c7a707e0bb21:image.png)

**Madhav**

I experimented with the following questions. I also ran some prompt sensitivity experiments (removing the baseline, confidence intervals).

1.) Do quantities (num actors) show more expert disagreement than percentiles? (Human experts showed higher expert variance in quantities than probability estimates) 

2.) Does the expert variance for quantities increase with the difficulty of benchmark task?

3.) For the same attack step, does expert disagreement on probability estimates remain stable when we vary benchmark task difficulty (LOW vs MED vs HIGH)? 

4.) Does model choice create more variance than expert persona for quantity estimates ? Previous result found yes 3x to 5x more variance for probability elicitation. The trends follows for number of actors estimates as well

**Method**: Fréchet ANOVA with Wasserstein distance on fitted distributions. ICC_F = proportion of variance explained by grouping factor.

| Estimate Type | Persona ICC_F | Model ICC_F | Ratio | Model p-value |
| --- | --- | --- | --- | --- |
| Probability | 0.12 | 0.58 | ~5× | <0.001 |
| Quantity (num_actors) | 0.09 | 0.24 | ~2.6× | 0.005 |

*Persona ICC: 10 personas as groups, pooled across 3 models (N=300). Model ICC: 3 models as groups, pooled across 10 personas (N=300)*

**Conclusion**: Based on this experiment, model choice seems to create more variance than persona choice for both estimate types (quantity and probability).

**Limitation**: Quantity analysis uses PERT fitting (bounded), which may not suit unbounded integers.

I spent most of my time setting up and running the experiments for number of actors and varying benchmark task difficulty, prompts. For each experiment, I have 10 experts and 10 runs each and repeat for three models (Gemini 2.5 Pro, Claude Sonnet 4.5 and GPT 4o) to compare if model choice agrees on final conclusions or not. 

**Confusions:**

-For each expert’s (p25, p50, p75) estimation on numeric quantity like number of actors, as it’s an unbounded estimate, I assume bounded support and numerically fit PERT(a, m, b) whose CDF equals 0.25, 0.5, 0.75 at those values. However, there might be better alternatives (lognormal?)

-For probability elicitation, we have different baselines for a scenario (we chose 30%, 50%, 80%). However, for quantities, I believe each quantity is per scenario hence we’ll have one quantity elicitation for one scenario. This makes it hard to compare the expert variance between quantity estimation and probability estimation. Coefficient of variance is scale independent, hence comparing it for probabilities and quantities seems fine, but averaging CVs of probability estimation across different baselines and comparing them with the CV of quantity is questionable. 

-For computing the variance in expert elicitation of quantities, I’m figuring out which approach is the best. I was thinking of two options:

1.) Simple coefficient of variation: For each expert take p50 from each run and compute their mean. We get 10 expert means and compute the cv as std(expert means) / mean(expert_means)*100%. This would measured how spread out expert central estimates are scaled by the median. However, there are many limitations, for instance we are only using p50 values, computing the mean of CVs doesn’t make sense, no distribution used, mean of medians etc. If I pick p25 or p75 instead of p50 for all the computations, the results are similar though.

2.) ICC_F: Fit the PERT distribution for three percentiles. V_pooled = mean over all observations of W2² from each Qᵢ to the pooled Fréchet mean μ_pooled. V_within = weighted sum of within-group variances (W2² of each Qᵢ to its group’s Fréchet mean). F_n = V_pooled − V_within (between-group component). ICC_F = F_n / V_pooled (bounded to 0–1). This would measure the fraction of total distributional variance that is between experts vs within experts.

**Results** (this might likely change based on methodological choices for computing/comparing variances for numeric estimates. I need to reiterate the computations and results based on what we decide. ) *Each experiment was run for 10 experts 10 times for each 3 models, hence each cell represents calculations from 100 data points. I am using Coefficient of Variance on p50 for now (which has limitations), but the conclusions seem strong, hence I put it here anyway.*

## H1: Quantities Show More Expert Variance Than Probabilities

**Question**: Do quantity estimates (num_actors) show more expert variance than probability estimates?

**Design**:

- Quantity CV: averaged across 3 benchmark difficulty levels (LOW/MED/HIGH) (this is definitely questionable)
- Probability CV: averaged across 3 attack steps with different baselines (T1657 30%, TA0002 50%, TA0007 85%)
- Benchmark tasks: cURL (CVSS 5.3), Imaginairy (CVSS 7.5), MLFlow0 (CVSS 10.0)

**Results**:

| Model | Quantity CV | Probability CV | Ratio |
| --- | --- | --- | --- |
| GPT-4o | 4.7% | 1.9% | 2.5× |
| Gemini | 6.0% | 2.2% | 2.7× |
| Claude | 6.2% | 1.7% | 3.6× |

All three models show higher variance for quantities. Matches human pattern where human expert variance was higher for quantities than probability estimates.

**Limitation**: Probability CV averages across different baselines (30%, 50%, 85%), which may not be directly comparable due to ceiling/floor effects, as well as averaging CV doesn’t really make sense. However, the ~3× ratio is large enough that this probably might not change the conclusion? But I need to come up with better comparison between probability variance and quantity variance next week.

## H2: Quantity Variance Increases With Difficulty

**Question**: Does expert variance for quantity estimates increase with benchmark task difficulty?

**Design**:

- Same elicitation target: num_actors
- Varying: benchmark task difficulty (LOW/MED/HIGH)
- 10 runs per experiment, 10 experts per model

**Results**:

| Model | Easy (22) | Medium (38) | Hard (42) | Trend |
| --- | --- | --- | --- | --- |
| GPT-4o | 18.7% | 17.4% | 16.5% |  Decreasing |
| Gemini | 16.8% | 15.0% | 13.1% |  Decreasing |
| Claude | 12.1% | 17.2% | 19.8% |  Increasing |

**Conclusion**: Unlike humans, no consistent increase in variance with task difficulty was observed.

**Data quality**:

- GPT-4o: 10 values/expert (complete)
- Claude: 9-10 values/expert (near-complete)
- Gemini: 5-10 values/expert (some API failures)

## H3: Probability Variance Stable Across Difficulty

**Question**: For the same attack step, does expert variance on probability estimates remain stable when we vary benchmark task difficulty?

**Design**:

- Attack step: TA0002 (Execution) held constant
- Baseline: 50% held constant
- Varying: benchmark task difficulty only (LOW/MED/HIGH)
- This isolates the difficulty effect by controlling for baseline and step

**Results**:

| Model | LOW (CVSS 5.3) | MED (CVSS 7.5) | HIGH (CVSS 10.0) | Trend |
| --- | --- | --- | --- | --- |
| GPT-4o | 1.96% | 1.62% | 1.89% | Stable |
| Claude | 1.07% | 0.69% | 1.18% | Stable |
| Gemini | 2.21% | 3.53% | no data | — |

CV ranges: GPT-4o = 0.34%, Claude = 0.49%.

## **Prompt Sensitivity for Probability Estimates**

For this experiment, I started with 10 runs for a single expert on Claude Sonnet 4.5. This was done with TA0002  Execution, 50% baseline with AI/ML Security Researcher as expert fixed. 

-Include baseline and CI

-Remove baseline, include CI

-Remove CI, include baseline

-Remove baseline and CI

Removing the baseline seems to have the biggest impact, as baseline provides the central anchor.

**Each run:** (p25, p50, p75) → fit Beta(alpha, beta) → quantile function

**Compare distributions:** Using Wasserstein distance (W^2)

**Fréchet ANOVA:** Test if condition Fréchet means differ

W^2 Distance Between Condition Fréchet Means

*(How different are the "typical" distributions?)*

| Comparison | W^2 Distance |
| --- | --- |
| Control ↔ No confidence | 0.016 (small) |
| Control ↔ No baseline | 0.176 (large) |
| Control ↔ No baseline no CI | 0.233 (largest) |

### 2. Within-Condition W^2 Variance

| Condition | W^2 Variance | vs Control |
| --- | --- | --- |
| **Control** | 0.00084 | 1× |
| **No confidence** | 0.00029 | 0.3× |
| **No baseline** | 0.01462 | 17× |
| **No baseline no CI** | 0.00991 | 12× |

![prompt_sensitivity_chart.png](attachment:c9fac010-ae4d-4b9b-ba2d-59b0cf557f76:prompt_sensitivity_chart.png)

![w1_chart.png](attachment:83a32674-8417-4b43-adf1-e6af0ca29638:w1_chart.png)

the results are similar for w1 and w2; w2 amplifies it more.

Fréchet ANOVA

- Removing baseline: Shifts distributions significantly (W^2 = 0.18 from control)
- Removing CI only: Minimal shift (W^2 = 0.02)
- Removing both: Produces the largest shift (W^2 = 0.23)