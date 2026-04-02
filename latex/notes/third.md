# Week 3

## Jeff

Spent some time thinking about the best metric to use that would capture consistency across the entire distribution instead of just comparing the point estimates (though that is still useful). I think the best approach is to use [wasserstein distance](https://en.wikipedia.org/wiki/Wasserstein_metric) (W_1 = earthmovers distance) instead of KL divergence as this makes a much friendlier metric to use for a loss function and is more intuitive. It spans from 0 to 1 where 0 would be perfect agreement (total overlap) and 1 would mean perfect disagreement. KL divergence is unbounded and can blow up because of tails we mostly don’t care about. It also isn’t symmetric while w_1 is, which is nice because there isn’t a clear ‘reference’ we’re trying to match. 

- Here’s some more [detailed literature](https://arxiv.org/pdf/2408.09770) on wasserstein and how to interpret different flavors/variations if I want to get into that

w_1 is the fraction of the total cdf that does not overlap between the generating functions. You could also use w_2 if you wanted to punish differences more strongly, but you lose that clean mapping to probability mass.

For w_1 like KL divergence you need a probability distribution, and you can either use the raw values with linear interp or fit a distribution. I did this both ways because I think the raw values are more defensible/interpretable but from what I understand the beta distribution is used eventually down stream so we might care about that and/or want to use that as our loss function.

I made some plots to visualize what is going on here because I think the above might be confusing. Here are comparisons from the same task/step/model for both approaches (w_1 value is included in title). These are the same data but the first uses the linear approach and the second fits a beta. 

![image.png](attachment:7b0d42f0-1be3-4c7a-b575-86df6557e963:image.png)

![image.png](attachment:bd9d6b9d-2489-4e5f-b357-cf110e21992a:9693d979-4810-40c0-8c0f-ed46bf24dace.png)

For comparison, this is the same thing but comparing different models (so more disagreement).

![image.png](attachment:6205446f-72f0-45db-959b-3be1076ab301:image.png)

![image.png](attachment:c0e9858a-c4e5-464c-9f61-dff2279a25e4:image.png)

(as a sense check the estimates above correspond to the 1st and 5th bars here)

![image.png](attachment:4429754d-2a28-4d72-8b4a-7acbff1b6e72:image.png)

Here I’m using the test-retest framing where we’re testing for consistency across model runs, but you could use the same approach to evaluate additivity. A model prompted to report failure instead of success should be reporting p_fail = 1-p_success, so can flip it around to see if it lines up with what is reported when asking for p_success directly.

![image.png](attachment:12c357d3-846c-4b29-a671-a33e358a071a:image.png)

![image.png](attachment:d88ea151-4c6f-4739-b235-0545e0db8881:image.png)

To turn this into a metric, you just calculate w1 for every pair and take the mean. I think this makes it a nice overall metric, and could be combined with divergence on the midpoint and divergence on the IQR to decompose the difference into central tendency vs. confidence differences.

I’m not sure whether there is a clean statistical test you could apply to w1 like you could do with the point estimates to answer a question like ‘there is no difference across model runs’. Also afaik there isn’t a straightforward ‘this is a good w1 score’ we can reference, but I think saying something like ‘the cdfs overlapped 98.7%’ is pretty compelling. And we could easily bootstrap a confidence interval to do something like determine whether a certain prompt variation provides measurable improvement on this metric.

Here are results from running the additivity comparison. For reference p50 divergence here is exactly the same definition as incoherence: mean(|p_s + p_f - 1|). 

![image.png](attachment:e03f7a4e-68df-4345-a945-2e4cbf79d3b6:image.png)

**Madhav**

**Frechet Mean and Variance**

For general metric space valued random variables, Frechet provides the direct generalization of means, which implies a corresponding generalization of variance that may be used to quantify the spread of distribution of metric space valued random variable (https://arxiv.org/pdf/1710.02761). This can be used to quantify the spread of distribution of metric spaced valued random variables or objects. The frechet mean resides in the object space and therefore doesn’t need to obey algebraic operations. 

The population Frechet mean  µF of Y and the sample Frechet mean  µ^F
for a random sample Y1, Y2, . . . , Yn of independent and identically distributed random variables
with the same distribution as Y are given by

$$
\mu_F = \arg\min_{\omega \in \Omega} \mathbb{E}[d^2(\omega, Y)]
$$

$$
\hat{\mu}_F = \arg\min_{\omega \in \Omega} \frac{1}{n}\sum_{i=1}^{n} d^2(\omega, Y_i)
$$

The Frechet variance quantifies the spread of the random variable Y around
its Frechet mean  µF . The population Frechet variance VF and its sample version Vˆ
F are

$$
V_F = \mathbb{E}\left[d^2(\mu_F, Y)\right]
$$

$$
\hat{V}_F = \frac{1}{n} \sum_{i=1}^{n} d^2(\hat{\mu}_F, Y_i)
$$

**Frechet ANOVA:**

Standard ANOVA compares group means of numbers. We need to compare group means of probability distributions. Freche ANOVA generalizes the F-test by replacing the arithmetic mean with Frechet mean (t*he distribution minimizing squared Wasserstein distances to all group members)* and replacing scalar variance with Frechet Variance (t*he average squared Wasserstein distance to the Fréchet mean).* We can use the W2 **(L2-Wasserstein) as the distance metric** and apply this to our data points which are fitted beta distributions represented as quantile function.

**Implementation  (beta fitted with least square on p25, p50, p75)**

$$
\begin{aligned}                                                                            
  &\text{1. Fit Beta distributions}\\                                                        
  &\text{For each LLM response, fit } \mathrm{Beta}(\alpha,\beta) \text{ from } (p_{25},     
  p_{50}, p_{75}) \text{ via least squares:}\\                                               
  &\quad \min_{\alpha,\beta} \sum_{q \in \{0.25, 0.50, 0.75\}} \bigl( \mathrm{CDF}(p_q;      
  \alpha, \beta) - q \bigr)^2\\                                                              
  &\text{Reject fits where } \max_q |\mathrm{CDF}(p_q) - q| > 0.05 \text{ (internally        
  inconsistent elicitations).}\\[6pt]                                                        
                                                                                             
  &\text{2. Represent as quantile functions}\\                                               
  &\quad Q(u) = F^{-1}(u), \quad u \in [0.001,\, 0.999]\\                                    
  &\text{Evaluated on a uniform grid of 201 points.}\\[6pt]                                  
                                                                                             
  &\text{3. Compute pairwise Wasserstein distances}\\                                        
  &\quad d_W^2(F,G) = \int_0^1 \bigl(Q_F(u) - Q_G(u)\bigr)^2\, du\\                          
  &\text{Approximated via trapezoidal integration over the quantile grid.}\\[6pt]            
                                                                                             
  &\text{4. Calculate Fréchet means}\\                                                       
  &\quad \bar{Q}(u) = \frac{1}{n}\sum_{i=1}^{n} Q_i(u)\\                                     
  &\text{Computed pointwise for each group (persona or model) and for pooled data.}\\[6pt]   
                                                                                             
  &\text{5. Calculate Fréchet variances}\\                                                   
  &\quad \hat{V}_j = \frac{1}{n_j} \sum_{i=1}^{n_j} d_W^2(Q_{ij},\, \bar{Q}_j) \quad         
  \text{(within-group)}\\                                                                    
  &\quad \hat{V}_p = \frac{1}{N} \sum_{j}\sum_{i} d_W^2(Q_{ij},\, \bar{Q}_p) \quad           
  \text{(pooled)}\\[6pt]                                                                     
                                                                                             
  &\text{6. Compute test statistic } T_n \text{ (Dubey \& Müller, 2019)}\\                   
  &\quad F_n = \hat{V}_p - \sum_{j=1}^{k} \lambda_j \hat{V}_j \quad \text{(between-group     
  variance)}\\                                                                               
  &\quad U_n = \sum_{1 \le i < j \le k} \lambda_i \lambda_j (\hat{V}_i - \hat{V}_j)^2 \quad  
  \text{(heteroscedasticity correction)}\\                                                   
  &\quad T_n = N \cdot F_n + N \cdot U_n\\                                                   
  &\quad \text{where } \lambda_j = n_j / N \text{ (group proportions)}\\[6pt]                
                                                                                             
  &\text{7. Permutation test}\\                                                              
  &\text{Shuffle group labels 5000 times, recompute } T_n \text{ each time.}\\               
  &\quad p = \frac{\#\{T_n^{\text{perm}} \ge T_n^{\text{obs}}\} + 1}{5000 + 1}               
  \end{aligned}                                                                              
                  
$$

I plotted some fitted distributions (10 runs for each 10 experts per model per baseline based on percentile elicitation and **least square fit for beta distribution** parameters)

## **1.)   Expert Persona Belief Distributions (Claude Sonnet 4.5)**

                                ****

![beta_persona_distributions_percentile.png](attachment:24176ac0-260d-4745-ab17-71f8f2f8370f:beta_persona_distributions_percentile.png)

-No visible separation between expert personna. The 10 curves seem to overlap at most places.

## **2.)  Cross-Model Belief Distributions**

![beta_cross_model_distributions_percentile.png](attachment:52225e06-e6d3-4b22-baab-ad5afa81caf0:beta_cross_model_distributions_percentile.png)

-Visible separation in width and center for three different models on each baseline. 

## **3.) Expert Persona Distributions Across Models and Attack Steps**

![beta_full_distribution_grid_percentile.png](attachment:f9cc81c8-af2a-41ab-9954-43f0ae8cf2f2:beta_full_distribution_grid_percentile.png)

-In most boxes, all 10 curves overlap, signalling that expert personna distributions are close

-It’s worth noting that Gemini 2.5 Pro at 50% baseline has some variance on expert personna, but the ICC_F is still under 18-25%

-At extreme baselines (85% and 30%), distributions become narrower and more concentrated,
reflecting reduced uncertainty near the boundaries

-Across models (rows), clusters shift visibly: model choice matters 

**Experimental Results**

**H₀:** All 10 expert personas produce the same belief distribution.

### Claude Sonnet 4.5

| Step | Probability | N | T_n | p-value | ICC_F |
| --- | --- | --- | --- | --- | --- |
| TA0002 | 50% | 100 | 0.008 | 0.400 | 9.6% |
| TA0007 | 85% | 80 | 0.003 | 0.195 | 13.9% |
| T1657 | 30% | 100 | 0.020 | 0.803 | 7.2% |

### GPT-4o

| Step | Probability | N | T_n | p-value | ICC_F |
| --- | --- | --- | --- | --- | --- |
| TA0002 | 50% | 97 | 0.016 | 0.899 | 5.7% |
| TA0007 | 85% | 96 | 0.007 | 0.708 | 7.5% |
| T1657 | 30% | 100 | 0.043 | 0.215 | 11.8% |

### Gemini 2.5 Pro

| Step | Probability | N | T_n | p-value | ICC_F |
| --- | --- | --- | --- | --- | --- |
| TA0002 | 50% | 88 | 0.043 | 0.030 | 18.3% |
| TA0007 | 85% | 96 | 0.007 | 0.002 | 24.9% |
| T1657 | 30% | 88 | 0.005 | 0.585 | 8.8% |

**Result:** No significant persona effects for Claude or GPT-4o. Gemini shows significant persona variation at two steps (p < 0.05).

---

# Experiment 2: Model Variance (Across Models)

**H₀:** All 3 models (Claude, GPT-4o, Gemini) produce the same belief distribution.

| Step | Probability | N | T_n | p-value | ICC_F |
| --- | --- | --- | --- | --- | --- |
| TA0002 | 50% | 285 | 0.729 | <0.001 | 55.3% |
| TA0007 | 85% | 272 | 0.131 | <0.001 | 48.2% |
| T1657 | 30% | 288 | 1.515 | <0.001 | 69.4% |

**Result:** All steps show significant model effects (p < 0.001). Model choice explains **48–69% of distributional variance**.

---

**Key Takeaway**

Model variance (48–69% ICC) >> Persona variance (6–25% ICC). Expert personas seem

statistically indistinguishable within Claude and GPT-4o; model choice seems the dominant

source of variation.

**Symmetric CI Fitting (Old Approach Based on min/max and confidence: Mightn’t be relevant since we are using percentiles now. Just keeping it here for reference)**

We get maximum_estimate (hi), minimum_estimate (lo), confidence_in_range and most_likely_ estimate (mode). We ignore the mode and treat lo, hi as a symmetric confidence interval with equal probabilities

$$
\mathrm{CDF}(lo) = \frac{1 - \text{confidence}}{2}
$$

$$
\mathrm{CDF}(hi) = \frac{1 + \text{confidence}}{2}
$$

As Matt pointed out, the initial ellicitation asks for support bounds (min/max) and then interprets them as confidence intervals which is contradictory. We can resolve this by treating bounds as the quantities of fitted distribution. I think the distribution still makes sense without using the mode because mode creates an overconstrain (3 constraints for 2 parameters), experts struggle with reasoning mode in skewed distributions, and mode also seems to be inconsistent with CI bounds. In the implementation, bounds are clamped to [0.005, 0.995] to avoid edge cases where lo=0 or hi=1 (which make symmetric CI constraints unsolvable)

1.) **Expert Persona Belief Distributions**

![beta_persona_distributions.png](attachment:e669f9b2-4033-455c-ad8d-38b588f24a3c:beta_persona_distributions.png)

All 10 persona distributions overlap within each step

- At TA0002 (50% baseline), personas show the widest spread in both location and shape, but even here the aggregate (dashed black) sits comfortably within the cluster
- At TA0007 (85%) and T1657 (30%), all 10 curves collapse into nearly identical shapes, consistent with a ceiling/floor floor effect: as the baseline moves toward 0 or 1, *there's less room for personas to disagree (this is an incorrect interpretation for 30%. maybe a floor effect for 30% similar to ceiling effect for 85%? )*

 

**2.) Cross-Model Belief Distributions**

![beta_cross_model_distributions.png](attachment:e3b15621-60f6-413c-893c-3700fa07a43c:beta_cross_model_distributions.png)

- Visible separation between models at all three baselines, in both location (mode shift) and shape (width/peakedness). *Note: The curves shown represent the pointwise average of individual Beta PDFs fitted to each elicitation (100 per model)*

**3.) Individual Run Distributions per Persona**

![beta_individual_run_distributions.png](attachment:edd61211-3ad0-462e-921a-b5bd4b6a5e4d:beta_individual_run_distributions.png)

- Shows the within-persona run-to-run variability (faint lines = individual runs, bold = mean)
- Run-to-run variability within each persona (W1 = 0.093) is comparable to differences

      between personas (W1 = 0.095). 

**4.) Expert Persona Distributions Across Models and Steps**

- Within any cell, the 10 persona curves cluster tightly around the aggregate persona variance seems like noise
- Across rows (same step, different models), the entire cluster shifts in location and width

![beta_full_distribution_grid.png](attachment:a1dce811-9cad-4ffd-99e6-89d5c9e5c774:beta_full_distribution_grid.png)

**Experiment 1: Persona Variance (within-model)**

H₀: All 10 expert personas produce the same distribution.

| **Model** | **Step** | **N** | **T_n** | **p (perm)** | **ICC_F** |
| --- | --- | --- | --- | --- | --- |
| Claude Sonnet 4.5 | TA0002 (50%) | 100 | 0.104 | 0.120 | 13.3% |
| Claude Sonnet 4.5 | TA0007 (85%) | 90 | 0.006 | 0.650 | 8.5% |
| Claude Sonnet 4.5 | T1657 (30%) | 97 | 0.007 | 0.772 | 6.8% |
| GPT-4o | TA0002 (50%) | 100 | 0.063 | 0.063 | 13.9% |
| GPT-4o | TA0007 (85%) | 100 | 0.004 | 0.691 | 7.5% |
| GPT-4o | T1657 (30%) | 100 | 0.021 | 0.247 | 11.1% |

**None significant at p < 0.05.** Persona choice explains only 7-14% of distributional variance.

**Experiment 2: Model Variance (across models)**

H₀: All 3 models (Claude, GPT-4o, Gemini) produce the same distribution.

| **Step** | **N** | **T_n** | **p (perm)** | **ICC_F** |
| --- | --- | --- | --- | --- |
| TA0002 (50%) | 300 | 0.799 | 0.0002 | 27.1% |
| TA0007 (85%) | 290 | 0.196 | 0.0002 | 41.7% |
| T1657 (30%) | 297 | 0.272 | 0.0002 | 36.1% |

**All significant at p < 0.001.** Model choice explains 27-42% of distributional variance.

**Key takeaway:** Model variance (27-42% ICC) >> Persona variance (7-14% ICC). Expert personas are statistically indistinguishable; model choice is the dominant source of variation in elicited probability distributions.


