# Evaluation Metrics for Distributional Forecasts

## Context

We're forecasting probabilities (attack success rates) by eliciting three quantiles from an LLM: the 25th, 50th, and 75th percentiles. From these three numbers we fit a Beta distribution that represents the LLM's uncertainty. The ground truth is a single observed probability.

The question is how to measure forecast quality when the forecast is a distribution and the outcome is a point.

The metric needs to do something that might seem contradictory at first: reward forecasts that are both confident and correct, while not being too harsh on forecasts that are uncertain but reasonable.

A forecast of "50% ± 1%" that turns out right should score better than "50% ± 40%", but if the truth is actually 20%, the second forecast should be penalized less severely. Most importantly, the metric shouldn't be gameable—it should be strictly proper, meaning the LLM's best strategy is to report its true beliefs.

---

## Continuous Ranked Probability Score

The CRPS has a straightforward intuition. Imagine sweeping through every possible threshold $y$ from 0 to 1 and asking a yes/no question: "Is the true value at or below $y$?"

The forecast answers with a probability $F(y)$, where $F$ is the CDF of our fitted Beta distribution. Reality answers with 0 or 1. The Brier score at that threshold is $(F(y) - \mathbf{1}\{y \geq o\})^2$, where $o$ is the observed outcome.

The CRPS integrates this squared error over all thresholds:

$$
\text{CRPS}(F, o) = \int_0^1 \left(F(y) - \mathbf{1}\{y \geq o\}\right)^2 dy
$$

There's an equivalent form that's easier to compute and more interpretable:

$$
\text{CRPS}(F, o) = \mathbb{E}_F|X - o| - \frac{1}{2}\mathbb{E}_F|X - X'|
$$

where $X$ and $X'$ are independent draws from the forecast distribution.

The first term measures how far the forecast is from the truth on average. The second term measures how spread out the forecast is. A tight forecast has a small second term, so it needs to be accurate to score well. A diffuse forecast has a large second term that partially offsets the error penalty—hedging is allowed but not free.

The units are the same as the outcome, so a CRPS of 0.05 means the forecast is, in an integrated sense, about 5 percentage points away from a perfect point mass at the truth. This makes it directly comparable to mean absolute error.

In fact, if the forecast were deterministic (a point mass), CRPS reduces exactly to absolute error. This property is useful: a distributional forecast and a point forecast can be compared on the same scale.

One practical advantage is robustness. Unlike the logarithmic score, CRPS doesn't blow up when the outcome lands in a low-probability region. If the LLM forecasts tight mass around 0.6 but the truth is 0.1, the score is large but finite.

Gneiting and Raftery's case study in Section 8 of their paper shows CRPS giving stable, interpretable results where log score was unstable and other alternatives produced nonsensical rankings.

The main caveat is computational: the integral doesn't have a trivial closed form for arbitrary distributions. For Beta distributions it can be computed exactly using incomplete beta functions, but in practice Monte Carlo is simple and accurate enough.

Drawing 10,000 samples from Beta(a,b) and computing the kernel form takes milliseconds and converges well.

---

## Brier Score on a Point Estimate

The Brier score is $(p - o)^2$, where $p$ is a probability forecast and $o$ is the observed outcome. It's widely used, well understood, and strictly proper for point probability forecasts.

The problem is that we don't have a point forecast—we have a distribution.

One option is to extract a single number, say the median of our fitted Beta, and score that. This works, but it discards the uncertainty we elicited.

A forecast of "median 50%, very tight" and "median 50%, very wide" would receive identical Brier scores even though they represent fundamentally different beliefs. The first forecast is making a strong claim; the second is hedging. If the truth is 50%, both should do well, but if the truth is 30%, arguably the second forecast was less wrong.

The Brier score isn't wrong per se—if you only cared about the median prediction, it would be appropriate. But we went to the effort of eliciting three percentiles precisely to capture uncertainty, and this approach throws that information away.

---

## Interval Score

The interval score evaluates a prediction interval $[L, U]$ by combining width and coverage:

$$
\text{IS}_\alpha(L, U; o) = (U - L) + \frac{2}{\alpha}(L - o)\mathbf{1}\{o < L\} + \frac{2}{\alpha}(o - U)\mathbf{1}\{o > U\}
$$

where $\alpha$ is the exceedance probability (e.g., $\alpha = 0.1$ for a 90% interval).

The score rewards narrow intervals but penalizes the forecast if the outcome falls outside. The penalty scales inversely with $\alpha$: missing a 90% interval is worse than missing a 50% interval.

This is clean and interpretable. For our setup, we could use the 25th and 75th percentiles as a 50% interval and compute $\text{IS}_{0.5}$.

The issue is that we elicit three percentiles but only use two of them. The median carries information—if the 25th is 0.2, the 50th is 0.3, and the 75th is 0.6, the distribution is right-skewed—but the interval score doesn't see that.

The interval score is proper, so it's not a bad choice, just somewhat less efficient than using the full distribution. It might make sense as a supplementary metric if we care specifically about central interval performance, but it seems like leaving some information on the table.

---

## Logarithmic Score

The log score evaluates the density at the observed outcome:

$$
\text{LogS}(f, o) = \log f(o)
$$

where $f$ is the probability density function of the forecast.

It has strong theoretical appeal—connections to likelihood, entropy, and information theory—and it's strictly proper. The LLM maximizes expected log score by reporting its true predictive density.

The difficulty is that it's harsh and sometimes fragile. If the outcome lands in a region where the forecast density is low, the score can become very negative.

For Beta distributions on $[0,1]$, this is particularly concerning near the boundaries. If the LLM forecasts Beta(20, 2), which is tightly concentrated near 1, but the truth is 0.05, the density $f(0.05)$ will be tiny and $\log f(0.05)$ will be a large negative number. One outlier can dominate the mean.

Gneiting and Raftery specifically mention this issue in their Section 8 case study. They note that the log score "was unstable when the predictive density was tight near the observation."

For our use case, where outcomes are probabilities that can reasonably be near 0 or 1, this instability might be a problem. A single unexpected ground truth could make the metric hard to interpret.

---

## Mean Absolute Error

If we just take the median of the forecast and compute $|\text{median}(F) - o|$, we get mean absolute error. It's simple and interpretable, but it suffers from the same issue as Brier on a point estimate: all the uncertainty information is discarded.

MAE isn't even a proper scoring rule for distributional forecasts because it doesn't distinguish between tight and wide distributions with the same median.

That said, MAE on the median could be useful as a baseline. It represents what we'd get if we completely ignored the uncertainty.

If CRPS is only marginally better than this, it suggests the elicited uncertainty isn't adding much value. Conversely, if CRPS is much better, it validates the effort of eliciting distributions rather than point forecasts.

---

## Baselines and Context

Gneiting and Raftery recommend evaluating forecasts relative to reference strategies. Two baselines seem natural here.

First, a uniform forecast: predict Beta(1,1) for every cell. This is maximally uncertain, a flat distribution on $[0,1]$. Any method that performs worse than this is arguably subtracting information.

Second, an empirical baseline: predict a point mass at the observed source bin pass rate, with no LLM involved. This is the no-elicitation benchmark. If the LLM elicitation doesn't beat this, what's the point?

Reporting the mean CRPS alongside these baselines gives context. A CRPS of 0.08 might seem good or bad in isolation, but if the uniform baseline is 0.25 and the empirical baseline is 0.12, it's clearly an improvement.

---

## Recommendation

For this project, CRPS seems like the most appropriate primary metric. It uses all three elicited percentiles through the fitted Beta distribution, it's strictly proper, it handles uncertainty in a balanced way, and it's robust to the occasional surprising outcome.

The units are interpretable (probability points), and it reduces to absolute error when forecasts are deterministic, which means it sits naturally on a scale we already understand.

The logarithmic score has theoretical elegance but might be too brittle for outcomes that can lie near 0 or 1. Brier or MAE on the median would work but seem wasteful given that we're eliciting distributions. The interval score is reasonable but uses less information than CRPS.

A practical approach might be: compute CRPS as the main metric, report mean CRPS relative to uniform and empirical baselines, and perhaps compute the interval score for the central 50% interval as a supplementary check.

This way the primary evaluation uses the full distribution, but we can also see how the central credible interval performs on its own terms.

For implementation, Monte Carlo sampling from the fitted Beta distributions should be sufficient. Drawing 10,000 samples per cell, computing the kernel form of CRPS, and averaging across cells is straightforward and accurate enough for comparing methods or experimental conditions.

---

## References

Gneiting, T., & Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. *Journal of the American Statistical Association*, 102(477), 359-378.
