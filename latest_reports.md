Week 8

# **eff**

I’ve spent some time this week going through the available data from the [cyber time horizons paper](https://lyptusresearch.org/research/offensive-cyber-time-horizons) I linked in slack to see whether this seems like a useful replacement for the inconsistent benchmark data we’ve been trying to use with the intra-benchmark comparison. 

*Benchmarks included*

![image.png](attachment:97e518e9-d7cd-4791-a04d-8219bef1907e:image.png)

*Model performance*

The focus of this paper is on measuring model capability using a similar time horizons approach to the METR general coding capabilities time horizons. But this could be very useful for us because it establishes a mostly objective difficulty ranking (human time to completion) which can let us use a similar ranking approach to select examples as what is done in the inter-benchmark experiment.

Performance is  computed as the fraction of tasks within a time bin successfully completed by the model. 

![image.png](attachment:1611a03c-3ddb-48b0-8c11-c669d698e145:image.png)

Conversely, could use the benchmark level performance, which is also provided here. I think using the time bins is more in line with the eventual goal of predicting performance on an arbitrary cyber attack step though.

![image.png](attachment:2a3e450a-6391-45eb-a374-eb3ac39c0c5b:image.png)

*Tasks*

They have solid estimates for 291 tasks coming from either actual completion time, first blood time, or human expert estimates. They call this the ‘headline’ set, and probably it would make sense for us to just use those tasks (though there are an additional 630 tasks with model generated difficulty levels).

They went out of their way to provide the prompts for each of these in a useful format, so it would be pretty straightforward to pass in a combination of task prompt + difficulty (in terms of expected human time) and pass rate among tasks within that difficulty bin.

For each of these tasks we have the the model x task success values, so we could in principle use a similar approach of saying ‘this is the hardest task the model passed’. However I don’t really think this is the best approach, as the model might randomly perform well on a very long timeframe task and this isn’t necessarily representative of it’s future performance.

*Outline for potentially using this*

The goal we’re still trying to fill is having a good source for ground-truth data that we can use for improving the elicitation strategy. My conception of the goal here is to get a useful test bed of eliciting p(success on task of unknown difficulty | performance on tasks of known difficulty). This is somewhat different than p(MITRE step | performance on tasks) and probably substantially different from p(MITRE step | benchmark score). If we care more about the latter option here, it might make sense to change the approach I’m outlining to be more focused on benchmark level performance. This could still be done using the same dataset, and the benchmarks themselves span a sort of difficulty range.

- Provide anchoring context
    - Either manually select or sample a set of tasks from each time bin
        - I would lean towards selecting representative tasks. Benchmarks have issues so there may be tasks that have a 0% pass rate despite being in an easy bin (or 100% in a hard bin) and it would be unfortunate to use these as the template tasks. Would want to make sure the example tasks were actually representative by having reasonable pass rates distributions across models. Fortunately we have this data because we have task level pass values for every task x model.
    - Provide task prompt, difficulty tier (either in terms of minutes or ranking 1-5), and pass rate for that difficulty tier
        - “This model passed 80% of the tasks in difficulty bin 2, which had tasks like: {task description from prompt}. It had a 50% pass rate on tasks of difficulty 3, like: {another task description}”
- Select test task from a target difficulty bin. Elicit expected pass rate on that task/set
    - “What is the probability of this model passing the following task: {task description}”
- Compare elicited pass rates to actual pass rates.
    - Here we are comparing the elicited probability on the target task to the model’s pass rate on the target bin.
        - this assumes that the target task is representative of the overall bin, so it is a fair comparison. Some alternatives are discussed below.
    - This goes through several cycles:
        - Elicit using target tasks from each of the difficulty bins (5x)
        - Elicit using pass rates from various models (15x)
        - Gives a total of 75 datapoints, which can be plotted with the y axis being estimated pass rate (model x bin) and the x axis being actual pass rate (for that model in that bin)
- Sensitivity tests it would be good to run
    - How much does the selection of context tasks or target tasks matter
        - ideally, different tasks from the same bins should be relatively interchangeable
    - Make sure changing the pass rates meaningfully impacts the elicited probabilities
        - the existing model spread might be good enough to cover this

*Potential wrinkles*

- It would be helpful to have some more thoughts about how this relates to the longer term goal of using these elicitations in practice
    - This dataset is clean and nice, but it doesn’t extend to new model releases unless you run the model on this task set to get the difficulty bin pass rates
    - A slightly different approach that tried to elicit using the overall benchmark pass rate might be better, because those values could be taken directly from a model card or public leaderboard.
        - This could be done using this dataset, as you can compute the overall benchmark difficulty from the average task difficulty, and can get benchmark level pass rates. I think you would still want to pass some representative task descriptions in to the context.
- We are trying to quantify expected pass rate of an arbitrary task.
    - sampling a task from a given difficulty bin, we could approximate the expected pass rate of that task as the pass rate of the bin containing it. This may not be strictly true (maybe some tasks are slow but easy?)
    - could instead provide several representative tasks and ask it to estimate the pass rate of those tasks in aggregate? This more cleanly maps onto the structure of the bin level task pass rates, but less cleanly to the eventual goal of p(MITRE step | task level performance)

**Post meeting note:**


Week 9 
# Week 9

**Madhav**

Thought about the design choice for per-LLM vs averaged across LLM ground truth

**Single target-task variance**

As discussed, single target task for a specific model resembles our P(MITRE| benchmark performance) more closely than other approaches. Task nature might be highly variable for a sinlg task/model. The predictions on single target tasks  might jump a lot based on the specific nature of the task. One potential approach could be picking K = 3 samples per bins. This way we can get the within bin-variance, but not sure if it fixes the underlying issues that well. The other alternative as would be Elicit P(aggregate pass rate over a sample of N target tasks) instead of P(one specific task). This collapses the within-bin noise on both sides (prompt and ground truth match). The other concern is representativeness assumption. I think Jeff had a simple solution for this: use per-task Brier.

I was discussing with Jakub about predictions with individual LLMs and while I like the idea of averaging across LLMs too based on your explanation, I am still unsure if this actually resembles our main goal. Ground truth values might vary a lot depending on the LLM.  When we average the ground truth across target tasks and all LLMs, we are modeling a generic LLM on a generic task rather than a specific task and a specific LLM which might be noisy. But at the same time, some models are good, some are bad, and I am not sure smoothing the results out by averaging scores (for instance GPT 2 model score and Opus 4.7 score) might defeat the purpose. On our main elicitation P(MITRE step | benchmark score), do we care more about one specific frontier LLM assisting attacks in the wild or an overall AI capability in general?  I am also wondering if the variance we get out of individual LLMs (claude, Gpt, Gemini etc depending on the model) a useful signal (expected) or a noise, and whether smoothing them is good or not. If we decide to go with the LLM averaged ground truth, that would mean we would have a brier score from 5 data points which might be another problem? That might probably mean a weaker statistical power for comparing brier scores across different conditions. 

**Uncertainty of P_i into Brier Score**

I came across this paper that might be relevant. There might be other approaches that are better. https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

We need a metric that takes forecast distribution and observed outcome as input, and returns a number. The metric should reward forecast that are confident as well as correct, and penalize the ones that are confident and wrong. Wrong and non-confident should be in between. Plain brier won’t work since it will ignore the elicited uncertainty. 50% +- 1% will get the same brier as 50% +- 40%. Gneiting and Raftery (2007) paper above suggests that variance scaled brier or other fixes don’t always work as forecaster can still get away with a better score by reporting something other than its true belief. They show that linear and spherical scores produce qualitatively wrong rankings. Log score on the beta density fails when outcome lands near 0 or 1. They suggest interval score or CPRS for “interval estimation that addresses width
as well as coverage"

1.) **Interval Score**

The simplest is interval score. Three forecasts for a cell whose true outcome is o = 0.20 would look like this. 

IS_alpha(L, U; o) = (U - L) + (2/alpha) * (L - o) * 1{o < L}  + (2/alpha) * (o - U) * 1{o > U}  

-Confident and wrong. L = 0.70, U = 0.80, o = 0.20.
IS = (0.80 - 0.70) + 4 * (0.70 - 0.20) = 0.10 + 2.00 = 2.10.

-Wide and uncertain. L = 0.10, U = 0.90, o = 0.20.
Outcome falls inside, so only the width term contributes. IS = 0.80 - 0.10 = 0.80.

-Tight and correct. L = 0.15, U = 0.25, o = 0.20.
IS = 0.25 - 0.15 = 0.10.

2.) **CRPS**

A potentially better approach would be to use the continuous ranked probability score (CRPS) from the fitted beta of the forecast and the actual outcome. We’d compute this by integrating the squared difference between fitted beta CDF and the step function at observed outcome. It generalises absolute error. When the forecast is a point mass, CRPS reduces to |forecast - o|. So a point forecaster and a distributional forecaster can be put on the same axis and compared directly. This is the reason Gneiting and Raftery highlight CRPS over the logarithmic score for general use (p. 367, last paragraph). Intuitively, I think CRPS measures how far the distribution is from being a perfect spike at the outcome; **CRPS is the area under the squared-difference curve between the forecast CDF and the step function that jumps from 0 to 1 at the actual outcome**.

*“CRPS provides a direct way to compare deterministic and probabilistic forecasts”*

CRPS(F_i, o_i) = integral (F_i(y) - 1{y >= o_i})^2 dy   

It is the integral of the Brier scores for the binary forecasts {Y >= y} at every threshold y. So CRPS aggregates Brier scores over all possible thresholds, weighted equally (Matheson and Winkler 1976; Hersbach 2000). If F collapses to a point mass at a single value, CRPS reduces to the absolute error |forecast - x|. So CRPS is a direct generalisation of mean absolute error to distributional forecasts. The integral often can also be evaluated in closed form as

```
CRPS*(F, x) = E_F |X - x|  -  (1/2) E_F |X - X'|
```

where X and X' are independent copies of a random variable with distribution F, assumed to have finite first moment. Equation (21) in the paper.

Prompt:

https://github.com/lyptus-research/cyber-task-horizons-data/tree/main/data/methodology

 

# 

## 

## Continuous Ranked Probability Score (recommended)

The CRPS has a straightforward intuition. Imagine sweeping through every possible threshold y from 0 to 1 and asking a yes/no question: "Is the true value at or below y?"

The forecast answers with a probability F(y), where F is the CDF of our fitted Beta distribution. Reality answers with 0 or 1. The Brier score at that threshold is where o is the observed outcome.

$$
(F(y) - \mathbf{1}\{y \geq o\})^2
$$

The CRPS integrates this squared error over all thresholds:

$$
\text{CRPS}(F, o) = \int_0^1 \left(F(y) - \mathbf{1}\{y \geq o\}\right)^2 dy
$$

There's an equivalent form that's easier to compute and more interpretable:

$$
\text{CRPS}(F, o) = \mathbb{E}_F|X - o| - \frac{1}{2}\mathbb{E}_F|X - X'|
$$

where X and X' are independent draws from the forecast distribution.

The first term measures how far the forecast is from the truth on average. The second term measures how spread out the forecast is. A tight forecast has a small second term, so it needs to be accurate to score well. A diffuse forecast has a large second term that partially offsets the error penalty.

The units are the same as the outcome, so a CRPS of 0.05 means the forecast is, in an integrated sense, about 5 percentage points away from a perfect point mass at the truth. This makes it directly comparable to mean absolute error. if the forecast were deterministic (a point mass), CRPS reduces exactly to absolute error. This property is useful: a distributional forecast and a point forecast can be compared on the same scale.

One practical advantage is robustness. Unlike the logarithmic score, CRPS doesn't blow up when the outcome lands in a low-probability region. If the LLM forecasts tight mass around 0.6 but the truth is 0.1, the score is large but finite. Gneiting and Raftery's case study in Section 8 of their paper shows CRPS giving stable, interpretable results where log score was unstable and other alternatives produced nonsensical rankings.

The main caveat is computational the integral doesn't have a trivial closed form for arbitrary distributions. For Beta distributions it can be computed exactly though.

## Brier Score on a Point Estimate

The Brier score is (p - o)^2, where p is a probability forecast and o is the observed outcome. The problem is that we don't have a point forecast, but since we have a distribution, one option is to extract a single number, say the median of our fitted Beta, and score that. This works, but it discards the uncertainty we elicited. A forecast of "median 50%, very tight" and "median 50%, very wide" would receive identical as Jakub mentioned.

## Interval Score

The interval score evaluates a prediction interval $[L, U]$ by combining width and coverage:

$$
\text{IS}_\alpha(L, U; o) = (U - L) + \frac{2}{\alpha}(L - o)\mathbf{1}\{o < L\} + \frac{2}{\alpha}(o - U)\mathbf{1}\{o > U\}
$$

The score rewards narrow intervals but penalizes the forecast if the outcome falls outside. The penalty scales inversely with alpha: missing a 90% interval is worse than missing a 50% interval.

This is clean and interpretable. For our setup, we could use the 25th and 75th percentiles as a 50% interval, but we elicit three percentiles but only use two of them. The median carries information. if the 25th is 0.2, the 50th is 0.3, and the 75th is 0.6, the distribution is right-skewed, but the interval score doesn't see that. The interval score is proper, so it's not a bad choice, just somewhat less efficient than using the full distribution. 

## Logarithmic Score

The log score evaluates the density at the observed outcome:

$$
\text{LogS}(f, o) = \log f(o)
$$

where f is the probability density function of the forecast.

It has connections to likelihood, entropy, and information theory, and it's strictly proper. The LLM maximizes expected log score by reporting its true predictive density.The difficulty is that it's harsh and sometimes fragile. If the outcome lands in a region where the forecast density is low, the score can become very negative. Gneiting and Raftery specifically mention this issue in their Section 8 case study. They note that the log score "was unstable when the predictive density was tight near the observation."

## Mean Absolute Error

If we just take the median of the forecast and compute $|\text{median}(F) - o|$, we get mean absolute error. It's simple and interpretable, but it suffers from the same issue as Brier on a point estimate: all the uncertainty information is discarded. MAE isn't even a proper scoring rule for distributional forecasts because it doesn't distinguish between tight and wide distributions with the same median.



