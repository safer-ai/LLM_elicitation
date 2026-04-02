# Week 2

## Jeff

Notes on self-consistency checks

**types of self consistency**

- additivity (p(success) = 1- p(failure))
- test-retest reliability (spread of forecasts across multiple runs with identical settings)
- conjunctive/disjunctive consistency (p(A&B) ≤ P(A||B))
- decomposition-aggregation consistency
    - What is probability of x
    - what is p(x|y)*p(y) + p(x|!y)*p(!y)
    - $L_{\text{decomp}} = \left| P(A) - \sum_i P(A \mid B_i), P(B_i) \right|$

**Notes from literature**

- violations in complementary symmetry ([Zhu 2025](https://arxiv.org/pdf/2505.07883))
    - this is probably easiest to implement, but requires some prompt engineering to make sure the question is asked both ways
    - defined incoherence as |P + ¬P − 1|, found incoherence of ~0.13
        - *aside* this 0.13 is much higher than the .013 I found with sonnet 4.6, though I didn’t run a ton of samples and this was the easiest possible implementation.
    - normalizing (divide p and !p by sum of both) was very useful for reducing incoherence.
- sycophantic agreement ([Dhuliawala et al., 2023](https://arxiv.org/pdf/2309.11495)) they state that yes/no question formulation has a yes bias, but I can’t find the actual data for this
- ([Xiong et al. 2024](https://arxiv.org/abs/2306.13063)) used average confidence across multiple runs and found this improved self-consistency. Not identical to what I’m measuring here though.
- [OpenEstimate](https://github.com/alanarenda/openestimate) ([paper](https://arxiv.org/pdf/2510.15096)) is a benchmark for evaluating probabilistic estimation. This would very much be worth looking in to. It’s a bit more straightforward/faster, and has some convenient things built in, but might not be as directly relevant because I’d be building tests around a different pipeline.

**Tests I’m focusing on**

- test-retest consistency
    - This is the easiest and most generalizable. Comparing p(A) across multiple runs is straightforward, and also allows for things like evaluating the impact of various prompts.
        - Measure Cronback’s alpha?
- Additivity/incoherence
    - this one is slightly less straightforward because it requires manipulating the prompt/question format to work in the opposite direction, and keeping these labeled differently

**Results**

*Test-retest*

*Additivity*

I tested this in the simplest possible way - I change the prompt in initial_probability_estimation to as for the probability the model FAILED a given step instead of succeeded, and reran the same tasks with that prompt instead and everything else the same. 

Additivity was surprisingly well respected. The mean incoherence was 1.33% (SD 1%) across 48 pairs (4x4 runs on 3 tasks) for sonnet 4.6.

Limitations

This is a little hacky, because there are many other things in the prompt that encourage the model to think about success (e.g., thinking in terms of uplift, or the base rates being in terms of success). So the model might just make a given prediction and then subtract it from one to get the negative prediction. But it’s sufficient to test the analysis

Issues

- had some issues with the parsing not working as well for the negative prompt? This only happened on one run, and the issue was the model not outputting the right tags. Little weird though.
- storing the API keys in the yaml feels a little risky if I’m using the yamls to config experiments. Would be better to have these in a .env, but this would require some other changes to the main src
- the workflow chokes on recent GPT models because they use different max token language (max_completion_tokens instead of max_tokens)
    - this seems to be hardcoded in [workflow.py](http://workflow.py) and then passed as required in llm_api.py. I think this is worth fixing but maybe not for me to do right now. Because this code wont be part of the main project we are working on
    - Instead I just took the lazy route and commented that parameter out of openai calls in llm_api.py

**Madhav**

Pilot study on variance.

Based on the initial results,

1. Persona assignment does not matter. Across 6 experiments (2 models × 3 baselines, 600
estimates), persona explains ≤14% of variance. Five of six show no significant effect; the one
nominally significant result fails Bonferroni correction and does not replicate.
2. Model identity matters. Across 3 experiments (3 models × 3 baselines, ∼890 estimates),
model choice explains 28–65% of variance (all p < 0.0001). Model F-statistics are 15–311×
larger than persona F-statistics

**Full Report:**

[experiment_report_standalone (1).pdf](attachment:1b707115-e5da-41ea-b5fd-f4dabc053d0a:experiment_report_standalone_(1).pdf)

Issues to be fixed: 

-I am computing the means of medians and using them in statistical tests, hence that should be properly addressed

-I need to look at individual distributions for any abnormalities/skweness