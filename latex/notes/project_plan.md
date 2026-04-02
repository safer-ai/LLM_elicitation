# Project plan

[Jakub initial notes](https://www.notion.so/Jakub-initial-notes-3085f8b9219680f4b512c1ee8273cb3b?pvs=21)

# Motivation

In fields for which potential harm is significant (financial, aviation, nuclear), risk modelling (concrete mapping from hazards—the source of potential risk—to harms—real adverse events) is the pillar of risk management systems. It allows organisations and regulators to establish common standards concerning best practices for risk management. Risk modelling represents a major gap in existing risk management methodologies for AI systems, which are fast becoming a high-risk industry. 

One reason for the lack of development in quantitative risk models for AI systems is the disconnect between evaluations/benchmarks and the real-life risk scenarios that they’re supposed to be a proxy of. Currently, we resolve this issue by using judgemental estimation—a procedure in which we elicit quantitative estimates for parameters in a risk model through structured surveys and workshops with domain experts. However, this approach has several limitations: being a domain expert in a risk-relevant field does not make someone good at probabilistic inference, and such expertise may be rare, making expert time extremely costly.

Another reason why risk modelling is challenging in the context of AI systems is their sociotechnical positioning. This technology is and may increasingly become pervasive across society, integrating into and shaping many complex and interdependent systems. This means that high-fidelity risk models incur a massive complexity burden—to accurately capture sophisticated risk dynamics, modellers are pushed to use many parameters to make use of all available information and resolve questions of dependency between variables. This interacts poorly with the constraints of judgemental estimation—compounding of errors, rising costs, and increasingly specific and challenging to estimate parameters all come naturally with scaling risk models in this way.

In order to address these challenges, we propose the use of LLM-simulated domain expert estimators for the quantification of risk models. By using these models, we can create verifiable, and reproducible elicitation experiments that would not be possible with humans, potentially asking many more questions of them at a fraction of the cost. The issue is that we need to demonstrate that they can work.

# Objectives

While the team at SaferAI has conducted some initial experiments with the use of LLM-simulated experts, our investigation thus far has lacked a systematic approach to the evaluation of how effective this approach can be. This means that we cannot be confident in the use of these methods. The goal of this project is to:

1. Establish clear methods for controlling and examining the outputs of LLM estimators
2. Develop a procedure to optimise LLM-simulated experts for the elicitation task that can be run periodically when new models are released
3. Improve the state of LLM elicitation as a substitution for human expertise with clear insight into what we can learn from LLM-estimated risk models and what we can’t learn.

# Approach

## Step 1: Establish Diverse Estimators

Currently we make use of “simulated expert profiles” to define a set of distinct experts with diverse outputs over estimated quantities. However, the effect of these profiles has not been investigated comprehensively, and we have some evidence to believe that these profiles are not, in fact generating any diversity in terms of elicited values. The objective of this step is to develop a procedure for estimating the effect of personality prompting on the output of the estimation process. We can evaluate the variance of the expert estimates across several identical rollouts to compare cross-expert and per-expert diversity in responses. Ideally, this would establish a concrete “expert diversity temperature” parameter, implemented via distinct prompt formats, which can be used to control the influence of the expert prompt on the elicited outcomes → lower temperature makes experts give more similar answers to each other, higher temperature makes them provide more different answers. 

## Step 2: Statistical probe of LLM elicitations

Another feature of our current estimators that has not been explored is the statistical profile of responses given. We know that they are capable of producing plausible sounding numbers, but we are unsure if these numbers can be taken as the output of some reasoning process or if they are just replication of statistical patterns. We need to validate this question statistically.

We currently elicit three model parameters at each estimation step: the estimated mode of the distribution and the 5th and 95th percentile confidence estimates. We have found that even when prompted with baseline confidence intervals, sometimes the estimator does not attend to these and gives responses that contradict this baseline without sufficient justification to support such divergence. It would be meaningful to observe if such divergences follow any particular statistical pattern. I.e., on the first order, are the models simply adding and subtracting 10% to their modal estimate? Can we establish clear patterns as to when such divergences occur? Is there other statistical regularities that we observe in model estimates that should not be there, given the diversity in the set of parameters being elicited? 

Another key line of insight here would involve testing the effects of modifying prompts—intuitively if the model outputs are sensitive to small, potentially irrelevant changes in prompt or prompt structure, it is unlikely that we can take the results as solid estimates. Relatedly, if we modify anchoring information in the plots (such as baseline probability), how does this affect downstream estimates?

## Step 3: Establish a set of self-consistency checks

In the absence of clear ground-truth signals, it is important that our estimators are robust to changes in how a particular elicitation is framed. For example, model estimates should be self-consistent in expectation regardless of if we ask “what is the probability of ***success*** of the threat actor evading spam filters?” or “what is the probability of ***failure*** of the threat actor evading spam filters?”. Such self-consistency checks support the overall statistical validity of the process - the model is coming to similar conclusions from different angles, and provides some confidence that the estimation procedure is not just hallucination, but is taking key real considerations into account. Establishing an effective set of self-consistency checks would likely go beyond simple inversion—we can observe other forms of self-consistency such as robust aggregation, including conjunctive/disjunctive accuracy: “is the probability of A and/or B lower/higher than the probability of each individually”?

- Note to self regarding internal the consistency checks
    
    For a binary outcome like ‘will X happen or not’, the self-consistency condition is very simple: $P(X)+P(¬X)= 1$. But in the case of LLM elicitation, X is not a binary outcome — it’s a distribution over probabilities, i.e. ‘the probability that an AI system will/will not be able to accomplish some step in the attack killchain’. Thus, we need to work with continuous probability distributions: $P(X\le x)=\int_0^x f(X)dX$ and $P(X>x)=\int_x^1 f(X)dX$.
    To check consistency, we can ask the LLM estimator: ‘what are the 25th/50th/75th percentiles of the probability distribution that the AI will complete this attack step’. Let’s denote the results as $\vec{a}=(a_1, a_2, a_3)$. Then, we ask:  ‘what are the 75th/50th/25th percentiles of the probability distribution that the AI will NOT complete this attack step’ and denote this as $\vec{b}=(b_1, b_2, b_3)$. The two sets of estimates are related by the following simple condition: $\vec{b}=(1-a_1, 1-a_2, 1-a_3)$.
    In other words, the point-wise percentiles have to individually sum to 1. If we want to turn this into a scalar loss function, we simply do the L2 norm:
    
    $L=\frac{1}{\sqrt{n}} ||\vec{a}+\vec{b}-\vec{1}||_2$,
    
    where the sqrt(n) is for normalisation, so that this works with any number *n* of elicited percentiles.
    
    Alternatively, we can also fit two Beta distributions — one called *f(X*) based on $\vec{a}$ and one called *g(X)* based on $\vec{b}$ — and the loss function then becomes:
    
     $L = \int_0^1|f(X) - g(1-X)|\, dX$
    

## Step 4: Develop an “adjacent ground truth” accuracy estimation procedure

Even self consistency is not sufficient to establish that models are effective predictors. For this, we need to establish a source of “ground truth”, empirical observations that the model does not have access to which validate that estimates made by the model are accurate (or reflect *how* inaccurate the model is). Some work has been invested in using LLMs for forecasting tasks, which shares some similarities with our judgemental estimation procedure and has the benefits of readily available forecasting data. However, this is a highly contested space, with several organisations funding multiple researchers to investigate how to make LLMs good at such tasks. Furthermore, it is not exactly the same question as the one we are trying to ask: we want to know “conditioned on a particular capability metric, how likely is the agent to be able to complete (or support an actor to complete) this step on the pathway from hazard to harm?”. 

This is a form of conditional estimation, rather than arbitrary prediction of future events. It can take advantage of structural assumptions (particularly in estimating current capabilities) that are not possible for arbitrary forecasting. In this way, we escape the retrieval problem highlighted in previous work. One approach for gathering ground truth data in our setting is to estimate the conditional probability of a model being able to complete a task in one benchmark, given its score in another related benchmark. This provides a question of very similar form to the one that we are asking, but with the benefit of verifiable data.

It would be highly valuable to come up with other such experiments where (1) the ground truth is available (2) the prediction relates to benchmark scores/probabilities of success on a task (3) the prediction is ideally in the domains we are interested in (cyber and CBRN). 

## Step 5: Establish an optimisation procedure

Equipped with both internal consistency checks and external validation, we can proceed to develop an optimisation procedure for fitting the LLM estimators to optimise these scores. This procedure will mean that even if LLMs are not currently capable of effectively replacing human expert judgement in this context, we will have a process by which we can test an estimate of how good they are. We have to be careful during this phase not to overfit, perhaps some of the statistics developed in step 2 will provide some indication of this, as well as held-out elements of the scoring methods above.

## Step 6 (stretch): Comparison with humans

With a clear statistical characterisation, evaluation methods, and optimisation procedure for the LLM experts, the final test would be to compare the judgemental estimation capabilities of the models with real human experts. We could consider comparison across two groups of experts: expert forecasters and domain experts in a particular task. We would have to establish some metric for comparison informed by the above to determine relative capabilities of the two/three groups of estimators, held out-data which neither the models nor the humans have access to.

# Resources

## SaferAI

- https://www.safer-ai.org/beyond-benchmarks-in-ai-cyber-risk-assessment ← this is an overview of the 3 risk modelling papers we released in December. You can also find the links to the full papers inside. **Please read this carefully so that you understand the idea behind our risk modelling methodology, why we need expert elicitation and why we want to do it with LLMs.**
- https://www.safer-ai.org/technical-report-llm-simulated-expert-judgement-for-quantitative-ai-risk-estimation ← some early results from last year about testing LLM elicitation for risk modelling. You can think of this as v0 of our work in this project.

## Risk Modelling

- Our risk modelling methodology as applied to cybersecurity risk - mostly done with LLM estimators: https://arxiv.org/abs/2512.08864
- **Hemming, V. et al. (2018).** "A Practical Guide to Structured Expert Elicitation Using the IDEA Protocol." [*Methods in Ecology and Evolution*](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.12857)
- **Barons, M.J. et al. (2022).** "Balancing the Elicitation Burden and the Richness of Expert Input When Quantifying Discrete Bayesian Networks." [*Risk Analysis*.](https://onlinelibrary.wiley.com/doi/full/10.1111/risa.13772)
- https://onlinelibrary.wiley.com/doi/book/10.1002/0470033312

## LLM Estimators + Forecasting

- A blog post we did with initial results on LLM estimators: https://www.safer-ai.org/technical-report-llm-simulated-expert-judgement-for-quantitative-ai-risk-estimation
- **Paleka, D. et al. (2025).** “Pitfalls in Evaluating Language Model Forecasters” [*arXiv*](https://arxiv.org/abs/2506.00723).
- **Schoenegger, P. et al. (2024).** "Wisdom of the Silicon Crowd: LLM Ensemble Prediction Capabilities Rival Human Crowd Accuracy." [*Science Advances*](https://www.science.org/doi/10.1126/sciadv.adp1528)
- **Shorinwa, O. et al. (2025).** "A Survey on Uncertainty Quantification of Large Language Models." [*ACM Computing Surveys*](https://dl.acm.org/doi/full/10.1145/3744238)
- **Epstein, E.L. et al. (2025).** "LLMs are Overconfident: Evaluating Confidence Interval Calibration." [*arXiv*](https://arxiv.org/abs/2510.26995).
- **Lu, J. (2025).** "Evaluating LLMs on Real-World Forecasting Against Expert Forecasters." [*arXiv*](https://arxiv.org/abs/2507.04562).
- **Schoenegger, P. et al. (2024).** "LLM Assistants Improve Human Forecasting Accuracy." [*ACM Transactions on Interactive Intelligent Systems*](https://dspace.mit.edu/handle/1721.1/158063).
- **Halawi, D. et al. (2024).** "Approaching Human-Level Forecasting with Language Models." *arXiv*.
- **Perrault, A. et al. (2024).** "Can Language Models Use Forecasting Strategies?" [*arXiv*](https://arxiv.org/abs/2406.04446).
- https://arxiv.org/abs/2402.10811
- https://arxiv.org/abs/2401.16646
- https://arxiv.org/abs/2602.08889