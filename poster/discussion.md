# SPAR Discussion **May 12, 2026 11:00 AM (GMT+1)**

![image.png](attachment:accb9544-ad63-4189-aa4e-cae9b4517978:image.png)

---

Firstly, let’s clarify the confusion regarding K - `n_target_tasks_per_cell`. There are three possible interpretations of this setting:

1. (Madhav’s interpretation) Only one target task is included in the prompt for each elicitation. We do `K` separate elicitations, with one target task per elicitation. The CSV gets appended by `K` rows.
2. (Jeff’s interpretation) All `K` target tasks are passed within a *single elicitation.* The LLM estimator is asked to make `K` separate predictions for each target task, all part of the same LLM completion. The CSV gets appended by `K` rows.
    1. I’m not sure how qualitatively different this is from (1). 
3. `K` tasks are passed within a single prompt *as example tasks*. I.e. something like ‘you are asked to estimate the probability of solving task X, but here are `K` easier tasks that represent this bin’.
    1. This interpretation doesn’t make sense in our setting, as we are eliciting the probability of a model solving a single task, not the average probability of solving the target bin.

**The correct interpretation is (1).** Note that this is in contrast to how *source* bins work, where the setting `n_examples_per_source_bin`actors according to (3).

---

Now, what sort of experiment are we actually trying to run? Let’s establish some notation:

- *l -* a given LLM
- *t* - a given task, either from source or target bin
- *O -* observed outcome (i.e. ground truth)
- *P(s(X)) -* probability of success in a setup X
- *i(…) -* information (i.e. context) about results on other bins
- *TB* - target bin
- *SB* - source bin(s)

With this notation, we have the following experiments:

## A: Single model, single target task

We are measuring: $P(s(t_{TB}, l) | i(l, t_{SB}))$ — we elicit a single P(…) from the LLM forecaster, and average these *manually* over all target tasks so that we can compare to ground truth:

Our ground truth is: $O(t_{TB}, l)$ — a single pass/fail observation *O*(…) for a single LLM, on a single target task.

Then, we manually aggregate these over all target tasks in TB to get a single Brier score:

$\frac{1}{|TB|} \sum_{t\in{TB}}^{|TB|} (P-O)^2$

This approach is gonna give us a set of Brier scores, one per LLM. We can then also aggregate these Brier score into a single number.

- t_SB can be either a single task from a single source bin, multiple tasks from multiple source bins, or a variation of these settings. This is controlled by `n_examples_per_source_bin` and `source_profile`.
- Problems:
    - large variation from task to task
    - requires a lot of elicitations — expensive
- Advantages:
    - avoids the ‘non-stationary’ problem (see below)
- This is the last plot that Madhav uploaded:
    
    ![image.png](attachment:1f626500-4a9c-4be7-8bac-294b11ed70ee:image.png)
    

## B: An ‘average model’, single target task

We are measuring: $P(s(t_{TB}, \bar{l})\, |\, i(\bar{l}, t_{SB}))$, where: $\bar{l}$ is an average over LLMs’ performance. So in this case, $i(\bar{l}, t_{SB})$ means information about the *averaged* performance on source task(s) of all 12 Lyptus LLMs. In the elicitation prompts, we’re asking the forecaster to predict the performance of an ‘typical’ LLM.

Our ground truth is $GT = \frac{1}{|LLMs|} \sum_{l\in\{LLMs\}}^{|LLMs|} O(t_{TB}, l)$

Our metric here would be some kind of a distance metric between P and GT, averaged over all target tasks — maybe the CRPS?

$\frac{1}{|TB|} \sum_{t\in{TB}}^{|TB|} CRPS\left[ P(t_{TB})-GT(t_{TB}) \right]$

- advantage: cheaper than above
- disadvantage: it suffers from the ‘non-stationary’ problem (see below)

## C: Single model, an ‘average target task’

We are measuring: $P(s(\bar{t}_{TB}, l)\, |\, i(l, \bar{t}_{SB}))$

Our ground truth is: $GT = \frac{1}{|TB|} \sum_{t\in{TB}} O(t_{TB}, l)$

Similarly to B, our metric here would be something like:

$\frac{1}{|LLMs|} \sum_{l\in{\{LLMs\}}}^{|LLMs|} CRPS\left[ P(l)-GT(l) \right]$

---

### Summary

I don’t really have a preference for one of these options and I don’t think that averaging over models is be default bad — see the explanation below.

The only thing I want to remark at this point is that in the original plots Madhav uploaded, I think we were mixing two cases:

![image.png](attachment:01d8c7b5-9438-4b8b-b910-d52e6bac33f0:image.png)

That is, we were performing elicitation according to A, but displaying ground truth according to C. This doesn’t really make sense, I think. Jeff alluded to this on Slack:

> I think for the line plots, this comparison is not particularly helpful. Unless I'm misunderstanding, the predicted probability (dashed line) is for a single task/model combo at each point, while the ground truth (solid line) is the average pass rate for the entire bin (for a single model at each point). A good match here would only occur if the correct prediction for the target task was exactly equal to the average pass rate within the bin for each model, and that assumption isn't justified.
> 

## The ‘non-stationary’ problem

The issue with averaging across multiple LLMs - in particular, LLMs from different generations - is that their capability profiles are going to be different. Ultimately, when we are doing P(MITRE step | benchmark tasks), we are asking the LLM estimator to extrapolate from a given observed slice of this model’s capability profile to a different corner of its capabilities.

If we now start averaging over LLMs like GPT-4 and GPT-5.5, these models might have very different capability profiles, which we will lose this information. For example, if we expect earlier models to have a more jagged frontier (e.g. o3 was good at maths but struggled with counting ‘r’s in ‘strawberry’), then including this model in our experiment will skew the ground-truth. Ultimately, we are interested in doing risk models for the latest LLMs, not for LLMs that came out 2 years ago, so we shouldn’t be including the old ones in our experiments and asking the forecasters to reason over them.

In practical terms, I think it’s fine to just be a bit more selective and choose LLMs that came out within the last ~12 months. Averaging across LLMs from the same generation should be fine. In fact it might be desired. This brings me to a ‘philosophical’ point.

## Should we be modelling a specific LLM or a ‘generic frontier LLM’?

When we are doing LLM elicitation for our risk models, what are we actually trying to get out of it?

1. are we interested in one specific LLM?
2. or are we interested in a generic frontier LLM?

Of course, (2) wouldn’t cover the whole spectrum of latest releases, e.g. we wouldn’t include something like Llama-70B or biological models. But to give a concrete example: for cyber risk, are we interested in (1) Claude Mythos, or (2) frontier cyber-capable models, e.g. Claude Mythos / Opus 4.7 / Gemini 3 Pro?

I know Jeff seemed to prefer option (1), on the other hand I’m leaning towards option (2). My reasons are:

- from the perspective of policymaking/regulation/prioritising defensive mitigations, we don’t really care about a single LLM’s performance — if you’re trying to prepare society for an onslaught of offensive cyber capabilities, you just want the whole of ‘frontier cyber risk’ to be modelled, not a specific LLM. You cannot predict in advance that someone is going to hack you specifically with Mythos or GPT-5.5. Most of the time, you wouldn’t even know that.
    - a counter-argument here is that governments might be interested in setting regulatory thresholds for the release of new models. For example, every new model released by Anthropic on the EU market could undergo an audit, as part of which we would do risk modelling on it, and if the outputs of this exercise exceed some acceptable risk threshold, the model is not permitted. There are two issues with this idea:
        - I think we are nowhere near the level of precision that would allow us to make such go/no-go decisions for model releases
        - It seems like the European Commission is not that interested in this approach anyway (Matt can say more about this)
- it might seem that when we do P(MITRE step | score on benchmark A, score on benchmark B, …), we are describing a single, specific LLM that happens to achieve these scores on these benchmarks. But is it really so? In my opinion, the full capability profile of an LLM can be very complicated, and we can imagine a situation where two LLMs, say Opus 4.7 and GPT5.5, achieve identical scores on BountyBench+CyBench, yet perform very differently on other cyberbenchmarks, have different METR time horizons, etc.
This is to say, I think when we do P(MITRE step | …), we are still performing a lot of implicit averaging over all other *unobserved* capabilities.
My point is, when we’re thinking about which of the experimental settings to choose, I don’t think it’s right to think that we must study single LLMs, because at the end of the day our risk modelling procedure involves modelling a single LLM only. I think this is a flawed interpretation. But feel free to disagree.

Madhav:
-Estimate the cost for model sweep and run the model sweep experiments with setup A

-Prepare the poster by Thursday