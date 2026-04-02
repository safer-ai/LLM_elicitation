# Week 1

**Jeff**

Thoughts reading through the project plan

- I think step 4 is interesting in general, and would be worth knowing whether in sample performance to predict out of sample performance would be possible.
    - There are a lot of ways to break this down as well.
        - Extrapolating from benchmark scores: here is the definition of the benchmark(s) and the types of tasks, what is the expected performance on this other benchmark? This is useful because you can take published estimates (e.g., from frontier model cards) instead of running the benchmarks manually, which is required for the next option. If this works well it’s likely more useful since the turnaround would be faster for new models/benchmarks, and less overhead is required.
        - Feed in 2 questions and directly ask “if model x succeeded at question 1, what is the probability it also succeeded on question 2?”. This could generate a ton of data very easily.
        - Different approach: use LLMs to grade question difficulty and relevance, then feed those in to a regression or some other mathematical model. This is a little less black boxy than the first option, though it includes more steps. I’m not sure which would be expected to be more accurate a priori
        - 
- Some points of concern
    - Comparing benchmark vs. benchmark may not translate well when asking about more sensitive cyber issues, particularly if labs are training against this as an undesired behavior.
        - e.g., Ask Opus 4.6 if Opus 5 will perform well on a certain benchmark and it says yes, but ask if it will be capable of performing <some undesirable offensive task> and it says no because it has been trained to not consider those tasks.
        - This is just a hunch and isn’t based on anything real
    - Data on uplift is extremely spotty, and self-report or expert estimated uplift has overestimated actual uplift significantly consistently, as far as I’m aware (see [here](https://arxiv.org/abs/2602.16703) and [here](https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/)). Uplift seems pretty core to this methodology, and it’s importantly different from ‘model capability’ which is really what it seems like this project is eliciting.

**Madhav**

 Takeaway:

- Among the 6 steps mentioned, developing the ground truth seems important and the most challenging. LLMs have been criticized for relying too heavily on memorizing forecasting data instead of reasoning about it. This paper shows that LLMs have memorized economic and financial data, recalling precise values with high accuracy, [see here](https://arxiv.org/abs/2504.14765). Since published benchmark scores are present in the training data, conditional framing of benchmark-to-benchmark tasks can reduce, but not eliminate, the retrieval problem.
- (Step 1) Telling the LLM to act like different experts ('Malware Engineer', 'Threat Analyst') may fail to capture meaningful diversity, as persona variables account for <10% of output variance in most tasks ([Hu & Collier, 2024](https://arxiv.org/abs/2402.10811)). However, personas are most effective when the role genuinely predicts different reasoning which is an open empirical question for cybersecurity risk estimation. Rather than surface role labels, structural reasoning diversity (giving different analytical frameworks per persona) is more likely to generate genuine output differences ([Hu et al., EMNLP 2025](https://arxiv.org/abs/2412.15238))
- (Step 3) LLM have a incoherent probability judgement
    
    [paper](https://arxiv.org/abs/2401.16646) However, they consistently exhibit systematic biases that mirror human cognition. The authors suggest that this happens because the LLMs' autoregressive training causes them to act as an implicit “Bayesian Sampler” that pulls information from a learned prior rather than calculating the strict math. Once we get the flawed results, we can use probabilistic identities (like testing conjunctions and disjunctions) to measure the exact degree of incoherence and recalibrate the output.
    
- (Step 2) Different scaffolding and prompt formats seem to generate different behavioral and statistical profiles, making a simple, universal characterization of a model's bias difficult. For instance, during my automated grading work, LLMs would be optimistic about student work and assign higher scores to everyone unless explicitly told not to do so (the bias was systematic and fixable, but that’s for one particular task). In the case of Safer AI’s risk modelling, the report noted that LLMs were more “conservative” in assigning probabilities. In this https://arxiv.org/abs/2507.04562, the authors tried narrative prompt and direct prompts. The models were overconfident with the direct prompt and vice versa. It seems dependent on task, model and prompts complicating things with no single answer.