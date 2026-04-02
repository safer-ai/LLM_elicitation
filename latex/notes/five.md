# Week 5

**Jeff**

**Madhav**

I experimented with changing baseline percentage of attack success from 10% to 90% (removed the CI) and kept everything else. I ran 10 experiments for each baseline percentage under one expert with Sonnet 4.6. The result:

![baseline_uplift_plot.png](attachment:2e32278b-c64f-45b2-a231-1c5a2f301400:baseline_uplift_plot.png)

I wasn’t expecting this graph. The above linear-like graph seems interesting. As discussed in the meeting, the probabilities seem to anchor to the baseline. Towards higher percentages, the uplift is smaller (ceiling effect?). I can’t confidently say whether this is additive, multipicative, or something else (might need more experiments on different tasks/benchmarks. I will push it forward if we are interested to test this rigorously). I tried plotting a similar plot with number of actors:

![numactors_baseline_uplift_plot.png](attachment:d843d01e-4ba7-42c6-af83-767415947493:numactors_baseline_uplift_plot.png)

With the number of actors as well, it seems a strong linear fit. There’s no ceiling effect as the number of actors is unbounded. This also might require a larger sample size (tested across different tasks/benchmarks) to be confident about any conclusions. Based on this small test, it seems that LLM anchors heavily to the baseline given. This raises a question whether LLMs are reasoning independently about the true probability and number of actors, or just simply uplifting from a baseline.

**Prompt Sensitivity**

Finished the remaining experiments from last week about changing the prompts. I tried four cases with trimming. The main prompt seems to have five main steps:

-Initial scenario and benchmark context

-4-section capability breakdown (task decomposition, difficulty assessment, capability correlation, boundaries)

-Estimation task (baseline, step name, CI)

-Reasoning (establish ranges, check confidence, reality check)

-Note on distribution and output format

I tried seven conditions below:

| Condition | What Was Removed |
| --- | --- |
| **control** | Nothing (full prompt) |
| **no_ci** | Removed uncertainty range (5th/95th percentile: "20%-90%") from baseline |
| **no_baseline** | Removed point estimate ("50% chance of success") from baseline |
| **no_baseline_no_ci** | Removed entire baseline statement |
| **skip_analysis** | Removed detailed capability analysis instructions (4-section breakdown of task difficulty, capability correlation, boundaries) |
| **trim_reasoning** | Removed 3-phase reasoning scaffold (Phase 1: establish ranges, Phase 2: check confidence, Phase 3: reality check) |
| **trim_all** | Removed capability analysis + reasoning scaffold + technical analysis output |

The findings with trimming prompts are very surprising. 

![w1_chart.png](attachment:aa297566-037c-47bf-ab5d-e84a1a2a0f51:w1_chart.png)

In my preliminary experiment, trimming the entire capability analysis, reasoning scaffold and technical analysis output didn’t change model response much, which is quite surprising. Also, it’s worth noting that model’s responses converge as we trim the prompts, possibly due to reduction in noise due to shorter/straightforward prompts. If we can rigorously test and verify that trim_all works consistently across different tasks/benchmarks/models, and if the findings are consistent across many experiments, there might be a chance that we can reduce the word count of prompt from 615 words to around 100 words, though proper validation is required before drawing any strong conclusions. 

**Intra-benchmark**

Spent some time looking into the livebench dataset (https://huggingface.co/datasets/livebench/model_judgment). As discussed on slack, having per task difficulty seems a good thing to have, which this dataset does, but it’s within the same benchmark. There’s also a reasonable spread in model performance on tasks. The model scores on each task range from 0 to 1. There seem two possibilities, whether to treat them as continious or pick a threshold score ≥ 0.5 to binarize. I was thinking something like this:

We have 195 models in the dataset, 3 domains, individual model scores on individual tasks. 

Setup (oversimplified, ignores capability ceiling, might)

-Pick source task A (from domain 1, e.g., coding). Pick target task B (from domain 2, e.g., language) Ask LLM: "Model solved task A. What's P(solves task B)?" Ground truth: (# models that solved both A and B) / (# models that solved A). This could potentially help us understand the correlation structure between domains.