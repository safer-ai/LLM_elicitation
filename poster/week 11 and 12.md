# week 11

## Madhav

I was looking through the intra-benchmark code Jakub shared. The results from some conditions Jakub ran was interesting.

All runs use claude-sonnet-4-6, 2 experts, 1 Delphi round, 5 bins, K=1 target task per cell.
Metrics computed by `analyse_results.py` (groups by condition_id, keeps final Delphi round per cell).

These were the experimental conditions. A, B and C were run were made by Jakub. I ran a pilot on E. I might be wrong, but based on what I understand each of these conditions are as follows:

**A.) all_except_target, thinking OFF**

- For each target bin j, show all 4 other bins as source context (i.e. 4 bins × 3 tasks = 12 source tasks).
- Extended thinking disabled (temperature=0.8 based on config filename, but thinking off).
- 5 cells per model (one per target bin). 60 total cells, 240 API calls.
- The forecaster sees a broad capability profile spanning the whole benchmark.

**B.)  all_except_target, thinking ON**

- Identical to A but with extended thinking enabled (budget=10k tokens).
- Same 5 cells per model, 60 total, 240 API calls.
- Tests whether chain-of-thought reasoning improves calibration vs. no-thinking.

**C.) single_bin, all 20 pairs, thinking ON**

- For each (source bin i, target bin j) pair where i≠j, show only source bin i as context.
- All 20 ordered pairs are run: 5×4=20 pairs × 12 models = 240 cells, 960 API calls.
- The forecaster sees a focused, single-bin capability profile rather than the whole benchmark.
- Much more expensive but tests every possible source→target bin direction.

**E.) closest_bin, 5 pairs only, thinking ON**

- For each target bin j, show only the single source bin whose mean pass rate is closest to bin j's.
- This gives exactly 5 (i,j) pairs (one per target bin), same cost as A/B.
- The forecaster sees only the most similar source bin, maximally relevant context.
- Key idea: showing a closely matched reference bin gives the forecaster the best analogical anchor. This aligns with our previous findings where  we saw that providing too much context to LLMs might confuse them instead of helping them.

F.) Full loop on 20 source/target pairs with number of target tasks = 3

| **Metric** | **A — all_bins, no think** | **B — all_bins, think** | **C — single_bin, K=1** | **E — closest_bin, K=1** | **F — full loop, single bin, K=3 *(partial)*** |
| --- | --- | --- | --- | --- | --- |
| N analyzed | 60 | 60 | 240 | **60** | 1389 |
| **Brier ↓** | 0.2336 | 0.2255 | 0.1736 | **0.1030** | 0.1925 |
| **CRPS ↓** | 0.3113 | 0.3014 | 0.2601 | **0.1908** | 0.2821 |
| ECE ↓ | 0.1732 | 0.1755 | **0.0888** | 0.1428 | **0.0701** |
| **Spearman rho ↑** | 0.31 | 0.35 | 0.54 | **0.77** | 0.49 |
| **Kendall tau ↑** | 0.26 | 0.29 | 0.45 | **0.64** | 0.41 |
| MAE ↓ | 0.3892 | 0.3765 | 0.3334 | **0.2555** | 0.3587 |
| Bias | −0.074 | −0.075 | −0.032 | +0.010 | +0.058 |

## Key findings

-E dominates A, B, C on Brier, CRPS, MAE, RMSE, Spearman rho, Kendall tau, and Bias. However, for P(MITRE | benchmark performance) we **don't know** where a MITRE step falls on the difficulty spectrum upfront, hence this mightn’t be directly relevant? However, this tells us that providing an accurate, relevant, clear and concise prompt seems to perform better than a long and generic prompt with too much details.

-C still has the best ECE (0.0888 vs 0.1428), meaning the full 20-pair loop produces better-calibrated interval widths, but E is far more accurate pointwise.

-E ran 4× faster than C (120 vs 480 API calls) while achieving nearly 2× lower Brier (0.1030 vs 0.1736).

-Bias is essentially zero in E (+0.010), vs systematic underconfidence in all prior runs.

As a further sanity check for method E (close bin pairs), I looked at Jakub’s results from method C (loop over 20 pairs), and compared the overall brier score of all experiment E (20 pairs) vs brier score of bin pairs that are close (5 pairs). 

We get the following results

| **Slice** | **N** | **Brier** | **Spearman rho** | **Bias** |
| --- | --- | --- | --- | --- |
| C — all 20 pairs | 240 | 0.1736 | 0.54 | −0.032 |
| **C — closest 5 pairs only** | 60 | **0.1046** | **0.79** | −0.002 |
| C — other 15 pairs | 180 | 0.1966 | 0.45 | −0.042 |
| E — closest_bin run | 60 | 0.1030 | 0.77 | +0.010 |

This suggests that the prediction is more accurate when we anchor from a specific bin that is of similar difficulty level as the target task bin, but I think this approach doesn’t fulfills our main goal of P(MITRE | benchmark performance). In real life, we won’t know where the MITRE attack step lies or what’s it’s difficulty is, right? So we won’t know which benchmark this task might be close to, or do we? I am not sure how looping over 20 bin pair would work as well in real life.  *"Here are tasks from Lyptus that [model] solved and failed, across difficulty levels. Given this, what's P([model] succeeds at this MITRE attack step?"* So seems like we might need to provide all the bins as anchor in the prompt. However, if we somehow justify  “based on the target task description (we don’t know it’s difficulty in real life), bin X out of 5 bins is the closest”, and use that bin as anchor instead of providing context about all four source bins, the forecasts seem to be better.

# Number of Target Tasks = 3

# week 12

**Madhav**

### 1.) Breakdown of two experts

Dr Capability

| **Metric** | **A** | **B** | **C** | **E** | **F** |
| --- | --- | --- | --- | --- | --- |
| N analyzed | 60 | 60 | 240 | 60 | 718 |
| Brier ↓ | 0.2275 | 0.2245 | 0.1743 | 0.1050 | 0.1903 |
| CRPS ↓ | 0.3061 | 0.3035 | 0.2617 | 0.1927 | 0.2785 |
| ECE ↓ | 0.1768 | 0.1687 | 0.0881 | 0.1465 | 0.0630 |
| Spearman rho ↑ | 0.32 | 0.34 | 0.54 | 0.76 | 0.50 |
| Kendall tau ↑ | 0.27 | 0.29 | 0.44 | 0.63 | 0.41 |
| MAE ↓ | 0.3865 | 0.3793 | 0.3353 | 0.2575 | 0.3544 |
| Bias | −0.062 | −0.080 | −0.033 | +0.001 | +0.057 |

**Prof. Psychometrics & Test Design**

| **Metric** | **A** | **B** | **C** | **E** | **F** |
| --- | --- | --- | --- | --- | --- |
| N analyzed | 59 | 60 | 240 | 59 | 720 |
| Brier ↓ | 0.2249 | 0.2245 | 0.1774 | 0.1018 | 0.1888 |
| CRPS ↓ | 0.3041 | 0.3029 | 0.2641 | 0.1912 | 0.2790 |
| ECE ↓ | 0.1790 | 0.1830 | 0.0907 | 0.1220 | 0.0765 |
| Spearman rho ↑ | 0.37 | 0.33 | 0.53 | 0.78 | 0.50 |
| Kendall tau ↑ | 0.31 | 0.28 | 0.43 | 0.65 | 0.41 |
| MAE ↓ | 0.3810 | 0.3797 | 0.3378 | 0.2566 | 0.3550 |
| Bias | −0.076 | −0.073 | −0.039 | +0.006 | +0.060 |

There seems to be a tiny difference between elicitation results between two experts. We probably mightn’t need two experts.

#### 2.) Model Sweep

We sweep across different models. 

### 3.) Ground Truth Plots

![01_model_comparison.png](attachment:1f4f80d7-cfcf-4919-8d95-8f982f89e643:01_model_comparison.png)

In this plot, the LLM forecaster seems to underestimate the pass rate for easier bins and slightly overestimate the pass rate for the hardest bins. However, the ground truth for these plots are misleading. Sorry about the confusion before. In the plots, the ground truth is computed as number of tasks solved by model M in the bin / total number of tasks in the bin. While this makes the visualization simpler, we discussed earlier that this is not right, and is not a fair comparison since this assumes that a single task is representative of the entire bin. However, in our analysis scripts, the brier score is computed at a row level (per each task) and ground truth is treated as row level too, which seems to make more sense than the bin level ground truth. 

If we take the same model/task combo, and plot the ground truth at row level for each task (instead of bin level), then the figure looks something like this:

![01_model_comparison.png](attachment:a41ad015-82a0-4dc6-a8f7-ed14ccc151ff:01_model_comparison.png)

### 4.) Variance Due to the Choice of Target Task

Keeping bin 1 as the target bin and passing the information from bins 2-5 as an anchor, there seems to be a significant difference in elicited outcome between 20 tasks in the target bin. 

![task_variance_analysis_bin1.png](attachment:16ef9528-9533-4aa0-9ec4-06f74867b3ec:task_variance_analysis_bin1.png)

The choice of a specific target task has a significant impact on the elicited outcome. This suggests that we might need a sample of target tasks from the target bin, maybe low, medium and high, K = 3 to balance out the noise. 

I repeated the same experiment keeping bin 3 as the target. The results show a similar pattern. The source of variance seems to be the task specific LLM miscalibration, suggesting that K = 1 elicitation might not be perfect.

![task_variance_analysis.png](attachment:aa97f5b9-9036-4eba-bdac-ec9aaae14c1c:task_variance_analysis.png)

To see whether the choice of speific task in the target bin matters, we have 20 real Brier scores (one per task, computed across all 12 models) for all 20 tasks in bin 3. When we just have one target task (K = 1) per bin, the elicitation is heavily task dependent and noisy. For K=3: randomly pick 3 of those 20 target tasks, average their 3 Brier scores, that's one simulated "K=3 experiment result". Repeat this 5000 times, get a distribution of what K=3 *could* give you. Same for K=10. I think this is a simple statistical fact for the K>1 case to have a lower variance in this case, so the histogram is expected, nothing new I guess. Since this doesn't require new API calls, I did it to see how different cases would differ.

A more interesting result would probably be the one you mentioned above where we elicit the mean performance of the bin based on a task and repeat this for 3 tasks in a bin to get the overall mean
