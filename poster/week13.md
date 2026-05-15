**Model Sweep Results**

Some of the GPT 5 runs didn’t succeed as those tasks were flagged by GPT 5 as cyberattack risks. A slight change in prompt could probably bypass that, but I think that would lead to an inconsistency in experimental setups for a fair comparison.

| **Model** | **N valid** | **Brier ↓** | **CRPS ↓** |
| --- | --- | --- | --- |
| `gpt55` | 190 | **0.1008** | **0.1712** |
| `opus47` | 290 | 0.1370 | 0.2094 |
| `sonnet46` | 299 | 0.1387 | 0.2119 |
| `haiku45` | 299 | 0.1712 | 0.2594 |
| `gemini25flash` | 300 | 0.1822 | 0.2450 |

Except the GPT 5.5 results (which we can’t say for sure if it’s good as some runs didn’t suceed), the rest of the models produce a decent result. All the models performed better than the baseline 0.25 for Brier and 0.33 with Uniform Beta (1,1) of CRPS . Opus 4.7 and Sonnet 4.6 had a similar performance. This is probably a good news for us as we did most of our experiments on Sonnet due to Opus being 2x expensive.