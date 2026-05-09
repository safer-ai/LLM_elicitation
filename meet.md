SPAR Project 2 Meeting
Invited jefftmohl@gmail.com Matt Smith Jakub Kryś Jack Kengott mkhanal@rollins.edu
Attachments SPAR Project 2 Meeting


Summary
Review of experimental settings and cost analysis confirmed K=1 settings for future project efficiency.

Code Stability and Results
Code runs stably with no API key errors and a budget of 750 dollars. Concise anchors in case C proved more effective than complex multi bin inputs.

Experimental Settings Evaluation
Setting E risks information leakage while setting F yields results similar to case C but at higher cost. Prioritizing K=1 settings maximizes efficiency and accuracy.

Future Baselines and Optimization
Team will implement linear regression baselines and run iterative tests to ensure score confidence. Future experiments will be capped at 100 dollars per run.

Rate this Summary: Helpful or Not Helpful


Decisions
We've updated the Decisions section using your feedback.
Let us know what you think: Helpful or Not Helpful

ALIGNED
Performance baselines for metric validation The team will establish performance baselines, including linear regression and random guessing, to effectively evaluate and validate Brier and CRPS metric results.
K=1 configuration selected for experiments The research will utilize the K=1 configuration for future experiments to optimize API costs and maintain consistency across results.
Five-run minimum for uncertainty analysis Experimental procedures require a minimum of five runs per configuration to enable uncertainty interval calculations and ensure statistical significance.


Next steps
[Madhav Khanal] Complete Runs: Ensure all 1,389 runs are completed for data consistency.
[Jeff Mohl] Plot Ground Truth: Plot the actual ground truth data for the current experiments. Examine the general appearance of the results.
[Jeff Mohl] Compute Baselines: Calculate brier scores for naive 0.5 prediction baseline. Calculate scores for the mean bin prediction baseline.
[Jeff Mohl] Calculate Linear Fit: Fit a linear regression to the inverse sigmoid ground truth plot. Calculate the resulting brier score.
[Madhav Khanal] Run Model Sweep: Conduct a scaling law experiment using different model families. Track brier score changes across model versions, likely using K=1 setup.
[Madhav Khanal] Analyze Per Expert: Reproduce the existing results table, calculating metrics separately for each expert. Determine if differences exist to potentially eliminate 1 expert.
[Madhav Khanal] Run Random Task: Run a simple experiment selecting 1 target task randomly from the bin.
[Madhav Khanal] Run 20 Tasks: Select 1 target bin and test 20 different tasks within it. Analyze the brier score fluctuation across the cases.
[Madhav Khanal] Update Poster: Draft and continually update the project poster with new results.
[Madhav Khanal] Rerun Experiments: Perform at least 5 runs of each new experiment. Estimate the monetary cost beforehand and proceed if below 100 dollars.
[Jakub Kryś] Merge Branches: Merge Madhavs branch and new code changes onto Spar Spring 2026.
[Jakub Kryś] Notify Merge: Inform the team when the code branches have been merged and are ready for use.
[Madhav Khanal] Analyze Results: Reflect on the meaning of current brier scores. Propose clever methods for testing forecasting capabilities.


Details
Past Week Review and Code Stability: Jakub Kryś apologized for their delay and reported finishing the code, while Madhav Khanal successfully ran some experiments. The code is running in a stable manner, with no reported API key errors. Jakub Kryś also noted the code was less expensive than anticipated, and they have $750 available for spending over the next two weeks.
Experimental Settings and Results Overview: Jakub Kryś mentioned sending three settings in a zipped file and noted that Madhav Khanal also ran "setting E," though the status of "setting D" is unknown. They asked Madhav Khanal to describe the experimental settings E and F and discuss the initial results, particularly concerning case C. Madhav Khanal summarized that case A involved thinking turned off with 12 models and 60 total cells, case B involved thinking on, and case C utilized source-bin pairs where one source and one target bin were used, looping over 20 pairs.
Analysis of Case C and the Effect of Context: Madhav Khanal reported that the results for case C, which involved passing only one source bin as an anchor, appeared "really nice" compared to cases A and B. They inferred that providing a concise, shorter anchor seems to be more beneficial than passing information from all four bins, which might confuse the model, an observation consistent with earlier results.
Concerns Regarding Setting E (Closest Anchor): Madhav Khanal detailed setting E, which involved selecting five source/target bin pairs that were the closest in terms of average model solve rate, yielding a better Brier score. However, Jakub Kryś expressed concern that this approach involves "leaking a bunch of information" by providing the closest bin, which may artificially inflate the Brier score and not accurately reflect forecasting capability, as the model could be predicting close to the anchor's solve rate. Madhav Khanal agreed that this setting is likely just anchoring to the baseline and may not translate well to the project's general goal.
Analysis of Setting F (K=3 Samples) and Median Selection: Madhav Khanal explained that setting F looped over all 20 bins and used a K=3 setting, involving three separate predictions for low, medium, and high difficulty tasks from the target bin. The results from K=3 were very similar to case C (K=1), but choosing only the median task (as in case C) yielded a better Brier score than choosing a range of difficulties, suggesting that language models might struggle more with predicting the difficulty of easier and harder tasks. They noted that given the similar results, the cheaper cost of K=1 is preferable.
Discussion on Data Confidence and Experimental Setup: Jakub Kryś stated they have "zero intuition" for the stability of the current Brier scores and suggested running a loop of at least 10 iterations to gain confidence. Jeff Mohl agreed that more predictions are needed, and suggested adding confidence intervals, as they have 60 Brier scores available. Jeff Mohl also questioned why the target bin is held out when predicting a target task, noting that excluding it may leave a hole in the interpolation for the model. Jakub Kryś acknowledged that there is no obvious reason for this setup, but reiterated the practical constraint that the difficulty of the target task is unknown in the real-world application of predicting microsteps.
Establishing Baselines and Future Experiments: To contextualize the Brier scores, Jakub Kryś emphasized the need to establish a baseline, such as fitting a linear regression to the ground truth data. Other proposed experiments include a scaling law experiment sweeping through different models from the oldest available (e.g., 3.5) to the newest (e.g., 4.7) to see if the Brier score decreases. Jeff Mohl offered to calculate both the naive baseline (predicting 0.5) and the slightly higher standard of predicting the mean of a bin.
Evaluation Metrics and Brier Score Interpretation: The discussion included a review of various metrics, where Madhav Khanal highlighted that continuous ranked probability scores (CRPS) for some experiments were worse than predicting the same thing every time, while Brier scores seemed decent. Jeff Mohl noted that the Brier score baseline for random guessing is 0.25 (since it is the square of 0.5), which makes the current results appear worse than initially thought. Madhav Khanal noted that a CRPS score below 0.33 is considered better than the naive case of predicting 0.5 with a uniform interval.
Optimizing Experiments and Cost Management: Jakub Kryś suggested focusing on the K=1 setting for future experiments due to the better results and lower cost. They proposed running the existing results again but separating them by expert to see if one expert can be eliminated, which would halve the costs. Madhav Khanal proposed running a simple experiment where the target task is selected randomly to avoid information leakage.
Final Commitments and Next Steps: Jakub Kryś committed to merging the current branch with changes that allow for setting the number of repeats (e.g., 20) in the code automatically to simplify reruns. They instructed Madhav Khanal to estimate the cost of proposed experiments, allowing them to proceed if the cost is below $100 per experiment, or to consult first if the cost is significantly higher. They agreed to keep in touch on Slack .