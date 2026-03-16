"""
CV vs ICC_F: Why do they give opposite results?

This script demonstrates the fundamental difference between:
- CV: Absolute spread of expert means
- ICC_F: Proportion of variance that's between-expert
"""

import numpy as np

def demonstrate_difference():
    """Show why CV and ICC_F can give opposite conclusions."""

    print("=" * 80)
    print("WHY CV AND ICC_F GIVE OPPOSITE RESULTS")
    print("=" * 80)

    print("""
SCENARIO: Compare Integer estimates vs Probability estimates

For illustration, let's say we have 3 experts, each with 3 runs.
    """)

    # Integer estimates - experts differ, but each expert is also noisy
    print("INTEGER ESTIMATES (num_actors):")
    print("-" * 40)
    int_data = {
        'Expert A': [7, 9, 5],   # mean=7, high within-expert variance
        'Expert B': [8, 10, 6],  # mean=8, high within-expert variance
        'Expert C': [9, 11, 7],  # mean=9, high within-expert variance
    }
    for expert, values in int_data.items():
        print(f"  {expert}: {values} → mean = {np.mean(values):.1f}")

    # Expert means
    int_expert_means = [np.mean(v) for v in int_data.values()]
    int_cv = np.std(int_expert_means, ddof=1) / np.mean(int_expert_means) * 100

    # Total variance and between-expert variance
    all_int_values = [v for vals in int_data.values() for v in vals]
    int_total_var = np.var(all_int_values, ddof=1)
    int_between_var = np.var(int_expert_means, ddof=1) * 3  # multiply by n_runs
    int_iccf = int_between_var / int_total_var * 100

    print(f"\n  Expert means: {int_expert_means}")
    print(f"  CV = std/mean = {np.std(int_expert_means, ddof=1):.2f}/{np.mean(int_expert_means):.2f} = {int_cv:.1f}%")
    print(f"  Total variance: {int_total_var:.2f}")
    print(f"  Between-expert variance (approx): {int_between_var:.2f}")
    print(f"  ICC_F ≈ {int_iccf:.1f}%")

    # Probability estimates - experts similar, each expert is consistent
    print("\n\nPROBABILITY ESTIMATES:")
    print("-" * 40)
    prob_data = {
        'Expert A': [0.50, 0.51, 0.49],  # mean=0.50, low within-expert variance
        'Expert B': [0.52, 0.53, 0.51],  # mean=0.52, low within-expert variance
        'Expert C': [0.54, 0.55, 0.53],  # mean=0.54, low within-expert variance
    }
    for expert, values in prob_data.items():
        print(f"  {expert}: {values} → mean = {np.mean(values):.2f}")

    prob_expert_means = [np.mean(v) for v in prob_data.values()]
    prob_cv = np.std(prob_expert_means, ddof=1) / np.mean(prob_expert_means) * 100

    all_prob_values = [v for vals in prob_data.values() for v in vals]
    prob_total_var = np.var(all_prob_values, ddof=1)
    prob_between_var = np.var(prob_expert_means, ddof=1) * 3
    prob_iccf = prob_between_var / prob_total_var * 100

    print(f"\n  Expert means: {prob_expert_means}")
    print(f"  CV = std/mean = {np.std(prob_expert_means, ddof=1):.3f}/{np.mean(prob_expert_means):.2f} = {prob_cv:.1f}%")
    print(f"  Total variance: {prob_total_var:.4f}")
    print(f"  Between-expert variance (approx): {prob_between_var:.4f}")
    print(f"  ICC_F ≈ {prob_iccf:.1f}%")

    print("\n" + "=" * 80)
    print("COMPARISON:")
    print("=" * 80)
    print(f"""
                    │ Integers │ Probabilities │ Which is higher?
    ────────────────┼──────────┼───────────────┼─────────────────
    CV              │ {int_cv:5.1f}%   │ {prob_cv:5.1f}%         │ {'Integers' if int_cv > prob_cv else 'Probabilities'}
    ICC_F           │ {int_iccf:5.1f}%   │ {prob_iccf:5.1f}%         │ {'Integers' if int_iccf > prob_iccf else 'Probabilities'}
    """)

    print("""
KEY INSIGHT:
------------
- Integers: High TOTAL variance (experts noisy), but experts DO differ
- Probabilities: Low TOTAL variance (experts consistent), experts also differ

- CV: Measures absolute spread of expert means
      → Integers have LARGER spread (1 unit) than probabilities (0.02)
      → But normalized by mean, integers win: 1/8 > 0.02/0.52

- ICC_F: Measures what PROPORTION of variance is between-expert
      → Integers: Much variance is WITHIN-expert (noise)
      → Probabilities: Most variance is BETWEEN-expert (signal)
      → So ICC_F is HIGHER for probabilities!

THE PARADOX:
------------
Integers have MORE absolute disagreement between experts (CV higher)
But LESS of their total variance is due to expert identity (ICC_F lower)

This happens because integers have more RUN-TO-RUN NOISE.
""")


def which_is_better():
    """Discuss which measure is more appropriate."""
    print("\n" + "=" * 80)
    print("WHICH MEASURE IS BETTER FOR THE PAPER'S CLAIM?")
    print("=" * 80)

    print("""
Paper says: "quantities exhibit a much GREATER VARIANCE across experts"

INTERPRETATION 1: Greater ABSOLUTE spread
→ CV is appropriate
→ Result: Integers > Probabilities ✓ SUPPORTED

INTERPRETATION 2: Greater PROPORTION of variance is between-expert
→ ICC_F is appropriate
→ Result: Integers < Probabilities ✗ NOT SUPPORTED

MY RECOMMENDATION:
------------------
The paper's phrasing "greater variance across experts" suggests ABSOLUTE
disagreement, not proportion. When we say "experts disagree more on X than Y",
we typically mean the absolute spread of their opinions, not the signal-to-noise
ratio.

Therefore, CV is more appropriate for testing the paper's claim.

HOWEVER: The fact that ICC_F is LOWER for integers is also meaningful!
It tells us: Integer estimates have more RUN-TO-RUN NOISE.
This means LLM expert personas are LESS CONSISTENT on integer estimates.

BOTH findings are interesting:
1. CV higher for integers → Experts disagree more on integers (absolute)
2. ICC_F lower for integers → More noise in integer estimates (relative)
""")


def recommendation():
    """Final recommendation."""
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    print("""
FOR TESTING THE PAPER'S CLAIM:
------------------------------
Use CV of expert-mean p50 values.

Justification:
1. "Greater variance across experts" = absolute spread = CV
2. CV is scale-independent (can compare integers to probabilities)
3. Simpler and more interpretable than ICC_F

FOR COMPLETENESS:
-----------------
Also report ICC_F from Fréchet ANOVA, noting:
- ICC_F measures signal-to-noise ratio
- Lower ICC_F for integers means more run-to-run noise
- This is a different (but also interesting) finding

WHAT ABOUT DISTRIBUTION FITTING?
--------------------------------
Distribution fitting (Beta/PERT) is used for ICC_F calculation.
For CV, we don't need distribution fitting - just use p50.

If you want to use distribution fitting for CV-like measure:
- Compute Fréchet mean (average distribution in Wasserstein space)
- Compute average Wasserstein distance from each expert to Fréchet mean
- This would be "variance in distribution space"
- But this is more complex and may not add much over CV of p50
""")


if __name__ == "__main__":
    demonstrate_difference()
    which_is_better()
    recommendation()
