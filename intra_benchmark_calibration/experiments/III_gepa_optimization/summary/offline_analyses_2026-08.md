# Offline analyses (zero API cost) — 2026-08-29

Three analyses that only read logged data. Reproduce from the gepa repo:

```bash
uv run python scripts/prompt_feature_scan.py    # section 1
uv run python scripts/offline_prescreen.py      # sections 2-3
```

## 1. Prompt-feature table (Matt's clustering idea)

All 57 evolved prompts from the three runs, tagged with a deterministic
regex rubric, scores parsed-only. The headline is not any single feature —
it is the split between the sealed-validated winners and the sealed nulls:

| | July winners (cands 12, 15, 20; edges +0.027/+0.014/+0.017) | v2 candidates (all edges ≤ 0) |
|---|---|---|
| numeric probability bands | **all 3** | none |
| extreme prescriptions (0.9x / 0.0x) | **all 3** | none |
| anti-hedging rules | all 3 | present |
| task-type heuristics | all 3 | present |

The features v2's softened instruction banned (numeric bands, extremes) are
exactly the ones that separate winners from nulls; the "clean" features the
v2 prompts kept (anti-hedging language, task-type talk) produced no gain on
their own. Directional single-feature deltas on val agree (anti-hedging
−0.004, task-type −0.002, platform terms −0.003, all toward better), and
shorter prompts run slightly better than longer ones.

Caveats: prompts within a run share lineage (not independent samples);
features co-occur; val deltas are single-pass (rerun sd 0.0065). This is
screening in Matt's sense — which feature clusters to test properly — not
hypothesis testing.

## 2. Acceptance-rule replay (Sec. 4 idea 4) — a real candidate setting

Scope caveat first (Jakub's point, correct): a replay grades a rule against
the decisions we logged; it cannot simulate the run that rule would have
produced. Truth for an accept = did the child's parsed-only val beat its
parent's (itself single-pass, so noisy truth).

July run, 25 accepts (12 good / 13 bad by that truth), 20-cell gate:

| rule | good kept | bad kept | rejects newly admitted |
|---|---|---|---|
| native `sum>0` | 12/12 | 13/13 | 3/15 |
| `min_task_wins` alone, any k ≤ 8 | ~all | ~all | **8–15/15 (near-vacuous)** |
| **`sum>0 AND wins≥8`** | **11/12** | **8/13** | 3/15 |
| `sum>0 AND wins≥9` | 9/12 | 4/13 | 3/15 |

The joint criterion at k≈8 of 20 keeps nearly every good accept while
cutting ~40% of bad ones and admitting nothing new — the first acceptance
variant to actually pass screening. (`min_task_wins` alone stays vacuous,
confirming the earlier retro finding; the k=6 default helps nothing.)
Worth one live run if the group wants an acceptance ablation.

## 3. (model, bin) Pareto aggregation (Sec. 4 idea 5, Jeff's) — double-edged

Rebuilding July's val scoreboard at both granularities (parsed-only):

| granularity | instances | frontier holders | top-3 share | spearman(wins, quality) |
|---|---|---|---|---|
| per-cell (native) | 84 | 24 of 26 | 0.31 | +0.16 |
| (model, bin) | 20 | 10 | 0.50 | **+0.32** |

Aggregation makes instance-wins track true quality twice as well and
concentrates parent selection — the intended effect. But two warnings:
the extremity tilt does not shrink (+0.07 → +0.29, n=12, not significant),
and two of the three sealed-validated winners would hold **zero** aggregated
instances (cand 15: 2 cell-wins, 0 agg-wins; cand 9 likewise) — under
aggregation those lineages starve and cand 15 is never found. Verdict:
promising for selection sharpness, risky for diversity; if tried live, pair
it with something that preserves exploration (e.g. keep the frontier
filter at 1 instance).

## What this changes in the plan

- The reflection-feedback ablation stays the most promising paid idea (not
  affected by these results).
- If the group wants a second ablation, the joint acceptance rule
  (`aggregate_sum_and_min_task_wins`, k≈8/20) is now the screened pick —
  already implemented in the codebase, config-only change.
- Plain (model,bin) aggregation should not be run as-is; it would likely
  have discarded cand 15.
