#!/usr/bin/env python3
"""fig18: every search run's measured prompts on one axis — held-out (sealed)
Brier, temp-0 multi-pass, with the seed and the ground-truth table.

Bars = mean over that prompt's sealed evaluation passes (each pass = the
same 230 held-out questions from the training benchmarks); whisker =
min-max over passes. Data: the temp-0 noise-study cell logs in the gepa
repo (committed on the results/* branches).
"""
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1a1a1a"; MUTED = "#736f6c"; SOFT = "#94918e"; HAIR = "#ebebeb"
GREEN = "#034f46"; GOLD = "#b8860b"
plt.rcParams.update({"font.family": "Geist", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})

GEPA = Path(os.environ.get("GEPA_DIR", Path(__file__).resolve().parents[4].parent / "gepa"))
FILES = ["noise_cells_temp0.jsonl", "noise_cells_temp0_confirm.jsonl",
         "noise_cells_clean_rerun.jsonl", "noise_cells_feature_ablation.jsonl",
         "noise_cells_accept_joint.jsonl", "noise_cells_pareto_modelbin.jsonl"]

per = defaultdict(list)  # prompt -> [pass means]
for i, f in enumerate(FILES):
    d = defaultdict(lambda: defaultdict(list))
    path = GEPA / "runs/noise_study" / f
    if not path.exists():
        continue
    for l in path.open():
        r = json.loads(l)
        if r["set"] != "sealed":
            continue
        d[(i, r["repeat"])][r["prompt"]].append(r["brier"] if r["brier"] is not None else 1.0)
    for _key, dd in d.items():
        for p, v in dd.items():
            per[p].append(statistics.mean(v))

TABLE = 0.1034  # ground-truth LOO bin-mean table on the same 230 held-out questions

bars = [  # (key, label line 1 = prompt, line 2 = which search run produced it)
    ("seed",            "starting\nprompt",        MUTED),
    ("july_cand12",     "cand 12\nrun 1 · native", GREEN),
    ("july_cand20",     "cand 20\nrun 1 · native", GREEN),
    ("july_cand15",     "cand 15\nrun 1 · native", GREEN),
    ("clean_cand7",     "cand 7\nrun 2 · native",  GREEN),
    ("clean_winner",    "cand 3\nrun 2 · native",  GREEN),
    ("joint_cand5",     "cand 5\ngate k = 8",      GREEN),
    ("joint_cand16",    "cand 16\ngate k = 8",     GREEN),
    ("modelbin_cand18", "cand 18\nmodel_bin",      GREEN),
    ("modelbin_cand10", "cand 10\nmodel_bin",      GREEN),
]

fig, ax = plt.subplots(figsize=(9.4, 4.6), dpi=220)
fig.subplots_adjust(top=0.76, bottom=0.16, left=0.09, right=0.965)
fig.text(0.05, 0.965, "ALL SEARCH RUNS  ·  held-out (sealed) set, training benchmarks, "
         "230 questions, temperature 0", fontsize=8, color=MUTED, ha="left", va="top")
fig.text(0.05, 0.925, "Every measured prompt, by the run that produced it",
         fontsize=12.5, color=INK, ha="left", va="top", fontweight="bold")
fig.text(0.05, 0.862, "a question = will one AI model solve one benchmark task? "
         "(Brier; lower = better) · bar = mean over that prompt's evaluation passes, "
         "whisker = min–max", fontsize=8.5, color=MUTED, ha="left", va="top")
fig.text(0.965, 0.965, "SAME 230 QUESTIONS FOR EVERY BAR", fontsize=8, color=INK,
         ha="right", va="top", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=INK, lw=1.1))

for x, (k, label, c) in enumerate(bars):
    v = per[k]
    m = statistics.mean(v)
    ax.bar(x, m, width=0.58, color=c,
           yerr=[[m - min(v)], [max(v) - m]], capsize=3.5,
           error_kw={"lw": 1.0, "ecolor": INK})
    ax.text(x, 0.0915, f"{m:.3f}", ha="center", fontsize=8.5, color="white",
            fontweight="bold")
    ax.text(x, max(v) + 0.0012, f"{len(v)}", ha="center", fontsize=7, color=SOFT)
ax.axhline(TABLE, ls="--", lw=1.3, color=GOLD)
ax.text(-0.35, 0.1392, f"gold dashed line = ground-truth lookup table ({TABLE:.3f})",
        fontsize=8, color=GOLD, ha="left", va="top")
ax.set_xticks(range(len(bars)), [l for _k, l, _c in bars], fontsize=8.5, color=INK)
ax.set_ylim(0.09, 0.14)
ax.set_ylabel("held-out Brier score\n(lower = better)", fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", color=HAIR, lw=0.8); ax.set_axisbelow(True)
ax.tick_params(colors=MUTED, labelsize=8.5); ax.tick_params(axis="x", length=0)
fig.text(0.05, 0.012,
         "small number above each whisker = evaluation passes for that prompt · "
         "runs: native rule (runs 1–2), acceptance gate k = 8 (aggregate_sum_and_min_task_wins), "
         "model_bin (pareto_instance)\nevery run also proposed ~20 further prompts never "
         "measured here · lookup table = mean solve rate of the model's other tasks in the "
         "same difficulty bin (uses ground truth)",
         fontsize=7, color=SOFT, ha="left", va="bottom")
fig.savefig("fig18_all_runs_sealed.png")
print("fig18 written · " + " · ".join(f"{k} {statistics.mean(per[k]):.4f}(n={len(per[k])})"
                                      for k, _l, _c in bars))
