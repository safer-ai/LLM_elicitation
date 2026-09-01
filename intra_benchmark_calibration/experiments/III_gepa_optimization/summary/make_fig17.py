#!/usr/bin/env python3
"""fig17: the reserved held-out test (CVEBench + CyberGym), measured once.

Bars = mean Brier per prompt over the 1,033 test questions (94 frozen tasks
x 11 AI models), whiskers = min-max over the 2 repeat evaluations; the
ground-truth bin-mean baseline is recomputed from the same logged cells.
Data: runs/noise_study/noise_cells_test_set.jsonl in the gepa repo
(results/sweep-arms-2026-09-01 branch holds the committed copy).
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
GREEN = "#034f46"; VERM = "#d64e2e"; GOLD = "#b8860b"
plt.rcParams.update({"font.family": "Geist", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})

GEPA = Path(os.environ.get("GEPA_DIR", Path(__file__).resolve().parents[4].parent / "gepa"))
rows = [json.loads(l) for l in (GEPA / "runs/noise_study/noise_cells_test_set.jsonl").open()
        if '"test"' in l]

per_rep = defaultdict(lambda: defaultdict(list))   # prompt -> repeat -> briers
outcome, binof = {}, {}
for r in rows:
    per_rep[r["prompt"]][r["repeat"]].append(r["brier"] if r["brier"] is not None else 1.0)
    if r["prompt"] == "seed" and r["repeat"] == 0:
        c = (r["task_id"], r["model"]); outcome[c] = r["outcome"]; binof[c] = r["bin"]

# ground-truth leave-one-out bin-mean baseline on the same cells (no LLM)
tasks_in_bin = defaultdict(set)
for (t, m), _y in outcome.items():
    tasks_in_bin[binof[(t, m)]].add(t)
gt = []
for (t, m), y in outcome.items():
    others = [outcome[(t2, m)] for t2 in tasks_in_bin[binof[(t, m)]]
              if t2 != t and (t2, m) in outcome]
    if others:
        gt.append((statistics.mean(others) - y) ** 2)
TABLE = statistics.mean(gt)

bars = [("seed", "starting prompt\n(no forecasting guidance)", MUTED),
        ("clean_cand7", "optimized prompt B\n(held-out gain +0.016 in-distribution)", VERM),
        ("july_cand12", "optimized prompt A\n(held-out gain +0.026 in-distribution)", VERM)]

fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=220)
fig.subplots_adjust(top=0.76, bottom=0.185, left=0.10, right=0.965)
fig.text(0.055, 0.965, "THE RESERVED TEST  ·  CVEBench + CyberGym, first and only use",
         fontsize=8, color=MUTED, ha="left", va="top")
fig.text(0.055, 0.925, "On the reserved benchmarks, the optimized prompts fall behind "
         "the starting prompt", fontsize=12.5, color=INK, ha="left", va="top",
         fontweight="bold")
fig.text(0.055, 0.862, "1,033 questions = 94 frozen tasks × 11 AI models · a question = will "
         "one AI model solve one benchmark task? (Brier; lower = better)",
         fontsize=8.5, color=MUTED, ha="left", va="top")
fig.text(0.965, 0.965, "MEASURED ONCE · 2 REPEAT EVALUATIONS", fontsize=8, color=INK,
         ha="right", va="top", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=INK, lw=1.1))

for x, (k, label, c) in enumerate(bars):
    reps = [statistics.mean(v) for v in per_rep[k].values()]
    m = statistics.mean(reps)
    ax.bar(x, m, width=0.52, color=c,
           yerr=[[m - min(reps)], [max(reps) - m]], capsize=4,
           error_kw={"lw": 1.1, "ecolor": INK})
    ax.text(x, m - 0.006, f"{m:.3f}", ha="center", va="top", fontsize=11,
            color="white", fontweight="bold")
ax.axhline(TABLE, ls="--", lw=1.3, color=GOLD)
ax.text(-0.42, 0.2125, f"gold dashed line = ground-truth lookup table ({TABLE:.3f}):\n"
        "mean solve rate of the model's other tasks in the same difficulty bin",
        fontsize=8, color=GOLD, ha="left", va="top")
ax.set_xticks(range(3), [l for _k, l, _c in bars], fontsize=9, color=INK)
ax.set_ylim(0.10, 0.215)
ax.set_ylabel("Brier score on the reserved test set\n(lower = better)", fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", color=HAIR, lw=0.8); ax.set_axisbelow(True)
ax.tick_params(colors=MUTED, labelsize=8.5); ax.tick_params(axis="x", length=0)
fig.text(0.055, 0.012,
         "whisker = min–max over the 2 repeat evaluations (repeats differ by ≤0.003) · "
         "in-distribution the order was reversed: prompt A 0.105 < prompt B 0.115 < starting 0.131\n"
         "paired on the same cells, both optimized prompts are worse (t = 9.3 and 6.9, n = 1,033) · "
         "every prompt sees evidence from the training benchmarks only\n"
         "preliminary: one reserved test set, one measurement session",
         fontsize=7, color=SOFT, ha="left", va="bottom")
fig.savefig("fig17_test_set.png")
print(f"fig17 written · table {TABLE:.4f} · " +
      " · ".join(f"{k} {statistics.mean([statistics.mean(v) for v in per_rep[k].values()]):.4f}"
                 for k, _l, _c in bars))
