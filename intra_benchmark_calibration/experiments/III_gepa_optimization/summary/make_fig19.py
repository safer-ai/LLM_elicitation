#!/usr/bin/env python3
"""fig19: the grand comparison — every candidate prompt (set C) on the three
evaluation sets, one grouped bar chart.

Sets: search (84 questions, used during optimization), held-out (230
questions, same benchmarks), reserved test (1,033 questions, CVEBench +
CyberGym). Bar = mean over that prompt's evaluation passes on that set,
whisker = min-max over passes. The three ground-truth lookup-table baselines
(leave-one-out bin-mean, recomputed from the same logged cells) are dashed
lines in the matching colors. All measurements temperature 0.
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
GREEN = "#034f46"; VERM = "#d64e2e"; GRAY = "#b7b4b1"
plt.rcParams.update({"font.family": "Geist", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})

GEPA = Path(os.environ.get("GEPA_DIR", Path(__file__).resolve().parents[4].parent / "gepa"))
NS = GEPA / "runs/noise_study"

C = ["seed", "v2_cand13", "clean_cand7", "modelbin_cand18", "joint_cand5", "july_cand12"]
FILES = {  # set -> files holding temp-0 cells for it
    "search": ["noise_cells_grand_val.jsonl"],
    "sealed": ["noise_cells_temp0.jsonl", "noise_cells_temp0_confirm.jsonl",
               "noise_cells_clean_rerun.jsonl", "noise_cells_feature_ablation.jsonl",
               "noise_cells_accept_joint.jsonl", "noise_cells_pareto_modelbin.jsonl",
               "noise_cells_v2_sealed.jsonl"],
    "test": ["noise_cells_test_set.jsonl", "noise_cells_test_ext_winners.jsonl",
             "noise_cells_test_ext_modelbin.jsonl"],
}
SETKEY = {"search": "val", "sealed": "sealed", "test": "test"}

passes = defaultdict(lambda: defaultdict(list))   # set -> prompt -> [pass means]
gt_cells = {}                                     # set -> {(task, model): (bin, outcome)}
for setname, files in FILES.items():
    for f in files:
        p = NS / f
        if not p.exists():
            continue
        d = defaultdict(lambda: defaultdict(list))
        for l in p.open():
            r = json.loads(l)
            if r["set"] != SETKEY[setname] or r["prompt"] not in C:
                continue
            d[(f, r["repeat"])][r["prompt"]].append(
                r["brier"] if r["brier"] is not None else 1.0)
            gt_cells.setdefault(setname, {})[(r["task_id"], r["model"])] = (r["bin"], r["outcome"])
        for _k, dd in d.items():
            for prompt, v in dd.items():
                passes[setname][prompt].append(statistics.mean(v))


def loo_table(cells):
    tasks_in_bin = defaultdict(set)
    for (t, _m), (b, _y) in cells.items():
        tasks_in_bin[b].add(t)
    briers = []
    for (t, m), (b, y) in cells.items():
        others = [cells[(t2, m)][1] for t2 in tasks_in_bin[b]
                  if t2 != t and (t2, m) in cells]
        if others:
            briers.append((statistics.mean(others) - y) ** 2)
    return statistics.mean(briers)


TABLES = {s: loo_table(gt_cells[s]) for s in FILES if s in gt_cells}
COLORS = {"search": GRAY, "sealed": GREEN, "test": VERM}
LABELS = {"search": "search set (84 questions, used during optimization)",
          "sealed": "held-out, same benchmarks (230 questions)",
          "test": "reserved test, new benchmarks (1,033 questions)"}
XLAB = {"seed": "starting\nprompt", "v2_cand13": "v2_cand13\n(numbers banned)",
        "clean_cand7": "cand 7\n(run 2 · native)", "modelbin_cand18": "cand 18\n(model_bin)",
        "joint_cand5": "cand 5\n(gate k = 8)", "july_cand12": "cand 12\n(run 1 · native)"}

fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=220)
fig.subplots_adjust(top=0.775, bottom=0.155, left=0.085, right=0.965)
fig.text(0.05, 0.968, "SET C  ·  every candidate prompt × every evaluation set",
         fontsize=8, color=MUTED, ha="left", va="top")
fig.text(0.05, 0.928, "Gains hold on held-out questions from the training benchmarks "
         "and vanish on the reserved benchmarks", fontsize=12.5, color=INK,
         ha="left", va="top", fontweight="bold")
fig.text(0.05, 0.868, "a question = will one AI model solve one benchmark task? "
         "(Brier; lower = better) · bar = mean over evaluation passes at temperature 0, "
         "whisker = min–max", fontsize=8.5, color=MUTED, ha="left", va="top")
fig.text(0.965, 0.968, "DASHED LINES = GROUND-TRUTH LOOKUP TABLES", fontsize=8,
         color=INK, ha="right", va="top", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=INK, lw=1.1))

W = 0.26
for si, setname in enumerate(["search", "sealed", "test"]):
    xs, ms, lo, hi = [], [], [], []
    for pi, prompt in enumerate(C):
        v = passes[setname].get(prompt, [])
        if not v:
            continue
        m = statistics.mean(v)
        xs.append(pi + (si - 1) * W); ms.append(m)
        lo.append(m - min(v)); hi.append(max(v) - m)
    ax.bar(xs, ms, width=W - 0.02, color=COLORS[setname],
           label=f"{LABELS[setname]}  ·  table {TABLES[setname]:.3f}",
           yerr=[lo, hi], capsize=2.5, error_kw={"lw": 0.9, "ecolor": INK})
    if setname in TABLES:
        ax.axhline(TABLES[setname], ls="--", lw=1.1, color=COLORS[setname], alpha=0.9)
ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK,
          borderaxespad=0.2)
ax.set_xticks(range(len(C)), [XLAB[p] for p in C], fontsize=8.5, color=INK)
ax.set_xlim(-0.55, 5.68)
ax.set_ylim(0.07, 0.245)
ax.set_ylabel("Brier score (lower = better)", fontsize=9.5)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", color=HAIR, lw=0.8); ax.set_axisbelow(True)
ax.tick_params(colors=MUTED, labelsize=8.5); ax.tick_params(axis="x", length=0)
fig.text(0.05, 0.012,
         "passes: search 3 · held-out 3–17 · test 2 · dashed line = leave-one-out bin-mean "
         "of ground-truth solve rates over that set's own tasks\n"
         "the all-benchmarks table of earlier figures scores 0.103 on the held-out set · "
         "preliminary: one reserved test set, one measurement day per set",
         fontsize=7, color=SOFT, ha="left", va="bottom")
fig.savefig("fig19_grand_comparison.png")
for setname in ["search", "sealed", "test"]:
    row = " · ".join(f"{p} {statistics.mean(v):.4f}(n={len(v)})"
                     for p, v in passes[setname].items())
    print(f"{setname}: table {TABLES.get(setname, float('nan')):.4f} · {row}")
