#!/usr/bin/env python3
"""fig20: one LLM, two questions — elicit the probability vs elicit the
task's human time (converted to probability by a 12-parameter curve fit on
the training benchmarks). Same cells, paired arms, house style.

Bars: seed probability prompt; best-transfer GEPA-optimized probability
prompt (cand 18); time-elicitation pipeline (arm T, pre-registered).
Dashed gold lines = within-set leave-one-out ground-truth lookup tables.
Sources: gepa runs/noise_study cell logs + the frozen head-to-head scorer
output (audit workspace, two independent derivations).
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
GREEN = "#034f46"; VERM = "#d64e2e"; GOLD = "#b8860b"; GRAY = "#b7b4b1"
plt.rcParams.update({"font.family": "Geist", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})

GEPA = Path(os.environ.get("GEPA_DIR", Path(__file__).resolve().parents[4].parent / "gepa"))
NS = GEPA / "runs/noise_study"
WS = Path("/Users/madhav/.claude/projects/-Users-madhav-SaferAI-LLM-elicitation/audit-workspace")
HH = json.load(open(WS / "analysis/freeze/headtohead_scores.json"))


def cells(fname, prompt, setname):
    per_rep = defaultdict(list)
    for l in open(NS / fname):
        r = json.loads(l)
        if r["set"] == setname and r["prompt"] == prompt:
            per_rep[r["repeat"]].append(r["brier"] if r["brier"] is not None else 1.0)
    return [statistics.mean(v) for v in per_rep.values()]


seed_sealed = 0.1305   # pooled 17 passes (audit-verified); printed as 0.131
seed_test = 0.1633
c18_sealed = statistics.mean(cells("noise_cells_pareto_modelbin.jsonl", "modelbin_cand18", "sealed"))
c18_test = statistics.mean(cells("noise_cells_test_ext_modelbin.jsonl", "modelbin_cand18", "test"))
t_sealed = HH["T_sealed"]["brier"]
t_test = HH["T_test"]["brier"]
TABLES = {"sealed": 0.1179, "test": 0.1488}

ARMS = [("seed prompt\n(asks for the probability)", GRAY, {"sealed": seed_sealed, "test": seed_test}),
        ("cand 18 — the GEPA prompt that transfers\nbest (asks for the probability)", VERM, {"sealed": c18_sealed, "test": c18_test}),
        ("same LLM asked for the task's\nhuman time + fitted curve", GREEN, {"sealed": t_sealed, "test": t_test})]
SETS = [("sealed", "sealed set — held-out questions,\ntraining benchmarks (230 cells)"),
        ("test", "reserved test — new benchmarks\nCVEBench + CyberGym (1,033 cells)")]

fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=220)
fig.subplots_adjust(top=0.745, bottom=0.155, left=0.09, right=0.965)
fig.text(0.055, 0.965, "ONE LLM, TWO QUESTIONS  ·  identical cells, temperature 0  ·  "
         "reserved-test comparison pre-registered", fontsize=8, color=MUTED, ha="left", va="top")
fig.text(0.055, 0.925, "Asking for the probability wins at home; asking for the task's "
         "human time transfers", fontsize=12.5, color=INK, ha="left", va="top", fontweight="bold")
fig.text(0.055, 0.862, "a cell = will one AI model solve one benchmark task? (Brier; lower = better)\n"
         "time answers become probabilities via a 12-parameter curve fit on the training benchmarks",
         fontsize=8.5, color=MUTED, ha="left", va="top")

W = 0.25
for si, (skey, slabel) in enumerate(SETS):
    for ai, (label, color, vals) in enumerate(ARMS):
        x = si + (ai - 1) * W
        ax.bar(x, vals[skey], width=W - 0.03, color=color,
               label=label if si == 0 else None)
        ax.text(x, vals[skey] + 0.002, f"{vals[skey]:.3f}", ha="center",
                fontsize=8.5, color=INK)
    ax.hlines(TABLES[skey], si - 1.5 * W, si + 1.5 * W, ls="--", lw=1.2, color=GOLD)
ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
ax.set_xticks(range(len(SETS)), [s for _k, s in SETS], fontsize=9, color=INK)
ax.set_ylim(0.09, 0.20)
ax.set_ylabel("Brier score (lower = better)", fontsize=9.5)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", color=HAIR, lw=0.8); ax.set_axisbelow(True)
ax.tick_params(colors=MUTED, labelsize=8.5); ax.tick_params(axis="x", length=0)
fig.text(0.055, 0.012,
         "gold dashed = (model × difficulty-bin) solve-rate lookup fit on the set's own labels — "
         "perfect rates, no per-task reading, so per-task signal can beat it\n"
         "time arm, reserved test: paired vs seed −0.012, task-clustered t = −2.4 · sealed: time arm "
         "0.1515 vs seed 0.1305 (n.s. at 21 tasks)\n"
         "curve = logistic in log(median estimated minutes); 12 params = 11 per-model abilities + 1 "
         "shared slope · preliminary: two held-out benchmark families, one dataset",
         fontsize=7, color=SOFT, ha="left", va="bottom")
fig.savefig("fig20_ask_swap.png")
print(f"fig20 written · sealed: seed {seed_sealed:.4f} c18 {c18_sealed:.4f} T {t_sealed:.4f} · "
      f"test: seed {seed_test:.4f} c18 {c18_test:.4f} T {t_test:.4f}")
