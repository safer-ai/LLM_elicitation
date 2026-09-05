#!/usr/bin/env python3
"""fig20: Brier by elicitation method on the sealed set and the reserved
test. Standard grouped bars; details live in the team message, not the
figure."""
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1a1a1a"; MUTED = "#736f6c"; HAIR = "#ebebeb"
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


vals = {
    "Seed prompt (probability)": (GRAY, 0.1305, 0.1633),
    "GEPA cand 18 (probability)": (VERM,
        statistics.mean(cells("noise_cells_pareto_modelbin.jsonl", "modelbin_cand18", "sealed")),
        statistics.mean(cells("noise_cells_test_ext_modelbin.jsonl", "modelbin_cand18", "test"))),
    "Time elicitation + fitted curve": (GREEN, HH["T_sealed"]["brier"], HH["T_test"]["brier"]),
}
TABLES = [0.1179, 0.1488]
SETS = ["Sealed set\n(230 cells)", "Reserved test\n(1,033 cells)"]

fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=220)
fig.subplots_adjust(top=0.97, bottom=0.12, left=0.11, right=0.97)

W = 0.25
for ai, (label, (color, sealed, test)) in enumerate(vals.items()):
    for si, v in enumerate([sealed, test]):
        ax.bar(si + (ai - 1) * W, v, width=W - 0.03, color=color,
               label=label if si == 0 else None)
        ax.text(si + (ai - 1) * W, v + 0.002, f"{v:.4f}", ha="center",
                fontsize=8.5, color=INK)
for si, t in enumerate(TABLES):
    ax.hlines(t, si - 1.5 * W, si + 1.5 * W, ls="--", lw=1.2, color=GOLD,
              label="Bin-rate lookup table (in-set)" if si == 0 else None)
ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
ax.set_xticks(range(2), SETS, fontsize=9.5, color=INK)
ax.set_ylim(0.09, 0.20)
ax.set_ylabel("Brier score (lower is better)", fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", color=HAIR, lw=0.8); ax.set_axisbelow(True)
ax.tick_params(colors=MUTED, labelsize=8.5); ax.tick_params(axis="x", length=0)
fig.savefig("fig20_ask_swap.png")
print("fig20 written")
