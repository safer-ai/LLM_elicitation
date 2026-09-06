#!/usr/bin/env python3
"""fig23: reliability diagram on the reserved test (standard axes, no prose).
One curve per prompt: mean predicted p50 per equal-width bin vs observed
solve rate; dashed diagonal = perfect calibration."""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1a1a1a"; MUTED = "#736f6c"; HAIR = "#ebebeb"
GREEN = "#034f46"; VERM = "#d64e2e"; GOLD = "#b8860b"; GRAY = "#b7b4b1"; BLUE = "#2b5d8a"
plt.rcParams.update({"font.family": "Geist", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})

GEPA = Path(os.environ.get("GEPA_DIR", Path(__file__).resolve().parents[4].parent / "gepa"))
files = ["noise_cells_test_set.jsonl", "noise_cells_test_ext_winners.jsonl",
         "noise_cells_test_ext_ablation.jsonl", "noise_cells_test_ext_modelbin.jsonl"]
rows = [json.loads(l) for f in files for l in open(GEPA / "runs/noise_study" / f) if l.strip()]
df = pd.DataFrame(rows)
df = df[df["set"] == "test"]
cell = df.groupby(["prompt", "task_id", "model"], as_index=False).agg(
    p50=("p50", "mean"), y=("outcome", "first"))

SHOW = [("seed", "Seed prompt", GRAY),
        ("july_cand12", "GEPA cand 12", VERM),
        ("joint_cand5", "GEPA cand 5", GOLD),
        ("v2_cand13", "GEPA v2 (no numbers)", BLUE)]
EDGES = np.linspace(0, 1, 11)

fig, ax = plt.subplots(figsize=(5.6, 5.2), dpi=220)
fig.subplots_adjust(top=0.97, bottom=0.11, left=0.12, right=0.97)
ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color=MUTED)
for key, label, color in SHOW:
    c = cell[cell["prompt"] == key]
    b = np.clip(np.digitize(c["p50"], EDGES[1:-1]), 0, 9)
    xs, ys = [], []
    for i in range(10):
        m = b == i
        if m.sum() >= 15:
            xs.append(c["p50"][m].mean()); ys.append(c["y"][m].mean())
    ax.plot(xs, ys, marker="o", ms=4, lw=1.8, color=color, label=label)
ax.set_xlabel("Mean predicted P(solve)", fontsize=10)
ax.set_ylabel("Observed solve rate", fontsize=10)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
ax.grid(color=HAIR, lw=0.8); ax.set_axisbelow(True)
ax.tick_params(colors=MUTED, labelsize=8.5)
fig.savefig("fig23_reliability_test.png")
print("fig23 written")
