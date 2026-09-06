#!/usr/bin/env python3
"""fig24: calibration lines in log-odds (Cox fit per prompt), reserved test.
Fitted logit(observed) = a + b*logit(predicted); identity = perfect."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1a1a1a"; MUTED = "#736f6c"; HAIR = "#ebebeb"
GOLD = "#b8860b"; VERM = "#d64e2e"; GRAY = "#b7b4b1"; BLUE = "#2b5d8a"
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


def lgt(p):
    return np.log(p / (1 - p))


SHOW = [("seed", "Seed", GRAY), ("july_cand12", "GEPA cand 12", VERM),
        ("joint_cand5", "GEPA cand 5", GOLD), ("v2_cand13", "GEPA v2", BLUE)]

fig, ax = plt.subplots(figsize=(5.8, 5.4), dpi=220)
fig.subplots_adjust(top=0.97, bottom=0.11, left=0.13, right=0.97)
ax.plot([-4, 4], [-4, 4], ls="--", lw=1.2, color=MUTED)
for key, label, color in SHOW:
    c = cell[cell["prompt"] == key]
    z = lgt(np.clip(c["p50"], 0.01, 0.99)).to_numpy()
    y = c["y"].to_numpy()
    X = np.column_stack([np.ones(len(z)), z])
    r = optimize.minimize(lambda w: np.sum(np.logaddexp(0, X @ w) - y * (X @ w)),
                          [0.0, 1.0], method="L-BFGS-B")
    a, b = r.x
    xs = np.array([np.quantile(z, 0.02), np.quantile(z, 0.98)])
    ax.plot(xs, a + b * xs, lw=2.0, color=color,
            label=f"{label}  (slope {b:.2f}, intercept {a:+.2f})")
ax.set_xlabel("log-odds of predicted P(solve) — reserved test, 1,033 cells", fontsize=9.5)
ax.set_ylabel("log-odds of observed solve rate (Cox fit)", fontsize=9.5)
ax.set_xlim(-4, 4); ax.set_ylim(-5.5, 4); 
ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
ax.grid(color=HAIR, lw=0.8); ax.set_axisbelow(True)
ax.tick_params(colors=MUTED, labelsize=8.5)
fig.savefig("fig24_calibration_logit.png")
print("fig24 written")
