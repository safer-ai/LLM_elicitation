#!/usr/bin/env python3
"""fig21: best eval-set Brier vs iteration, sabotage-seed run vs real-seed
run. Standard line chart; details live in the team message."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1a1a1a"; MUTED = "#736f6c"; HAIR = "#ebebeb"
VERM = "#d64e2e"; GRAY = "#b7b4b1"
plt.rcParams.update({"font.family": "Geist", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})

GEPA = Path(os.environ.get("GEPA_DIR", Path(__file__).resolve().parents[4].parent / "gepa"))


def trajectory(run_dir):
    xs, ys = [0], [None]
    for l in open(GEPA / run_dir / "diagnostics.jsonl"):
        r = json.loads(l)
        if r.get("kind") == "iteration":
            xs.append(r["iteration"])
            ys.append(r["best_val_brier_so_far"])
    for l in open(GEPA / run_dir / "candidates.jsonl"):
        r = json.loads(l)
        if r.get("kind") == "valset_evaluation" and r.get("candidate_idx") == 0:
            ys[0] = r["val_mean_brier"]
            break
    return xs, ys


xs_s, ys_s = trajectory("runs/pilot_sabotage_seed")
xs_c, ys_c = trajectory("runs/pilot_baseline_clean")

fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=220)
fig.subplots_adjust(top=0.97, bottom=0.15, left=0.12, right=0.97)
ax.plot(xs_s, ys_s, color=VERM, lw=2.0, marker="o", ms=2.5, label="Sabotaged seed")
ax.plot(xs_c, ys_c, color=GRAY, lw=1.6, marker="o", ms=2.0, label="Real seed")
ax.legend(loc="upper right", frameon=False, fontsize=9.5, labelcolor=INK,
          title="Search started from:", title_fontsize=9.5, alignment="left")
ax.set_xlabel("GEPA iteration", fontsize=10)
ax.set_ylabel("Best eval-set Brier so far (lower is better)", fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", color=HAIR, lw=0.8); ax.set_axisbelow(True)
ax.tick_params(colors=MUTED, labelsize=8.5)
fig.savefig("fig21_sabotage_recovery.png")
print("fig21 written")
