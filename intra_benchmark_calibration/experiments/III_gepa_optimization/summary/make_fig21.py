#!/usr/bin/env python3
"""fig21: the sabotage positive control — best val Brier so far vs search
iteration, sabotage-seed run vs the good-seed reference run (identical
search settings; only the starting prompt differs).

Sources: diagnostics.jsonl of runs/pilot_sabotage_seed and
runs/pilot_baseline_clean (per-iteration best_val_brier_so_far).
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1a1a1a"; MUTED = "#736f6c"; SOFT = "#94918e"; HAIR = "#ebebeb"
GREEN = "#034f46"; VERM = "#d64e2e"; GOLD = "#b8860b"; GRAY = "#b7b4b1"
plt.rcParams.update({"font.family": "Geist", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})

GEPA = Path(os.environ.get("GEPA_DIR", Path(__file__).resolve().parents[4].parent / "gepa"))


def trajectory(run_dir):
    xs, ys = [0], [None]
    start = None
    for l in open(GEPA / run_dir / "diagnostics.jsonl"):
        r = json.loads(l)
        if r.get("kind") == "iteration":
            xs.append(r["iteration"])
            ys.append(r["best_val_brier_so_far"])
    for l in open(GEPA / run_dir / "candidates.jsonl"):
        r = json.loads(l)
        if r.get("kind") == "valset_evaluation" and r.get("candidate_idx") == 0:
            start = r["val_mean_brier"]
            break
    ys[0] = start
    return xs, ys, start


xs_s, ys_s, start_s = trajectory("runs/pilot_sabotage_seed")
xs_c, ys_c, start_c = trajectory("runs/pilot_baseline_clean")

fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=220)
fig.subplots_adjust(top=0.755, bottom=0.175, left=0.09, right=0.965)
fig.text(0.055, 0.965, "POSITIVE CONTROL  ·  identical search, only the starting prompt differs",
         fontsize=8, color=MUTED, ha="left", va="top")
fig.text(0.055, 0.925, "Given a deliberately broken starting prompt, does the search repair it?",
         fontsize=12.5, color=INK, ha="left", va="top", fontweight="bold")
fig.text(0.055, 0.875, "y = best evaluation-set Brier found so far (84 questions the search "
         "optimizes on; lower = better)\nsabotage seed = real seed + one sentence capping every "
         "forecast at 0.50", fontsize=8.5, color=MUTED, ha="left", va="top")

ax.plot(xs_s, ys_s, color=VERM, lw=2.0, marker="o", ms=2.5,
        label=f"search started from the SABOTAGED seed (starts at {start_s:.3f})")
ax.plot(xs_c, ys_c, color=GRAY, lw=1.6, marker="o", ms=2.0,
        label=f"reference: same search from the real seed (starts at {start_c:.3f})")
ax.axhline(start_c, ls=":", lw=1.0, color=SOFT)
ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK)
ax.set_xlabel("search iteration", fontsize=9.5)
ax.set_ylabel("best eval-set Brier so far\n(lower = better)", fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", color=HAIR, lw=0.8); ax.set_axisbelow(True)
ax.tick_params(colors=MUTED, labelsize=8.5)
fig.text(0.055, 0.012,
         "dotted line = the real seed's own starting level · sabotage run stopped at iteration 37 "
         "on the shared 4,000-call budget\nfinal verdicts use the standard 3-pass sealed study "
         "(separate table), not this curve · preliminary: one search per seed condition",
         fontsize=7, color=SOFT, ha="left", va="bottom")
fig.savefig("fig21_sabotage_recovery.png")
print(f"fig21 written · sabotage start {start_s:.4f} end {ys_s[-1]:.4f} · "
      f"clean start {start_c:.4f} end {ys_c[-1]:.4f}")
