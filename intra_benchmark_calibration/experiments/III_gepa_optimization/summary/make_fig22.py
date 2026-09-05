#!/usr/bin/env python3
"""fig22: sabotage-run recovery verdict on the sealed set — same-session
3-pass sealed means (tag sabotage_rec), one bar per arm, standard form."""
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

per = defaultdict(lambda: defaultdict(list))
for l in open(GEPA / "runs/noise_study/noise_cells_sabotage_rec.jsonl"):
    r = json.loads(l)
    if r["set"] == "sealed":
        per[r["prompt"]][r["repeat"]].append(r["brier"] if r["brier"] is not None else 1.0)
mean3 = {p: statistics.mean(statistics.mean(v) for v in reps.values())
         for p, reps in per.items()}

BARS = [("Sabotaged seed\n(start of the run)", VERM, mean3["sabotage_seed"]),
        ("Real seed", GRAY, mean3["seed"]),
        ("Winner recovered\nfrom sabotage", GREEN, mean3["sab_cand_a"]),
        ("cand 12\n(clean-seed winner)", GREEN, mean3["july_cand12"])]
TABLE = 0.1179

fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=220)
fig.subplots_adjust(top=0.96, bottom=0.15, left=0.12, right=0.96)
for x, (label, color, v) in enumerate(BARS):
    ax.bar(x, v, width=0.55, color=color)
    ax.text(x, v + 0.004, f"{v:.3f}", ha="center", fontsize=9, color=INK)
ax.axhline(TABLE, ls="--", lw=1.2, color=GOLD)
ax.text(3.35, TABLE + 0.004, "Bin-rate lookup\ntable (in-set)", fontsize=8,
        color=GOLD, ha="right", va="bottom")
ax.set_xticks(range(len(BARS)), [b[0] for b in BARS], fontsize=9, color=INK)
ax.set_ylim(0.0, 0.30)
ax.set_ylabel("Sealed-set Brier, 3-pass mean (lower is better)", fontsize=9.5)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", color=HAIR, lw=0.8); ax.set_axisbelow(True)
ax.tick_params(colors=MUTED, labelsize=8.5); ax.tick_params(axis="x", length=0)
fig.savefig("fig22_sabotage_verdict.png")
print("fig22 written · " + " · ".join(f"{p} {v:.4f}" for p, v in mean3.items()))
