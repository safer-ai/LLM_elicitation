#!/usr/bin/env python3
"""fig11: best-validation-Brier-so-far per iteration for the two main GEPA
runs, inside the seed's re-measurement band. Reads candidates.jsonl of each
run. The curve is a running minimum over single noisy passes: flatness does
not imply convergence, and the clean run's sealed-verified winner (cand 7)
never appears on the curve at all (its val draw was worse than the seed's)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK="#1a1a1a"; MUTED="#736f6c"; SOFT="#94918e"; HAIR="#ebebeb"
GREEN="#034f46"; VERM="#d64e2e"
plt.rcParams.update({"font.family":"Geist","figure.facecolor":"white","savefig.facecolor":"white"})

GEPA = Path("/Users/madhav/SaferAI/gepa")
BAND = (0.0956, 0.1137)  # seed prompt, 5 fresh val passes, same cells

def series(run):
    props = [json.loads(l) for l in (GEPA/f"runs/{run}/candidates.jsonl").open()
             if '"proposal"' in l]
    seed = next(-p["parent_val_score"] for p in props
                if p.get("parent_idx") == 0 and p.get("parent_val_score") is not None)
    xs, ys = [0], [seed]
    best = seed
    accepted = {}
    for p in props:
        if p["status"] == "accepted" and p.get("val_mean_score") is not None:
            b = -p["val_mean_score"]
            accepted[p["candidate_idx"]] = (p["iteration"], b)
            if b < best:
                best = b
                xs.append(p["iteration"]); ys.append(best)
    xs.append(40); ys.append(best)
    return xs, ys, accepted

fig, ax = plt.subplots(figsize=(8.0, 3.4), dpi=220)
ax.axhspan(*BAND, color=HAIR, alpha=0.75, lw=0,
           label="seed re-measurement range (5 passes, same cells)")

for run, color, label in (("pilot_baseline", GREEN, "run 1 (July)"),
                          ("pilot_baseline_clean", VERM, "run 2 (clean pipeline)")):
    xs, ys, acc = series(run)
    ax.step(xs, ys, where="post", color=color, lw=1.8, label=label)

# sealed-verified winners at their acceptance points
xs1, ys1, acc1 = series("pilot_baseline")
it, b = acc1[12]
ax.scatter([it], [b], color=GREEN, zorder=5, s=34)
ax.annotate("winner (held-out +0.027)", (it, b), textcoords="offset points",
            xytext=(6, -13), fontsize=8, color=GREEN)
xs2, ys2, acc2 = series("pilot_baseline_clean")
it, b = acc2[7]
ax.scatter([it], [b], color=VERM, zorder=5, s=34, facecolors="white", linewidths=1.6)
ax.annotate("winner (held-out +0.016);\nnever a val best", (it, b),
            textcoords="offset points", xytext=(10, -22), fontsize=8, color=VERM)

ax.set_xlabel("iteration", fontsize=9.5)
ax.set_ylabel("best validation Brier so far", fontsize=9.5)
ax.set_xlim(0, 40); ax.set_ylim(0.09, 0.12)
ax.legend(loc="upper right", frameon=False, fontsize=8.5)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", color=HAIR, lw=0.8); ax.set_axisbelow(True)
ax.tick_params(colors=MUTED, labelsize=8.5)
fig.tight_layout()
fig.savefig("fig11_best_so_far_curves.png", bbox_inches="tight", pad_inches=0.2)
print("fig11 written")
