#!/usr/bin/env python3
"""fig10 for the one-page synthesis: (a) held-out Brier of seed, causal arms,
winners vs the difficulty-oracle table; (b) output shift vs accuracy gain.
Reads gepa/runs/feature_report_data.json (rerun feature_report_data.py first).
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK="#1a1a1a"; MUTED="#736f6c"; SOFT="#94918e"; HAIR="#ebebeb"
GREEN="#034f46"; VERM="#d64e2e"; GOLD="#b8860b"
plt.rcParams.update({"font.family":"Geist","figure.facecolor":"white","savefig.facecolor":"white"})

d = json.load(open("/Users/madhav/SaferAI/gepa/runs/feature_report_data.json"))["sealed"]["prompts"]
SEED = d["seed"]["brier"]; TABLE = 0.1034

order = [("seed","seed",MUTED),
         ("ablate_numeric_bands_only","seed + numeric bands",VERM),
         ("ablate_procedure_only","seed + anchoring procedure",VERM),
         ("clean_cand7","cand 7 (clean run)",GREEN),
         ("july_cand12","cand 12 (July run)",GREEN)]
order = [(k,l,c) for k,l,c in order if k in d]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 3.3), dpi=220,
                               gridspec_kw={"width_ratios":[1.25,1]})
xs = range(len(order))
ax1.bar(xs, [d[k]["brier"] for k,_,_ in order], color=[c for _,_,c in order], width=0.62)
ax1.axhline(TABLE, ls="--", lw=1.3, color=GOLD, label="difficulty-oracle table (no LLM)")
ax1.axhline(SEED, ls=":", lw=1.1, color=MUTED, label="seed prompt")
ax1.legend(loc="upper right", frameon=False, fontsize=8)
for x,(k,l,c) in zip(xs, order):
    ax1.text(x, d[k]["brier"]+0.0012, f"n={d[k]['n_passes']}", fontsize=7, color=SOFT, ha="center")
ax1.set_xticks(list(xs), [l for _,l,_ in order], fontsize=8.6, color=INK)
ax1.tick_params(axis="x", length=0, rotation=12)
ax1.set_ylim(0.095, 0.135)
ax1.set_ylabel("held-out Brier (230 cells, lower is better)", fontsize=9.5)
ax1.set_title("a  Causal decomposition of the winning recipe", fontsize=11, loc="left", color=INK)

pts = [("seed",0,0,MUTED)]
for k,l,c in order[1:]:
    pts.append((l, d[k]["w1_p50_vs_seed"], SEED-d[k]["brier"], c))
cw = d.get("clean_winner")
if cw: pts.append(("cand 3 (val winner)", cw["w1_p50_vs_seed"], SEED-cw["brier"], SOFT))
c15 = d.get("july_cand15")
if c15: pts.append(("cand 15", c15["w1_p50_vs_seed"], SEED-c15["brier"], GREEN))
for l, x, y, c in pts:
    ax2.scatter(x, y, s=48, color=c, zorder=3)
    dy = -11 if l.startswith("cand 3") else 4
    ax2.annotate(l, (x, y), textcoords="offset points", xytext=(6,dy), fontsize=8, color=INK)
ax2.axhline(0, lw=1.0, color=HAIR)
ax2.set_xlabel("output shift vs seed  ($W_1$ of matched-cell $p_{50}$)", fontsize=9.5)
ax2.set_ylabel("Brier improvement over seed", fontsize=9.5)
ax2.set_title("b  Shifting outputs vs actually helping", fontsize=11, loc="left", color=INK)
ax2.set_xlim(-0.003, 0.055)

for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=HAIR, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.5)
fig.tight_layout()
fig.savefig("fig10_feature_synthesis.png", bbox_inches="tight", pad_inches=0.25)
print("fig10 written")
