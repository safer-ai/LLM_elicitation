#!/usr/bin/env python3
"""fig10: one bar chart. Held-out Brier of the seed, the two single-feature
prompts, and the best evolved prompt, with min-max whiskers over passes and
the difficulty-oracle table as a reference line. Pass means recomputed from
the committed per-cell files."""
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK="#1a1a1a"; MUTED="#736f6c"; SOFT="#94918e"; HAIR="#ebebeb"
GREEN="#034f46"; VERM="#d64e2e"; GOLD="#b8860b"
plt.rcParams.update({"font.family":"Geist","figure.facecolor":"white","savefig.facecolor":"white"})

NS = Path("/Users/madhav/SaferAI/gepa/runs/noise_study")
FILES = ["noise_cells_temp0.jsonl", "noise_cells_temp0_confirm.jsonl",
         "noise_cells_clean_rerun.jsonl", "noise_cells_feature_ablation.jsonl"]

pass_means = defaultdict(list)
for f in FILES:
    per = defaultdict(list)
    for line in (NS / f).open():
        r = json.loads(line)
        if r["set"] != "sealed":
            continue
        per[(r["prompt"], r["repeat"])].append(r["brier"] if r["brier"] is not None else 1.0)
    for (prompt, _rep), briers in per.items():
        pass_means[prompt].append(statistics.mean(briers))

TABLE = 0.1034
bars = [("seed", "seed\n(minimal prompt)", MUTED),
        ("ablate_procedure_only", "seed +\nanchoring procedure", VERM),
        ("ablate_numeric_bands_only", "seed +\nnumeric bands", VERM),
        ("july_cand12", "best evolved\nprompt", GREEN)]

fig, ax = plt.subplots(figsize=(8.2, 3.0), dpi=220)
xs = range(len(bars))
means = [statistics.mean(pass_means[k]) for k, _, _ in bars]
los = [means[i] - min(pass_means[bars[i][0]]) for i in xs]
his = [max(pass_means[bars[i][0]]) - means[i] for i in xs]
ax.bar(xs, means, color=[c for _, _, c in bars], width=0.58,
       yerr=[los, his], capsize=4, error_kw={"lw": 1.1, "ecolor": INK})
ax.axhline(TABLE, ls="--", lw=1.3, color=GOLD,
           label="difficulty-oracle lookup table (no LLM)")
ax.legend(loc="upper right", frameon=False, fontsize=9)
for x, (k, _, _) in zip(xs, bars):
    ax.text(x, means[x] + his[x] + 0.0012, f"n={len(pass_means[k])} passes",
            fontsize=7.5, color=SOFT, ha="center")
ax.set_xticks(list(xs), [l for _, l, _ in bars], fontsize=9.5, color=INK)
ax.set_ylim(0.095, 0.14)
ax.set_ylabel("held-out Brier (lower is better)", fontsize=9.5)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", color=HAIR, lw=0.8); ax.set_axisbelow(True)
ax.tick_params(colors=MUTED, labelsize=8.5); ax.tick_params(axis="x", length=0)
fig.tight_layout()
fig.savefig("fig10_feature_synthesis.png", bbox_inches="tight", pad_inches=0.2)
print("fig10 written; whisker = min-max of per-pass means")
for k, l, _ in bars:
    print(f"  {k:<28} mean {statistics.mean(pass_means[k]):.4f} "
          f"[{min(pass_means[k]):.4f}, {max(pass_means[k]):.4f}] n={len(pass_means[k])}")
