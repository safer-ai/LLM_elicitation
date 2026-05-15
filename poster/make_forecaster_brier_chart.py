"""
Generate a cross-forecaster Brier + CRPS bar chart for the poster.

Reads each forecaster's `plots/statistics.txt` (produced by
intra_benchmark_calibration/analyse_results.py) and plots grouped bars:
  - Brier-on-p50 (solid)
  - Mean CRPS (hatched)
with two reference lines:
  - Brier chance baseline = 0.25
  - CRPS Uniform Beta(1,1) baseline = 0.333

All numbers come from the same Beta-fit + numerical-CRPS pipeline.
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/Users/madhav/SaferAI/LLM_elicitation/intra_benchmark_calibration/experiments/G_model_sweep/results")
OUT  = Path("/Users/madhav/SaferAI/LLM_elicitation/poster/forecaster_brier_sweep.png")

# (folder, display label, partial-completion flag, brand colour)
FORECASTERS = [
    ("gpt55",         "GPT-5.5*",          True,  "#10A37F"),  # OpenAI green
    ("opus47",        "Claude Opus 4.7",   False, "#A8431B"),  # Anthropic deep ember
    ("sonnet46",      "Claude Sonnet 4.6", False, "#D97757"),  # Anthropic ember
    ("haiku45",       "Claude Haiku 4.5",  False, "#E8B299"),  # Anthropic light ember
    ("gemini25flash", "Gemini 2.5 Flash",  False, "#4285F4"),  # Google blue
]

BRIER_CHANCE = 0.25
CRPS_UNIFORM = 1/3  # Uniform Beta(1,1)

# --- parse statistics.txt for each forecaster --------------------------------
def parse_stats(path: Path):
    text = path.read_text()
    n      = int(re.search(r"N elicitations analysed.*?:\s*(\d+)", text).group(1))
    brier  = float(re.search(r"Brier-on-p50:\s*([0-9.]+)", text).group(1))
    crps   = float(re.search(r"Mean CRPS:\s*([0-9.]+)", text).group(1))
    return n, brier, crps


def bootstrap_ci(csv_path: Path, n_boot: int = 10_000, seed: int = 42) -> dict:
    """95% bootstrap CIs for mean Brier-on-p50 and mean CRPS."""
    df = pd.read_csv(csv_path)
    brier_arr  = ((df["p50"] - df["outcome"]) ** 2).dropna().values
    crps_arr   = df["crps"].dropna().values
    rng = np.random.default_rng(seed)
    nb, nc = len(brier_arr), len(crps_arr)
    boot_brier = brier_arr[rng.integers(0, nb, size=(n_boot, nb))].mean(axis=1)
    boot_crps  = crps_arr[rng.integers(0, nc, size=(n_boot, nc))].mean(axis=1)
    return {
        "brier_lo": float(np.percentile(boot_brier, 2.5)),
        "brier_hi": float(np.percentile(boot_brier, 97.5)),
        "crps_lo":  float(np.percentile(boot_crps, 2.5)),
        "crps_hi":  float(np.percentile(boot_crps, 97.5)),
    }


records = []
for folder, label, partial, color in FORECASTERS:
    plots_dir  = ROOT / folder / "plots"
    stats_file = plots_dir / "statistics.txt"
    csv_file   = plots_dir / "scored_with_crps.csv"
    n, brier, crps = parse_stats(stats_file)
    ci = bootstrap_ci(csv_file)
    records.append({"label": label, "brier": brier, "crps": crps,
                    "n": n, "partial": partial, "color": color, "ci": ci})
    print(f"{label:22s}  Brier={brier:.4f} [{ci['brier_lo']:.4f}, {ci['brier_hi']:.4f}]"
          f"  CRPS={crps:.4f} [{ci['crps_lo']:.4f}, {ci['crps_hi']:.4f}]  N={n}")

# Sort by Brier ascending
records.sort(key=lambda r: r["brier"])

# --- plot: 2 vertical subplots — Brier (top) and CRPS (bottom) ----------------
fig, (ax_b, ax_c) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
fig.subplots_adjust(hspace=0.08)

x      = np.arange(len(records))
width  = 0.55

ANNOT_PAD  = 0.012
ANNOT_BBOX = dict(boxstyle="round,pad=0.15", facecolor="white",
                  alpha=0.85, edgecolor="none")

for i, r in enumerate(records):
    ci = r["ci"]
    # ── Brier subplot ─────────────────────────────────────────────────────────
    brier_err = [[r["brier"] - ci["brier_lo"]], [ci["brier_hi"] - r["brier"]]]
    ax_b.bar(i, r["brier"], width, color=r["color"],
             edgecolor="black", linewidth=0.6)
    ax_b.errorbar(i, r["brier"], yerr=brier_err,
                  fmt="none", ecolor="black", elinewidth=1.2, capsize=5)
    ax_b.text(i, ci["brier_hi"] + ANNOT_PAD, f"{r['brier']:.3f}",
              ha="center", va="bottom", fontsize=9, bbox=ANNOT_BBOX)
    # ── CRPS subplot ──────────────────────────────────────────────────────────
    crps_err = [[r["crps"] - ci["crps_lo"]], [ci["crps_hi"] - r["crps"]]]
    ax_c.bar(i, r["crps"], width, color=r["color"],
             edgecolor="black", linewidth=0.6)
    ax_c.errorbar(i, r["crps"], yerr=crps_err,
                  fmt="none", ecolor="black", elinewidth=1.2, capsize=5)
    ax_c.text(i, ci["crps_hi"] + ANNOT_PAD, f"{r['crps']:.3f}",
              ha="center", va="bottom", fontsize=9, bbox=ANNOT_BBOX)

# ── Brier reference line (blended transform: x in axes fraction, y in data) ──
blend_b = ax_b.get_yaxis_transform()
ax_b.axhline(BRIER_CHANCE, ls="--", c="grey", lw=1)
ax_b.text(0.99, BRIER_CHANCE + 0.005, "Chance (0.25)",
          color="grey", ha="right", va="bottom", fontsize=9, transform=blend_b)
ax_b.set_ylabel("Brier score on p50  (lower = better)", fontsize=11)
max_brier_hi = max(r["ci"]["brier_hi"] for r in records)
ax_b.set_ylim(0, max(max_brier_hi + ANNOT_PAD + 0.025, BRIER_CHANCE + 0.055))
ax_b.tick_params(bottom=False)

# ── CRPS reference line ───────────────────────────────────────────────────────
blend_c = ax_c.get_yaxis_transform()
ax_c.axhline(CRPS_UNIFORM, ls=":", c="#666", lw=1)
ax_c.text(0.99, CRPS_UNIFORM + 0.005, "Uniform Beta(1,1) = 0.333",
          color="#555", ha="right", va="bottom", fontsize=9, transform=blend_c)
ax_c.set_ylabel("Mean CRPS  (lower = better)", fontsize=11)
max_crps_hi = max(r["ci"]["crps_hi"] for r in records)
ax_c.set_ylim(0, max(max_crps_hi + ANNOT_PAD + 0.025, CRPS_UNIFORM + 0.045))

# ── shared x-axis ─────────────────────────────────────────────────────────────
ax_c.set_xticks(x)
ax_c.set_xticklabels([f"{r['label']}\nN={r['n']}" for r in records], fontsize=10)

# ── overall title ─────────────────────────────────────────────────────────────
fig.suptitle("Cross-forecaster calibration on Lyptus\n"
             "(K=5 target tasks × 12 target LLMs × 5 bins, 1 expert, all-except-target)",
             fontsize=11, y=1.01)

plt.tight_layout()
fig.text(0.02, -0.02,
         "*GPT-5.5 completed 190/300 elicitations (cyber-flagged tasks refused); others completed all.  "
         "Error bars: 95% bootstrap CI (n = 10,000 resamples).",
         fontsize=8, color="#555", ha="left")
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"\nSaved: {OUT}")
