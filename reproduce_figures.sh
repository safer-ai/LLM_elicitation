#!/usr/bin/env bash
# reproduce_figures.sh
#
# Generates all figures for the final report in one shot.
# Output goes to figures_for_report/ at the repo root.
#
# Usage:
#   cd /path/to/LLM_elicitation
#   bash reproduce_figures.sh
#
# Requirements: pip install -r requirements.txt
#
# ── What this script does ─────────────────────────────────────────────────────
#
# MIDTERM FIGURES (all 7 from the midterm report PDF)
#   Fig 1  — W1 CDF illustration            [committed PNG, no script]
#   Fig 2  — Self-consistency bar chart      [committed PNG, no script]
#   Fig 3  — Cross-approach invariance       [committed PNG, no script]
#       (Figs 1-3 live only as inline outputs in
#        consistency_checks/consistency_experiments.ipynb.
#        Re-extract with: bash reproduce_figures.sh --reextract-notebook)
#   Fig 4  — Expert persona Beta dists       [regenerated: frechet_anova script]
#   Fig 5  — Cross-model Beta dists          [regenerated: frechet_anova script]
#   Fig 6  — Prompt sensitivity W1 chart     [regenerated: make_w1_chart.py]
#   Fig 7  — Baseline anchoring              [committed PNG, source CSVs not in repo]
#
# POST-MIDTERM FIGURES
#   New 1  — Model sweep vs baselines        [committed PNG: report_analyses/]
#   New 2  — Model sweep vs baselines per bin [committed PNG: report_analyses/]
#   New 3  — All-tasks model sweep           [committed PNG: report_analyses/]
#   New 4  — METR-style logFST (Sonnet 4.6)  [committed PNG: G_model_sweep/]
#   New 5  — p50 vs GT solve rate, bin 1      [committed PNG: H_task_variance_bin1/]
#   New 6  — Per-task Brier, bin 1            [committed PNG: H_task_variance_bin1/]
#
# TABLE CSVs
#   Condition A/B/C/E/F comparison: numbers are hardcoded in poster/context.md §6.4
#   Model sweep scored results and baseline comparison CSVs also copied.
#
# ─────────────────────────────────────────────────────────────────────────────

set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
OUT="$REPO/figures_for_report"
mkdir -p "$OUT"

# ── Optional: re-extract notebook inline figures ──────────────────────────────
if [[ "$1" == "--reextract-notebook" ]]; then
    echo "[notebook] Re-extracting Figs 1-3 from consistency_experiments.ipynb..."
    python3 << 'PYEOF'
import json, base64
from pathlib import Path

OUT = Path("/Users/madhav/SaferAI/LLM_elicitation/latex/figures")
nb_path = Path("/Users/madhav/SaferAI/LLM_elicitation/consistency_checks/consistency_experiments.ipynb")

with open(nb_path) as f:
    nb = json.load(f)

to_extract = {
    13: (0, "fig_midterm_w1_illustration.png"),
    11: (0, "fig_midterm_selfconsistency_bar.png"),
    16: (0, "fig_midterm_invariance_w1.png"),
    17: (0, "fig_midterm_invariance_w1_beta.png"),
}
for cell_idx, (output_idx, fname) in to_extract.items():
    cell = nb["cells"][cell_idx]
    png_outputs = [o for o in cell.get("outputs", []) if "image/png" in o.get("data", {})]
    data = png_outputs[output_idx]["data"]["image/png"]
    if isinstance(data, list):
        data = "".join(data)
    (OUT / fname).write_bytes(base64.b64decode(data))
    print(f"  Saved: {fname}")
PYEOF
fi

echo "=== Regenerating figures from committed data ==="

# ── Figs 4 + 5: Fréchet ANOVA Beta distributions ─────────────────────────────
# Source: output_data/experiments/frechet_anova/  (committed CSVs + JSON)
# Output: output_data/experiments/frechet_anova/beta_*_percentile.png
echo "[1/3] Frechet ANOVA Beta distributions (Figs 4 + 5)..."
python3 "$REPO/output_data/experiments/frechet_anova/plot_beta_distributions_percentile.py"

# ── Fig 6: Prompt sensitivity — W1 shift across 6 ablation conditions ─────────
# Source: prompt_sensitivity/output/runs/  (committed run data)
# Output: latex/figures/w1_chart (1)_week5.png
echo "[2/3] Prompt sensitivity W1 chart (Fig 6)..."
python3 "$REPO/poster/make_w1_chart.py"

# ── Post-midterm: Cross-forecaster Brier + CRPS bar chart ─────────────────────
# Source: intra_benchmark_calibration/experiments/G_model_sweep/results/*/plots/
#         (statistics.txt + scored_with_crps.csv, committed)
# Output: poster/forecaster_brier_sweep.png
echo "[3/3] Cross-forecaster Brier sweep chart..."
python3 "$REPO/poster/make_forecaster_brier_chart.py"

echo ""
echo "=== Collecting all figures → figures_for_report/ ==="

# ── MIDTERM: Figs 1-3 (committed PNGs extracted from notebook) ───────────────
cp "$REPO/latex/figures/fig_midterm_w1_illustration.png"     "$OUT/midterm_fig1_w1_illustration.png"
cp "$REPO/latex/figures/fig_midterm_selfconsistency_bar.png" "$OUT/midterm_fig2_selfconsistency.png"
cp "$REPO/latex/figures/fig_midterm_invariance_w1.png"       "$OUT/midterm_fig3a_invariance_w1.png"
cp "$REPO/latex/figures/fig_midterm_invariance_w1_beta.png"  "$OUT/midterm_fig3b_invariance_w1_beta.png"

# ── MIDTERM: Figs 4-5 (Fréchet ANOVA, just regenerated) ─────────────────────
cp "$REPO/output_data/experiments/frechet_anova/beta_persona_distributions_percentile.png" \
   "$OUT/midterm_fig4_persona_betas.png"
cp "$REPO/output_data/experiments/frechet_anova/beta_cross_model_distributions_percentile.png" \
   "$OUT/midterm_fig5_crossmodel_betas.png"

# ── MIDTERM: Fig 6 (prompt sensitivity W1, just regenerated) ─────────────────
cp "$REPO/latex/figures/w1_chart (1)_week5.png" "$OUT/midterm_fig6_prompt_sensitivity_w1.png"

# ── MIDTERM: Fig 7 (baseline anchoring, committed — source CSVs not in repo) ──
cp "$REPO/latex/figures/baseline_uplift_plot_week5.png"    "$OUT/midterm_fig7a_baseline_anchoring.png"
cp "$REPO/latex/figures/numactors_baseline_uplift_plot.png" "$OUT/midterm_fig7b_numactors_anchoring.png"

# ── POST-MIDTERM: Model sweep vs baselines ────────────────────────────────────
cp "$REPO/report_analyses/results/model_sweep_baseline/score_comparison.png" \
   "$OUT/new_fig1a_model_sweep_vs_baselines.png"
cp "$REPO/report_analyses/results/model_sweep_baseline/target_bin_score_comparison.png" \
   "$OUT/new_fig1b_model_sweep_per_bin.png"
cp "$REPO/report_analyses/results/model_sweep_baseline_all_tasks/score_comparison.png" \
   "$OUT/new_fig1c_all_tasks_comparison.png"

# ── POST-MIDTERM: Sonnet 4.6 + bin-1 diagnostic figures ──────────────────────
SONNET="$REPO/intra_benchmark_calibration/experiments/G_model_sweep/results/sonnet46/plots"
cp "$SONNET/metr_style_logfst.png"    "$OUT/new_fig2_metr_logfst_sonnet46.png"
BIN1="$REPO/intra_benchmark_calibration/experiments/H_task_variance_bin1/results"
cp "$BIN1/fig_exp_h_bin1_p50_vs_gt_solve_rate.png" "$OUT/new_fig3_p50_vs_gt_solve_rate_bin1.png"
cp "$BIN1/fig_exp_h_bin1_per_task_brier.png"        "$OUT/new_fig4_per_task_brier_bin1.png"

# ── TABLE CSVs ────────────────────────────────────────────────────────────────
# Condition A/B/C/E/F: numbers are in poster/context.md §6.4 (hardcoded, no CSV)
cp "$REPO/intra_benchmark_calibration/experiments/G_model_sweep/results/sonnet46/plots/scored_with_crps.csv" \
   "$OUT/table_sonnet46_scored.csv"
cp "$REPO/intra_benchmark_calibration/experiments/G_model_sweep/results/sonnet46/plots/statistics.txt" \
   "$OUT/table_sonnet46_statistics.txt"
cp "$REPO/report_analyses/results/model_sweep_baseline/comparison_table.csv" \
   "$OUT/table_model_sweep_baseline_comparison.csv"
cp "$REPO/report_analyses/results/model_sweep_baseline_all_tasks/comparison_table.csv" \
   "$OUT/table_model_sweep_all_tasks_comparison.csv"

echo ""
echo "=== DONE. $(ls "$OUT" | wc -l | tr -d ' ') files written to: $OUT ==="
echo ""
echo "Contents:"
ls "$OUT"
