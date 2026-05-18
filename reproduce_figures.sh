#!/usr/bin/env bash
# reproduce_figures.sh
#
# Regenerates all figures for the final report from existing data (no API calls).
# Collects them into figures_for_report/ at the repo root.
#
# Usage:
#   cd /path/to/LLM_elicitation
#   bash reproduce_figures.sh
#
# Requirements: pip install -r requirements.txt

set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
OUT="$REPO/figures_for_report"
mkdir -p "$OUT"

echo "=== Regenerating figures ==="

# ── 1. Prompt sensitivity: W₂ shift + output stability ────────────────────────
# Output: latex/figures/prompt_sensitivity_chart_week4.png
# Source data: prompt_sensitivity/output/runs/  (committed)
echo "[1/5] Prompt sensitivity chart..."
python3 "$REPO/poster/make_prompt_sensitivity_chart.py"

# ── 2. W₁ distribution shift bar chart ────────────────────────────────────────
# Output: latex/figures/w1_chart (1)_week5.png
# Source data: prompt_sensitivity/output/runs/  (committed)
echo "[2/5] W1 chart..."
python3 "$REPO/poster/make_w1_chart.py"

# ── 3. Fréchet ANOVA: cross-model + persona Beta distributions ─────────────────
# Output: output_data/experiments/frechet_anova/beta_cross_model_distributions_percentile.png
#         output_data/experiments/frechet_anova/beta_persona_distributions_percentile.png
# Source data: output_data/experiments/frechet_anova/  (committed CSVs)
echo "[3/5] Frechet ANOVA distribution plots..."
python3 "$REPO/output_data/experiments/frechet_anova/plot_beta_distributions_percentile.py"

# ── 4. Baseline anchoring ──────────────────────────────────────────────────────
# Output PNGs are committed and do not need regeneration.
# Source CSVs were excluded from git (too large). To regenerate from scratch:
#   python3 output_data/experiments/baseline_uplift/execution_task/analyze_results.py
#   python3 output_data/experiments/baseline_uplift/initial_access/analyze_results.py
echo "[4/5] Baseline anchoring plots (using committed figures, no regeneration needed)..."

# ── 5. Cross-forecaster Brier + CRPS (model sweep) ───────────────────────────
# Output: poster/forecaster_brier_sweep.png
# Source data: intra_benchmark_calibration/experiments/G_model_sweep/results/*/plots/
#              (statistics.txt + scored_with_crps.csv already committed)
echo "[5/5] Forecaster Brier sweep chart..."
python3 "$REPO/poster/make_forecaster_brier_chart.py"

echo ""
echo "=== Collecting figures → figures_for_report/ ==="

# ── Midterm figures ────────────────────────────────────────────────────────────
cp "$REPO/latex/figures/prompt_sensitivity_chart_week4.png"    "$OUT/fig_prompt_sensitivity.png"
cp "$REPO/latex/figures/w1_chart (1)_week5.png"               "$OUT/fig_w1_shift.png"
cp "$REPO/output_data/experiments/frechet_anova/beta_cross_model_distributions_percentile.png" \
                                                               "$OUT/fig_frechet_cross_model.png"
cp "$REPO/output_data/experiments/frechet_anova/beta_persona_distributions_percentile.png" \
                                                               "$OUT/fig_frechet_persona.png"
cp "$REPO/output_data/experiments/baseline_uplift/execution_task/results/baseline_uplift_plot.png" \
                                                               "$OUT/fig_baseline_anchoring.png"
cp "$REPO/output_data/experiments/baseline_uplift/execution_task/numactors/results/numactors_uplift_only.png" \
                                                               "$OUT/fig_numactors_anchoring.png"

# ── New figures (post-midterm) ─────────────────────────────────────────────────
cp "$REPO/poster/forecaster_brier_sweep.png"                   "$OUT/fig_model_sweep_brier.png"

# Per-model diagnostic plots from the model sweep
for MODEL in sonnet46 opus47 haiku45 gemini25flash gpt55; do
    PLOTS="$REPO/intra_benchmark_calibration/experiments/G_model_sweep/results/$MODEL/plots"
    cp "$PLOTS/per_model_brier.png"       "$OUT/fig_${MODEL}_per_model_brier.png"
    cp "$PLOTS/calibration_scatter.png"   "$OUT/fig_${MODEL}_calibration_scatter.png"
    cp "$PLOTS/reliability_diagram.png"   "$OUT/fig_${MODEL}_reliability.png"
    cp "$PLOTS/metr_style_logfst.png"     "$OUT/fig_${MODEL}_metr_logfst.png"
    cp "$PLOTS/brier_heatmap.png"         "$OUT/fig_${MODEL}_brier_heatmap.png"
done

# ── Table CSVs ─────────────────────────────────────────────────────────────────
for MODEL in sonnet46 opus47 haiku45 gemini25flash gpt55; do
    PLOTS="$REPO/intra_benchmark_calibration/experiments/G_model_sweep/results/$MODEL/plots"
    cp "$PLOTS/statistics.txt"        "$OUT/table_${MODEL}_statistics.txt"
    cp "$PLOTS/scored_with_crps.csv"  "$OUT/table_${MODEL}_scored.csv"
done

echo ""
echo "=== DONE. Files written to: $OUT ==="
echo ""

# ── NOTE: condition A/B/C/E comparison ────────────────────────────────────────
# The condition A vs E vs C comparison table (week 11-12 results) is NOT
# regenerable from this repo — those runs were on Jakub's machine.
# The numbers are documented in poster/context.md §6.4 and poster/week 11 and 12.md.
# If you need to re-run:
#   - Condition A: python intra_benchmark_calibration/run_calibration.py -c <config_A.yaml>
#   - Condition E: python intra_benchmark_calibration/run_calibration.py -c <config_E.yaml>
#   See intra_benchmark_calibration/QUICK_START.md for full setup.
echo "NOTE: Condition A/B/C/E comparison figures are not regenerable from this"
echo "      repo (data on Jakub's machine). Numbers are in poster/context.md §6.4."
echo "      See intra_benchmark_calibration/QUICK_START.md to re-run."
