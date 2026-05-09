# SPAR S26 Poster — Build Spec

Lean, paste-into-Canva spec. One landscape page, three columns + a header, modelled after the Stuxbench S25 poster: five standard sections (Abstract, Introduction, Methods, Results, Discussion), ~400 body words on the poster total, three figures, no roadmap or sub-panels.

- **Format:** landscape, ~1920 × 1080, readable on a laptop screen.
- **Layout:**
  - Header strip (full-width).
  - Column 1: Abstract, Introduction, Methods.
  - Column 2: Result 1 + figure, Result 2 + figure.
  - Column 3: Result 3 + figure, Discussion, References.
- **Total target body words on the poster:** 380–420.

---

## Header

**Title (largest, ~64 pt):**

> Measuring LLM Forecasters for Quantitative Risk Modelling of Cyber Misuse

**Authors (subtitle):**

> Jeff T. Mohl · Madhav Khanal · Jakub Kryś · Matt Smith

**Affiliation:**

> SaferAI · Supervised Program for Alignment Research, Spring 2026

Right side of header: SaferAI logo (asset to add).

---

## Abstract  (~55 words)

> Quantitative cyber-risk models translate AI-benchmark scores into forecasts of attack-step capability uplift, but the translation is built by slow, expensive **human expert elicitation**. We test whether **LLM-simulated experts** can replace humans by quantifying their consistency, prompt sensitivity, and calibration against a verifiable proxy. Forecasts are robust to most knobs; the leverage points are **model identity** and **baseline anchoring**.

---

## Introduction  (~60 words)

> Risk modelling is the backbone of management in finance, aviation and nuclear, but for AI it is a major gap. SaferAI's nine cyber-risk models map benchmark scores to MITRE ATT&CK uplift via judgemental estimation by domain experts (Barrett et al. 2025) — a process that is slow, costly, and error-compounding as risk models scale. We propose **LLM-simulated experts** to make elicitation verifiable, reproducible, and cheap.

---

## Methods  (~70 words)

> For each (model, attack step), we elicit the **25th, 50th and 75th percentiles** of P(success | benchmark task) and fit a **Beta(α, β)** distribution. We compare runs and conditions with **W1 / W2 Wasserstein** distance (on quantiles and Beta-fit CDFs). **Fréchet ANOVA** (Dubey & Müller 2019) on the Wasserstein space attributes variance to *model*, *persona*, *temperature* or *prompt*. To probe accuracy without human labels, we calibrate against a benchmark-transfer task with known ground truth: **LiveBench LCB-Generation → Coding-Completion**.

---

## Results

Three findings. One figure each. ~50 words per finding.

### 1. Model identity dominates persona and temperature

> Across three MITRE steps, Fréchet-ANOVA ICC_F (between-group share of total Fréchet variance): **cross-model 48–69 %** (p < 0.001) ≫ persona 6–25 % ≫ sampling-temperature ≤ 12 % (mostly n.s.). Ten expert personas overlap tightly within a single model; three frontier models separate cleanly across the same steps.

**Figure 1:** [output_data/experiments/frechet_anova/beta_cross_model_distributions_percentile.png](output_data/experiments/frechet_anova/beta_cross_model_distributions_percentile.png)
*Caption:* Cross-model belief distributions for three attack steps (Beta fits to elicited percentiles). Models separate; personas (not shown) overlap.

### 2. Prompts are mostly invariant — baselines anchor strongly

> We compared seven prompt ablations to a control prompt (~615 words). Removing the supplied **baseline value** shifts the elicited distribution by W1 ≈ 0.17 (~5× within-control variance); removing **baseline + CI** gives W1 ≈ 0.24. Trimming the reasoning and analysis scaffolding is essentially free. Sweeping the baseline 10 %→90 % yields a near-linear shift in the elicited median.

**Figure 2:** [prompt_sensitivity/output/wasserstein_combined_all_conditions.png](prompt_sensitivity/output/wasserstein_combined_all_conditions.png)
*Caption:* W1 / W2 distance from each prompt ablation to the control. Dashed lines = within-control variance.

### 3. Empirical anchors close the calibration gap

> Predicting per-task solve rate on Coding-Completion from a model's capability bin on LCB-Generation: with no anchors the LLM **overestimates 26/28 cells**. Three empirical median-percentile anchors cut **MAE 0.270 → 0.196** and **bias +0.259 → +0.022**. The pre-estimation analysis stage adds no accuracy (Wilcoxon p = 0.72), so it can be dropped. Weak-tier bins remain overestimated (+0.29).

**Figure 3:** [inter_benchmark_calibration/results/comparison_no_anchor_vs_few_shot.png](inter_benchmark_calibration/results/comparison_no_anchor_vs_few_shot.png)
*Caption:* Calibration scatter — LLM forecast vs ground-truth solve rate, without anchors (left) and with three anchors (right).

---

## Discussion  (~80 words)

> LLM forecasters are robust to most prompt, persona and temperature choices, so those are not where the leverage is. The dominant axes are **model identity** and the **baseline anchor** — and the anchoring effect appears in **both** the MITRE elicitation and the benchmark-transfer calibration, suggesting a single underlying mechanism. Next, we scale the ground-truth pipeline to the **cyber-time-horizons dataset** (291 tasks × 15 models × 5 difficulty bins) and score forecasts with **CRPS** on the Beta-fit (Gneiting & Raftery 2007), unlocking head-to-head comparison with human experts.

---

## References  (footer, ~12 pt, single line)

> Barrett et al. 2025 (arXiv:2512.08864) · Panaretos & Zemel 2019 · Dubey & Müller 2019 · Gneiting & Raftery 2007 · Lyptus Offensive Cyber Time Horizons.

Right side of footer: code link and contact, e.g. `github.com/<org>/LLM_elicitation` · `matt@safer-ai.org`.

---

## Figure inventory

All three figures already exist in the repo at the cited paths:

1. [output_data/experiments/frechet_anova/beta_cross_model_distributions_percentile.png](output_data/experiments/frechet_anova/beta_cross_model_distributions_percentile.png) — Result 1.
2. [prompt_sensitivity/output/wasserstein_combined_all_conditions.png](prompt_sensitivity/output/wasserstein_combined_all_conditions.png) — Result 2.
3. [inter_benchmark_calibration/results/comparison_no_anchor_vs_few_shot.png](inter_benchmark_calibration/results/comparison_no_anchor_vs_few_shot.png) — Result 3.

Figures intentionally **dropped** from this version (kept available if a panel feels light): the persona-overlap plot ([beta_persona_distributions_percentile.png](output_data/experiments/frechet_anova/beta_persona_distributions_percentile.png)), the temperature grid ([beta_temperature_means_grid.png](output_data/experiments/frechet_anova/beta_temperature_means_grid.png)), the all-conditions bar chart ([comparison_all_conditions_final.png](inter_benchmark_calibration/results/comparison_all_conditions_final.png)), and mid-term Figs 2 & 7 from the PDF.

---

## Visual notes

- **Type hierarchy:** title 64 pt → section heading 32 pt → body 20–22 pt → caption 16 pt → footer 12 pt. Body text never below 18 pt.
- **One accent colour** (suggest SaferAI brand or deep teal) for headings and figure callouts. Body in dark grey on near-white.
- **Per-section word cap on the poster:** 80 words. The reader should finish a section in <10 seconds.
- **Figures:** preserve native aspect ratios; ensure axis text is legible at final print/render size.
- **Pre-flight (3 checks):** (i) all bold numbers in Results match the source files; (ii) the *baseline anchoring* idea visibly links Result 2 and Result 3 (same accent colour or a thin connecting arrow); (iii) the SPAR submission form is filled with title, link, "Whole Team", and all four collaborators.
