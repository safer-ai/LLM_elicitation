# Measuring LLM Forecasters for Quantitative Risk Modelling of Cyber Misuse

**Jeff T. Mohl · Madhav Khanal · Jakub Kryś · Matt Smith** — SaferAI · SPAR Spring 2026

---

## Abstract  *(~35 words)*

Cyber-risk models need expert probability elicitation, but experts don't scale. We test **LLM forecasters**: reproducible and persona-robust, but **anchored on user baselines**. All 5 frontier forecasters reach **Brier 0.10–0.18** on Lyptus, beating the **0.25** uniform baseline.

---

## Introduction  *(~30 words)*

MITRE-style cyber-risk models need **P(success | capability)** at a scale humans can't supply. **Can LLMs replace expert forecasters?** A viable substitute must be **consistent**, **calibrated**, and **discriminating**.

---

## Methods  *(~50 words)*

Forecasters return **25/50/75 percentiles** → fit Beta. Variance attributed via **Fréchet ANOVA** in W₂-space across persona / temperature / repeat. Calibration tested on **Lyptus offensive-cyber benchmark** under the **all-except-target** condition: **K=5 target tasks × 12 target LLMs × 5 difficulty bins = 300 elicitations** per forecaster, 1 expert. Metrics: **Brier on p50**, **CRPS (Beta-fit to percentiles)**.

---

## Results & Discussion  *(~90 words)*

Forecaster **model identity** dominates distributional variance (**48–69%** Fréchet ANOVA), far above persona (**6–25%**) or temperature (**<5%**). The single largest prompt factor is the **baseline value**: removing it shifts outputs by **W² = 0.176–0.233** with **17.3× more within-condition variance** than the control — forecasts simply inherit the supplied prior (Fig 1).

> **Fig 1.** `latex/figures/prompt_sensitivity_chart_week4.png`

On Lyptus, all **5 frontier forecasters** beat the **0.25 / 0.333** uniform baselines (Brier **0.10–0.18**, CRPS **0.17–0.26**). **Sonnet 4.6 ≈ Opus 4.7** at half cost — one expert suffices. GPT-5.5 excluded (190/300; cyber-refusals). (Fig 2)

> **Fig 2.** `poster/forecaster_brier_sweep.png`

---

## Conclusion  *(~40 words)*

Frontier LLMs are **viable substitutes for expert elicitation** at scale, conditional on carefully supplying baseline anchors. **Sonnet 4.6** is the practical default. Next: **MITRE-step elicitation** where ground truth is sparse, and stress tests against adversarial / refusal-prone framings.

---

## References  *(footer, small, 8pt)*

[1] Payne, Miller & Peters, *Offensive Cybersecurity Time Horizons*, Lyptus Research, 2026. [2] Dubey & Müller, *Fréchet ANOVA*, Ann. Statist. 2019. [3] Gneiting & Raftery, *Proper Scoring Rules*, JASA 2007. [4] MITRE ATT&CK.

---

### Figure decision rationale (delete before printing)

- **2 figures, 6 sections** (Abstract / Intro / Methods / Results & Discussion / Conclusion / References). Results & Discussion merged.
- **Fig 1** = `latex/figures/prompt_sensitivity_chart_week4.png` — two-panel baseline-ablation chart from midterm report (W² shift + relative variability).
- **Fig 2** = `poster/forecaster_brier_sweep.png` — cross-forecaster Brier + CRPS bar chart, generated from each forecaster's canonical `plots/statistics.txt` via `poster/make_forecaster_brier_chart.py`.
- R1 (model > persona) stays text-only — its available figure uses 2024-era models.
- **Closest-bin / condition E results are dropped** — final experimental setup is **all-except-target (condition A)** per discussion-A framing.

**Total body ≈ 340 words.** Bold every number.
