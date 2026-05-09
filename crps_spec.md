# CRPS for the intra-benchmark experiment

Specification adapted from Gneiting and Raftery (2007), Section 4.2.

---

## 1. The four things in play, in our setting

Before any equations, here is what every symbol stands for.

| Symbol | What it is | What it means in our setting |
|---|---|---|
| $o$ | the **outcome** | One number in $[0, 1]$. The actual per-LLM bin pass rate for one (model, target bin) cell. Example: model $M$ passed 7 of the 10 tasks in target bin $B$, so $o = 0.7$. |
| $F$ | the **forecast CDF** | The cumulative distribution function of the Beta we fit from the LLM's elicited 25th, 50th, 75th percentiles. Encodes the LLM's full uncertainty about the pass rate. |
| $F(y)$ | a **value of that CDF** at point $y$ | The LLM's probability that the true pass rate is at most $y$. Example: if $F(0.5) = 0.30$, the LLM thinks there is a 30% chance the pass rate is $0.5$ or lower. |
| $y$ | a **threshold** | A value in $[0, 1]$ we sweep over from $0$ to $1$. Not a special quantity. It is the dummy integration variable. Each $y$ corresponds to one yes/no question: is the pass rate at or below $y$? |

The forecast is a distribution. The outcome is a single number. CRPS measures how far the distribution is from being a perfect spike at the outcome.

---

## 2. The intuition: CRPS averages many Brier scores

For any threshold $y \in [0, 1]$, we can ask one yes/no question:

> Is the true pass rate at or below $y$?

The LLM's forecast answers this with a probability, $F(y)$.
Reality answers with $0$ or $1$: it is $1$ if the outcome $o$ is at or below $y$, and $0$ otherwise. Write this indicator as $\mathbf{1}\{y \geq o\}$.

The squared error of this single yes/no forecast is the Brier score at threshold $y$:

$$
\mathrm{brier}(y) \;=\; \bigl(F(y) - \mathbf{1}\{y \geq o\}\bigr)^{2}
$$

CRPS is the integral of $\mathrm{brier}(y)$ over all thresholds $y \in [0, 1]$:

$$
\mathrm{CRPS}(F, o) \;=\; \int_{0}^{1} \bigl(F(y) - \mathbf{1}\{y \geq o\}\bigr)^{2} \, dy
$$

This is equation (20) of Gneiting and Raftery, restricted to our $[0, 1]$ support. CRPS is the area under the squared-difference curve between the forecast CDF and the step function that jumps from $0$ to $1$ at the actual outcome.

A picture in words: draw the LLM's forecast CDF as a smooth S-curve from $(0, 0)$ to $(1, 1)$. Draw the perfect-hindsight CDF as a step that is flat at $0$ until $o$ and then jumps to $1$. CRPS is the area of the squared gap between these two curves.

---

## 3. The two equivalent formulas

The paper gives two formulas for the same number: equations (20) and (21).

**Integral form** (equation 20):

$$
\mathrm{CRPS}(F, o) \;=\; \int_{0}^{1} \bigl(F(y) - \mathbf{1}\{y \geq o\}\bigr)^{2} \, dy
$$

Use this when you have the CDF in closed form. We do, because we fit a Beta.

**Kernel form** (equation 21):

$$
\mathrm{CRPS}(F, o) \;=\; \mathbb{E}\,|X - o| \;-\; \tfrac{1}{2}\,\mathbb{E}\,|X - X'|
$$

where $X$ and $X'$ are two independent random draws from the forecast distribution $F$.

What each piece means in plain words:

- $\mathbb{E}\,|X - o|$: average distance between a draw from the LLM's forecast and the actual outcome. The "are you close to the truth" term.
- $\mathbb{E}\,|X - X'|$: average distance between two independent draws from the LLM's forecast. A measure of how spread out the forecast is. The factor $\tfrac{1}{2}$ rewards tighter forecasts.

So in words:

$$
\mathrm{CRPS} \;=\; (\text{average error}) \;-\; \tfrac{1}{2}\,(\text{forecast spread})
$$

A confident forecast has a small spread term, so it must be accurate to keep CRPS low. A diffuse forecast has a large spread term that subtracts off some of the error penalty: hedging is allowed but not free.

The two forms give the same number. The kernel form is easier when you have samples; the integral form is easier when you have a CDF.

---

## 4. Why this is the right metric for us

The paper highlights three properties (Section 4.2, p. 367), all of which match our needs.

1. **It scores a distribution against a number.** Our forecast is a Beta over $[0, 1]$; our outcome is one number in $[0, 1]$. CRPS uses the whole CDF, so the spread between $q_{25}$ and $q_{75}$ actually enters the score. Squared error on the Beta mean would throw the spread away.

2. **It is strictly proper.** The LLM minimises its expected CRPS only by reporting its true predictive distribution. It cannot game CRPS by widening or narrowing its quantiles relative to its actual belief.

3. **It reduces to absolute error when the forecast is a point.** If $F$ is a point mass at value $p$, then $\mathrm{CRPS}(F, o) = |p - o|$. A hypothetical point forecaster and our distributional forecaster sit on the same axis and can be compared directly.

The Section 8 case study in the paper makes the same argument empirically: improper alternatives (linear score, probability score) gave nonsensical optima; the logarithmic score was unstable when the predictive density was tight near the observation; CRPS gave a stable, interpretable answer.

---

## 5. What we compute, per cell

For each (model $M$, target task $t$) cell:

1. **Inputs from elicitation:** three numbers $q_{25}, q_{50}, q_{75}$, each in $[0, 1]$.
2. **Input from data:** one number $o \in [0, 1]$ (per-LLM bin pass rate).
3. **Fit step:** find $\mathrm{Beta}(a, b)$ whose 25th, 50th, 75th percentiles match $(q_{25}, q_{50}, q_{75})$ as closely as possible. Call its CDF $F$.
4. **Score step:** compute

$$
\mathrm{CRPS}(F, o) \;=\; \int_{0}^{1} \bigl(F(y) - \mathbf{1}\{y \geq o\}\bigr)^{2} \, dy
$$

5. Record the cell's CRPS.

---

## 6. What we report, per condition

Each experimental condition (one choice of source bins, prompt variant, number of source bins, etc.) has 75 cells: 15 models times 5 target bins. Index them $i = 1, \ldots, 75$. Headline number for the condition:

$$
\overline{\mathrm{CRPS}} \;=\; \frac{1}{75} \sum_{i=1}^{75} \mathrm{CRPS}(F_i, o_i)
$$

Lower is better. Units are the same as $o$, so $\overline{\mathrm{CRPS}} = 0.05$ means the average forecast distribution sits, in an integrated-CDF sense, $0.05$ probability units away from a point mass at the observed pass rate.

---

## 7. Two baselines to anchor the numbers

A value of $\overline{\mathrm{CRPS}} = 0.05$ means nothing on its own. The paper recommends comparison against reference forecasts (Section 2.3). Two natural references in our setting:

- **Uniform baseline.** Predict $\mathrm{Beta}(1, 1)$ (flat on $[0, 1]$) for every cell. Any condition that does not beat this is uninformative.
- **Source-data baseline.** Predict a point mass at model $M$'s observed pass rate on the source bins. This is the no-LLM benchmark. The elicitation procedure must beat this to justify itself.

Both reduce to single numbers we can plot as horizontal reference lines on the comparison chart.

---

## 8. Reference

Gneiting, T. and Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. *Journal of the American Statistical Association* 102(477), 359-378. CRPS in Section 4.2, equations (20) and (21), p. 367.
