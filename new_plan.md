The purpose of this document is to set a plan for the remainder of the project, with a focus on finding automated improvements to prompts for LLM forecasters.

- what is $ the cost per cell? I’m not sure if the estimates Claude came up with below are right
- what N is appropriate for the grand Brier? Currently, we’re using N=300, but I’d like to decrease this to save money. Could we do something like:
    - use N=300 as a reference point (even though we have a single BS with N=300, not a distribution of BS_300)
    - randomly subsample, say, N=200 and bootstrap this so that we get a histogram showing BS_50
    - repeat this for N=150, N=100, N=50, … and see at which point the boostrapped histogram is all over the place. This will tell us what is the lowest N we can get away with
- when we write code for the GEPA thing, we need to make sure that:
    - everything is logged, so things can be easily restarted
    - metrics (Brier/CRPS and their Murphy/Yates decompositions) are tracked throughout the training process
    - there’s some automatic cutoff, i.e. if the Brier is not improving after some number of initial iterations, the code stops in order not to waste tokens
- regarding train/test split, are we happy just using the 7 benchmarks within the Lyptus dataset, or should we look for another 2 benchmarks? I think it will be hard to find 2 other benchmarks that meet all the requirements we have, and Lyptus contains enough questions, so I’d say we just go for that. Regarding doing train/test on Lyptus:
    1. there are two options for how we do the split:
        1. hold 1-2 sub-benchmarks as the test set
        2. don’t care about which sub-benchmark a task belongs to, just do an 80/20 split
        
        I think (a) is better, because then we can also test the transferability of LLM forecaster quality to new, unseen contexts. If we train on a combination of all 7 benchmarks, and then evaluate on some other subset of these 7 benchmarks, I think that classifies as (a weaker form of) data leakage
        
    2. it might be useful to hold out the hardest tasks (in terms of FST) because that fits our objective of making predictions about future levels of risk
    - so, an ideal scenario to combine (1) and (2) would be to check whether 1 or 2 of the seven Lyptus benchmarks occupy the highest bins of the FST space, and use them as the test set
- i think doing Murphy decomposition + Yates decomposition (see Sec. 0) is a good idea, it’s more informative than just BS / CRPS on their own
- [ ]  convince myself that the Murphy decomposition is exact for discrete variables
- [ ]  then, for continuous varsriables, figure out how this inexactness biases the estimator and whether that will lead to different behaviour during optimisation than the original Brier
- [ ]  challenge Claude on whether the post-hoc re-calibration should be monotonic or not. If not, we’re gonna get better results, but what’s the meaning of that?
- [ ]  do recalibration on the train set, apply to the test set and see how it improves over the raw prompt
    - [ ]  then repeat the same with GEPA- train on the train set apply to the test set
    - [ ]  compare the two approaches. If the latter approach is worse, then spending money on GEPA is pointless
- [ ]  test whether the 1-stage analysis is different to the 2-stage analysis
- [ ]  if the 1-stage analysis is the same, figure out what is the best starting prompt for GEPA. We have identified that pretty much nothing else in the prompts matters, so we could (1) remove those and leave analysis as the only main component of the prompt; (2) keep the irrelevant elements anyway, so that GEPA has a ‘richer’ starting point (i.e. let it decide). Matt’s intuition is that (1) is better because LLM-based optimisers will most likely keep *adding* new stuff, not subtracting them. But we can read more about how exactly GEPA works and what the best practices are

**<BEGIN CLAUDE OUTPUT>**

## **1. Methodological caveat (raise with the team).**

Because each forecast is scored against a 0/1 outcome, grand Brier conflates calibration and resolution. “Improving the prompt” is ambiguous unless you decide which you’re optimising. Recommend decomposing the grand Brier two ways — **Murphy** (standard ML-venue language) and **Yates bias/slope/scatter** (the more communicative form for our invariance narrative). Report both.

### Murphy decomposition (reliability–resolution–uncertainty)

For N forecasts $f_i \in [0, 1]$ with binary outcomes  $o_i \in \{0, 1\}$, grand Brier is

$\text{BS} = \tfrac{1}{N}\sum_{i=1}^{N}(f_i - o_i)^2 .$

Group forecasts into $K$ probability bins; bin $k$ has $n_k$ forecasts, mean forecast $\bar{f}_k$, observed frequency $\bar{o}_k$. With overall base rate $\bar{o}$:

$\text{BS} = \underbrace{\tfrac{1}{N}\sum_k n_k(\bar f_k - \bar o_k)^2}_{\text{reliability (calibration), } \downarrow}
- \underbrace{\tfrac{1}{N}\sum_k n_k(\bar o_k - \bar o)^2}_{\text{resolution, } \uparrow}
+ \underbrace{\bar o(1-\bar o)}_{\text{uncertainty, fixed}}$

- **Reliability**: does a forecast of $p$ come true $p$ of the time. The *only* term post-hoc recalibration can move (see below).
- **Resolution**: does the forecaster sort easy tasks from hard ones at all. Recalibration cannot create resolution; only genuine forecaster signal can.
- **Uncertainty**: irreducible base-rate variance, independent of the forecaster.

Use Murphy to attribute any Brier change to the right mechanism: recalibration → reliability; better prompting/signal → resolution.

### Yates decomposition (bias / slope / scatter)

Yates is the covariance decomposition of the same mean probability score, repackaged into three interpretable quantities. Let $\bar{f}$ be the mean forecast, $\bar{o}$ the base rate, and let $(\bar{f}_1, \bar{f}_0)$ be the mean forecast on tasks that *did* and *did not* succeed:

- **Bias** $(\bar{f}-\bar{o})$: systematic tendency to forecast high or low relative to the realised base rate. A constant offset. This can be removed by post-hoc recalibration (see below)
- **Slope** $(\bar{f}_1 - \bar{f}_0)$: mean forecast on solved tasks minus mean forecast on unsolved tasks. The discrimination signal, as a single signed number; the Yates analogue of resolution.
- **Scatter**: residual within-outcome variance of forecasts not explained by the slope — noise.

**Why Yates is worth adding for us specifically.** Our narrative is “forecasters anchor hard and are insensitive to most manipulations.” Bias/slope/scatter lets findings be stated as e.g. *“manipulation X shifted bias by Δ but left slope unchanged”* — i.e. it moved forecasts up/down without improving discrimination. That sentence is more communicative than a change in a reliability sum-of-squares, and it directly separates “the prompt recalibrated the forecaster” from “the prompt made the forecaster genuinely more discriminating” — the exact distinction that determines whether prompt optimisation is doing anything real. Slope is the single most useful number to track across all manipulations.

---

## 2. Which automated methods fit — and which don’t

The methods split cleanly into three families. Only one family is a serious recommendation; the other two are useful framing/baselines.

### Family A — Reflective / evolutionary prompt search (RECOMMENDED)

These treat the prompt as a discrete object, run it on a scored dataset, feed the **score + execution traces** to an LLM that proposes a revised prompt, and iterate. No gradients. Human-legible output.

- **GEPA (Genetic-Pareto), arXiv:2507.19457, ICLR 2026 Oral.** The strongest fit for this project. A reflection LLM reads execution traces (here: the forecaster’s rationale + the per-task Brier outcome), diagnoses *why* a forecast was bad, and proposes a targeted edit. Maintains a **Pareto front** over instances rather than greedily chasing one global best, which is what gives it generalisation and avoids the local optima that plague greedy refinement. Reported as **~10–35× more sample-efficient than GRPO-style RL** (hundreds–low-thousands of rollouts, not tens of thousands) and produces **shorter, human-readable** prompts. It accepts **any scalar metric**, so Brier/CRPS drop straight in. It is integrated into DSPy and has a standalone library (`gepa-ai/gepa`).
- **DSPy (MIPROv2 + GEPA optimisers).** DSPy is the *framework*; MIPROv2 and GEPA are *optimisers* inside it. MIPROv2 jointly optimises two things: the **instruction text** and the **selection of few-shot exemplars**. The exemplar part is independently valuable to us. Our current prompt hand-picks n=2 examples per source bin + 1 anchor. MIPROv2 instead:
    1. **bootstraps** a candidate pool of exemplars by running the current forecaster over the train tasks and keeping those whose elicitation traces score well (good Brier on their own cell);
    2. treats “which exemplars, and in what combination, go in the prompt” as a search problem;
    3. uses a **Bayesian / TPE search** over the joint space of {instruction candidates × exemplar sets}, evaluating each configuration’s Brier on a train minibatch and proposing promising next configurations. Concretely for us, this directly answers a question we currently resolve by hand and by accident: *which representative tasks should the prompt show, to most improve forecasts of held-out tasks?* Use it as the principled replacement for manual exemplar choice, and as a cheaper comparison point to GEPA (instruction+exemplar search vs. open-ended reflective rewriting). **Recommendation: use DSPy as the harness, run GEPA as the primary optimiser, MIPROv2 as a cheaper comparison and specifically to optimise exemplar selection.**
- **ProTeGi / “APO with textual gradients” (arXiv:2305.03495).** This is the original “textual gradient” paper. The “gradient” is a metaphor: an LLM produces a natural-language critique of failures (“the gradient”), then edits the prompt in the opposite direction (“descent”), with beam search over candidates. **No actual gradients, no open weights needed** — your friend was right that the name is bad. GEPA is essentially a more sample-efficient, Pareto-aware descendant. Worth citing as lineage; not worth implementing separately if you have GEPA.

### Family B — Jailbreaking / red-teaming methods (USEFUL FOR FRAMING, mostly not directly)

Your instinct to borrow from automated red-teaming is sound, but with an important split by *legibility*:

- **Rainbow Teaming (arXiv:2402.16822).** Quality-diversity search (MAP-Elites) with an LLM mutator + LLM judge, black-box, producing a *diverse archive* of effective prompts along chosen feature axes. Legible. The QD idea is genuinely transferable: instead of one optimised prompt, you could maintain an archive spanning your hand-defined feature axes (baseline/no-baseline × analysis/no-analysis × …) and read off which regions of feature space score well. This is a principled upgrade of your idea (3) grid search (see below). But it’s heavier to set up than GEPA and the diversity objective isn’t really what you want (you want the single best prompt, not coverage).
- **“The Attacker Moves Second” (arXiv:2510.09023).** This is a *survey/framework* of adaptive attacks (gradient / RL / random-search-genetic / human). Its value to you is conceptual: it’s the cleanest statement that **black-box search with an LLM-suggested genetic mutation + an LLM-as-judge** is competitive with everything else for navigating discrete prompt space without gradient access. The concrete “search attack” (genetic algorithm + LLM mutation) ≈ GEPA without the Pareto machinery. Cite for motivation; don’t reimplement.
- **Best-of-N jailbreaking (arXiv:2412.03556).** **Not useful here, and instructive about why.** Best-of-N works by sampling many random augmentations (capitalisation, shuffling, token noise) and keeping whichever happens to trip the model. The augmentations are **non-legible** and, more fundamentally, it’s pure undirected sampling with **no learning signal carried between attempts**. For your problem that’s both uninterpretable (you’d learn nothing about *which prompt features matter* — which is the actual research question) and sample-inefficient. Skip.

**Your collaborator’s key warning applies directly:** the **judge / scoring pipeline is the load-bearing component**, because reward hacking and false positives dominate failure modes. Your advantage over the jailbreaking setting is that **your reward is not an LLM judge — it’s a Brier/CRPS against held-out binary ground truth.** That largely immunises you against reward hacking *of the metric*. The residual risk is **overfitting the optimiser to the train split** (the prompt learns idiosyncrasies of the train benchmark, not transferable forecasting skill). That risk is handled by the train/test design (see Sec. 5), not by the judge.

### Family C — Gradient / white-box methods (RULED OUT — see §4)

GCG-style and FGSM-analogue methods. Correctly excluded.

### Anything missed?

- **Minimal-prompt baseline (already on your list — keep it, it’s important).** “Here are 5 tasks + scores, predict the 6th.” This is the single most valuable cheap experiment: if a 50-word prompt matches or beats the 615-word scaffold on Brier, that’s a *publishable negative result on its own* and it sets the floor any optimiser must beat. Run this **first**, before any optimisation, as the reference point. It also de-risks the whole project: if minimal ≈ complex, the “what prompt features matter” question has a clean answer (almost none beyond the data), and you pivot the paper to that finding.
- **Calibration-only post-hoc fit (cheap, strong baseline).** Before optimising the *prompt*, fit a monotone recalibration map (isotonic regression or Platt/beta calibration) on the train split’s (p50, outcome) pairs and apply to test. This is the Madhav/Jeff “linear fit baseline” generalised. It often captures most of the achievable Brier improvement at ~zero LLM cost, and any prompt optimisation should be reported *net of* recalibration — otherwise you risk attributing to clever prompting what was really a fixable calibration offset.
- **Self-consistency / ensembling over elicitations.** Note this is *not* the same as the grand-Brier cell set — those cells are the evaluation set (one 0/1 outcome each), not repeated runs of one cell. Ensembling here means re-running the forecaster (k) times on the *same* cell and averaging p50 before scoring, to cut per-cell sampling noise. Minor lever; given the cost picture (§3) it multiplies an already-expensive eval by (k), so deprioritise unless cheap.

---

## 3. Per-method: legibility + cost

The relevant unit for automated prompt optimisation is **not** “one task” — it is **one grand Brier score**, computed over many cells. The exact call count is a *formula* of the config knobs, not a single magic number:

> **calls per grand Brier = n_cells × n_experts × n_calls_per_expert**, where
**n_cells = n_source_bins × n_target_bins × n_forecasted_models × K** (K = `n_target_tasks_per_cell`),
and n_calls_per_expert = 2 per Delphi round (analysis + estimation) × delphi_rounds.
> 

Two real parameterisations bracket the range:

| Parameterisation | Cells | × experts | × calls/expert | **Calls / grand Brier** |
| --- | --- | --- | --- | --- |
| Shipped `config_example.yaml` default (single_bin, K=1, 2 experts, 1 round) | 5×4×12×1 = 240 | ×2 | ×2 | **960** |
| Madhav’s `model_sweep` (5 target tasks/bin, K=5, experts not multiplied) | 5×5×12 = 300 | ×1 | ×2 | **600** |
| `all_except_target` mode (source index collapsed, K=1, 2 experts, 1 round) | 5×12×1 = 60 | ×2 | ×2 | **240** |

A **reflective optimiser needs one scalar score per candidate prompt it evaluates**. If that score is a full grand Brier (take ~960 calls, the shipped default), then GEPA at 200–500 candidates costs **~190k–480k calls per train split** ≈ **~$1k–5k** at a blended ~$0.005–0.01/call — i.e. it can exhaust the entire budget on one method, one fold.

**Mitigation 1 — proxy-Brier minibatching during search.** Do **not** compute the full grand Brier per candidate. Give the optimiser a stratified **subsample of cells** (e.g. ~50 cells ≈ 100–200 calls) as its per-candidate feedback, and reserve the full grand Brier for the handful of Pareto finalists. GEPA’s and MIPROv2’s minibatch settings do exactly this; minibatch size is the dominant cost knob. (You flagged you’ll check 50-vs-300-cell ranking stability first — correct: verify the proxy preserves candidate ordering before trusting it.)

**Mitigation 2 — run the search in `all_except_target` mode (240 calls/full-Brier, 4× cheaper).** The source-index iteration is largely redundant for a fixed target anyway (per the config notes), so optimising in this mode is both cheaper and arguably cleaner, then validate the winning prompt under the full single_bin condition.

**Cost model:** `cost ≈ (#candidates) × (cells per candidate) × n_calls_per_cell × $/call`, plus a one-off `(#finalists) × (calls per full grand Brier) × $/call` for validation.

Costs below assume:

- forecaster on the **cheapest viable model** during search,
- proxy-Brier of ~50 cells/candidate,
- `all_except_target` mode,
- optimiser/reflection LLM on a capable model (far fewer calls than the forecaster).

Replace `$/call` with Madhav’s measured per-call cost before committing.

| Method | Legible output? | Eval cost per candidate | Candidates | Indicative total (1 train split) | Notes |
| --- | --- | --- | --- | --- | --- |
| Minimal-prompt baseline | n/a (you write it) | 1× full Brier (~240–960 calls) | 1 | **~$2–10** | Do first. Sets the floor. |
| Post-hoc recalibration | n/a (a fit, not a prompt) | reuses existing forecasts | 0 new | **~$0** | Highest value/$. Report everything net of this. |
| Feature grid sweep (idea 3) | Yes — you define the axes | 1× full Brier | 2^k configs | k=5 → 32 × ~960 ≈ **31k calls ≈ $150–300** | Each config is a full Brier (few configs, no proxy needed). Fractional factorial if k>5. |
| MIPROv2 (via DSPy) | Yes — instruction + chosen exemplars | proxy ~50 cells (~100–200 calls) | ~100–200 | **~$50–400** | Best for exemplar selection. Cheaper than GEPA. |
| **GEPA (via DSPy)** | **Yes** — evolved prompts + diagnosis trail | proxy ~50 cells (~100–200 calls) | ~150–300 (capped) | **~$75–600** *with proxy*; **~$1k–5k without** | Primary recommendation — **only viable with proxy-Brier minibatching + `all_except_target`.** Cap rollout budget; stop when Pareto front stalls. |
| ProTeGi / textual-gradient (2305.03495) | Yes — NL critiques + edits | proxy ~50 cells | ~hundreds | similar to GEPA | Superseded by GEPA; cite, don’t build. |
| Rainbow Teaming (QD archive) | Yes — diverse legible prompts | proxy ~50 cells | ~1k+ (archive fill) | **~$500–2k** | Heavier; gives a feature-space map, not a single best prompt. Optional. |
| Best-of-N (2412.03556) | **No** — random augmentations | — | thousands | wasteful | Skip. |
| GCG / white-box | **No** — gibberish suffixes | — | huge + GPU | n/a | Ruled out (§4). |

The headline: **with proxy-Brier minibatching (and `all_except_target` mode), GEPA and the feature grid both land at ~$100–600 per train split and the project fits comfortably under $5k. Without these mitigations, GEPA alone can exhaust the budget.** Budget a full grand Brier (~240–960 calls) only for finalists and for final validation on the real 2-per-provider model set.

**On the feature grid (your idea 3) vs. GEPA — they’re complementary, not rivals.**

- The grid is a **designed experiment**: clean main-effects + interaction estimates over *axes you choose*, directly answering “does the analysis stage matter? does the baseline matter?” with interpretable coefficients. This is exactly what reviewers at a policy/workshop venue want. Downside: combinatorial (2^k), and it can only test features you thought of. **But see §5 — if the minimal-prompt gate shows structural features are inert, the grid collapses (nothing to vary), while GEPA may still have non-structural content to optimise.**
- GEPA is **open-ended search**: it can discover edits you didn’t pre-specify (wording, how pass-rate evidence is presented, how percentiles are requested, anchor framing), and is far more sample-efficient than a full grid, but the output is a single evolved prompt, not a clean factorial table.
- **Recommended split:** run the **grid (or a fractional-factorial subset)** to get the interpretable feature-effects table that anchors the paper’s narrative; run **GEPA** to get the best-achievable prompt and to discover features outside your grid. Report both. A full 2^k grid is wasteful if k>5 — use a **fractional factorial / Plackett-Burman design** to get main effects at a fraction of the runs.

---

## 4. Are gradient-based methods infeasible? — Yes, but the previous reasoning was slightly off

**Conclusion: agree, rule them out.** But sharpen *why*, because the earlier Claude conversation located the obstacle in the wrong place.

The earlier note said the gradient “breaks at the token-extraction step” (logits → discrete token → parsed float) and proposed a constrained decile-output workaround to restore differentiability. That diagnosis is **half right and misleading**:

1. **The token-extraction non-differentiability is real but not the binding constraint.** Yes, parsing a float from sampled text is non-differentiable, and yes, a constrained output head (softmax-weighted sum over decile tokens) makes the *output→Brier* step differentiable. But this is the *minor* obstacle.
2. **The binding constraint is the object you’d be optimising.** In FGSM/universal-perturbation, you optimise a **continuous** δ in input (pixel) space — gradients in ℝ^n are meaningful and you can take a step. Your optimisation variable is the **prompt**, which lives in **discrete token space**. Even with a fully differentiable output→loss path, ∂loss/∂(prompt tokens) does not give you a usable update, because there’s no continuous prompt to step. This is precisely why GCG exists — it uses gradients only to *rank candidate token swaps*, then does discrete search. So “make Brier differentiable” doesn’t buy you prompt optimisation; it would only buy you *soft-prompt / weight* tuning.
3. **To actually use gradients you’d need white-box weights** (open model) **and** you’d be doing either soft-prompt tuning or GCG-style discrete search. Soft prompts aren’t human-legible and don’t transfer across the API models you actually deploy (Sonnet/Opus/GPT/Gemini). GCG produces gibberish and transfers poorly. Neither answers your research question (“which *legible* prompt features improve forecasts”).
4. **The deeper mismatch:** your loss is computed **across many tasks aggregated into one Brier/CRPS**, exactly like a universal perturbation averages over examples — that part is fine and not the problem. The problem is purely that the **decision variable is discrete and the deployment models are black-box**.

**So:** gradients are infeasible *and* undesirable here, for the right reasons. The previous conversation’s Option 2 (“don’t backprop, use Brier as an eval metric and search prompts”) is the correct framing — and that is exactly what Family A (GEPA/DSPy) operationalises. The decile-output workaround is a solution to a problem you don’t have.

---

## 5. Mild recommendation: the most feasible direction given the budget

**A disciplined, staged plan stays well within budget** because the expensive part (optimisation) is gated behind cheap experiments that might make it unnecessary.

Recommended ordering, cheapest/highest-signal first:

1. **Reframe existing invariance checks in Brier terms on Lyptus** (already assigned to Madhav). Confirms persona/analysis/quantile invariance *transfers to accuracy*, not just to W₁ consistency. A few full-Brier evals (one per condition). **~$10–50.**
2. **Minimal-prompt baseline.** Establishes the floor. If minimal ≈ full scaffold, that *is* a result and you down-scope the optimisation. **~$3–6.**
3. **Post-hoc recalibration baseline.** The forecaster’s p50 values may be miscalibrated in a *monotone* way (e.g. it says 0.7 when the true solve frequency is 0.55). A recalibration map $g: [0, 1] \to [0, 1]$ corrects this: fit *g* on the train split’s (p50, outcome) pairs and apply the **same** *g* to test-split p50s before scoring. Standard choices:
    1. **Platt** (logistic, 2 params, sigmoidal distortion)
    2. **beta calibration** (3 params, flexible, natural on [0,1] — good default)
    3. **isotonic** (non-parametric, any monotone map, overfits on small N). The Madhav/Jeff “linear fit” is the crudest member of this family.
    
    Crucially, recalibration can only move the **reliability/bias** term, never **resolution/slope** (§0) — so if it alone closes most of the Brier gap, the forecaster’s problem was a calibration offset, not lack of signal, and elaborate prompting that “improves Brier” may just be doing implicit recalibration a 2-param fit does for free. **Report every prompt-optimisation gain net of recalibration**, else you credit the prompt for what a logistic fit already captured. Costs ~0 API calls (one-time fit on data you already have); highest value-per-dollar single action. ~$0.
    
4. **Fractional-factorial feature sweep** over your hand-defined axes (baseline, analysis, reasoning scaffold, Delphi rounds, #source-bin examples). Gives the interpretable main-effects table. Use a reduced design, not full 2^k. Each config is a full Brier; ~$150–300 for k=5.
5. **GEPA via DSPy** *(with proxy-Brier minibatching + `all_except_target` mode — see §3, non-negotiable for budget)*, train on a leave-one-benchmark-out split of Lyptus’s 7 sub-benchmarks (Jeff’s leave-one-out idea — preferable to a blind 80/20 because it tests *cross-domain* transfer, which is the real claim). Cap candidate budget low (~150–250) first; only scale if the Pareto front is still moving. Cheap forecaster during search; validate the final prompt with a full grand Brier on the held-out benchmark and the 2-per-provider model set. **~$75–600 per fold with mitigations; ~$1k–5k per fold without.**
6. **Transferability check:** take the prompt GEPA optimised on a cheap forecaster and evaluate it unchanged on Opus-4.8 / GPT-5.5. The *delta* is a clean, cheap, publishable result about whether prompt optimisation is model-portable or needs re-running per model. One full-Brier eval per model. **~$10–20.**

**Total realistic spend: ~$300–1,500** with proxy-Brier minibatching (one or two folds), comfortably under $5k, with the expensive step (5) gated behind cheap ones (2,3) that may make it unnecessary. Team time is dominated by harness setup (DSPy + metric plumbing + proxy-subsample logic + train/test splits), a one-time ~10–20 hr cost; after that the runs are largely unattended.

**The go/no-go gate (this is the key structural point).** Steps 2–3 are not just baselines — they *gate* whether steps 4–5 should run at all:
- **If minimal-prompt + recalibration leaves no resolution/slope headroom** (no candidate prompt beats it on the discrimination terms of §0), then the structural features are confirmed inert, the **grid collapses (nothing to vary)**, *and* GEPA has nothing meaningful to optimise either — because there’s no signal gap for content edits to close. You **stop**, and the paper is the negative result: “forecaster accuracy here is signal-bound, not prompt-bound.” Do not spend the GEPA budget.
- **If there is a gap** (some richer prompt beats minimal on slope), then GEPA is justified: the grid may still be flat on your *named* structural axes while GEPA finds *non-structural content* edits (wording, evidence presentation, percentile elicitation, anchor framing) that move slope. This is exactly the regime where GEPA earns its cost and a pure feature grid would miss the win.

In short: **you cannot know whether GEPA has anything to optimise until the gate is passed**, which is why cheap-first ordering is logically necessary, not merely frugal.

**Honest risk flags:**
- If steps 2–3 show minimal-prompt ≈ complex-prompt and recalibration captures most gains, then **prompt optimisation has little headroom** and the paper’s contribution becomes “LLM forecasters in this domain are robustly insensitive to prompt structure; accuracy is bounded by calibration, not prompting” — a legitimate and policy-relevant negative result, consistent with your existing invariance findings.
- The single biggest threat to validity is **train/test leakage via benchmark contamination** (raised in the 12 Jun meeting). Leave-one-benchmark-out partially mitigates; document contamination as a limitation regardless.
- **Proxy-Brier noise:** a ~50-cell proxy is noisier than the full grand Brier (240–960 cells depending on config), so the optimiser’s feedback signal is noisier. Mitigate by stratifying the proxy across bins/models and re-drawing the subsample periodically; validate finalists on the full cell set. Verify up front (your planned 50-vs-300 check) that the proxy preserves candidate *ranking*, not just absolute Brier.

---

## Appendix: reference list

- GEPA — arXiv:2507.19457 (ICLR 2026 Oral); lib `github.com/gepa-ai/gepa`
- ProTeGi / “APO with Gradient Descent and Beam Search” — arXiv:2305.03495
- DSPy / MIPROv2 — `dspy.ai`
- Rainbow Teaming — arXiv:2402.16822
- Adaptive attacks survey (“The Attacker Moves Second”) — arXiv:2510.09023
- Best-of-N Jailbreaking — arXiv:2412.03556 (cited as *not* applicable)
- Metaculus, “Automated Prompt Engineering for Forecasting” (notebook 38421) — domain-adjacent prior art worth reading before step 5
- Murphy (1973), “A new vector partition of the probability score” — reliability/resolution/uncertainty
- Yates (1982; refined 1994) — covariance decomposition; bias/slope/scatter form
- Stewart & Lusk (1994) — lens-model decomposition (noted, less applicable)
- O’Hagan et al., *Uncertain Judgements: Eliciting Experts’ Probabilities* — textbook treatment of the above scoring-rule decompositions

**<END CLAUDE OUTPUT>**