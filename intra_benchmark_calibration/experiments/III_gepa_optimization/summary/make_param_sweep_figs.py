#!/usr/bin/env python3
"""fig12-16: one figure per search parameter considered for the next runs.

Wording is calibrated against five zero-context reviewer passes (agents shown
only the image): every term is defined on-figure in the labels themselves —
question, starting prompt, parent prompt, win, tie handling, denominators —
and each figure states in the corner chip whether it is recomputed from the
logs of completed runs or comes from completed runs directly.

Data: sweep_report_data.json (gepa repo scripts/sweep_report_data.py — a
replay of recorded runs, no simulation).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1a1a1a"; MUTED = "#736f6c"; SOFT = "#94918e"; HAIR = "#ebebeb"
GREEN = "#034f46"; VERM = "#d64e2e"
plt.rcParams.update({"font.family": "Geist", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})

D = json.load(open(Path(__file__).parent / "sweep_report_data.json"))
R1, R2 = "pilot_baseline", "pilot_baseline_clean"

QUESTION = "a question = will one AI model solve one benchmark task? (scored by Brier; lower = better)"


def header(fig, n, code, title, frame, chip, gloss=""):
    # the exact code parameter name, highlighted in a monospace chip
    t1 = fig.text(0.055, 0.968, f"PARAMETER {n} OF 5", fontsize=8, color=MUTED,
                  ha="left", va="top")
    renderer = fig.canvas.get_renderer()
    x = t1.get_window_extent(renderer).x1 / (fig.get_figwidth() * fig.dpi) + 0.012
    t2 = fig.text(x, 0.974, code, fontsize=9, color=INK, ha="left", va="top",
                  fontweight="bold",
                  fontfamily=["JetBrains Mono", "Menlo", "monospace"],
                  bbox=dict(boxstyle="round,pad=0.32", fc="#f2f0ea", ec=MUTED, lw=0.8))
    if gloss:
        x2 = t2.get_window_extent(renderer).x1 / (fig.get_figwidth() * fig.dpi) + 0.014
        fig.text(x2, 0.968, gloss, fontsize=8, color=MUTED, ha="left", va="top")
    fig.text(0.055, 0.922, title, fontsize=12.5, color=INK, ha="left", va="top",
             fontweight="bold")
    fig.text(0.055, 0.862, frame, fontsize=8.5, color=MUTED, ha="left", va="top")
    fig.text(0.965, 0.974, chip, fontsize=8, color=INK, ha="right",
             va="top", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=INK, lw=1.1))


def footer(fig, text):
    fig.text(0.055, 0.012, text, fontsize=7, color=SOFT, ha="left", va="bottom")


def clean_axes(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(HAIR); ax.spines["bottom"].set_color(MUTED)
    ax.grid(axis="y", color=HAIR, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.5)


# ---------------------------------------------------------------- fig12
def fig12():
    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=220)
    fig.subplots_adjust(top=0.775, bottom=0.21, left=0.115, right=0.965)
    header(fig, 1, "n_train_tasks_per_iter",
           "Best confirmed improvement, by size of the entry test",
           "a proposed prompt joins the pool only if it beats its parent prompt "
           "(the prompt it was edited from) on this test",
           "FROM COMPLETED RUNS",
           gloss="value 1 → a 20-question test · value 5 → 100")
    G = D["gate_size"]
    for x, r in enumerate(G["runs"]):
        b = r["best"]
        if b is not None:
            ax.bar(x, b["mean"], width=0.5, color=GREEN,
                   yerr=[[b["mean"] - b["min"]], [b["max"] - b["mean"]]],
                   capsize=4, error_kw={"lw": 1.1, "ecolor": INK})
            ax.text(x, b["mean"] - 0.0018, f"+{b['mean']:.3f}",
                    ha="center", va="top", fontsize=11, color="white",
                    fontweight="bold")
            ax.text(x, b["max"] + 0.0012, f"{b['n_repeats']} repeat evaluations",
                    ha="center", fontsize=7.5, color=SOFT)
        else:
            ax.bar(x, 0.0004, width=0.5, color=HAIR, edgecolor=MUTED,
                   lw=1.0, ls="--")
            ax.text(x, 0.0035, "none of its 40 proposals beat the starting\nprompt even "
                    "on the search questions —\nnothing to evaluate held-out",
                    ha="center", fontsize=8.5, color=MUTED)
    ax.set_xticks([0, 1, 2], ["20-question test\nrun 1", "20-question test\nrun 2",
                              "100-question test\nrun 3"], fontsize=9.5, color=INK)
    ax.set_ylim(0, 0.034)
    ax.set_ylabel("improvement in held-out Brier score\nover the starting prompt",
                  fontsize=9)
    clean_axes(ax); ax.tick_params(axis="x", length=0)
    footer(fig, "improvement = starting prompt's Brier − this prompt's Brier, on the same 230 "
                "held-out questions never used during the search · bar = mean, whisker = "
                "min–max\nover repeat evaluations · confirmed = better in every repeat · "
                "starting prompt's held-out Brier ≈ 0.131 · every run proposes 40 prompts\n"
                + QUESTION + " · an entry test = (value × 5 difficulty bins) tasks × 4 AI models")
    fig.savefig("fig12_sweep_gate_size.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig13/14
FRAME_1314 = ("today's entry rule: a proposed prompt joins the pool if its total score on 20 "
              "questions beats its parent prompt's\n(parent = the prompt it was edited from) · "
              "a win = a better score than the parent on one question")


def _accept_lines(ax, table, pooled, reject_label):
    ks = list(range(0, 21))
    g = [100 * table[str(k)]["good_kept"] / pooled["n_good"] for k in ks]
    b = [100 * table[str(k)]["bad_kept"] / pooled["n_bad"] for k in ks]
    a = [100 * table[str(k)]["rejects_admitted"] / pooled["n_reject"] for k in ks]
    ax.plot(ks, g, color=GREEN, lw=2, marker="o", ms=3.5,
            label=f"accepted then — later confirmed better than its parent  (n={pooled['n_good']})")
    ax.plot(ks, b, color=VERM, lw=2, marker="o", ms=3.5,
            label=f"accepted then — later found worse than its parent  (n={pooled['n_bad']})")
    ax.plot(ks, a, color=INK, lw=1.4, ls=":", marker="o", ms=3, label=reject_label)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5,
              handlelength=2.2, labelcolor=INK, borderaxespad=0.2)
    ax.set_xlim(-0.3, 20.3); ax.set_ylim(-4, 108)
    ax.set_xticks(range(0, 21, 4))
    ax.set_xlabel("required question wins,  k  (of the 20 entry-test questions)", fontsize=9.5)
    ax.set_ylabel("share of each group this rule\nwould accept (%)", fontsize=9.5)


def fig13():
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=220)
    fig.subplots_adjust(top=0.75, bottom=0.185, left=0.105, right=0.965)
    header(fig, 2, "acceptance_criterion: min_task_wins",
           "Entry by “win at least k of the 20 questions” alone, replayed",
           FRAME_1314, "RECOMPUTED FROM RUN LOGS", gloss="swept: k")
    A = D["acceptance"]
    _accept_lines(ax, A["wins_only"], A["pooled"],
                  f"rejected then  (n={A['pooled']['n_reject']})")
    clean_axes(ax)
    footer(fig, "replays the 77 recorded accept/reject decisions of the two completed searches "
                "under the alternative rule · better/worse = the prompt's later score on all 84 "
                "search questions\n(different tasks from the entry tests — no overlap) · "
                "rejected prompts never got that evaluation, so the dotted line carries no "
                "better/worse label\n" + QUESTION)
    fig.savefig("fig13_sweep_wins_only_gate.png")
    plt.close(fig)


def fig14():
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=220)
    fig.subplots_adjust(top=0.75, bottom=0.185, left=0.105, right=0.965)
    header(fig, 3, "aggregate_sum_and_min_task_wins",
           "Keeping today's rule AND requiring k question wins, replayed",
           FRAME_1314, "RECOMPUTED FROM RUN LOGS · NEW RUN PLANNED",
           gloss="swept: k · new run: k = 8")
    A = D["acceptance"]
    _accept_lines(ax, A["joint"], A["pooled"],
                  f"rejected then (n={A['pooled']['n_reject']}) — none re-accepted; "
                  "the AND only tightens")
    ax.axvline(8, color=GREEN, lw=1.2, ls="--", ymax=0.80)
    ax.text(7.7, 32, "k = 8, chosen for the new run:\nkeeps 15 of 18 better,\n"
            "22 of 32 worse", fontsize=8.5, color=GREEN, ha="right",
            fontweight="bold")
    clean_axes(ax)
    footer(fig, "replays the 77 recorded accept/reject decisions of the two completed searches "
                "· better/worse = the prompt's later score on all 84 search questions, not the "
                "20 entry-test\nquestions ('task' in the parameter name = question) · k = 8 is "
                "the largest k keeping ≥ 80% of the confirmed-better group · a changed early "
                "acceptance\nchanges every later proposal, so the full-run effect needs the new "
                "run · " + QUESTION)
    fig.savefig("fig14_sweep_joint_gate.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig15
def fig15():
    fig, ax = plt.subplots(figsize=(8.6, 4.9), dpi=220)
    fig.subplots_adjust(top=0.775, bottom=0.225, left=0.105, right=0.965)
    header(fig, 4, "n_cells_won_needed_for_pareto_frontier",
           "Questions won vs confirmed improvement, per measured prompt",
           "only prompts winning ≥ k questions may be edited further · "
           "win = best, or tied for best, score on one question among the pool",
           "RECOMPUTED FROM RUN LOGS",
           gloss="current: 1 · 'cell' = question")
    pts = {"o": [], "^": []}
    for marker, run in (("o", R1), ("^", R2)):
        F = D["frontier"][run]
        for i, d in F["sealed_candidates"].items():
            pts[marker].append((d["cells_won"], d["sealed"]))
        rug = [w for i, w in F["cells_won"].items()
               if i not in F["sealed_candidates"] and int(i) != 0]
        ax.plot(rug, [-0.0135] * len(rug), marker="|", ms=8, mew=1.2, lw=0,
                color=SOFT, alpha=0.8)
    for marker, data in pts.items():
        xs, ys = zip(*data)
        ax.plot(xs, ys, marker, ms=9, color=INK, lw=0,
                label={"o": "run 1", "^": "run 2"}[marker])
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.text(29.6, 0.0008, "no better than\nstarting prompt", fontsize=7.5,
            color=SOFT, ha="right", va="bottom")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK,
              borderaxespad=0.2)
    ax.set_xlim(0, 30); ax.set_ylim(-0.016, 0.032)
    ax.set_xlabel("questions won at the end of the run  (of 84) — a threshold at k removes "
                  "prompts left of k from parenthood", fontsize=9.5)
    ax.set_ylabel("confirmed held-out improvement\nover the starting prompt (Brier)",
                  fontsize=9)
    clean_axes(ax)
    footer(fig, "| = prompt never measured held-out (improvement unknown) · the 7 measured "
                "prompts were the promising-looking ones, not a random sample\n"
                "ties count for every tied prompt · raising the bar to k = 2 leaves run 1 "
                "unchanged through iteration 34 of 40, run 2 through 28 of 40; beyond that the "
                "logs cannot say\n" + QUESTION)
    fig.savefig("fig15_sweep_frontier_bar.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig16
def fig16():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 5.0), dpi=220, sharey=True)
    fig.subplots_adjust(top=0.745, bottom=0.175, left=0.09, right=0.965, wspace=0.18)
    header(fig, 5, "pareto_instance",
           "Who gets edited further, under the two scoring units",
           "a prompt's share = fraction of scoring units where it has the best (or "
           "tied-best) score; prompts are picked for further editing in proportion to it",
           "RECOMPUTED FROM RUN LOGS · NEW RUN PLANNED",
           gloss="current value: cell · new run: model_bin")
    fig.text(0.055, 0.795, "prompt color:", fontsize=8, color=INK)
    fig.text(0.145, 0.795, "confirmed better than starting prompt", fontsize=8,
             color=GREEN, fontweight="bold")
    fig.text(0.39, 0.795, "≈ no difference", fontsize=8, color=MUTED,
             fontweight="bold")
    fig.text(0.50, 0.795, "confirmed worse", fontsize=8, color=VERM,
             fontweight="bold")
    fig.text(0.625, 0.795, "thin gray = never measured held-out", fontsize=8,
             color=SOFT)
    for ax, run, panel in ((axes[0], R1, "run 1"), (axes[1], R2, "run 2")):
        P = D["pareto_instance"][run]
        cw, gw = P["cell_wins"], P["group_wins"]
        tc, tg = sum(cw.values()), sum(gw.values())
        sealed = {int(i): v for i, v in P["sealed"].items()}
        n_cand = max(int(i) for i in list(cw) + list(gw)) + 1
        for i in range(1, n_cand):
            if i in sealed:
                continue
            ax.plot([0, 1], [100 * cw.get(str(i), 0) / tc, 100 * gw.get(str(i), 0) / tg],
                    color=SOFT, lw=0.9, alpha=0.55)
        labels = []
        for i, gain in sorted(sealed.items(), key=lambda kv: -kv[1]):
            x0 = 100 * cw.get(str(i), 0) / tc
            x1 = 100 * gw.get(str(i), 0) / tg
            c = GREEN if gain >= 0.01 else (VERM if gain < 0 else MUTED)
            ax.plot([0, 1], [x0, x1], color=c, lw=2.4, zorder=3)
            ax.plot([0, 1], [x0, x1], "o", ms=6, color=c, zorder=4)
            labels.append((x0, f"{gain:+.3f}", c))
        labels.sort(key=lambda t: t[0])
        ys = []
        for y, _t, _c in labels:
            if ys and y - ys[-1] < 1.7:
                y = ys[-1] + 1.7
            ys.append(y)
        for (y0, t, c), y in zip(labels, ys):
            ax.annotate(t, (0, y0), xytext=(-0.07, y), ha="right", va="center",
                        fontsize=8.5, color=c, fontweight="bold")
        ax.set_xlim(-0.45, 1.25); ax.set_ylim(-1.5, 30)
        ax.set_xticks([0, 1], ["84 single questions\n(current:  cell)",
                               "20 groups of questions,\none per AI model ×\ndifficulty level "
                               "(model_bin)"], fontsize=8.5, color=INK)
        ax.set_title(panel, fontsize=9.5, color=INK, pad=8)
        clean_axes(ax)
        ax.tick_params(axis="x", length=0, labelsize=8.5)
        ax.tick_params(axis="y", labelleft=True)
    axes[0].set_ylabel("share of scoring units where the prompt\nis best — its editing weight (%)",
                       fontsize=9)
    footer(fig, "line = one prompt; label = its confirmed held-out improvement over the starting "
                "prompt (≥3 repeat evaluations; ≈ no difference = repeats straddle zero)\n"
                "run 1 / run 2 = independent searches, same settings, shares at the final state "
                "· a group's score = the mean over its member questions\n"
                "run 1: grouping lifts the +0.027 prompt (15 → 25%) and the −0.007 prompt "
                "(5 → 10%), zeroes +0.014 and +0.005 · run 2: the only confirmed-better prompt "
                "(+0.016) drops to zero;\nthe largest group shares go to never-measured prompts · "
                + QUESTION)
    fig.savefig("fig16_sweep_pareto_instance.png")
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    fig12(); fig13(); fig14(); fig15(); fig16()
    print("written: fig12..fig16 param sweep figures")
