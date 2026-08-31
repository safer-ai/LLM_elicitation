#!/usr/bin/env python3
"""fig12-16: one figure per search parameter in the points-4/5 sweep decision.

Each figure answers one question, states online/offline in the corner chip,
and reads from sweep_report_data.json (produced by gepa repo
scripts/sweep_report_data.py — pure replay of the logged runs).
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


def header(fig, n, code, title, chip, chip_color):
    fig.text(0.055, 0.955, f"PARAMETER {n} OF 5  ·  {code}",
             fontsize=8, color=MUTED, ha="left", va="top")
    fig.text(0.055, 0.915, title, fontsize=13, color=INK, ha="left", va="top",
             fontweight="bold")
    fig.text(0.965, 0.955, chip, fontsize=8.5, color=chip_color, ha="right",
             va="top", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=chip_color, lw=1.2))


def footer(fig, text):
    fig.text(0.055, 0.012, text, fontsize=7, color=SOFT, ha="left", va="bottom")


def clean_axes(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(HAIR); ax.spines["bottom"].set_color(MUTED)
    ax.grid(axis="y", color=HAIR, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.5)


# ---------------------------------------------------------------- fig12
def fig12():
    fig, ax = plt.subplots(figsize=(8.6, 4.0), dpi=220)
    fig.subplots_adjust(top=0.76, bottom=0.20, left=0.10, right=0.965)
    header(fig, 1, "n_train_tasks_per_iter  —  size of the accept/reject test",
           "A 5× bigger accept test did not find a better prompt",
           "LIVE RUNS · COMPLETE → SETTLED", INK)
    runs = D["gate_size"]["runs"]
    xs = range(3)
    for x, r in zip(xs, runs):
        if r["best_gain"] is not None:
            ax.bar(x, r["best_gain"], width=0.52, color=GREEN)
            ax.text(x, r["best_gain"] + 0.0008, f"+{r['best_gain']:.3f}",
                    ha="center", fontsize=11, color=INK, fontweight="bold")
            ax.text(x, r["best_gain"] + 0.0032, r["passes"],
                    ha="center", fontsize=7.5, color=SOFT)
        else:
            ax.bar(x, 0.0004, width=0.52, color=HAIR, edgecolor=MUTED,
                   lw=1.0, ls="--")
            ax.text(x, 0.0022, "no prompt beat the seed,\neven on the search set",
                    ha="center", fontsize=9, color=VERM, fontweight="bold")
    ax.set_xticks(list(xs), ["20-cell test\nsearch run 1", "20-cell test\nsearch run 2",
                             "100-cell test\none run · 5× cost per decision"],
                  fontsize=9.5, color=INK)
    ax.set_ylim(0, 0.032)
    ax.set_ylabel("best verified held-out gain vs seed\n(Brier, higher = better)",
                  fontsize=9)
    clean_axes(ax); ax.tick_params(axis="x", length=0)
    footer(fig, "gain = paired mean Brier improvement vs the seed on the 230 held-out pairs, "
                "≥3 paired passes (run-1 winner: 8)")
    fig.savefig("fig12_sweep_gate_size.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig13/14
def _accept_lines(ax, table, pooled, admit_label):
    ks = list(range(0, 21))
    g = [100 * table[str(k)]["good_kept"] / pooled["n_good"] for k in ks]
    b = [100 * table[str(k)]["bad_kept"] / pooled["n_bad"] for k in ks]
    a = [100 * table[str(k)]["rejects_admitted"] / pooled["n_reject"] for k in ks]
    ax.plot(ks, g, color=GREEN, lw=2.2,
            label=f"known-GOOD pool entries kept  (n={pooled['n_good']})")
    ax.plot(ks, b, color=VERM, lw=2.2,
            label=f"known-BAD pool entries kept  (n={pooled['n_bad']})")
    ax.plot(ks, a, color=INK, lw=1.6, ls=":", label=admit_label)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5,
              handlelength=2.4, labelcolor=INK, borderaxespad=0.2)
    ax.set_xlim(0, 20); ax.set_ylim(-4, 108)
    ax.set_xticks(range(0, 21, 4))
    ax.set_xlabel("required cell wins  k  (of the 20 test cells)", fontsize=9.5)
    ax.set_ylabel("share of each group (%)", fontsize=9.5)


def fig13():
    fig, ax = plt.subplots(figsize=(8.6, 4.3), dpi=220)
    fig.subplots_adjust(top=0.76, bottom=0.145, left=0.09, right=0.965)
    header(fig, 2, "acceptance_criterion: min_task_wins  —  pool entry by cell wins alone",
           "“Win ≥ k cells” alone cannot replace “better on average”",
           "OFFLINE REPLAY → DEAD", VERM)
    A = D["acceptance"]
    _accept_lines(ax, A["wins_only"], A["pooled"],
                  f"formerly-REJECTED proposals it would admit  (n={A['pooled']['n_reject']})")
    clean_axes(ax)
    footer(fig, "replay of all 77 recorded accept/reject decisions from the two completed "
                "20-cell-gate searches · no k reproduces the recorded gate (≥23 of 77 "
                "decisions change at every k)")
    fig.savefig("fig13_sweep_wins_only_gate.png")
    plt.close(fig)


def fig14():
    fig, ax = plt.subplots(figsize=(8.6, 4.3), dpi=220)
    fig.subplots_adjust(top=0.76, bottom=0.145, left=0.09, right=0.965)
    header(fig, 3, "aggregate_sum_and_min_task_wins  —  current rule AND ≥ k cell wins",
           "Tighter pool entry, admits nothing new — only a live run can judge it",
           "OFFLINE REPLAY → LIVE RUN · ARM A", GREEN)
    A = D["acceptance"]
    _accept_lines(ax, A["joint"], A["pooled"],
                  "formerly-rejected admitted — 0 at every k")
    ax.axvline(8, color=GREEN, lw=1.2, ls="--", ymax=0.88)
    ax.text(8, 103, "arm A:  k = 8\nkeeps 15/18 good · cuts 10/32 bad",
            fontsize=8.5, color=GREEN, ha="center", fontweight="bold")
    clean_axes(ax)
    footer(fig, "same replay as parameter 2 · the added condition only tightens: it can "
                "block accepts, never admit rejects · whether tighter breeding helps is "
                "path-dependent → live run")
    fig.savefig("fig14_sweep_joint_gate.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig15
def fig15():
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 4.8), dpi=220, sharex=True)
    fig.subplots_adjust(top=0.74, bottom=0.22, left=0.055, right=0.965, hspace=0.35)
    header(fig, 4, "n_cells_won_needed_for_pareto_frontier  —  bar to be bred (parent eligibility)",
           "The eligibility bar removes verified winners before verified losers",
           "OFFLINE REPLAY → DEAD", VERM)
    fig.text(0.5, 0.795, "a bar at k makes every prompt LEFT of k ineligible for breeding "
             "— the current setting (k = 1) keeps all", fontsize=8.5, color=MUTED,
             ha="center")
    for ax, run, row_label in ((axes[0], R1, "search run 1"), (axes[1], R2, "search run 2")):
        F = D["frontier"][run]
        verified = {int(i): d for i, d in F["sealed_candidates"].items()}
        others = [w for i, w in F["cells_won"].items() if int(i) not in verified and int(i) != 0]
        ax.plot(others, [0] * len(others), marker="|", ms=13, mew=1.3, lw=0,
                color=SOFT, alpha=0.8)
        for idx, d in sorted(verified.items(), key=lambda kv: kv[1]["cells_won"]):
            gain = d["sealed"]
            c = GREEN if gain >= 0.01 else (VERM if gain < 0 else MUTED)
            ax.plot(d["cells_won"], 0, "o", ms=11, color=c, zorder=3)
            ax.annotate(f"{gain:+.3f}", (d["cells_won"], 0), xytext=(0, 11),
                        textcoords="offset points", ha="center", fontsize=9,
                        color=c, fontweight="bold")
        ax.set_ylim(-0.6, 1.1); ax.set_yticks([])
        ax.set_xlim(0, 30)
        ax.text(29.7, 0.75, row_label, fontsize=9, color=INK, ha="right")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8.5)
    axes[1].set_xlabel("cells where the prompt is the current best  (of 84 search cells)",
                       fontsize=9.5)
    footer(fig, "dot = held-out-verified prompt, labeled with its gain vs seed · gray tick = "
                "never held-out-measured\nany k ≥ 4 removes a +0.014 winner while keeping a "
                "−0.007 prompt to k = 13 · k = 2 leaves both runs unchanged until iteration "
                "34 / 28 of 40, then unknown")
    fig.savefig("fig15_sweep_frontier_bar.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig16
def fig16():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.8), dpi=220, sharey=True)
    fig.subplots_adjust(top=0.74, bottom=0.16, left=0.09, right=0.965, wspace=0.14)
    header(fig, 5, "pareto_instance: model_bin  —  what prompts compete on for breeding rights",
           "Group-level competition reshuffles breeding — evidence points both ways",
           "OFFLINE REPLAY → LIVE RUN · ARM B", GREEN)
    for ax, run, panel in ((axes[0], R1, "search run 1"), (axes[1], R2, "search run 2")):
        P = D["pareto_instance"][run]
        cw, gw = P["cell_wins"], P["group_wins"]
        tc, tg = sum(cw.values()), sum(gw.values())
        sealed = {int(i): v for i, v in P["sealed"].items()}
        n_cand = max(int(i) for i in list(cw) + list(gw)) + 1
        for i in range(1, n_cand):
            x0 = 100 * cw.get(str(i), 0) / tc
            x1 = 100 * gw.get(str(i), 0) / tg
            if i in sealed:
                continue
            ax.plot([0, 1], [x0, x1], color=SOFT, lw=0.9, alpha=0.55)
        labels = []
        for i, gain in sorted(sealed.items(), key=lambda kv: -kv[1]):
            x0 = 100 * cw.get(str(i), 0) / tc
            x1 = 100 * gw.get(str(i), 0) / tg
            c = GREEN if gain >= 0.01 else (VERM if gain < 0 else MUTED)
            ax.plot([0, 1], [x0, x1], color=c, lw=2.4, zorder=3)
            ax.plot([0, 1], [x0, x1], "o", ms=6, color=c, zorder=4)
            labels.append((x0, f"{gain:+.3f}", c))
        # de-overlap left-side gain labels
        labels.sort(key=lambda t: t[0])
        ys = []
        for y, _t, _c in labels:
            if ys and y - ys[-1] < 1.7:
                y = ys[-1] + 1.7
            ys.append(y)
        for (y0, t, c), y in zip(labels, ys):
            ax.annotate(t, (0, y0), xytext=(-0.07, y), ha="right", va="center",
                        fontsize=8.5, color=c, fontweight="bold",
                        arrowprops=None)
        ax.set_xlim(-0.45, 1.25); ax.set_ylim(-1.5, 30)
        ax.set_xticks([0, 1], ["84 single\ncells", "20 (model ×\ndifficulty) groups"],
                      fontsize=9, color=INK)
        ax.set_title(panel, fontsize=9.5, color=INK, pad=8)
        clean_axes(ax); ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("share of breeding-lottery tickets (%)", fontsize=9.5)
    axes[0].text(0.03, 0.985, "one line = one prompt · label = its verified held-out gain vs seed",
                 transform=axes[0].transAxes, fontsize=7.5, color=SOFT, va="top")
    footer(fig, "run 1: grouping promotes the +0.027 champion (15 → 25% of tickets) but zeroes "
                "the +0.014 and +0.005 prompts and lifts a −0.007 prompt (6 → 10%)\n"
                "run 2: the only verified winner (+0.016) drops to zero — "
                "unverified prompts take over the lottery")
    fig.savefig("fig16_sweep_pareto_instance.png")
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    fig12(); fig13(); fig14(); fig15(); fig16()
    print("written: fig12..fig16 param sweep figures")
