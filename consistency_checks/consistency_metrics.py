import itertools

import numpy as np
import pandas as pd
from scipy.stats import beta
from scipy.integrate import quad

def w1_distance(a: dict, b: dict) -> float:
    """Wasserstein-1 distance between two quartile distributions.

    Approximates the W1 distance by treating each distribution as piecewise
    linear between the 5 quantile points: 0, p25, p50, p75, 1. The area
    between the two inverse CDFs is computed analytically per segment.

    Args:
        a, b: dicts with keys 'p25', 'p50', 'p75'

    Returns:
        W1 distance (non-negative float)
    """
    xr = [0.0, a["p25"], a["p50"], a["p75"], 1.0]
    xs = [0.0, b["p25"], b["p50"], b["p75"], 1.0]
    h = 0.25
    total = 0.0

    for i in range(4):
        d0 = xr[i] - xs[i]
        d1 = xr[i + 1] - xs[i + 1]

        if d0 == 0 and d1 == 0:
            area = 0.0
        elif d0 * d1 >= 0:
            area = h * (abs(d0) + abs(d1)) / 2.0
        else:
            area = h * (d0 * d0 + d1 * d1) / (2.0 * (abs(d0) + abs(d1)))

        total += area

    return total


def p50_divergence(a: dict, b: dict) -> float:
    """Absolute difference in median (p50) between two distributions.

    Args:
        a, b: dicts with keys 'p25', 'p50', 'p75'

    Returns:
        |a['p50'] - b['p50']|
    """
    return abs(a["p50"] - b["p50"])


def iqr_divergence(a: dict, b: dict) -> float:
    """Absolute difference in interquartile range between two distributions.

    Args:
        a, b: dicts with keys 'p25', 'p50', 'p75'

    Returns:
        |(a['p75'] - a['p25']) - (b['p75'] - b['p25'])|
    """
    return abs((a["p75"] - a["p25"]) - (b["p75"] - b["p25"]))


def compute_pairwise_metrics(
    df1: pd.DataFrame, df2: pd.DataFrame | None = None, include_beta: bool = False
) -> dict:
    """Compute mean consistency metrics across all pairwise row combinations.

    For a single dataframe, pairs are every combination of two rows within df1.
    For two dataframes, pairs are every cross-product combination (one row from
    each dataframe).

    Args:
        df1: DataFrame with columns 'p25', 'p50', 'p75'
        df2: Optional second DataFrame with the same columns. If provided,
             pairs are drawn one from each dataframe.
        include_beta: If True, also compute W1 using Beta distributions fit to
                      each row's quantile points and include 'w1_beta' in the
                      returned dict alongside the standard linear metrics.

    Returns:
        dict with keys:
            'w1'             -- mean W1 distance across all pairs
            'p50_divergence' -- mean absolute p50 difference across all pairs
            'iqr_divergence' -- mean absolute IQR difference across all pairs
            'n_pairs'        -- number of pairs evaluated
            'w1_beta'        -- mean beta-fit W1 distance (only if include_beta=True)
    """
    records1 = df1[["p25", "p50", "p75"]].to_dict(orient="records")

    if df2 is None:
        pairs = list(itertools.combinations(records1, 2))
    else:
        records2 = df2[["p25", "p50", "p75"]].to_dict(orient="records")
        pairs = list(itertools.product(records1, records2))

    if not pairs:
        result = {
            "w1": float("nan"),
            "p50_divergence": float("nan"),
            "iqr_divergence": float("nan"),
            "n_pairs": 0,
        }
        if include_beta:
            result["w1_beta"] = float("nan")
        return result

    w1_vals = []
    p50_vals = []
    iqr_vals = []
    w1_beta_vals: list[float] = []

    for a, b in pairs:
        w1_vals.append(w1_distance(a, b))
        p50_vals.append(p50_divergence(a, b))
        iqr_vals.append(iqr_divergence(a, b))
        if include_beta:
            w1_beta_vals.append(w1_distance_beta(a, b))

    result = {
        "w1": np.mean(w1_vals),
        "p50_divergence": np.mean(p50_vals),
        "iqr_divergence": np.mean(iqr_vals),
        "n_pairs": len(pairs),
    }
    if include_beta:
        result["w1_beta"] = np.mean(w1_beta_vals)
    return result

def fit_beta(p25: float, p50: float, p75: float) -> tuple[float, float]:
    """Fit Beta(a, b) shape parameters to three quantile constraints.

    Minimises the sum of squared errors between the Beta distribution's
    quantile function and the supplied p25 / p50 / p75 values.  Parameters
    are optimised in log-space (via scipy.optimize.least_squares) so they
    remain strictly positive.

    Args:
        p25, p50, p75: Target quantile values in (0, 1).

    Returns:
        (a, b) — positive shape parameters.
    """
    from scipy.optimize import least_squares

    quantiles = np.array([0.25, 0.50, 0.75])
    targets = np.array([p25, p50, p75])

    def residuals(log_params):
        a, b = np.exp(log_params)
        return beta.ppf(quantiles, a, b) - targets

    result = least_squares(residuals, x0=[np.log(2.0), np.log(2.0)])
    return float(np.exp(result.x[0])), float(np.exp(result.x[1]))


def w1_distance_beta(a: dict, b: dict) -> float:
    """W1 distance using Beta distributions fit to quantile points.

    Fits a Beta(a, b) distribution to each set of quantile points then
    computes the exact W1 distance via numerical integration of the absolute
    CDF difference.

    Args:
        a, b: dicts with keys 'p25', 'p50', 'p75'

    Returns:
        W1 distance (non-negative float)
    """
    a1, b1 = fit_beta(a["p25"], a["p50"], a["p75"])
    a2, b2 = fit_beta(b["p25"], b["p50"], b["p75"])

    def integrand(x):
        return abs(beta.cdf(x, a1, b1) - beta.cdf(x, a2, b2))

    val, _ = quad(integrand, 0.0, 1.0, epsabs=1e-9, epsrel=1e-9)
    return val


def plot_pdf_pairs(
    df1: pd.DataFrame,
    df2: pd.DataFrame | None = None,
    ncols: int = 2,
    figsize_per_plot: tuple[float, float] = (5, 4),
    use_beta: bool = False,
    label1: str | None = None,
    label2: str | None = None,
):
    """Plot PDFs for each pairwise combination.

    Each subplot shows the two PDFs for one pair, with the region where
    distribution A lies above B filled in one colour and vice versa.
    The W1 distance for the pair is shown in the subplot title.

    For the piecewise-linear case (use_beta=False), the distribution implied
    by the five quantile points (0, p25, p50, p75, 1) is piecewise-constant
    (a histogram), so PDFs are drawn as step functions.  For use_beta=True,
    smooth Beta PDFs are used.

    Note: the filled area equals ∫|f_A − f_B| dx (total variation × 2), not
    the W1 distance.  W1 is still computed from CDFs and shown in the title.

    Args:
        df1: DataFrame with columns 'p25', 'p50', 'p75'. An optional 'model'
             column is used for labels when present.
        df2: Optional second DataFrame with the same columns.
        ncols: Number of subplot columns in the grid.
        figsize_per_plot: (width, height) in inches for each individual subplot.
        use_beta: If True, fit a Beta distribution to each row's quantile points
                  and plot smooth Beta PDFs.
        label1: Optional short name appended to auto-generated labels for df1 rows.
        label2: Optional short name appended to auto-generated labels for df2 rows.

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt

    COLOR_A = "steelblue"
    COLOR_B = "tomato"

    def _short_model(model_str: str) -> str:
        parts = model_str.split("-")
        while parts and parts[-1].isdigit():
            parts.pop()
        return "-".join(parts)

    def _label(df: pd.DataFrame, idx: int) -> str:
        run_num = idx + 1
        if "model" in df.columns:
            return f"{_short_model(str(df.iloc[idx]['model']))} R{run_num}"
        return f"R{run_num}"

    def _piecewise_pdf_at(row: pd.Series, xs: np.ndarray) -> np.ndarray:
        """Evaluate piecewise-constant PDF at arbitrary x values."""
        bps = np.array([0.0, row["p25"], row["p50"], row["p75"], 1.0])
        ys = np.zeros_like(xs, dtype=float)
        for i in range(4):
            dx = bps[i + 1] - bps[i]
            if dx > 0:
                upper = bps[i + 1] if i < 3 else bps[i + 1] + 1e-12
                mask = (xs >= bps[i]) & (xs < upper)
                ys[mask] = 0.25 / dx
        return ys

    def _piecewise_step_xy(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Step-function (x, y) arrays for clean line plotting."""
        bps = [0.0, row["p25"], row["p50"], row["p75"], 1.0]
        xs, ys = [], []
        for i in range(4):
            dx = bps[i + 1] - bps[i]
            d = 0.25 / dx if dx > 0 else 0.0
            xs += [bps[i], bps[i + 1]]
            ys += [d, d]
        return np.array(xs), np.array(ys)

    rows1 = list(range(len(df1)))
    if df2 is None:
        index_pairs = [(df1, i, df1, j) for i, j in itertools.combinations(rows1, 2)]
    else:
        rows2 = list(range(len(df2)))
        index_pairs = [(df1, i, df2, j) for i in rows1 for j in rows2]

    n = len(index_pairs)
    if n == 0:
        raise ValueError("No pairs to plot.")

    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
        squeeze=False,
    )

    for k, (dfa, ia, dfb, ib) in enumerate(index_pairs):
        ax = axes[k // ncols][k % ncols]
        row_a = dfa.iloc[ia]
        row_b = dfb.iloc[ib]
        label_a = _label(dfa, ia) + (f" ({label1})" if label1 else "")
        label_b_suffix = label2 if df2 is not None else label1
        label_b = _label(dfb, ib) + (f" ({label_b_suffix})" if label_b_suffix else "")

        if use_beta:
            xs = np.linspace(0.0, 1.0, 500)
            aa, ab = fit_beta(row_a["p25"], row_a["p50"], row_a["p75"])
            ba, bb = fit_beta(row_b["p25"], row_b["p50"], row_b["p75"])
            ya = beta.pdf(xs, aa, ab)
            yb = beta.pdf(xs, ba, bb)
            ax.fill_between(xs, ya, yb, where=ya >= yb, color=COLOR_A, alpha=0.35)
            ax.fill_between(xs, ya, yb, where=ya <= yb, color=COLOR_B, alpha=0.35)
            ax.plot(xs, ya, "-", color=COLOR_A, lw=1.8, label=label_a)
            ax.plot(xs, yb, "-", color=COLOR_B, lw=1.8, label=label_b)
            w1, _ = quad(
                lambda x: abs(beta.cdf(x, aa, ab) - beta.cdf(x, ba, bb)),
                0.0, 1.0, epsabs=1e-9, epsrel=1e-9,
            )
        else:
            bps_a = [0.0, row_a["p25"], row_a["p50"], row_a["p75"], 1.0]
            bps_b = [0.0, row_b["p25"], row_b["p50"], row_b["p75"], 1.0]
            xs = np.unique(np.concatenate([
                np.array(sorted(set(bps_a + bps_b))),
                np.linspace(0.0, 1.0, 300),
            ]))
            ya = _piecewise_pdf_at(row_a, xs)
            yb = _piecewise_pdf_at(row_b, xs)
            ax.fill_between(xs, ya, yb, where=ya >= yb, color=COLOR_A, alpha=0.35)
            ax.fill_between(xs, ya, yb, where=ya <= yb, color=COLOR_B, alpha=0.35)
            xa_step, ya_step = _piecewise_step_xy(row_a)
            xb_step, yb_step = _piecewise_step_xy(row_b)
            ax.plot(xa_step, ya_step, "-", color=COLOR_A, lw=1.8, label=label_a)
            ax.plot(xb_step, yb_step, "-", color=COLOR_B, lw=1.8, label=label_b)
            for bp in bps_a[1:-1]:
                ax.axvline(bp, color=COLOR_A, lw=0.7, ls=":", alpha=0.5)
            for bp in bps_b[1:-1]:
                ax.axvline(bp, color=COLOR_B, lw=0.7, ls=":", alpha=0.5)
            w1 = w1_distance(row_a.to_dict(), row_b.to_dict())

        ax.set_title(f"W₁ = {w1:.4f}  ({label_a} vs {label_b})", fontsize=9, fontweight="bold")
        ax.set_xlabel("Estimated probability", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.grid(linestyle=":", alpha=0.5)
        ax.legend(fontsize=7, loc="upper left")

    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    method_label = "Beta-fit PDFs" if use_beta else "Piecewise-constant PDFs"
    fig.suptitle(
        f"{method_label}  ·  Filled area = ∫|f_A − f_B| dx",
        fontsize=11,
    )
    top_margin = 1.0 - 0.4 / (figsize_per_plot[1] * nrows)
    plt.tight_layout(rect=(0, 0, 1, top_margin))
    return fig

def plot_w1_pairs(
    df1: pd.DataFrame,
    df2: pd.DataFrame | None = None,
    ncols: int = 2,
    figsize_per_plot: tuple[float, float] = (5, 4),
    use_beta: bool = False,
    label1: str | None = None,
    label2: str | None = None,
):
    """Plot quantile functions for each pairwise combination.

    Each subplot shows the two inverse CDFs (quantile functions) for one pair,
    with the region where distribution A lies above B filled in one colour and
    the region where B lies above A filled in another.  The W1 distance for
    the pair is shown in the subplot title.

    Pair generation follows the same logic as compute_pairwise_metrics: within
    df1 when df2 is None, or across df1 × df2 otherwise.

    Args:
        df1: DataFrame with columns 'p25', 'p50', 'p75'. An optional 'run_id'
             column is used for labels when present.
        df2: Optional second DataFrame with the same columns.
        ncols: Number of subplot columns in the grid.
        figsize_per_plot: (width, height) in inches for each individual subplot.
        use_beta: If True, fit a Beta distribution to each row's quantile points
                  and plot smooth Beta quantile functions.  W1 is also computed
                  from the fitted Beta distributions.  If False (default), use
                  piecewise-linear interpolation between the five quantile points.
        label1: Optional short name appended to auto-generated labels for df1
                rows (e.g. "normal").
        label2: Optional short name appended to auto-generated labels for df2
                rows. Ignored when df2 is None.

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt

    PROB_LEVELS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    COLOR_A = "steelblue"
    COLOR_B = "tomato"

    def _short_model(model_str: str) -> str:
        """Strip trailing numeric version parts (e.g. dates) from a model name."""
        parts = model_str.split("-")
        while parts and parts[-1].isdigit():
            parts.pop()
        return "-".join(parts)

    def _label(df: pd.DataFrame, idx: int) -> str:
        run_num = idx + 1
        if "model" in df.columns:
            return f"{_short_model(str(df.iloc[idx]['model']))} R{run_num}"
        return f"R{run_num}"

    def _quantiles(row: pd.Series) -> np.ndarray:
        return np.array([0.0, row["p25"], row["p50"], row["p75"], 1.0])

    def _insert_crossings(
        xs: np.ndarray, ya: np.ndarray, yb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Insert exact crossing points so fill_between colours are clean."""
        xs_out, ya_out, yb_out = [xs[0]], [ya[0]], [yb[0]]
        for i in range(len(xs) - 1):
            d0 = ya[i] - yb[i]
            d1 = ya[i + 1] - yb[i + 1]
            if d0 * d1 < 0:
                t = d0 / (d0 - d1)
                x_c = xs[i] + t * (xs[i + 1] - xs[i])
                y_c = ya[i] + t * (ya[i + 1] - ya[i])
                xs_out.append(x_c)
                ya_out.append(y_c)
                yb_out.append(y_c)
            xs_out.append(xs[i + 1])
            ya_out.append(ya[i + 1])
            yb_out.append(yb[i + 1])
        return np.array(xs_out), np.array(ya_out), np.array(yb_out)

    # Build index pairs
    rows1 = list(range(len(df1)))
    if df2 is None:
        index_pairs = [(df1, i, df1, j) for i, j in itertools.combinations(rows1, 2)]
    else:
        rows2 = list(range(len(df2)))
        index_pairs = [(df1, i, df2, j) for i in rows1 for j in rows2]

    n = len(index_pairs)
    if n == 0:
        raise ValueError("No pairs to plot.")

    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
        squeeze=False,
    )

    for k, (dfa, ia, dfb, ib) in enumerate(index_pairs):
        ax = axes[k // ncols][k % ncols]
        row_a = dfa.iloc[ia]
        row_b = dfb.iloc[ib]
        label_a = _label(dfa, ia) + (f" ({label1})" if label1 else "")
        if df2 is None:
            label_b = _label(dfb, ib) + (f" ({label1})" if label1 else "")
        else:
            label_b = _label(dfb, ib) + (f" ({label2})" if label2 else "")

        if use_beta:
            PROB_FINE = np.linspace(0.0, 1.0, 500)
            aa, ab = fit_beta(row_a["p25"], row_a["p50"], row_a["p75"])
            ba, bb = fit_beta(row_b["p25"], row_b["p50"], row_b["p75"])
            ya = beta.ppf(PROB_FINE, aa, ab)
            yb = beta.ppf(PROB_FINE, ba, bb)
            ax.fill_between(
                PROB_FINE, ya, yb, where=ya >= yb,
                color=COLOR_A, alpha=0.35, label=f"{label_a} > {label_b}",
            )
            ax.fill_between(
                PROB_FINE, ya, yb, where=ya <= yb,
                color=COLOR_B, alpha=0.35, label=f"{label_b} > {label_a}",
            )
            ax.plot(PROB_FINE, ya, "-", color=COLOR_A, lw=1.8, label=label_a)
            ax.plot(PROB_FINE, yb, "-", color=COLOR_B, lw=1.8, label=label_b)
            w1, _ = quad(lambda x: abs(beta.cdf(x, aa, ab) - beta.cdf(x, ba, bb)), 0.0, 1.0, epsabs=1e-9, epsrel=1e-9)
        else:
            qa = _quantiles(row_a)
            qb = _quantiles(row_b)
            xs, ya, yb = _insert_crossings(PROB_LEVELS, qa, qb)
            ax.fill_between(
                xs, ya, yb, where=ya >= yb,
                color=COLOR_A, alpha=0.35, label=f"{label_a} > {label_b}",
            )
            ax.fill_between(
                xs, ya, yb, where=ya <= yb,
                color=COLOR_B, alpha=0.35, label=f"{label_b} > {label_a}",
            )
            ax.plot(PROB_LEVELS, qa, "o-", color=COLOR_A, lw=1.8, ms=5, label=label_a)
            ax.plot(PROB_LEVELS, qb, "o-", color=COLOR_B, lw=1.8, ms=5, label=label_b)
            w1 = w1_distance(row_a.to_dict(), row_b.to_dict())
        ax.set_title(f"W₁ = {w1:.4f}  ({label_a} vs {label_b})", fontsize=9, fontweight="bold")
        ax.set_xlabel("Quantile level", fontsize=9)
        ax.set_ylabel("Estimated probability", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(PROB_LEVELS)
        ax.set_xticklabels(["0", "p25", "p50", "p75", "1"], fontsize=8)
        ax.grid(linestyle=":", alpha=0.5)
        ax.legend(fontsize=7, loc="upper left")

    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    method_label = "Beta-fit quantile functions" if use_beta else "Piecewise-linear quantile functions"
    fig.suptitle(
        f"{method_label}  ·  Filled area = W₁ distance",
        fontsize=11,
    )
    # Reserve space at the top for suptitle so it doesn't overlap subplots.
    # 0.4 in / total figure height gives a model-independent fraction.
    top_margin = 1.0 - 0.4 / (figsize_per_plot[1] * nrows)
    plt.tight_layout(rect=(0, 0, 1, top_margin))
    return fig
