import itertools

import numpy as np
import pandas as pd
from scipy.stats import beta


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _with_boundaries(
    pts: list[tuple[float, float]],
) -> tuple[list[float], list[float]]:
    """Sort (percentile, value) pairs and add (0, 0) and (1, 1) boundaries.

    Returns:
        (qs, vs) — percentile levels and corresponding values.
    """
    pts = sorted(pts)
    qs = [p for p, _ in pts]
    vs = [v for _, v in pts]
    if qs[0] != 0.0:
        qs.insert(0, 0.0)
        vs.insert(0, 0.0)
    if qs[-1] != 1.0:
        qs.append(1.0)
        vs.append(1.0)
    return qs, vs


def _interpolate_pair(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> tuple[list[float], np.ndarray, np.ndarray]:
    """Merge two quantile distributions onto a common grid.

    Returns:
        (all_q, ya, yb) — shared quantile levels and interpolated values.
    """
    qa, va = _with_boundaries(a)
    qb, vb = _with_boundaries(b)
    all_q = sorted(set(qa) | set(qb))
    return all_q, np.interp(all_q, qa, va), np.interp(all_q, qb, vb)


def _interp_quantile(pts: list[tuple[float, float]], q: float) -> float:
    """Interpolate a single quantile value from (percentile, value) pairs."""
    qs, vs = _with_boundaries(pts)
    return float(np.interp(q, qs, vs))


# ---------------------------------------------------------------------------
# Piecewise-linear Wasserstein distances
# ---------------------------------------------------------------------------

def w1_distance(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> float:
    """Wasserstein-1 distance between two quantile distributions.

    Treats each distribution as piecewise linear between the given quantile
    points, with boundary points (0.0, 0.0) and (1.0, 1.0) added implicitly.
    Supports an arbitrary number of quantile points; the two inputs need not
    share the same percentile levels (missing points are interpolated).
    The area between the two inverse CDFs is computed analytically per segment.

    Args:
        a, b: sequences of (percentile, value) pairs, e.g.
              [(0.25, 0.3), (0.5, 0.5), (0.75, 0.7)]

    Returns:
        W1 distance (non-negative float)
    """
    all_q, ya, yb = _interpolate_pair(a, b)

    total = 0.0
    for i in range(len(all_q) - 1):
        h = all_q[i + 1] - all_q[i]
        d0 = ya[i] - yb[i]
        d1 = ya[i + 1] - yb[i + 1]

        if d0 == 0 and d1 == 0:
            area = 0.0
        elif d0 * d1 >= 0:
            area = h * (abs(d0) + abs(d1)) / 2.0
        else:
            area = h * (d0 * d0 + d1 * d1) / (2.0 * (abs(d0) + abs(d1)))

        total += area

    return total


def w2_distance(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> float:
    """Wasserstein-2 distance between two quantile distributions.

    Treats each distribution as piecewise linear between the given quantile
    points, with boundary points (0.0, 0.0) and (1.0, 1.0) added implicitly.
    Supports arbitrary quantile grids; missing points are interpolated.
    """
    all_q, ya, yb = _interpolate_pair(a, b)

    total = 0.0
    for i in range(len(all_q) - 1):
        h = all_q[i + 1] - all_q[i]
        d0 = ya[i] - yb[i]
        d1 = ya[i + 1] - yb[i + 1]

        # Exact integral of the square of a linear function over the interval
        total += h * (d0 * d0 + d0 * d1 + d1 * d1) / 3.0

    return float(np.sqrt(total))


# ---------------------------------------------------------------------------
# Beta-fit Wasserstein distances (grid-based trapezoid rule)
# ---------------------------------------------------------------------------

_BETA_GRID_N = 2048
_BETA_GRID = np.linspace(0.0, 1.0, _BETA_GRID_N)


def fit_beta(pts: list[tuple[float, float]]) -> tuple[float, float]:
    """Fit Beta(a, b) shape parameters to quantile constraints.

    Minimises the sum of squared errors between the Beta distribution's
    quantile function and the supplied (percentile, value) pairs. Parameters
    are optimised in log-space (via scipy.optimize.least_squares) so they
    remain strictly positive.

    Args:
        pts: sequence of (percentile, value) pairs, e.g.
             [(0.25, 0.2), (0.5, 0.4), (0.75, 0.7)]

    Returns:
        (a, b) — positive shape parameters.
    """
    from scipy.optimize import least_squares

    pts_sorted = sorted(pts)
    quantiles = np.array([p for p, _ in pts_sorted])
    targets = np.array([v for _, v in pts_sorted])

    def residuals(log_params):
        a, b = np.exp(log_params)
        return beta.ppf(quantiles, a, b) - targets

    result = least_squares(residuals, x0=[np.log(2.0), np.log(2.0)])
    return float(np.exp(result.x[0])), float(np.exp(result.x[1]))


def _beta_grids(a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """Precompute CDF and inverse-CDF grids for a Beta(a, b) distribution."""
    cdf_vals = beta.cdf(_BETA_GRID, a, b)
    ppf_vals = beta.ppf(_BETA_GRID, a, b)
    return cdf_vals, ppf_vals


def _w1_from_cdf_grids(cdf1: np.ndarray, cdf2: np.ndarray) -> float:
    """W1 distance from precomputed CDF grids via trapezoid rule."""
    return float(np.trapz(np.abs(cdf1 - cdf2), _BETA_GRID))


def _w2_from_ppf_grids(ppf1: np.ndarray, ppf2: np.ndarray) -> float:
    """W2 distance from precomputed inverse-CDF grids via trapezoid rule."""
    return float(np.sqrt(np.trapz((ppf1 - ppf2) ** 2, _BETA_GRID)))


def w1_distance_beta(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> float:
    """W1 distance using Beta distributions fit to quantile points.

    Fits a Beta(a, b) distribution to each set of quantile points then
    computes the W1 distance via trapezoid integration of the CDF grids.

    Args:
        a, b: sequences of (percentile, value) pairs

    Returns:
        W1 distance (non-negative float)
    """
    a1, b1 = fit_beta(a)
    a2, b2 = fit_beta(b)
    cdf1, _ = _beta_grids(a1, b1)
    cdf2, _ = _beta_grids(a2, b2)
    return _w1_from_cdf_grids(cdf1, cdf2)


def w2_distance_beta(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> float:
    """W2 distance using Beta distributions fit to quantile points.

    Fits a Beta(alpha, beta) distribution to each set of quantile points, then
    computes the W2 distance via trapezoid integration of the inverse-CDF grids.

    Args:
        a, b: sequences of (percentile, value) pairs

    Returns:
        W2 distance (non-negative float)
    """
    a1, b1 = fit_beta(a)
    a2, b2 = fit_beta(b)
    _, ppf1 = _beta_grids(a1, b1)
    _, ppf2 = _beta_grids(a2, b2)
    return _w2_from_ppf_grids(ppf1, ppf2)


# ---------------------------------------------------------------------------
# Simple divergence metrics
# ---------------------------------------------------------------------------

def p50_divergence(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Absolute difference in median (p50) between two distributions."""
    return abs(_interp_quantile(a, 0.5) - _interp_quantile(b, 0.5))


def iqr_divergence(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Absolute difference in interquartile range between two distributions."""
    iqr_a = _interp_quantile(a, 0.75) - _interp_quantile(a, 0.25)
    iqr_b = _interp_quantile(b, 0.75) - _interp_quantile(b, 0.25)
    return abs(iqr_a - iqr_b)


# ---------------------------------------------------------------------------
# Pairwise computation
# ---------------------------------------------------------------------------

_METRIC_KEYS = ["w1", "w2", "p50_divergence", "iqr_divergence"]
_BETA_METRIC_KEYS = ["w1_beta", "w2_beta"]


def _compute_row_metrics(
    a: list[tuple[float, float]],
    b: list[tuple[float, float]],
    grids_a: tuple[np.ndarray, np.ndarray] | None = None,
    grids_b: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    """Compute all metrics for a single pair of distributions.

    If grids_a and grids_b are provided, beta metrics are included.
    """
    entry = {
        "w1": w1_distance(a, b),
        "w2": w2_distance(a, b),
        "p50_divergence": p50_divergence(a, b),
        "iqr_divergence": iqr_divergence(a, b),
    }
    if grids_a is not None and grids_b is not None:
        cdf_a, ppf_a = grids_a
        cdf_b, ppf_b = grids_b
        entry["w1_beta"] = _w1_from_cdf_grids(cdf_a, cdf_b)
        entry["w2_beta"] = _w2_from_ppf_grids(ppf_a, ppf_b)
    return entry


def row_to_pairs(estimates: dict) -> list[tuple[float, float]]:
    """Convert an estimates dict {percentile_int: value} to sorted (quantile_level, value) pairs.

    Args:
        estimates: dict with int percentile keys and float values, e.g. {20: 0.4, 60: 0.7}

    Returns:
        sorted list of (percentile / 100, value) pairs, e.g. [(0.2, 0.4), (0.6, 0.7)]
    """
    return sorted((int(p) / 100.0, v) for p, v in estimates.items() if v is not None)


def _prepare_records(
    df1: pd.DataFrame,
    df2: pd.DataFrame | None,
    include_beta: bool,
) -> tuple[
    list[list[tuple[float, float]]],
    list[list[tuple[float, float]]] | None,
    list[tuple[np.ndarray, np.ndarray]] | None,
    list[tuple[np.ndarray, np.ndarray]] | None,
]:
    """Convert dataframes to record lists and optionally precompute beta grids."""
    records1 = df1["estimates"].apply(row_to_pairs).tolist()
    records2 = df2["estimates"].apply(row_to_pairs).tolist() if df2 is not None else None

    grids1 = grids2 = None
    if include_beta:
        grids1 = [_beta_grids(*fit_beta(r)) for r in records1]
        if records2 is not None:
            grids2 = [_beta_grids(*fit_beta(r)) for r in records2]

    return records1, records2, grids1, grids2


def _iter_pairs(
    n1: int, n2: int | None
) -> list[tuple[int, int]]:
    """Generate index pairs: combinations within one group, or cross-product of two."""
    if n2 is None:
        return list(itertools.combinations(range(n1), 2))
    return list(itertools.product(range(n1), range(n2)))


def compute_pairwise_results(
    df1: pd.DataFrame, df2: pd.DataFrame | None = None, include_beta: bool = False
) -> pd.DataFrame:
    """Compute consistency metrics for every pairwise row combination.

    For a single dataframe, pairs are every combination of two rows within df1.
    For two dataframes, pairs are every cross-product combination (one row from
    each dataframe).

    Args:
        df1: DataFrame with an 'estimates' column.
        df2: Optional second DataFrame. If provided, pairs are drawn one from each.
        include_beta: If True, also compute W1 and W2 using Beta-fit grids.

    Returns:
        DataFrame with one row per pair and columns:
            'idx_a', 'idx_b', 'w1', 'w2', 'p50_divergence', 'iqr_divergence'
            and optionally 'w1_beta', 'w2_beta'.
    """
    records1, records2, grids1, grids2 = _prepare_records(df1, df2, include_beta)
    index1 = list(df1.index)
    index2 = list(df2.index) if df2 is not None else None
    two_group = df2 is not None

    pairs = _iter_pairs(len(records1), len(records2) if records2 is not None else None)

    rows = []
    for ri, rj in pairs:
        a = records1[ri]
        b = records2[rj] if two_group else records1[rj]
        ga = grids1[ri] if grids1 is not None else None
        gb = (grids2[rj] if two_group else grids1[rj]) if grids1 is not None else None

        row = _compute_row_metrics(a, b, ga, gb)
        row["idx_a"] = index1[ri]
        row["idx_b"] = index2[rj] if two_group else index1[rj]
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

_ZERO_METRICS = {k: 0.0 for k in _METRIC_KEYS}
_ZERO_METRICS_BETA = {**_ZERO_METRICS, **{k: 0.0 for k in _BETA_METRIC_KEYS}}


def _precompute_pairwise_cache(
    records1: list[list[tuple[float, float]]],
    records2: list[list[tuple[float, float]]] | None,
    include_beta: bool,
) -> dict[tuple[int, int], dict[str, float]]:
    """Precompute all unique pairwise distances between row distributions.

    For single-group (records2 is None): computes combinations(N, 2).
    For two-group: computes product(N1, N2).
    """
    grids1 = grids2 = None
    if include_beta:
        grids1 = [_beta_grids(*fit_beta(r)) for r in records1]
        if records2 is not None:
            grids2 = [_beta_grids(*fit_beta(r)) for r in records2]

    two_group = records2 is not None
    pairs = _iter_pairs(len(records1), len(records2) if records2 is not None else None)

    cache: dict[tuple[int, int], dict[str, float]] = {}
    for i, j in pairs:
        a = records1[i]
        b = records2[j] if two_group else records1[j]
        ga = grids1[i] if grids1 is not None else None
        gb = (grids2[j] if two_group else grids1[j]) if grids1 is not None else None
        cache[(i, j)] = _compute_row_metrics(a, b, ga, gb)

    return cache


def _percentile_ci(
    values: list[float], confidence_level: float
) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval."""
    if not values:
        return float("nan"), float("nan")

    alpha = 1.0 - confidence_level
    lower = float(np.quantile(values, alpha / 2.0))
    upper = float(np.quantile(values, 1.0 - alpha / 2.0))
    return lower, upper


def _bootstrap_mean_from_cache(
    cache: dict[tuple[int, int], dict[str, float]],
    sampled_indices: np.ndarray,
    two_group: bool,
    metric_keys: list[str],
) -> dict[str, float]:
    """Compute mean pairwise metrics for one bootstrap sample using cached distances."""
    zeros = {k: 0.0 for k in metric_keys}

    if two_group:
        pairs = list(itertools.product(sampled_indices, sampled_indices))
    else:
        pairs = list(itertools.combinations(sampled_indices, 2))

    if not pairs:
        return {k: float("nan") for k in metric_keys}

    totals = dict(zeros)
    for i, j in pairs:
        if not two_group:
            key = (min(i, j), max(i, j)) if i != j else None
        else:
            key = (i, j)

        entry = zeros if key is None else cache[key]
        for k in metric_keys:
            totals[k] += entry[k]

    n = len(pairs)
    return {k: totals[k] / n for k in metric_keys}


def _compute_pairwise_metric_ci(
    df1: pd.DataFrame,
    df2: pd.DataFrame | None = None,
    include_beta: bool = False,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | np.random.Generator | None = None,
) -> dict:
    """Compute bootstrap CIs for the mean pairwise metrics.

    Distances are precomputed once for all unique pairs, then each bootstrap
    iteration only performs index resampling and cache lookups.
    """
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    result = {
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap,
    }

    metric_keys = _METRIC_KEYS + (_BETA_METRIC_KEYS if include_beta else [])
    ci_keys = ["w1", "w2"] + (["w1_beta", "w2_beta"] if include_beta else [])

    if df1.empty or (df2 is not None and df2.empty):
        for k in ci_keys:
            result[f"{k}_ci_lower"] = float("nan")
            result[f"{k}_ci_upper"] = float("nan")
        return result

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    records1 = df1["estimates"].apply(row_to_pairs).tolist()
    two_group = df2 is not None
    if two_group:
        if len(df1) != len(df2):
            raise ValueError(
                "Bootstrap resampling with two dataframes requires the same number "
                "of rows so run pairs can be resampled jointly."
            )
        records2 = df2["estimates"].apply(row_to_pairs).tolist()
    else:
        records2 = None

    cache = _precompute_pairwise_cache(records1, records2, include_beta)

    n1 = len(df1)
    samples: dict[str, list[float]] = {k: [] for k in ci_keys}
    for _ in range(n_bootstrap):
        sampled_indices = rng.integers(0, n1, size=n1)
        means = _bootstrap_mean_from_cache(
            cache, sampled_indices, two_group, metric_keys
        )
        for k in ci_keys:
            samples[k].append(means[k])

    for k in ci_keys:
        result[f"{k}_ci_lower"], result[f"{k}_ci_upper"] = _percentile_ci(
            samples[k], confidence_level
        )
    return result


# ---------------------------------------------------------------------------
# Aggregated pairwise metrics
# ---------------------------------------------------------------------------

def compute_pairwise_metrics(
    df1: pd.DataFrame,
    df2: pd.DataFrame | None = None,
    include_beta: bool = False,
    compute_ci: bool = False,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | np.random.Generator | None = None,
) -> dict:
    """Compute mean consistency metrics across all pairwise row combinations.

    Args:
        df1: DataFrame with an 'estimates' column.
        df2: Optional second DataFrame. If provided, pairs are drawn one from each.
        include_beta: If True, also include 'w1_beta' and 'w2_beta' means.
        compute_ci: If True, compute bootstrap confidence intervals.
        n_bootstrap: Number of bootstrap resamples when compute_ci=True.
        confidence_level: Central confidence mass for the percentile interval.
        random_state: Optional seed or numpy Generator for bootstrap draws.

    Returns:
        dict with metric means, 'n_pairs', and optionally CI bounds.
    """
    results_df = compute_pairwise_results(df1, df2, include_beta=include_beta)

    metric_cols = _METRIC_KEYS + (_BETA_METRIC_KEYS if include_beta else [])

    if results_df.empty:
        result = {k: float("nan") for k in metric_cols}
        result["n_pairs"] = 0
        if compute_ci:
            ci_keys = ["w1", "w2"] + (["w1_beta", "w2_beta"] if include_beta else [])
            for k in ci_keys:
                result[f"{k}_ci_lower"] = float("nan")
                result[f"{k}_ci_upper"] = float("nan")
            result["confidence_level"] = confidence_level
            result["n_bootstrap"] = n_bootstrap
        return result

    result = results_df[metric_cols].mean().to_dict()
    result["n_pairs"] = len(results_df)
    if compute_ci:
        result.update(
            _compute_pairwise_metric_ci(
                df1,
                df2,
                include_beta=include_beta,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                random_state=random_state,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_COLOR_A = "steelblue"
_COLOR_B = "tomato"


def _make_pair_labels(
    a_len: int,
    b_len: int | None,
    labels_a: list[str] | None,
    labels_b: list[str] | None,
):
    """Build label-lookup callables for group A and group B."""
    def get_label_a(i):
        return labels_a[i] if labels_a is not None else f"A{i + 1}"

    def get_label_b(i):
        if b_len is None:
            return labels_a[i] if labels_a is not None else f"A{i + 1}"
        return labels_b[i] if labels_b is not None else f"B{i + 1}"

    return get_label_a, get_label_b


def _setup_pair_grid(
    a: list,
    b: list | None,
    ncols: int,
    figsize_per_plot: tuple[float, float],
    suptitle: str,
):
    """Create subplot grid and compute index pairs for pair-plotting functions.

    Returns:
        (fig, axes, index_pairs, get_dist_b, get_label_a, get_label_b, nrows, ncols)
    """
    import matplotlib.pyplot as plt

    idx_a = list(range(len(a)))
    if b is None:
        index_pairs = list(itertools.combinations(idx_a, 2))
        def get_dist_b(j):
            return a[j]
    else:
        idx_b = list(range(len(b)))
        index_pairs = [(i, j) for i in idx_a for j in idx_b]
        def get_dist_b(j):
            return b[j]

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

    return fig, axes, index_pairs, get_dist_b, nrows, ncols


def _finalize_pair_grid(fig, axes, n, nrows, ncols, figsize_per_plot, suptitle):
    """Hide unused subplots, add suptitle, and tighten layout."""
    import matplotlib.pyplot as plt

    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    fig.suptitle(suptitle, fontsize=11)
    top_margin = 1.0 - 0.4 / (figsize_per_plot[1] * nrows)
    plt.tight_layout(rect=(0, 0, 1, top_margin))


def plot_pdf_pairs(
    a: list[list[tuple[float, float]]],
    b: list[list[tuple[float, float]]] | None = None,
    ncols: int = 2,
    figsize_per_plot: tuple[float, float] = (5, 4),
    use_beta: bool = False,
    labels_a: list[str] | None = None,
    labels_b: list[str] | None = None,
):
    """Plot PDFs for each pairwise combination.

    Each subplot shows the two PDFs for one pair, with the region where
    distribution A lies above B filled in one colour and vice versa.
    The W1 distance for the pair is shown in the subplot title.

    Args:
        a: list of distributions, each a list of (percentile, value) pairs.
        b: optional second list of distributions.
        ncols: Number of subplot columns in the grid.
        figsize_per_plot: (width, height) in inches for each individual subplot.
        use_beta: If True, fit Beta distributions and plot smooth PDFs.
        labels_a: optional per-distribution labels for group a.
        labels_b: optional per-distribution labels for group b.

    Returns:
        matplotlib Figure
    """
    method_label = "Beta-fit PDFs" if use_beta else "Piecewise-constant PDFs"
    suptitle = f"{method_label}  \u00b7  Filled area = \u222b|f_A \u2212 f_B| dx"

    fig, axes, index_pairs, get_dist_b, nrows, ncols = _setup_pair_grid(
        a, b, ncols, figsize_per_plot, suptitle
    )
    get_label_a, get_label_b = _make_pair_labels(
        len(a), len(b) if b is not None else None, labels_a, labels_b
    )

    def _piecewise_pdf_at(pts, xs):
        qs, vs = _with_boundaries(pts)
        ys = np.zeros_like(xs, dtype=float)
        for i in range(len(qs) - 1):
            dq = qs[i + 1] - qs[i]
            dv = vs[i + 1] - vs[i]
            if dv > 0:
                upper = vs[i + 1] if i < len(qs) - 2 else vs[i + 1] + 1e-12
                mask = (xs >= vs[i]) & (xs < upper)
                ys[mask] = dq / dv
        return ys

    def _piecewise_step_xy(pts):
        qs, vs = _with_boundaries(pts)
        xs_out, ys_out = [], []
        for i in range(len(qs) - 1):
            dq = qs[i + 1] - qs[i]
            dv = vs[i + 1] - vs[i]
            d = dq / dv if dv > 0 else 0.0
            xs_out += [vs[i], vs[i + 1]]
            ys_out += [d, d]
        return np.array(xs_out), np.array(ys_out)

    for k, (ia, ib) in enumerate(index_pairs):
        ax = axes[k // ncols][k % ncols]
        dist_a = a[ia]
        dist_b = get_dist_b(ib)
        label_a = get_label_a(ia)
        label_b = get_label_b(ib)

        if use_beta:
            xs = np.linspace(0.0, 1.0, 500)
            aa, ab = fit_beta(dist_a)
            ba, bb = fit_beta(dist_b)
            ya = beta.pdf(xs, aa, ab)
            yb = beta.pdf(xs, ba, bb)
            ax.fill_between(xs, ya, yb, where=ya >= yb, color=_COLOR_A, alpha=0.35)
            ax.fill_between(xs, ya, yb, where=ya <= yb, color=_COLOR_B, alpha=0.35)
            ax.plot(xs, ya, "-", color=_COLOR_A, lw=1.8, label=label_a)
            ax.plot(xs, yb, "-", color=_COLOR_B, lw=1.8, label=label_b)
            w1 = w1_distance_beta(dist_a, dist_b)
        else:
            _, vs_a = _with_boundaries(dist_a)
            _, vs_b = _with_boundaries(dist_b)
            xs = np.unique(np.concatenate([
                np.array(sorted(set(vs_a + vs_b))),
                np.linspace(0.0, 1.0, 300),
            ]))
            ya = _piecewise_pdf_at(dist_a, xs)
            yb = _piecewise_pdf_at(dist_b, xs)
            ax.fill_between(xs, ya, yb, where=ya >= yb, color=_COLOR_A, alpha=0.35)
            ax.fill_between(xs, ya, yb, where=ya <= yb, color=_COLOR_B, alpha=0.35)
            xa_step, ya_step = _piecewise_step_xy(dist_a)
            xb_step, yb_step = _piecewise_step_xy(dist_b)
            ax.plot(xa_step, ya_step, "-", color=_COLOR_A, lw=1.8, label=label_a)
            ax.plot(xb_step, yb_step, "-", color=_COLOR_B, lw=1.8, label=label_b)
            for v in vs_a[1:-1]:
                ax.axvline(v, color=_COLOR_A, lw=0.7, ls=":", alpha=0.5)
            for v in vs_b[1:-1]:
                ax.axvline(v, color=_COLOR_B, lw=0.7, ls=":", alpha=0.5)
            w1 = w1_distance(dist_a, dist_b)

        ax.set_title(f"W\u2081 = {w1:.4f}  ({label_a} vs {label_b})", fontsize=9, fontweight="bold")
        ax.set_xlabel("Estimated probability", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.grid(linestyle=":", alpha=0.5)
        ax.legend(fontsize=7, loc="upper left")

    _finalize_pair_grid(fig, axes, len(index_pairs), nrows, ncols, figsize_per_plot, suptitle)
    return fig


def plot_cdf_pairs(
    a: list[list[tuple[float, float]]],
    b: list[list[tuple[float, float]]] | None = None,
    ncols: int = 2,
    figsize_per_plot: tuple[float, float] = (5, 4),
    use_beta: bool = False,
    labels_a: list[str] | None = None,
    labels_b: list[str] | None = None,
):
    """Plot quantile functions for each pairwise combination.

    Each subplot shows the two inverse CDFs (quantile functions) for one pair,
    with filled regions and the W1 distance in the subtitle.

    Args:
        a: list of distributions, each a list of (percentile, value) pairs.
        b: optional second list of distributions.
        ncols: Number of subplot columns in the grid.
        figsize_per_plot: (width, height) in inches for each individual subplot.
        use_beta: If True, fit Beta distributions and plot smooth quantile functions.
        labels_a: optional per-distribution labels for group a.
        labels_b: optional per-distribution labels for group b.

    Returns:
        matplotlib Figure
    """
    method_label = "Beta-fit quantile functions" if use_beta else "Piecewise-linear quantile functions"
    suptitle = f"{method_label}  \u00b7  Filled area = W\u2081 distance"

    fig, axes, index_pairs, get_dist_b, nrows, ncols = _setup_pair_grid(
        a, b, ncols, figsize_per_plot, suptitle
    )
    get_label_a, get_label_b = _make_pair_labels(
        len(a), len(b) if b is not None else None, labels_a, labels_b
    )

    def _insert_crossings(xs, ya, yb):
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

    for k, (ia, ib) in enumerate(index_pairs):
        ax = axes[k // ncols][k % ncols]
        dist_a = a[ia]
        dist_b = get_dist_b(ib)
        label_a = get_label_a(ia)
        label_b = get_label_b(ib)

        if use_beta:
            prob_fine = np.linspace(0.0, 1.0, 500)
            aa, ab = fit_beta(dist_a)
            ba, bb = fit_beta(dist_b)
            ya = beta.ppf(prob_fine, aa, ab)
            yb = beta.ppf(prob_fine, ba, bb)
            ax.fill_between(
                prob_fine, ya, yb, where=ya >= yb,
                color=_COLOR_A, alpha=0.35, label=f"{label_a} > {label_b}",
            )
            ax.fill_between(
                prob_fine, ya, yb, where=ya <= yb,
                color=_COLOR_B, alpha=0.35, label=f"{label_b} > {label_a}",
            )
            ax.plot(prob_fine, ya, "-", color=_COLOR_A, lw=1.8, label=label_a)
            ax.plot(prob_fine, yb, "-", color=_COLOR_B, lw=1.8, label=label_b)
            pts_a_q, pts_a_v = zip(*dist_a)
            pts_b_q, pts_b_v = zip(*dist_b)
            ax.plot(pts_a_q, pts_a_v, "o", color=_COLOR_A, ms=5, zorder=5)
            ax.plot(pts_b_q, pts_b_v, "o", color=_COLOR_B, ms=5, zorder=5)
            w1 = w1_distance_beta(dist_a, dist_b)
            prob_levels = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            tick_labels = ["0%", "25%", "50%", "75%", "100%"]
        else:
            qa, va = _with_boundaries(dist_a)
            qb, vb = _with_boundaries(dist_b)
            all_q = sorted(set(qa) | set(qb))
            va_interp = np.interp(all_q, qa, va)
            vb_interp = np.interp(all_q, qb, vb)
            xs, ya, yb = _insert_crossings(
                np.array(all_q), va_interp, vb_interp
            )
            ax.fill_between(
                xs, ya, yb, where=ya >= yb,
                color=_COLOR_A, alpha=0.35, label=f"{label_a} > {label_b}",
            )
            ax.fill_between(
                xs, ya, yb, where=ya <= yb,
                color=_COLOR_B, alpha=0.35, label=f"{label_b} > {label_a}",
            )
            ax.plot(qa, va, "o-", color=_COLOR_A, lw=1.8, ms=5, label=label_a)
            ax.plot(qb, vb, "o-", color=_COLOR_B, lw=1.8, ms=5, label=label_b)
            w1 = w1_distance(dist_a, dist_b)
            prob_levels = np.array([0.0, 0.25, 0.5, 0.75])
            tick_labels = ["0%", "25%", "50%", "75%"]

        ax.set_title(f"W\u2081 = {w1:.4f}  ({label_a} vs {label_b})", fontsize=9, fontweight="bold")
        ax.set_xlabel("Quantile level", fontsize=9)
        ax.set_ylabel("Estimated probability", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(prob_levels)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.grid(linestyle=":", alpha=0.5)
        ax.legend(fontsize=7, loc="upper left")

    _finalize_pair_grid(fig, axes, len(index_pairs), nrows, ncols, figsize_per_plot, suptitle)
    return fig
