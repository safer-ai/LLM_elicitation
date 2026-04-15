import itertools

import numpy as np
import pandas as pd
from scipy.stats import beta
from scipy.integrate import quad


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
    qa, va = _with_boundaries(a)
    qb, vb = _with_boundaries(b)

    all_q = sorted(set(qa) | set(qb))
    ya = np.interp(all_q, qa, va)
    yb = np.interp(all_q, qb, vb)

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
    qa, va = _with_boundaries(a)
    qb, vb = _with_boundaries(b)

    all_q = sorted(set(qa) | set(qb))
    ya = np.interp(all_q, qa, va)
    yb = np.interp(all_q, qb, vb)

    total = 0.0
    for i in range(len(all_q) - 1):
        h = all_q[i + 1] - all_q[i]
        d0 = ya[i] - yb[i]
        d1 = ya[i + 1] - yb[i + 1]

        # Exact integral of the square of a linear function over the interval
        total += h * (d0 * d0 + d0 * d1 + d1 * d1) / 3.0

    return float(np.sqrt(total))


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

_BETA_GRID_N = 2048
_BETA_X_GRID = np.linspace(0.0, 1.0, _BETA_GRID_N)
_BETA_U_GRID = np.linspace(0.0, 1.0, _BETA_GRID_N)


def _beta_grids(a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """Precompute CDF and inverse-CDF grids for a Beta(a, b) distribution."""
    cdf_vals = beta.cdf(_BETA_X_GRID, a, b)
    ppf_vals = beta.ppf(_BETA_U_GRID, a, b)
    return cdf_vals, ppf_vals


def _w1_from_cdf_grids(cdf1: np.ndarray, cdf2: np.ndarray) -> float:
    """W1 distance from precomputed CDF grids via trapezoid rule."""
    return float(np.trapz(np.abs(cdf1 - cdf2), _BETA_X_GRID))


def _w2_from_ppf_grids(ppf1: np.ndarray, ppf2: np.ndarray) -> float:
    """W2 distance from precomputed inverse-CDF grids via trapezoid rule."""
    return float(np.sqrt(np.trapz((ppf1 - ppf2) ** 2, _BETA_U_GRID)))


def _w1_distance_beta_params(
    a1: float, b1: float, a2: float, b2: float
) -> float:
    """W1 distance between two Beta distributions given pre-fit parameters."""
    def integrand(x):
        return abs(beta.cdf(x, a1, b1) - beta.cdf(x, a2, b2))

    val, _ = quad(integrand, 0.0, 1.0, epsabs=1e-9, epsrel=1e-9)
    return val


def _w2_distance_beta_params(
    a1: float, b1: float, a2: float, b2: float
) -> float:
    """W2 distance between two Beta distributions given pre-fit parameters."""
    def integrand(u):
        q1 = beta.ppf(u, a1, b1)
        q2 = beta.ppf(u, a2, b2)
        return (q1 - q2) ** 2

    val, _ = quad(integrand, 0.0, 1.0, epsabs=1e-9, epsrel=1e-9)
    return val**0.5


def w1_distance_beta(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> float:
    """W1 distance using Beta distributions fit to quantile points.

    Fits a Beta(a, b) distribution to each set of quantile points then
    computes the exact W1 distance via numerical integration of the absolute
    CDF difference.

    Args:
        a, b: sequences of (percentile, value) pairs

    Returns:
        W1 distance (non-negative float)
    """
    a1, b1 = fit_beta(a)
    a2, b2 = fit_beta(b)
    return _w1_distance_beta_params(a1, b1, a2, b2)

def w2_distance_beta(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> float:
    """W2 distance using Beta distributions fit to quantile points.

    Fits a Beta(alpha, beta) distribution to each set of quantile points, then
    computes the 1D Wasserstein-2 distance via numerical integration of the
    squared difference of the inverse CDFs.

    Args:
        a, b: sequences of (percentile, value) pairs

    Returns:
        W2 distance (non-negative float)
    """
    a1, b1 = fit_beta(a)
    a2, b2 = fit_beta(b)
    return _w2_distance_beta_params(a1, b1, a2, b2)


def p50_divergence(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Absolute difference in median (p50) between two distributions.

    Args:
        a, b: sequences of (percentile, value) pairs. The p50 value is obtained
              by linear interpolation at quantile level 0.5.

    Returns:
        |median(a) - median(b)|
    """
    def _median(pts):
        pts_sorted = sorted(pts)
        qs = [p for p, _ in pts_sorted]
        vs = [v for _, v in pts_sorted]
        return float(np.interp(0.5, qs, vs))

    return abs(_median(a) - _median(b))


def iqr_divergence(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Absolute difference in interquartile range between two distributions.

    Args:
        a, b: sequences of (percentile, value) pairs. The p25 and p75 values are
              obtained by linear interpolation at quantile levels 0.25 and 0.75.

    Returns:
        |IQR(a) - IQR(b)|
    """
    def _iqr(pts):
        pts_sorted = sorted(pts)
        qs = [p for p, _ in pts_sorted]
        vs = [v for _, v in pts_sorted]
        return float(np.interp(0.75, qs, vs)) - float(np.interp(0.25, qs, vs))

    return abs(_iqr(a) - _iqr(b))

def row_to_pairs(estimates: dict) -> list[tuple[float, float]]:
    """Convert an estimates dict {percentile_int: value} to sorted (quantile_level, value) pairs.

    Args:
        estimates: dict with int percentile keys and float values, e.g. {20: 0.4, 60: 0.7}

    Returns:
        sorted list of (percentile / 100, value) pairs, e.g. [(0.2, 0.4), (0.6, 0.7)]
    """
    return sorted((int(p) / 100.0, v) for p, v in estimates.items() if v is not None)


def compute_pairwise_results(
    df1: pd.DataFrame, df2: pd.DataFrame | None = None, include_beta: bool = False
) -> pd.DataFrame:
    """Compute consistency metrics for every pairwise row combination.

    For a single dataframe, pairs are every combination of two rows within df1.
    For two dataframes, pairs are every cross-product combination (one row from
    each dataframe).

    Each row is converted to (percentile, value) pairs via row_to_pairs, which
    extracts all 'percentile_Nth_mean' columns.

    Args:
        df1: DataFrame with 'percentile_Nth_mean' columns.
        df2: Optional second DataFrame with the same columns. If provided,
             pairs are drawn one from each dataframe.
        include_beta: If True, also compute W1 and W2 using Beta distributions
                      fit to each row's quantile points and include 'w1_beta'
                      and 'w2_beta' columns. Note this can be slow because of integration

    Returns:
        DataFrame with one row per pair and columns:
            'idx_a', 'idx_b'  -- index labels from the source dataframes
            'w1'              -- W1 distance
            'w2'              -- W2 distance
            'p50_divergence'  -- absolute p50 difference
            'iqr_divergence'  -- absolute IQR difference
            'w1_beta'         -- beta-fit W1 distance (only if include_beta=True)
            'w2_beta'         -- beta-fit W2 distance (only if include_beta=True)
    """
    index1 = list(df1.index)
    records1 = df1["estimates"].apply(row_to_pairs).tolist()

    if df2 is None:
        index_pairs = list(itertools.combinations(index1, 2))
        record_pairs = list(itertools.combinations(range(len(records1)), 2))
    else:
        index2 = list(df2.index)
        records2 = df2["estimates"].apply(row_to_pairs).tolist()
        index_pairs = list(itertools.product(index1, index2))
        record_pairs = list(itertools.product(range(len(records1)), range(len(records2))))

    # Pre-fit beta parameters and grids once per distribution
    if include_beta:
        beta_params1 = [fit_beta(r) for r in records1]
        grids1 = [_beta_grids(a, b) for a, b in beta_params1]
        if df2 is not None:
            beta_params2 = [fit_beta(r) for r in records2]
            grids2 = [_beta_grids(a, b) for a, b in beta_params2]
        else:
            grids2 = None

    rows = []
    for (ia, ib), (ri, rj) in zip(index_pairs, record_pairs):
        a = records1[ri]
        b = (records2[rj] if df2 is not None else records1[rj])
        row = {
            "idx_a": ia,
            "idx_b": ib,
            "w1": w1_distance(a, b),
            "w2": w2_distance(a, b),
            "p50_divergence": p50_divergence(a, b),
            "iqr_divergence": iqr_divergence(a, b),
        }
        if include_beta:
            cdf_i, ppf_i = grids1[ri]
            cdf_j, ppf_j = (grids2[rj] if df2 is not None else grids1[rj])
            row["w1_beta"] = _w1_from_cdf_grids(cdf_i, cdf_j)
            row["w2_beta"] = _w2_from_ppf_grids(ppf_i, ppf_j)
        rows.append(row)

    return pd.DataFrame(rows)


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

    Aggregates the output of compute_pairwise_results by taking the mean of
    each metric column. Optionally also computes bootstrap confidence
    intervals for the mean W1 and W2 metrics.

    Args:
        df1: DataFrame with 'percentile_Nth_mean' columns.
        df2: Optional second DataFrame with the same columns. If provided,
             pairs are drawn one from each dataframe.
        include_beta: If True, also include 'w1_beta' and 'w2_beta' means.
                      note this can be slow because of integration.
        compute_ci: If True, compute bootstrap confidence intervals for
                    'w1' and 'w2'.
        n_bootstrap: Number of bootstrap resamples to use when compute_ci=True.
        confidence_level: Central confidence mass for the percentile interval.
        random_state: Optional seed or numpy Generator for bootstrap draws.

    Returns:
        dict with keys:
            'w1'             -- mean W1 distance across all pairs
            'w2'             -- mean W2 distance across all pairs
            'p50_divergence' -- mean absolute p50 difference across all pairs
            'iqr_divergence' -- mean absolute IQR difference across all pairs
            'n_pairs'        -- number of pairs evaluated
            'w1_beta'        -- mean beta-fit W1 distance (only if include_beta=True)
            'w2_beta'        -- mean beta-fit W2 distance (only if include_beta=True)
            'w1_ci_lower'    -- bootstrap CI lower bound for mean W1
                                 (only if compute_ci=True)
            'w1_ci_upper'    -- bootstrap CI upper bound for mean W1
                                 (only if compute_ci=True)
            'w2_ci_lower'    -- bootstrap CI lower bound for mean W2
                                 (only if compute_ci=True)
            'w2_ci_upper'    -- bootstrap CI upper bound for mean W2
                                 (only if compute_ci=True)
    """
    results_df = compute_pairwise_results(df1, df2, include_beta=include_beta)

    if results_df.empty:
        result = {
            "w1": float("nan"),
            "w2": float("nan"),
            "p50_divergence": float("nan"),
            "iqr_divergence": float("nan"),
            "n_pairs": 0,
        }
        if include_beta:
            result["w1_beta"] = float("nan")
            result["w2_beta"] = float("nan")
        if compute_ci:
            result["w1_ci_lower"] = float("nan")
            result["w1_ci_upper"] = float("nan")
            result["w2_ci_lower"] = float("nan")
            result["w2_ci_upper"] = float("nan")
            result["confidence_level"] = confidence_level
            result["n_bootstrap"] = n_bootstrap
        return result

    metric_cols = ["w1", "w2", "p50_divergence", "iqr_divergence"]
    if include_beta:
        metric_cols += ["w1_beta", "w2_beta"]

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


def _resample_runs(
    df1: pd.DataFrame, df2: pd.DataFrame | None, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Bootstrap-resample the run-level inputs used by compute_pairwise_metrics.

    For a single dataframe, rows are resampled within that dataframe.
    For two dataframes, aligned row pairs are resampled jointly so the
    bootstrap unit remains the original run pair rather than a derived metric.
    """
    if df2 is None:
        if df1.empty:
            return df1, None
        sample_idx = rng.integers(0, len(df1), size=len(df1))
        return df1.iloc[sample_idx], None

    if len(df1) != len(df2):
        raise ValueError(
            "Bootstrap resampling with two dataframes requires the same number "
            "of rows so run pairs can be resampled jointly."
        )

    if df1.empty:
        return df1, df2

    sample_idx = rng.integers(0, len(df1), size=len(df1))
    return df1.iloc[sample_idx], df2.iloc[sample_idx]


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


def _precompute_pairwise_cache(
    records1: list[list[tuple[float, float]]],
    records2: list[list[tuple[float, float]]] | None,
    include_beta: bool,
) -> dict[tuple[int, int], dict[str, float]]:
    """Precompute all unique pairwise distances between row distributions.

    For single-group (records2 is None): computes all combinations(N, 2)
    distances. Keys are canonical (i, j) with i < j. Self-pairs (i, i)
    are implicitly zero and not stored.

    For two-group: computes all product(N1, N2) distances. Keys are (i, j)
    where i indexes into records1 and j indexes into records2.

    Returns:
        dict mapping (i, j) -> {"w1": ..., "w2": ..., "p50_divergence": ...,
        "iqr_divergence": ..., and optionally "w1_beta": ..., "w2_beta": ...}
    """
    cache: dict[tuple[int, int], dict[str, float]] = {}

    # Pre-fit beta parameters and grids once per distribution
    if include_beta:
        beta_params1 = [fit_beta(r) for r in records1]
        grids1 = [_beta_grids(a, b) for a, b in beta_params1]
        if records2 is not None:
            beta_params2 = [fit_beta(r) for r in records2]
            grids2 = [_beta_grids(a, b) for a, b in beta_params2]
        else:
            grids2 = None

    if records2 is None:
        for i, j in itertools.combinations(range(len(records1)), 2):
            a, b = records1[i], records1[j]
            entry = {
                "w1": w1_distance(a, b),
                "w2": w2_distance(a, b),
                "p50_divergence": p50_divergence(a, b),
                "iqr_divergence": iqr_divergence(a, b),
            }
            if include_beta:
                cdf_i, ppf_i = grids1[i]
                cdf_j, ppf_j = grids1[j]
                entry["w1_beta"] = _w1_from_cdf_grids(cdf_i, cdf_j)
                entry["w2_beta"] = _w2_from_ppf_grids(ppf_i, ppf_j)
            cache[(i, j)] = entry
    else:
        for i in range(len(records1)):
            for j in range(len(records2)):
                a, b = records1[i], records2[j]
                entry = {
                    "w1": w1_distance(a, b),
                    "w2": w2_distance(a, b),
                    "p50_divergence": p50_divergence(a, b),
                    "iqr_divergence": iqr_divergence(a, b),
                }
                if include_beta:
                    cdf_i, ppf_i = grids1[i]
                    cdf_j, ppf_j = grids2[j]
                    entry["w1_beta"] = _w1_from_cdf_grids(cdf_i, cdf_j)
                    entry["w2_beta"] = _w2_from_ppf_grids(ppf_i, ppf_j)
                cache[(i, j)] = entry

    return cache


_ZERO_METRICS = {"w1": 0.0, "w2": 0.0, "p50_divergence": 0.0, "iqr_divergence": 0.0}
_ZERO_METRICS_BETA = {**_ZERO_METRICS, "w1_beta": 0.0, "w2_beta": 0.0}


def _bootstrap_mean_from_cache(
    cache: dict[tuple[int, int], dict[str, float]],
    sampled_indices: np.ndarray,
    two_group: bool,
    include_beta: bool,
) -> dict[str, float]:
    """Compute mean pairwise metrics for one bootstrap sample using cached distances.

    Args:
        cache: Precomputed pairwise distances.
        sampled_indices: Resampled positional indices into the original rows.
        two_group: If True, pairs are cross-product (df1 x df2) with jointly
                   resampled aligned indices. If False, pairs are combinations
                   within a single group.
        include_beta: Whether beta metrics are present in the cache.

    Returns:
        dict with mean metric values across all pairs.
    """
    zeros = _ZERO_METRICS_BETA if include_beta else _ZERO_METRICS
    metric_keys = list(zeros.keys())

    if two_group:
        pairs = list(itertools.product(sampled_indices, sampled_indices))
    else:
        pairs = list(itertools.combinations(sampled_indices, 2))

    if not pairs:
        return {k: float("nan") for k in metric_keys}

    totals = {k: 0.0 for k in metric_keys}
    for i, j in pairs:
        if not two_group:
            key = (min(i, j), max(i, j)) if i != j else None
        else:
            key = (i, j)

        if key is None:
            entry = zeros
        else:
            entry = cache[key]

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
    """Compute bootstrap CIs for the mean pairwise W1 and W2 metrics.

    The bootstrap resamples the original run rows used to construct the
    pairwise comparisons. With one dataframe, rows are resampled within the
    dataframe. With two dataframes, aligned row pairs are resampled jointly.

    Distances are precomputed once for all unique pairs, then each bootstrap
    iteration only performs index resampling and cache lookups.

    Args:
        df1: DataFrame with 'percentile_Nth_mean' columns.
        df2: Optional second DataFrame with the same columns.
        include_beta: Passed through to compute_pairwise_metrics.
        n_bootstrap: Number of bootstrap resamples.
        confidence_level: Central confidence mass for the percentile interval.
        random_state: Optional seed or numpy Generator.

    Returns:
        dict containing the usual compute_pairwise_metrics output plus:
            'w1_ci_lower', 'w1_ci_upper' -- bootstrap CI for mean W1
            'w2_ci_lower', 'w2_ci_upper' -- bootstrap CI for mean W2
            'n_bootstrap'                -- number of bootstrap replicates used
            'confidence_level'           -- CI coverage level
    """
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    result = {
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap,
    }
    if df1.empty or (df2 is not None and df2.empty):
        result["w1_ci_lower"] = float("nan")
        result["w1_ci_upper"] = float("nan")
        result["w2_ci_lower"] = float("nan")
        result["w2_ci_upper"] = float("nan")
        return result

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    # Precompute all unique pairwise distances once
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
    w1_samples = []
    w2_samples = []
    w1_beta_samples = [] if include_beta else None
    w2_beta_samples = [] if include_beta else None
    for _ in range(n_bootstrap):
        sampled_indices = rng.integers(0, n1, size=n1)
        means = _bootstrap_mean_from_cache(
            cache, sampled_indices, two_group, include_beta
        )
        w1_samples.append(means["w1"])
        w2_samples.append(means["w2"])
        if include_beta:
            w1_beta_samples.append(means["w1_beta"])
            w2_beta_samples.append(means["w2_beta"])

    result["w1_ci_lower"], result["w1_ci_upper"] = _percentile_ci(
        w1_samples, confidence_level
    )
    result["w2_ci_lower"], result["w2_ci_upper"] = _percentile_ci(
        w2_samples, confidence_level
    )
    if include_beta:
        result["w1_beta_ci_lower"], result["w1_beta_ci_upper"] = _percentile_ci(
            w1_beta_samples, confidence_level
        )
        result["w2_beta_ci_lower"], result["w2_beta_ci_upper"] = _percentile_ci(
            w2_beta_samples, confidence_level
        )
    return result


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

    For the piecewise-linear case (use_beta=False), the distribution implied
    by the quantile points is piecewise-constant (a histogram), so PDFs are
    drawn as step functions.  For use_beta=True, smooth Beta PDFs are used.

    Note: the filled area equals ∫|f_A − f_B| dx (total variation × 2), not
    the W1 distance.  W1 is still computed from CDFs and shown in the title.

    Args:
        a: list of distributions, each a list of (percentile, value) pairs.
        b: optional second list of distributions. If provided, pairs are drawn
           one from each list; otherwise all pairwise combinations within a.
        ncols: Number of subplot columns in the grid.
        figsize_per_plot: (width, height) in inches for each individual subplot.
        use_beta: If True, fit a Beta distribution to each distribution's
                  quantile points and plot smooth Beta PDFs.
        labels_a: optional per-distribution labels for group a.
                  Defaults to "A1", "A2", ...
        labels_b: optional per-distribution labels for group b.
                  Defaults to "B1", "B2", ... (or "A1", "A2", ... when b is None).

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt

    COLOR_A = "steelblue"
    COLOR_B = "tomato"

    def _get_label_a(i):
        return labels_a[i] if labels_a is not None else f"A{i + 1}"

    def _get_label_b(i):
        if b is None:
            return labels_a[i] if labels_a is not None else f"A{i + 1}"
        return labels_b[i] if labels_b is not None else f"B{i + 1}"

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

    for k, (ia, ib) in enumerate(index_pairs):
        ax = axes[k // ncols][k % ncols]
        dist_a = a[ia]
        dist_b = get_dist_b(ib)
        label_a = _get_label_a(ia)
        label_b = _get_label_b(ib)

        if use_beta:
            xs = np.linspace(0.0, 1.0, 500)
            aa, ab = fit_beta(dist_a)
            ba, bb = fit_beta(dist_b)
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
            _, vs_a = _with_boundaries(dist_a)
            _, vs_b = _with_boundaries(dist_b)
            xs = np.unique(np.concatenate([
                np.array(sorted(set(vs_a + vs_b))),
                np.linspace(0.0, 1.0, 300),
            ]))
            ya = _piecewise_pdf_at(dist_a, xs)
            yb = _piecewise_pdf_at(dist_b, xs)
            ax.fill_between(xs, ya, yb, where=ya >= yb, color=COLOR_A, alpha=0.35)
            ax.fill_between(xs, ya, yb, where=ya <= yb, color=COLOR_B, alpha=0.35)
            xa_step, ya_step = _piecewise_step_xy(dist_a)
            xb_step, yb_step = _piecewise_step_xy(dist_b)
            ax.plot(xa_step, ya_step, "-", color=COLOR_A, lw=1.8, label=label_a)
            ax.plot(xb_step, yb_step, "-", color=COLOR_B, lw=1.8, label=label_b)
            for v in vs_a[1:-1]:
                ax.axvline(v, color=COLOR_A, lw=0.7, ls=":", alpha=0.5)
            for v in vs_b[1:-1]:
                ax.axvline(v, color=COLOR_B, lw=0.7, ls=":", alpha=0.5)
            w1 = w1_distance(dist_a, dist_b)

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
    with the region where distribution A lies above B filled in one colour and
    the region where B lies above A filled in another.  The W1 distance for
    the pair is shown in the subplot title.

    Args:
        a: list of distributions, each a list of (percentile, value) pairs.
        b: optional second list of distributions. If provided, pairs are drawn
           one from each list; otherwise all pairwise combinations within a.
        ncols: Number of subplot columns in the grid.
        figsize_per_plot: (width, height) in inches for each individual subplot.
        use_beta: If True, fit a Beta distribution to each distribution's
                  quantile points and plot smooth Beta quantile functions.
        labels_a: optional per-distribution labels for group a.
                  Defaults to "A1", "A2", ...
        labels_b: optional per-distribution labels for group b.
                  Defaults to "B1", "B2", ... (or "A1", "A2", ... when b is None).

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt

    COLOR_A = "steelblue"
    COLOR_B = "tomato"

    def _get_label_a(i):
        return labels_a[i] if labels_a is not None else f"A{i + 1}"

    def _get_label_b(i):
        if b is None:
            return labels_a[i] if labels_a is not None else f"A{i + 1}"
        return labels_b[i] if labels_b is not None else f"B{i + 1}"

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

    for k, (ia, ib) in enumerate(index_pairs):
        ax = axes[k // ncols][k % ncols]
        dist_a = a[ia]
        dist_b = get_dist_b(ib)
        label_a = _get_label_a(ia)
        label_b = _get_label_b(ib)

        if use_beta:
            PROB_FINE = np.linspace(0.0, 1.0, 500)
            aa, ab = fit_beta(dist_a)
            ba, bb = fit_beta(dist_b)
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
            w1, _ = quad(
                lambda x: abs(beta.cdf(x, aa, ab) - beta.cdf(x, ba, bb)),
                0.0, 1.0, epsabs=1e-9, epsrel=1e-9,
            )
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
                color=COLOR_A, alpha=0.35, label=f"{label_a} > {label_b}",
            )
            ax.fill_between(
                xs, ya, yb, where=ya <= yb,
                color=COLOR_B, alpha=0.35, label=f"{label_b} > {label_a}",
            )
            ax.plot(qa, va, "o-", color=COLOR_A, lw=1.8, ms=5, label=label_a)
            ax.plot(qb, vb, "o-", color=COLOR_B, lw=1.8, ms=5, label=label_b)
            w1 = w1_distance(dist_a, dist_b)
            prob_levels = np.array(sorted(set(qa) | set(qb)))
            tick_labels = [f"{q:.0%}" for q in prob_levels]

        ax.set_title(f"W₁ = {w1:.4f}  ({label_a} vs {label_b})", fontsize=9, fontweight="bold")
        ax.set_xlabel("Quantile level", fontsize=9)
        ax.set_ylabel("Estimated probability", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(prob_levels)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.grid(linestyle=":", alpha=0.5)
        ax.legend(fontsize=7, loc="upper left")

    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    method_label = "Beta-fit quantile functions" if use_beta else "Piecewise-linear quantile functions"
    fig.suptitle(
        f"{method_label}  ·  Filled area = W₁ distance",
        fontsize=11,
    )
    top_margin = 1.0 - 0.4 / (figsize_per_plot[1] * nrows)
    plt.tight_layout(rect=(0, 0, 1, top_margin))
    return fig
