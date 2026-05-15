#!/usr/bin/env python
"""Calculate Lyptus baseline predictions and scoring tables.

This module centralizes the baseline machinery used by the report notebooks.
It can be imported from notebooks or run directly as a small CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "intra_benchmark_calibration").exists():
            return candidate
    raise FileNotFoundError("Could not find repo root containing intra_benchmark_calibration")


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from intra_benchmark_calibration.binning import BinAssignment, compute_bins
from intra_benchmark_calibration.lyptus_data import LyptusDataset, load_lyptus_dataset


BaselineName = Literal["uninformed_0_5", "model_pass_rate", "model_bin_pass_rate", "irt_logistic_fit"]

DEFAULT_BASELINES: tuple[BaselineName, ...] = (
    "uninformed_0_5",
    "model_pass_rate",
    "model_bin_pass_rate",
    "irt_logistic_fit",
)

BASELINE_PREDICTION_COLUMNS: dict[str, str] = {
    "uninformed_0_5": "p_uninformed_0_5",
    "model_pass_rate": "p_model_pass_rate",
    "model_bin_pass_rate": "p_model_bin_pass_rate",
    "irt_logistic_fit": "p_irt_logistic_fit",
}

BASELINE_INTERVAL_COLUMNS: dict[str, tuple[str, str, str]] = {
    "uninformed_0_5": ("p25_uninformed_0_5", "p50_uninformed_0_5", "p75_uninformed_0_5"),
    "model_pass_rate": ("p25_model_pass_rate", "p50_model_pass_rate", "p75_model_pass_rate"),
    "model_bin_pass_rate": ("p25_model_bin_pass_rate", "p50_model_bin_pass_rate", "p75_model_bin_pass_rate"),
    "irt_logistic_fit": ("p25_irt_logistic_fit", "p50_irt_logistic_fit", "p75_irt_logistic_fit"),
}

RUN_REQUIRED_COLUMNS = [
    "forecasted_model",
    "target_task_id",
    "target_task_family",
    "target_fst_minutes",
    "target_bin",
    "outcome",
    "p50",
]

EVALUATION_CELL_COLUMNS = [
    "forecasted_model",
    "target_task_id",
    "target_task_family",
    "target_fst_minutes",
    "target_bin",
    "outcome",
]

DEFAULT_ORDERED_SCORE_COLUMNS = [
    "source_type",
    "source",
    "n",
    "mean_prediction",
    "empirical_pass_rate",
    "brier",
    "brier_ci_low",
    "brier_ci_high",
    "crps_beta",
    "crps_beta_ci_low",
    "crps_beta_ci_high",
    "bias",
]


@dataclass(frozen=True)
class BaselineConfig:
    lyptus_repo_dir: Path = Path("/home/jeffm/lyptus-data")
    drop_models: tuple[str, ...] = ("GPT-2", "GPT-3", "GPT-3.5")
    n_bins: int = 5
    binning_strategy: str = "equal_count"
    explicit_edges: tuple[float, ...] | None = None
    bootstrap_iterations: int = 1000
    bootstrap_seed: int = 20260515
    metric_bootstrap_quantiles: tuple[float, float] = (0.025, 0.975)
    irt_bootstrap_iterations: int = 250
    irt_bootstrap_seed: int | None = None
    refit_regularization: float = 1e-5
    refit_weight_col: str | None = "invsqrt_task_weight"

    @property
    def resolved_irt_bootstrap_seed(self) -> int:
        return self.bootstrap_seed + 1000 if self.irt_bootstrap_seed is None else self.irt_bootstrap_seed


@dataclass
class BaselineContext:
    config: BaselineConfig
    dataset: LyptusDataset
    bins: BinAssignment
    tasks: pd.DataFrame
    irt_fit_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    irt_fit_params: pd.DataFrame = field(default_factory=pd.DataFrame)
    irt_bootstrap_fit_params: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class RunComparisonData:
    raw_run_frames: dict[str, pd.DataFrame]
    run_frames: dict[str, pd.DataFrame]
    shared_run_frames: dict[str, pd.DataFrame]
    shared_panel: pd.DataFrame
    all_panel: pd.DataFrame
    run_index: pd.DataFrame
    target_count_summary: pd.DataFrame
    target_presence: pd.DataFrame
    incomplete_targets: pd.DataFrame
    cell_mismatches: pd.DataFrame
    common_target_ids: set[str]


def normalize_baselines(baselines: str | Iterable[str] | None = None) -> tuple[str, ...]:
    if baselines is None or baselines == "all":
        return DEFAULT_BASELINES
    if isinstance(baselines, str):
        requested = (baselines,)
    else:
        requested = tuple(baselines)
        if len(requested) == 1 and requested[0] == "all":
            return DEFAULT_BASELINES
    unknown = sorted(set(requested) - set(DEFAULT_BASELINES))
    if unknown:
        raise ValueError(f"Unknown baseline(s): {unknown}. Valid choices are: {list(DEFAULT_BASELINES)}")
    return requested


def logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def sigmoid(x: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def valid_run_rows(df: pd.DataFrame, *, last_round_only: bool = True) -> pd.DataFrame:
    """Keep rows with parsed p50, optionally restricted to each cell's final round."""
    out = df.dropna(subset=["p50"]).copy()
    if last_round_only:
        group_keys = ["condition_id"]
        for key in ("forecaster_model", "repeat_index", "expert_id"):
            if key in out.columns:
                group_keys.append(key)
        idx = out.groupby(group_keys, dropna=False)["delphi_round"].idxmax()
        out = out.loc[idx].reset_index(drop=True)
    return out


def fit_beta_to_percentiles(
    p25: float,
    p50: float,
    p75: float,
    init: tuple[float, float] = (2.0, 2.0),
) -> tuple[float, float] | None:
    targets = np.array([p25, p50, p75], dtype=float)
    if np.any(np.isnan(targets)) or not np.all((0.0 < targets) & (targets < 1.0)):
        return None
    if not (targets[0] <= targets[1] <= targets[2]):
        targets = np.sort(targets)

    def loss(params: np.ndarray) -> float:
        alpha, beta_param = params
        if alpha <= 0 or beta_param <= 0:
            return 1e9
        try:
            preds = beta_dist.ppf([0.25, 0.50, 0.75], alpha, beta_param)
        except Exception:
            return 1e9
        if np.any(np.isnan(preds)):
            return 1e9
        return float(np.sum((preds - targets) ** 2))

    result = minimize(
        loss,
        x0=np.array(init),
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 500},
    )
    if not result.success:
        return None
    alpha, beta_param = float(result.x[0]), float(result.x[1])
    if alpha <= 0 or beta_param <= 0:
        return None
    return alpha, beta_param


def crps_beta(alpha: float, beta_param: float, outcome: int) -> float:
    if outcome == 1:
        integrand = lambda y: beta_dist.cdf(y, alpha, beta_param) ** 2
    else:
        integrand = lambda y: (1.0 - beta_dist.cdf(y, alpha, beta_param)) ** 2
    val, _err = quad(integrand, 0.0, 1.0, limit=100)
    return float(val)


def load_task_weights(repo_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((repo_dir / "data" / "tasks").glob("*/*_tasks.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows.append(
                    {
                        "task_id": row.get("task_id"),
                        "equal_task_weight": row.get("equal_task_weight"),
                        "invsqrt_task_weight": row.get("invsqrt_task_weight"),
                    }
                )
    return pd.DataFrame(rows).drop_duplicates("task_id")


def load_irt_fit_data(repo_dir: Path) -> pd.DataFrame:
    model_runs = pd.read_parquet(repo_dir / "analysis" / "figures" / "data" / "model_runs.parquet")
    task_difficulties = pd.read_parquet(repo_dir / "analysis" / "figures" / "data" / "task_difficulties.parquet")
    task_weights = load_task_weights(repo_dir)

    headline_tasks = task_difficulties.dropna(subset=["best_available_minutes"])[
        ["task_id", "best_available_minutes", "best_available_source"]
    ]
    fit_data = (
        model_runs.merge(headline_tasks, on="task_id", how="inner")
        .merge(task_weights, on="task_id", how="left")
        .dropna(subset=["score_binarized", "best_available_minutes"])
        .copy()
    )
    fit_data["log2_minutes"] = np.log2(fit_data["best_available_minutes"].astype(float))
    return fit_data


def fit_logistic_1d(
    group: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    weight_col: str | None = None,
    regularization: float = 1e-5,
    x0: np.ndarray | None = None,
) -> dict[str, float | bool | str | int]:
    x = group[x_col].astype(float).to_numpy()
    y = group[y_col].astype(float).to_numpy()
    if weight_col and weight_col in group.columns:
        w = group[weight_col].astype(float).fillna(1.0).to_numpy()
    else:
        w = np.ones_like(y, dtype=float)
    w = w / np.mean(w)

    def objective(params: np.ndarray) -> float:
        coef, intercept = params
        z = coef * x + intercept
        bce = np.logaddexp(0.0, z) - y * z
        return float(np.average(bce, weights=w) + 0.5 * regularization * coef**2)

    result = minimize(objective, x0=np.array([-1.0, 0.0]) if x0 is None else x0, method="BFGS")
    coef, intercept = result.x
    out: dict[str, float | bool | str | int] = {
        "coefficient": float(coef),
        "intercept": float(intercept),
        "fit_success": bool(result.success),
        "fit_message": str(result.message),
        "bce_loss": float(objective(result.x)),
        "n_tasks": int(len(group)),
    }
    for pct, threshold in [(50, 0.50), (80, 0.80)]:
        horizon_log2 = (logit(threshold) - intercept) / coef
        out[f"p{pct}"] = float(2.0**horizon_log2)
    return out


def compute_irt_fit_params(fit_data: pd.DataFrame, config: BaselineConfig) -> pd.DataFrame:
    rows = []
    for model, group in fit_data.groupby("alias", sort=True):
        rows.append(
            {
                "agent": model,
                **fit_logistic_1d(
                    group,
                    x_col="log2_minutes",
                    y_col="score_binarized",
                    weight_col=config.refit_weight_col,
                    regularization=config.refit_regularization,
                ),
            }
        )
    return pd.DataFrame(rows).set_index("agent", drop=False)


def compute_irt_bootstrap_fit_params(
    fit_data: pd.DataFrame,
    point_params: pd.DataFrame,
    config: BaselineConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.resolved_irt_bootstrap_seed)
    rows = []
    for model, group in fit_data.groupby("alias", sort=True):
        if model not in point_params.index:
            continue
        group = group.reset_index(drop=True)
        n = len(group)
        point = point_params.loc[model]
        x0 = np.array([point["coefficient"], point["intercept"]], dtype=float)
        for bootstrap_idx in range(config.irt_bootstrap_iterations):
            sample = group.iloc[rng.integers(0, n, size=n)]
            fit = fit_logistic_1d(
                sample,
                x_col="log2_minutes",
                y_col="score_binarized",
                weight_col=config.refit_weight_col,
                regularization=config.refit_regularization,
                x0=x0,
            )
            rows.append(
                {
                    "agent": model,
                    "bootstrap_idx": bootstrap_idx,
                    "coefficient": fit["coefficient"],
                    "intercept": fit["intercept"],
                    "fit_success": fit["fit_success"],
                }
            )
    return pd.DataFrame(rows)


def build_tasks_table(dataset: LyptusDataset, bins: BinAssignment) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "task_id": [task.task_id for task in dataset.tasks],
            "task_family": [task.task_family for task in dataset.tasks],
            "fst_minutes": [task.fst_minutes for task in dataset.tasks],
            "target_bin": bins.bin_index_per_task,
        }
    )


def build_baseline_context(
    config: BaselineConfig | None = None,
    *,
    baselines: str | Iterable[str] | None = None,
) -> BaselineContext:
    config = config or BaselineConfig()
    baseline_names = normalize_baselines(baselines)
    dataset = load_lyptus_dataset(config.lyptus_repo_dir, drop_models=config.drop_models)
    bins = compute_bins(
        dataset.fst_array(),
        n_bins=config.n_bins,
        strategy=config.binning_strategy,
        explicit_edges=config.explicit_edges,
    )
    tasks = build_tasks_table(dataset, bins)

    if "irt_logistic_fit" in baseline_names:
        irt_fit_data = load_irt_fit_data(config.lyptus_repo_dir)
        irt_fit_params = compute_irt_fit_params(irt_fit_data, config)
        irt_bootstrap_fit_params = compute_irt_bootstrap_fit_params(irt_fit_data, irt_fit_params, config)
    else:
        irt_fit_data = pd.DataFrame()
        irt_fit_params = pd.DataFrame()
        irt_bootstrap_fit_params = pd.DataFrame()

    return BaselineContext(
        config=config,
        dataset=dataset,
        bins=bins,
        tasks=tasks,
        irt_fit_data=irt_fit_data,
        irt_fit_params=irt_fit_params,
        irt_bootstrap_fit_params=irt_bootstrap_fit_params,
    )


def sample_targets_by_bin(tasks: pd.DataFrame, *, n_targets_per_bin: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampled_targets = []
    for _bin_idx, group in tasks.groupby("target_bin", sort=True):
        n = min(n_targets_per_bin, len(group))
        chosen_positions = rng.choice(group.index.to_numpy(), size=n, replace=False)
        sampled_targets.append(group.loc[chosen_positions].sort_values("fst_minutes"))
    return pd.concat(sampled_targets, ignore_index=True)


def targets_from_task_ids(task_ids: Sequence[str], context: BaselineContext) -> pd.DataFrame:
    task_order = {task_id: idx for idx, task_id in enumerate(task_ids)}
    targets = context.tasks[context.tasks["task_id"].isin(list(task_order))].copy()
    missing = [task_id for task_id in task_ids if task_id not in set(targets["task_id"])]
    if missing:
        raise ValueError(f"Unknown task_id value(s): {missing[:20]}")
    targets["_task_order"] = targets["task_id"].map(task_order)
    return targets.sort_values("_task_order").drop(columns="_task_order").reset_index(drop=True)


def make_evaluation_panel(
    targets: pd.DataFrame,
    context: BaselineContext,
    *,
    forecasted_models: Sequence[str] | None = None,
) -> pd.DataFrame:
    rows = []
    models = list(forecasted_models or context.dataset.outcomes.models)
    for model in models:
        for target in targets.itertuples(index=False):
            task_id = getattr(target, "task_id", getattr(target, "target_task_id", None))
            outcome = context.dataset.outcomes.outcome(model, task_id)
            if outcome is None:
                continue
            task_family = getattr(target, "task_family", getattr(target, "target_task_family", None))
            fst_minutes = getattr(target, "fst_minutes", getattr(target, "target_fst_minutes", None))
            rows.append(
                {
                    "forecasted_model": model,
                    "target_task_id": task_id,
                    "target_task_family": task_family,
                    "target_fst_minutes": float(fst_minutes),
                    "target_bin": int(target.target_bin),
                    "outcome": int(outcome),
                }
            )
    return pd.DataFrame(rows)


def _stack_outcome_matrix(outcome_matrix: pd.DataFrame) -> pd.DataFrame:
    try:
        long = outcome_matrix.stack(future_stack=True).dropna().rename("outcome").reset_index()
    except TypeError:
        long = outcome_matrix.stack(dropna=True).rename("outcome").reset_index()
    long.columns = ["forecasted_model", "target_task_id", "outcome"]
    return long


def bootstrap_pass_rate_quantiles(
    values: pd.Series,
    *,
    rng: np.random.Generator,
    bootstrap_iterations: int,
) -> dict[str, float | int]:
    y = values.dropna().astype(float).to_numpy()
    if len(y) == 0:
        return {"p25": np.nan, "p50": np.nan, "p75": np.nan, "n": 0}
    draws = rng.choice(y, size=(bootstrap_iterations, len(y)), replace=True).mean(axis=1)
    return {
        "p25": float(np.quantile(draws, 0.25)),
        "p50": float(np.mean(y)),
        "p75": float(np.quantile(draws, 0.75)),
        "n": int(len(y)),
    }


def empirical_interval_table(
    long: pd.DataFrame,
    group_cols: list[str],
    prefix: str,
    *,
    config: BaselineConfig,
    seed_offset: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.bootstrap_seed + seed_offset)
    rows = []
    for key, group in long.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        quantiles = bootstrap_pass_rate_quantiles(
            group["outcome"],
            rng=rng,
            bootstrap_iterations=config.bootstrap_iterations,
        )
        rows.append(
            {
                **dict(zip(group_cols, key)),
                f"p25_{prefix}": quantiles["p25"],
                f"p50_{prefix}": quantiles["p50"],
                f"p75_{prefix}": quantiles["p75"],
                f"n_{prefix}": quantiles["n"],
            }
        )
    return pd.DataFrame(rows)


def irt_bootstrap_interval_table(panel: pd.DataFrame, context: BaselineContext) -> pd.DataFrame:
    if context.irt_bootstrap_fit_params.empty:
        return pd.DataFrame(
            columns=[
                "forecasted_model",
                "target_task_id",
                "p25_irt_logistic_fit",
                "p50_irt_logistic_fit",
                "p75_irt_logistic_fit",
                "n_irt_logistic_fit",
            ]
        )
    targets = panel[["forecasted_model", "target_task_id", "target_fst_minutes"]].drop_duplicates().copy()
    targets["log2_minutes"] = np.log2(targets["target_fst_minutes"].astype(float))

    draws = targets.merge(context.irt_bootstrap_fit_params, left_on="forecasted_model", right_on="agent", how="left")
    draws = draws[draws["fit_success"].fillna(False)].copy()
    draws["p_irt_draw"] = sigmoid(draws["coefficient"] * draws["log2_minutes"] + draws["intercept"])

    intervals = (
        draws.groupby(["forecasted_model", "target_task_id"])["p_irt_draw"]
        .quantile([0.25, 0.50, 0.75])
        .unstack()
        .rename(columns={0.25: "p25_irt_logistic_fit", 0.50: "p50_irt_logistic_fit", 0.75: "p75_irt_logistic_fit"})
        .reset_index()
    )
    intervals["n_irt_logistic_fit"] = draws.groupby(["forecasted_model", "target_task_id"]).size().to_numpy()
    return intervals


def add_baseline_predictions(
    panel: pd.DataFrame,
    context: BaselineContext,
    *,
    baselines: str | Iterable[str] | None = None,
) -> pd.DataFrame:
    baseline_names = normalize_baselines(baselines)
    out = panel.copy()

    if "uninformed_0_5" in baseline_names:
        out["p_uninformed_0_5"] = 0.5
        out["p25_uninformed_0_5"] = 0.25
        out["p50_uninformed_0_5"] = 0.50
        out["p75_uninformed_0_5"] = 0.75

    if {"model_pass_rate", "model_bin_pass_rate"} & set(baseline_names):
        task_bin = context.tasks.set_index("task_id")["target_bin"]
        long = _stack_outcome_matrix(context.dataset.outcomes.frame)
        long["target_bin"] = long["target_task_id"].map(task_bin)

        if "model_pass_rate" in baseline_names:
            model_intervals = empirical_interval_table(
                long,
                ["forecasted_model"],
                "model_pass_rate",
                config=context.config,
                seed_offset=0,
            )
            out = out.merge(model_intervals, on="forecasted_model", how="left")
            out["p_model_pass_rate"] = out["p50_model_pass_rate"]

        if "model_bin_pass_rate" in baseline_names:
            model_bin_intervals = empirical_interval_table(
                long,
                ["forecasted_model", "target_bin"],
                "model_bin_pass_rate",
                config=context.config,
                seed_offset=1,
            )
            out = out.merge(model_bin_intervals, on=["forecasted_model", "target_bin"], how="left")
            out["p_model_bin_pass_rate"] = out["p50_model_bin_pass_rate"]

    if "irt_logistic_fit" in baseline_names:
        if context.irt_fit_params.empty:
            raise ValueError("IRT baseline requested, but BaselineContext was built without IRT fit data.")
        fit_params = context.irt_fit_params[["coefficient", "intercept"]].rename(
            columns={"coefficient": "irt_coefficient", "intercept": "irt_intercept"}
        )
        out = out.join(fit_params, on="forecasted_model")
        log2_minutes = np.log2(out["target_fst_minutes"].astype(float))
        out["p_irt_logistic_fit"] = sigmoid(out["irt_coefficient"] * log2_minutes + out["irt_intercept"])
        out = out.merge(irt_bootstrap_interval_table(out, context), on=["forecasted_model", "target_task_id"], how="left")
        out["p50_irt_logistic_fit"] = out["p50_irt_logistic_fit"].fillna(out["p_irt_logistic_fit"])

    return out


@lru_cache(maxsize=None)
def cached_beta_fit(p25: float, p50: float, p75: float) -> tuple[float, float] | None:
    return fit_beta_to_percentiles(p25, p50, p75)


def beta_crps_contributions(group: pd.DataFrame, interval_cols: tuple[str, str, str]) -> pd.Series:
    group = group.reset_index(drop=True)
    p25_col, p50_col, p75_col = interval_cols
    crps_values = pd.Series(np.nan, index=group.index, dtype=float)
    needed = [p25_col, p50_col, p75_col, "outcome"]
    tmp = group.dropna(subset=needed).copy()
    if len(tmp) == 0:
        return crps_values

    eps = 1e-6
    tmp["p25"] = tmp[p25_col].clip(eps, 1.0 - eps)
    tmp["p50"] = tmp[p50_col].clip(eps, 1.0 - eps)
    tmp["p75"] = tmp[p75_col].clip(eps, 1.0 - eps)
    for idx, row in tmp.iterrows():
        fit = cached_beta_fit(float(row["p25"]), float(row["p50"]), float(row["p75"]))
        if fit is None:
            continue
        alpha, beta_param = fit
        crps_values.loc[idx] = crps_beta(alpha, beta_param, int(row["outcome"]))
    return crps_values


def bootstrap_mean_ci(
    values: pd.Series,
    *,
    rng: np.random.Generator,
    bootstrap_iterations: int,
    quantiles: tuple[float, float],
) -> tuple[float, float]:
    x = values.dropna().astype(float).to_numpy()
    if len(x) == 0:
        return (np.nan, np.nan)
    draws = rng.choice(x, size=(bootstrap_iterations, len(x)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, quantiles)
    return (float(low), float(high))


def score_prediction_source(
    df: pd.DataFrame,
    *,
    source_type: str,
    source: str,
    pred_col: str,
    interval_cols: tuple[str, str, str] | None,
    config: BaselineConfig,
    seed_offset: int,
) -> dict[str, float | int | str]:
    g = df.dropna(subset=[pred_col, "outcome"]).copy().reset_index(drop=True)
    p = g[pred_col].astype(float)
    y = g["outcome"].astype(float)
    err = p - y
    brier_contrib = err**2
    crps_contrib = beta_crps_contributions(g, interval_cols) if interval_cols else pd.Series(np.nan, index=g.index, dtype=float)

    rng = np.random.default_rng(config.bootstrap_seed + 10_000 + seed_offset)
    brier_low, brier_high = bootstrap_mean_ci(
        brier_contrib,
        rng=rng,
        bootstrap_iterations=config.bootstrap_iterations,
        quantiles=config.metric_bootstrap_quantiles,
    )
    crps_low, crps_high = bootstrap_mean_ci(
        crps_contrib,
        rng=rng,
        bootstrap_iterations=config.bootstrap_iterations,
        quantiles=config.metric_bootstrap_quantiles,
    )

    return {
        "source_type": source_type,
        "source": source,
        "n": int(len(g)),
        "mean_prediction": float(p.mean()) if len(g) else np.nan,
        "empirical_pass_rate": float(y.mean()) if len(g) else np.nan,
        "brier": float(brier_contrib.mean()) if len(g) else np.nan,
        "brier_ci_low": brier_low,
        "brier_ci_high": brier_high,
        "crps_beta": float(crps_contrib.mean()) if crps_contrib.notna().any() else np.nan,
        "crps_beta_ci_low": crps_low,
        "crps_beta_ci_high": crps_high,
        "bias": float(err.mean()) if len(g) else np.nan,
    }


def score_baselines(
    df: pd.DataFrame,
    group_cols: Sequence[str] | None = None,
    *,
    baselines: str | Iterable[str] | None = None,
    config: BaselineConfig | None = None,
    seed_offset: int = 0,
    source_columns: bool = False,
) -> pd.DataFrame:
    config = config or BaselineConfig()
    baseline_names = normalize_baselines(baselines)
    group_cols = list(group_cols or [])
    rows = []
    groups = [((), df)] if not group_cols else df.groupby(group_cols, dropna=False)

    for group_idx, (key, group) in enumerate(groups):
        if group_cols and not isinstance(key, tuple):
            key = (key,)
        group_values = dict(zip(group_cols, key)) if group_cols else {}
        for baseline_idx, baseline_name in enumerate(baseline_names):
            row = score_prediction_source(
                group,
                source_type="baseline",
                source=baseline_name,
                pred_col=BASELINE_PREDICTION_COLUMNS[baseline_name],
                interval_cols=BASELINE_INTERVAL_COLUMNS.get(baseline_name),
                config=config,
                seed_offset=seed_offset + 100 * group_idx + baseline_idx,
            )
            rows.append({**group_values, **row})

    out = pd.DataFrame(rows)
    if not source_columns and len(out):
        out = out.rename(columns={"source": "baseline"}).drop(columns=["source_type"])
    return out


def calculate_baselines(
    *,
    task_ids: Sequence[str] | None = None,
    panel: pd.DataFrame | None = None,
    context: BaselineContext | None = None,
    config: BaselineConfig | None = None,
    baselines: str | Iterable[str] | None = None,
    group_cols: Sequence[str] | None = None,
    forecasted_models: Sequence[str] | None = None,
    return_panel: bool = False,
    source_columns: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    baseline_names = normalize_baselines(baselines)
    context = context or build_baseline_context(config, baselines=baseline_names)
    if panel is None:
        targets = targets_from_task_ids(task_ids, context) if task_ids is not None else context.tasks
        panel = make_evaluation_panel(targets, context, forecasted_models=forecasted_models)
    panel_with_baselines = add_baseline_predictions(panel, context, baselines=baseline_names)
    scores = score_baselines(
        panel_with_baselines,
        group_cols,
        baselines=baseline_names,
        config=context.config,
        source_columns=source_columns,
    )
    if return_panel:
        return scores, panel_with_baselines
    return scores


def find_run_csv(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_file():
        if not path.name.endswith("_estimates.csv"):
            raise ValueError(f"Run file does not look like an estimates CSV: {path}")
        return path
    if not path.exists():
        raise FileNotFoundError(path)
    matches = sorted(path.glob("*_estimates.csv"))
    if not matches:
        raise FileNotFoundError(f"No *_estimates.csv found in {path}")
    return matches[-1]


def default_run_label(path: Path) -> str:
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return path.parent.name if path.parent.name else path.stem
    return path.name


def uniquify_labels(labels: Sequence[str]) -> list[str]:
    counts: dict[str, int] = {}
    out = []
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
        out.append(label if counts[label] == 1 else f"{label}_{counts[label]}")
    return out


def load_run_frames(path: Path, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_path = find_run_csv(path)
    raw = pd.read_csv(csv_path)

    missing = [col for col in RUN_REQUIRED_COLUMNS if col not in raw.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    scoreable = valid_run_rows(raw, last_round_only=True).copy()
    for frame in (raw, scoreable):
        for quantile_col in ["p25", "p50", "p75"]:
            if quantile_col not in frame.columns:
                frame[quantile_col] = np.nan
        frame["run_label"] = label
        frame["run_csv"] = str(csv_path)

    return raw.reset_index(drop=True), scoreable.reset_index(drop=True)


def target_task_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["target_task_id", "target_task_family", "target_fst_minutes", "target_bin"]]
        .drop_duplicates()
        .sort_values("target_task_id")
        .reset_index(drop=True)
    )


def evaluation_cell_table(df: pd.DataFrame) -> pd.DataFrame:
    cells = df[EVALUATION_CELL_COLUMNS].drop_duplicates().copy()
    conflicts = cells.groupby(["forecasted_model", "target_task_id"], dropna=False).size().rename("n_versions").reset_index()
    conflicts = conflicts[conflicts["n_versions"] > 1]
    if len(conflicts):
        raise ValueError(
            "A run has conflicting metadata/outcomes for the same forecasted_model and target_task_id:\n"
            f"{conflicts.to_string(index=False)}"
        )
    return cells.sort_values(["forecasted_model", "target_task_id"]).reset_index(drop=True)


def tuple_set(df: pd.DataFrame, cols: list[str]) -> set[tuple[Any, ...]]:
    return set(map(tuple, df[cols].itertuples(index=False, name=None)))


def load_shared_run_comparison(
    run_paths: Sequence[Path],
    *,
    run_labels: dict[Path, str] | None = None,
    strict_evaluation_cell_match: bool = True,
    require_shared_panel: bool = True,
) -> RunComparisonData:
    if not run_paths:
        raise ValueError("Set run_paths to one or more run directories or *_estimates.csv files.")

    run_labels = run_labels or {}
    raw_labels = [run_labels.get(Path(path), default_run_label(Path(path))) for path in run_paths]
    labels = uniquify_labels(raw_labels)
    loaded = {label: load_run_frames(Path(path), label) for path, label in zip(run_paths, labels)}
    raw_run_frames = {label: raw for label, (raw, _scoreable) in loaded.items()}
    run_frames = {label: scoreable for label, (_raw, scoreable) in loaded.items()}

    run_index = pd.DataFrame(
        [
            {
                "run_label": label,
                "n_raw_rows": len(raw_run_frames[label]),
                "n_valid_final_rows": len(df),
                "n_raw_target_tasks": raw_run_frames[label]["target_task_id"].nunique(),
                "n_scoreable_target_tasks": df["target_task_id"].nunique(),
                "n_scoreable_evaluation_cells": len(evaluation_cell_table(df)),
                "csv": df["run_csv"].iloc[0],
            }
            for label, df in run_frames.items()
        ]
    )

    target_summaries = {label: target_task_table(df) for label, df in raw_run_frames.items()}
    target_count_summary = (
        pd.concat([summary.assign(run_label=label) for label, summary in target_summaries.items()], ignore_index=True)
        .groupby(["run_label", "target_bin"], dropna=False)
        .size()
        .rename("n_target_tasks")
        .reset_index()
    )

    target_presence = pd.concat(
        [summary.assign(run_label=label) for label, summary in target_summaries.items()],
        ignore_index=True,
    )
    target_presence = (
        target_presence.groupby("target_task_id", dropna=False)
        .agg(
            target_task_family=("target_task_family", "first"),
            target_fst_minutes=("target_fst_minutes", "first"),
            target_bin=("target_bin", "first"),
            n_runs=("run_label", "nunique"),
            present_in=("run_label", lambda s: sorted(set(s))),
        )
        .reset_index()
    )
    all_run_labels = set(run_frames)
    target_presence["missing_from"] = target_presence["present_in"].map(lambda present: sorted(all_run_labels - set(present)))
    incomplete_targets = target_presence[target_presence["n_runs"] < len(run_frames)].copy()
    if len(incomplete_targets):
        missing_task_ids = sorted(incomplete_targets["target_task_id"].astype(str))
        missing_task_list = ", ".join(missing_task_ids[:20])
        if len(missing_task_ids) > 20:
            missing_task_list += f", ... (+{len(missing_task_ids) - 20} more)"
        warnings.warn(
            f"{len(incomplete_targets)} raw target tasks are not present in every run: {missing_task_list}. "
            "Inspect incomplete_targets for missing_from per task.",
            stacklevel=2,
        )

    common_target_ids = set(target_presence.loc[target_presence["n_runs"] == len(run_frames), "target_task_id"])
    if not common_target_ids and require_shared_panel:
        raise ValueError("No raw target_task_id values are present in every run.")

    all_cell_tables = {label: evaluation_cell_table(df) for label, df in run_frames.items()}
    all_cells = set.union(*(tuple_set(cells, EVALUATION_CELL_COLUMNS) for cells in all_cell_tables.values()))
    all_panel = pd.DataFrame(sorted(all_cells), columns=EVALUATION_CELL_COLUMNS).reset_index(drop=True)

    task_shared_run_frames = {
        label: df[df["target_task_id"].isin(common_target_ids)].copy().reset_index(drop=True)
        for label, df in run_frames.items()
    }

    cell_tables = {label: evaluation_cell_table(df) for label, df in task_shared_run_frames.items()}
    cell_sets = {label: tuple_set(cells, EVALUATION_CELL_COLUMNS) for label, cells in cell_tables.items()}
    shared_candidate_cells = set.union(*cell_sets.values())
    common_cells = set.intersection(*cell_sets.values())
    cell_mismatch_rows = []
    for label, cell_set in cell_sets.items():
        if cell_set != common_cells:
            cell_mismatch_rows.append(
                {
                    "run_label": label,
                    "n_cells_not_in_this_run": len(shared_candidate_cells - cell_set),
                    "n_excluded_from_shared_panel": len(cell_set - common_cells),
                }
            )
    cell_mismatches = pd.DataFrame(cell_mismatch_rows)

    if len(cell_mismatches):
        message = (
            "Scoreable evaluation cells differ across runs. "
            f"The shared scoring panel will use the {len(common_cells)} cells present in every run."
        )
        if strict_evaluation_cell_match:
            warnings.warn(message, stacklevel=2)
        else:
            print(f"Warning: {message}")

    if not common_cells and require_shared_panel:
        raise ValueError("No scoreable evaluation cells are present in every run after target-task filtering.")

    shared_panel = pd.DataFrame(sorted(common_cells), columns=EVALUATION_CELL_COLUMNS).reset_index(drop=True)
    shared_run_frames = {
        label: df.merge(shared_panel[EVALUATION_CELL_COLUMNS], on=EVALUATION_CELL_COLUMNS, how="inner").reset_index(drop=True)
        for label, df in task_shared_run_frames.items()
    }

    return RunComparisonData(
        raw_run_frames=raw_run_frames,
        run_frames=run_frames,
        shared_run_frames=shared_run_frames,
        shared_panel=shared_panel,
        all_panel=all_panel,
        run_index=run_index,
        target_count_summary=target_count_summary,
        target_presence=target_presence,
        incomplete_targets=incomplete_targets,
        cell_mismatches=cell_mismatches,
        common_target_ids=common_target_ids,
    )


def run_rows_for_scoring(df: pd.DataFrame, *, one_row_per_evaluation_cell: bool = False) -> pd.DataFrame:
    out = df.copy()
    if one_row_per_evaluation_cell:
        out = out.drop_duplicates(["forecasted_model", "target_task_id", "target_bin", "outcome"], keep="last")
    return out.reset_index(drop=True)


def score_runs(
    run_frames: dict[str, pd.DataFrame],
    *,
    config: BaselineConfig | None = None,
    one_row_per_evaluation_cell: bool = False,
) -> pd.DataFrame:
    config = config or BaselineConfig()
    rows = []
    for idx, (label, df) in enumerate(run_frames.items(), start=100):
        rows.append(
            score_prediction_source(
                run_rows_for_scoring(df, one_row_per_evaluation_cell=one_row_per_evaluation_cell),
                source_type="run",
                source=label,
                pred_col="p50",
                interval_cols=("p25", "p50", "p75"),
                config=config,
                seed_offset=idx,
            )
        )
    return pd.DataFrame(rows)


def score_elicited_predictions(
    df: pd.DataFrame,
    *,
    source: str = "LLM_forecaster_elicited",
    config: BaselineConfig | None = None,
    seed_offset: int = 3,
) -> pd.DataFrame:
    config = config or BaselineConfig()
    return pd.DataFrame(
        [
            score_prediction_source(
                df,
                source_type="run",
                source=source,
                pred_col="p50",
                interval_cols=("p25", "p50", "p75"),
                config=config,
                seed_offset=seed_offset,
            )
        ]
    )


def parse_task_ids(raw_task_ids: Sequence[str], task_id_file: Path | None) -> list[str] | None:
    task_ids = list(raw_task_ids)
    if task_id_file is not None:
        with task_id_file.open("r", encoding="utf-8") as fh:
            task_ids.extend(line.strip() for line in fh if line.strip())
    expanded: list[str] = []
    for item in task_ids:
        expanded.extend(part.strip() for part in item.split(",") if part.strip())
    return expanded or None


def _make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lyptus-repo-dir", type=Path, default=Path("/home/jeffm/lyptus-data"))
    parser.add_argument("--baselines", nargs="+", default=["all"], choices=[*DEFAULT_BASELINES, "all"])
    parser.add_argument("--task-ids", nargs="*", default=[])
    parser.add_argument("--task-id-file", type=Path)
    parser.add_argument("--sample-targets-per-bin", type=int)
    parser.add_argument("--target-sample-seed", type=int, default=20260514)
    parser.add_argument("--run-path", action="append", type=Path, default=[])
    parser.add_argument("--group-cols", nargs="*", default=[])
    parser.add_argument("--forecasted-models", nargs="*")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--irt-bootstrap-iterations", type=int, default=250)
    parser.add_argument("--output-csv", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _make_cli_parser().parse_args(argv)
    baselines = normalize_baselines(args.baselines)
    config = BaselineConfig(
        lyptus_repo_dir=args.lyptus_repo_dir,
        bootstrap_iterations=args.bootstrap_iterations,
        irt_bootstrap_iterations=args.irt_bootstrap_iterations,
    )

    context = build_baseline_context(config, baselines=baselines)
    if args.run_path:
        run_data = load_shared_run_comparison(args.run_path)
        panel = run_data.shared_panel
        baseline_scores, _panel_with_baselines = calculate_baselines(
            panel=panel,
            context=context,
            baselines=baselines,
            group_cols=args.group_cols,
            return_panel=True,
        )
        comparison = pd.concat(
            [
                baseline_scores.rename(columns={"baseline": "source"}).assign(source_type="baseline"),
                score_runs(run_data.shared_run_frames, config=config),
            ],
            ignore_index=True,
        )
        scores = comparison[[col for col in DEFAULT_ORDERED_SCORE_COLUMNS if col in comparison.columns]]
    else:
        task_ids = parse_task_ids(args.task_ids, args.task_id_file)
        if args.sample_targets_per_bin is not None:
            targets = sample_targets_by_bin(context.tasks, n_targets_per_bin=args.sample_targets_per_bin, seed=args.target_sample_seed)
            panel = make_evaluation_panel(targets, context, forecasted_models=args.forecasted_models)
            scores = calculate_baselines(panel=panel, context=context, baselines=baselines, group_cols=args.group_cols)
        else:
            scores = calculate_baselines(
                task_ids=task_ids,
                context=context,
                baselines=baselines,
                group_cols=args.group_cols,
                forecasted_models=args.forecasted_models,
            )

    scores = scores.sort_values([col for col in [*args.group_cols, "brier"] if col in scores.columns], kind="stable").reset_index(drop=True)
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(args.output_csv, index=False)
    print(scores.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
