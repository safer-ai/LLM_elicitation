"""
Compute bootstrap 95% CIs for Brier and CRPS on conditions E and F.

Raw estimates extracted from git history (commit cd91538) into /tmp/conditions_ci/.
A/B/C raw data was never committed and cannot be recovered without new runs.

Usage:
    python3 report_analyses/bootstrap_conditions_ef.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from typing import Tuple

# Import the canonical implementations
from intra_benchmark_calibration.analyse_results import (
    fit_beta_to_percentiles,
    crps_beta,
)

DATA_DIR = Path("/tmp/conditions_ci")
N_BOOT   = 10_000
SEED     = 20260515


def score_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per condition:
      - keep last Delphi round only
      - average p25/p50/p75 across expert_ids per (forecasted_model, target_task_id)
      - compute brier = (p50 - outcome)^2 and crps_beta
    Returns one scored row per unique (model, task) cell.
    """
    last = df["delphi_round"].max()
    df = df[df["delphi_round"] == last].copy()

    agg = (df.groupby(["forecasted_model", "target_task_id"])
             .agg(p25=("p25", "mean"),
                  p50=("p50", "mean"),
                  p75=("p75", "mean"),
                  outcome=("outcome", "first"))
             .reset_index())

    rows = []
    for _, r in agg.iterrows():
        brier = (float(r["p50"]) - float(r["outcome"])) ** 2
        params = fit_beta_to_percentiles(r["p25"], r["p50"], r["p75"])
        crps = crps_beta(params[0], params[1], int(r["outcome"])) if params else float("nan")
        rows.append({"brier": brier, "crps": crps})
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT,
                 seed: int = SEED) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    boots = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


# ── load, score, bootstrap ────────────────────────────────────────────────────

results = {}

for label, csv_path in {
    "E": DATA_DIR / "E_estimates.csv",
    "F": DATA_DIR / "F_estimates_full.csv",
}.items():
    if not csv_path.exists():
        print(f"  {label}: not found at {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    scored = score_df(df).dropna()
    n = len(scored)

    brier_mean = scored["brier"].mean()
    crps_mean  = scored["crps"].mean()
    b_lo, b_hi = bootstrap_ci(scored["brier"].values)
    c_lo, c_hi = bootstrap_ci(scored["crps"].values)

    results[label] = dict(n=n,
                          brier=brier_mean, brier_lo=b_lo, brier_hi=b_hi,
                          crps=crps_mean,  crps_lo=c_lo,  crps_hi=c_hi)
    print(f"Condition {label}  N={n:4d}  "
          f"Brier={brier_mean:.4f} [{b_lo:.4f}, {b_hi:.4f}]  "
          f"CRPS={crps_mean:.4f} [{c_lo:.4f}, {c_hi:.4f}]")

# ── final table ───────────────────────────────────────────────────────────────
print()
print("Condition comparison (A/B/C: pilot only, no archived raw data for CIs)")
print()

# Hardcoded from comparison_table.md (N = cells, not API calls)
known = {
    "A": dict(n=60,  brier=0.2336, crps=0.3113),
    "B": dict(n=60,  brier=0.2255, crps=0.3014),
    "C": dict(n=240, brier=0.1736, crps=0.2601),
}

header = f"{'Cond':<4}  {'N':>5}  {'Brier':>7}  {'[95% CI]':^21}  {'CRPS':>7}  {'[95% CI]':^21}"
print(header)
print("-" * len(header))

for label, d in known.items():
    print(f"{label:<4}  {d['n']:>5}  {d['brier']:>7.4f}  {'— no CI (data not archived) —':^21}  "
          f"{d['crps']:>7.4f}  {'':^21}")

for label, d in results.items():
    print(f"{label:<4}  {d['n']:>5}  {d['brier']:>7.4f}  "
          f"[{d['brier_lo']:.4f}, {d['brier_hi']:.4f}]  "
          f"{d['crps']:>7.4f}  [{d['crps_lo']:.4f}, {d['crps_hi']:.4f}]")

print(f"{'base':<4}  {'—':>5}  {0.2500:>7.4f}  {'(exact)':^21}  {0.3333:>7.4f}  {'(exact)':^21}")
