#!/usr/bin/env python3
"""Compare two Beta fitting methods for percentile-based elicitation.

Method 1: Exact fit on p25/p75, validate on p50
Method 2: Nonlinear least-squares on all three percentiles

Runs comprehensive comparison across all 9 experiment conditions.
"""

import csv
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import fsolve, least_squares

# Configuration
EXPERIMENTS_DIR = Path(__file__).parent.parent

DATA_DIRS = {
    ("Claude Sonnet 4.5", "TA0002 (50%)"): EXPERIMENTS_DIR / "percentile_claude_TA0002_50pct",
    ("Claude Sonnet 4.5", "TA0007 (85%)"): EXPERIMENTS_DIR / "percentile_claude_TA0007_85pct",
    ("Claude Sonnet 4.5", "T1657 (30%)"):  EXPERIMENTS_DIR / "percentile_claude_T1657_30pct",
    ("GPT-4o", "TA0002 (50%)"): EXPERIMENTS_DIR / "percentile_gpt4o_TA0002_50pct",
    ("GPT-4o", "TA0007 (85%)"): EXPERIMENTS_DIR / "percentile_gpt4o_TA0007_85pct",
    ("GPT-4o", "T1657 (30%)"):  EXPERIMENTS_DIR / "percentile_gpt4o_T1657_30pct",
    ("Gemini 2.5 Pro", "TA0002 (50%)"): EXPERIMENTS_DIR / "percentile_gemini_TA0002_50pct",
    ("Gemini 2.5 Pro", "TA0007 (85%)"): EXPERIMENTS_DIR / "percentile_gemini_TA0007_85pct",
    ("Gemini 2.5 Pro", "T1657 (30%)"):  EXPERIMENTS_DIR / "percentile_gemini_T1657_30pct",
}

STEPS = ["TA0002 (50%)", "TA0007 (85%)", "T1657 (30%)"]
MODELS = ["Claude Sonnet 4.5", "GPT-4o", "Gemini 2.5 Pro"]


# Method 1: Exact fit on p25/p75, validate p50
def fit_method1(p25: float, p50: float, p75: float) -> Optional[Tuple[float, float, float, str]]:
    """Fit Beta using p25 and p75, validate on p50.

    Returns (alpha, beta, p50_error, status) or None
    """
    if not (0 < p25 < p50 < p75 < 1):
        return None

    # Clamp
    p25 = max(p25, 0.005)
    p75 = min(p75, 0.995)
    p50 = max(min(p50, 0.995), 0.005)

    if not (p25 < p50 < p75):
        return None

    def equations(params):
        a, b = params
        if a <= 0.01 or b <= 0.01:
            return [1e10, 1e10]
        eq1 = sp_stats.beta.cdf(p25, a, b) - 0.25
        eq2 = sp_stats.beta.cdf(p75, a, b) - 0.75
        return [eq1, eq2]

    best_solution = None
    best_residual = float("inf")

    init_guesses = [
        (2.0, 2.0), (5.0, 5.0), (1.5, 1.5), (10.0, 10.0),
        (3.0, 8.0), (8.0, 3.0), (1.2, 0.8), (0.8, 1.2),
        (1.5, 3.0), (3.0, 1.5), (15.0, 5.0), (5.0, 15.0),
        (0.5, 0.5), (1.0, 2.0), (2.0, 1.0), (20.0, 20.0),
    ]

    for a0, b0 in init_guesses:
        try:
            sol, info, ier, msg = fsolve(equations, [a0, b0], full_output=True)
            a_sol, b_sol = sol
            if ier == 1 and a_sol > 0.01 and b_sol > 0.01:
                residual = sum(x**2 for x in info["fvec"])
                if residual < best_residual:
                    best_residual = residual
                    best_solution = (a_sol, b_sol)
                    if residual < 1e-20:
                        break
        except (ValueError, RuntimeError):
            continue

    if best_solution is None:
        return None

    if best_residual > 1e-6:
        return None

    # Validate p50
    a_fit, b_fit = best_solution
    p50_fitted_cdf = sp_stats.beta.cdf(p50, a_fit, b_fit)
    p50_error = abs(p50_fitted_cdf - 0.50)

    if p50_error > 0.05:
        return None

    return (a_fit, b_fit, p50_error, "success")


# Method 2: Nonlinear least-squares on all three
def fit_method2(p25: float, p50: float, p75: float) -> Optional[Tuple[float, float, float, str]]:
    """Fit Beta using least-squares on all three percentiles.

    Returns (alpha, beta, max_error, status) or None
    """
    if not (0 < p25 < p50 < p75 < 1):
        return None

    # Clamp
    p25 = max(p25, 0.005)
    p75 = min(p75, 0.995)
    p50 = max(min(p50, 0.995), 0.005)

    if not (p25 < p50 < p75):
        return None

    def residuals(params):
        a, b = params
        if a <= 0.01 or b <= 0.01:
            return [1e10, 1e10, 1e10]
        r1 = sp_stats.beta.cdf(p25, a, b) - 0.25
        r2 = sp_stats.beta.cdf(p50, a, b) - 0.50
        r3 = sp_stats.beta.cdf(p75, a, b) - 0.75
        return [r1, r2, r3]

    best_solution = None
    best_cost = float("inf")

    init_guesses = [
        (2.0, 2.0), (5.0, 5.0), (1.5, 1.5), (10.0, 10.0),
        (3.0, 8.0), (8.0, 3.0), (1.2, 0.8), (0.8, 1.2),
        (1.5, 3.0), (3.0, 1.5), (15.0, 5.0), (5.0, 15.0),
        (0.5, 0.5), (1.0, 2.0), (2.0, 1.0), (20.0, 20.0),
    ]

    for a0, b0 in init_guesses:
        try:
            result = least_squares(
                residuals,
                [a0, b0],
                bounds=([0.01, 0.01], [100, 100]),
                ftol=1e-12,
                xtol=1e-12
            )

            if result.success and result.cost < best_cost:
                best_solution = result.x
                best_cost = result.cost

        except (ValueError, RuntimeError):
            continue

    if best_solution is None:
        return None

    # Check all errors
    a_fit, b_fit = best_solution
    p25_err = abs(sp_stats.beta.cdf(p25, a_fit, b_fit) - 0.25)
    p50_err = abs(sp_stats.beta.cdf(p50, a_fit, b_fit) - 0.50)
    p75_err = abs(sp_stats.beta.cdf(p75, a_fit, b_fit) - 0.75)
    max_err = max(p25_err, p50_err, p75_err)

    if max_err > 0.05:
        return None

    return (a_fit, b_fit, max_err, "success")


def analyze_dataset(data_dir: Path):
    """Analyze both methods on a single dataset."""
    results = {
        "total": 0,
        "method1_success": 0,
        "method2_success": 0,
        "both_success": 0,
        "both_fail": 0,
        "method1_only": 0,
        "method2_only": 0,
        "method1_errors": [],
        "method2_errors": [],
        "percentile_stats": {
            "p25_values": [],
            "p50_values": [],
            "p75_values": [],
            "iqr_values": [],
        }
    }

    run_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])

    for run_dir in run_dirs:
        csv_path = run_dir / "detailed_estimates.csv"
        if not csv_path.exists():
            continue

        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                # Skip error rows
                if row.get("has_error", "").strip().lower() == "true":
                    continue

                try:
                    p25 = float(row["percentile_25th"])
                    p50 = float(row["percentile_50th"])
                    p75 = float(row["percentile_75th"])
                except (ValueError, KeyError):
                    continue

                results["total"] += 1
                results["percentile_stats"]["p25_values"].append(p25)
                results["percentile_stats"]["p50_values"].append(p50)
                results["percentile_stats"]["p75_values"].append(p75)
                results["percentile_stats"]["iqr_values"].append(p75 - p25)

                # Try both methods
                result1 = fit_method1(p25, p50, p75)
                result2 = fit_method2(p25, p50, p75)

                m1_success = result1 is not None
                m2_success = result2 is not None

                if m1_success:
                    results["method1_success"] += 1
                    results["method1_errors"].append(result1[2])

                if m2_success:
                    results["method2_success"] += 1
                    results["method2_errors"].append(result2[2])

                if m1_success and m2_success:
                    results["both_success"] += 1
                elif not m1_success and not m2_success:
                    results["both_fail"] += 1
                elif m1_success and not m2_success:
                    results["method1_only"] += 1
                elif not m1_success and m2_success:
                    results["method2_only"] += 1

    return results


def main():
    print("=" * 100)
    print("COMPREHENSIVE COMPARISON: Beta Fitting Methods")
    print("=" * 100)
    print()

    all_results = {}

    # Run analysis for each condition
    for model in MODELS:
        for step in STEPS:
            key = (model, step)
            data_dir = DATA_DIRS[key]

            if not data_dir.exists():
                print(f"WARNING: {data_dir} does not exist, skipping...")
                continue

            print(f"Analyzing: {model}, {step}")
            results = analyze_dataset(data_dir)
            all_results[key] = results

            total = results["total"]
            m1 = results["method1_success"]
            m2 = results["method2_success"]

            print(f"  Total estimates: {total}")
            print(f"  Method 1 success: {m1}/{total} ({100*m1/total:.1f}%)")
            print(f"  Method 2 success: {m2}/{total} ({100*m2/total:.1f}%)")
            print(f"  Both succeed: {results['both_success']} ({100*results['both_success']/total:.1f}%)")
            print(f"  Both fail: {results['both_fail']} ({100*results['both_fail']/total:.1f}%)")
            print(f"  Method 2 rescues: {results['method2_only']} ({100*results['method2_only']/total:.1f}%)")
            print(f"  Method 1 rescues: {results['method1_only']} ({100*results['method1_only']/total:.1f}%)")
            print()

    # Summary statistics
    print("=" * 100)
    print("OVERALL SUMMARY")
    print("=" * 100)
    print()

    # Aggregate by model
    print("BY MODEL:")
    print("-" * 100)
    for model in MODELS:
        total = sum(all_results[(model, step)]["total"] for step in STEPS if (model, step) in all_results)
        m1_success = sum(all_results[(model, step)]["method1_success"] for step in STEPS if (model, step) in all_results)
        m2_success = sum(all_results[(model, step)]["method2_success"] for step in STEPS if (model, step) in all_results)
        rescued = sum(all_results[(model, step)]["method2_only"] for step in STEPS if (model, step) in all_results)

        print(f"{model}:")
        print(f"  Total: {total}")
        print(f"  Method 1: {m1_success}/{total} ({100*m1_success/total:.1f}%)")
        print(f"  Method 2: {m2_success}/{total} ({100*m2_success/total:.1f}%)")
        print(f"  Rescued by Method 2: {rescued} ({100*rescued/total:.1f}%)")
        print()

    # Aggregate by step
    print("BY STEP:")
    print("-" * 100)
    for step in STEPS:
        total = sum(all_results[(model, step)]["total"] for model in MODELS if (model, step) in all_results)
        m1_success = sum(all_results[(model, step)]["method1_success"] for model in MODELS if (model, step) in all_results)
        m2_success = sum(all_results[(model, step)]["method2_success"] for model in MODELS if (model, step) in all_results)
        rescued = sum(all_results[(model, step)]["method2_only"] for model in MODELS if (model, step) in all_results)

        print(f"{step}:")
        print(f"  Total: {total}")
        print(f"  Method 1: {m1_success}/{total} ({100*m1_success/total:.1f}%)")
        print(f"  Method 2: {m2_success}/{total} ({100*m2_success/total:.1f}%)")
        print(f"  Rescued by Method 2: {rescued} ({100*rescued/total:.1f}%)")
        print()

    # Grand total
    print("GRAND TOTAL:")
    print("-" * 100)
    grand_total = sum(r["total"] for r in all_results.values())
    grand_m1 = sum(r["method1_success"] for r in all_results.values())
    grand_m2 = sum(r["method2_success"] for r in all_results.values())
    grand_rescued = sum(r["method2_only"] for r in all_results.values())
    grand_both_fail = sum(r["both_fail"] for r in all_results.values())

    print(f"Total estimates: {grand_total}")
    print(f"Method 1 success: {grand_m1}/{grand_total} ({100*grand_m1/grand_total:.1f}%)")
    print(f"Method 2 success: {grand_m2}/{grand_total} ({100*grand_m2/grand_total:.1f}%)")
    print(f"Data rescued by Method 2: {grand_rescued}/{grand_total} ({100*grand_rescued/grand_total:.1f}%)")
    print(f"Unrecoverable failures: {grand_both_fail}/{grand_total} ({100*grand_both_fail/grand_total:.1f}%)")
    print()

    # Error statistics
    print("=" * 100)
    print("ERROR STATISTICS")
    print("=" * 100)

    all_m1_errors = []
    all_m2_errors = []
    for r in all_results.values():
        all_m1_errors.extend(r["method1_errors"])
        all_m2_errors.extend(r["method2_errors"])

    if all_m1_errors:
        print("\nMethod 1 (p50 validation error):")
        print(f"  Mean: {np.mean(all_m1_errors):.4f}")
        print(f"  Median: {np.median(all_m1_errors):.4f}")
        print(f"  Std: {np.std(all_m1_errors):.4f}")
        print(f"  Max: {np.max(all_m1_errors):.4f}")
        print(f"  95th percentile: {np.percentile(all_m1_errors, 95):.4f}")

    if all_m2_errors:
        print("\nMethod 2 (max error across all percentiles):")
        print(f"  Mean: {np.mean(all_m2_errors):.4f}")
        print(f"  Median: {np.median(all_m2_errors):.4f}")
        print(f"  Std: {np.std(all_m2_errors):.4f}")
        print(f"  Max: {np.max(all_m2_errors):.4f}")
        print(f"  95th percentile: {np.percentile(all_m2_errors, 95):.4f}")

    # Percentile statistics
    print("\n" + "=" * 100)
    print("PERCENTILE ELICITATION STATISTICS")
    print("=" * 100)

    all_p25 = []
    all_p50 = []
    all_p75 = []
    all_iqr = []

    for r in all_results.values():
        all_p25.extend(r["percentile_stats"]["p25_values"])
        all_p50.extend(r["percentile_stats"]["p50_values"])
        all_p75.extend(r["percentile_stats"]["p75_values"])
        all_iqr.extend(r["percentile_stats"]["iqr_values"])

    print(f"\nInterquartile Range (IQR = p75 - p25):")
    print(f"  Mean: {np.mean(all_iqr):.4f}")
    print(f"  Median: {np.median(all_iqr):.4f}")
    print(f"  Min: {np.min(all_iqr):.4f}")
    print(f"  Max: {np.max(all_iqr):.4f}")
    print(f"  Very narrow (IQR < 0.05): {sum(1 for x in all_iqr if x < 0.05)}/{len(all_iqr)} ({100*sum(1 for x in all_iqr if x < 0.05)/len(all_iqr):.1f}%)")

    # Asymmetry check
    asymmetry = []
    for p25, p50, p75 in zip(all_p25, all_p50, all_p75):
        if p75 > p25:  # Avoid division by zero
            # Perfect symmetry: (p50 - p25) / (p75 - p25) = 0.5
            asym = abs((p50 - p25) / (p75 - p25) - 0.5)
            asymmetry.append(asym)

    print(f"\nDistribution Asymmetry (|p50 position - 0.5|):")
    print(f"  Mean: {np.mean(asymmetry):.4f}")
    print(f"  Median: {np.median(asymmetry):.4f}")
    print(f"  Highly asymmetric (>0.15): {sum(1 for x in asymmetry if x > 0.15)}/{len(asymmetry)} ({100*sum(1 for x in asymmetry if x > 0.15)/len(asymmetry):.1f}%)")

    print("\n" + "=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)

    rescue_pct = 100 * grand_rescued / grand_total

    print(f"\nMethod 2 (least-squares on all three percentiles) preserves {grand_rescued} additional")
    print(f"estimates ({rescue_pct:.1f}% of total data) compared to Method 1.")
    print()

    if rescue_pct > 5:
        print("RECOMMENDATION: Use Method 2 (least-squares)")
        print("  - Substantially more data preserved")
        print("  - Better handling of asymmetric elicitations")
        print("  - Minimal computational overhead")
    elif rescue_pct > 2:
        print("RECOMMENDATION: Use Method 2 (least-squares)")
        print("  - Moderate data preservation benefit")
        print("  - More robust to elicitation variability")
    else:
        print("RECOMMENDATION: Either method acceptable")
        print("  - Minimal difference in data preservation")
        print("  - Method 1 may be preferred for theoretical simplicity")

    # Save results to file
    output_file = Path(__file__).parent / "fitting_method_comparison.txt"
    with open(output_file, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("COMPREHENSIVE COMPARISON: Beta Fitting Methods\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Grand Total: {grand_total} estimates\n")
        f.write(f"Method 1 success: {grand_m1}/{grand_total} ({100*grand_m1/grand_total:.1f}%)\n")
        f.write(f"Method 2 success: {grand_m2}/{grand_total} ({100*grand_m2/grand_total:.1f}%)\n")
        f.write(f"Data rescued by Method 2: {grand_rescued}/{grand_total} ({100*grand_rescued/grand_total:.1f}%)\n")
        f.write(f"\nRecommendation: {'Use Method 2' if rescue_pct > 2 else 'Either method acceptable'}\n")

    print(f"\n✓ Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
