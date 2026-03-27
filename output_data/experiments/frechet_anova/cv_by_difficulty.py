#!/usr/bin/env python3
"""Analyze CV trends across difficulty levels for quantity nodes.

Checks whether CV increases with task difficulty.
"""

import csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPERIMENTS_DIR = Path(__file__).parent.parent

# Map difficulty level to data directory and task metadata
DIFFICULTY_LEVELS = {
    "Low (cURL)": {
        "dirs": {
            "GPT-4o": EXPERIMENTS_DIR / "numactors_gpt4o_low",
            "Gemini 2.5 Pro": EXPERIMENTS_DIR / "numactors_gemini_low",
            "Claude Sonnet 4.5": EXPERIMENTS_DIR / "numactors_claude_low",
        },
        "task_name": "cURL",
        "cvss": 5.3,
        "order": 1,
    },
    "Medium (Imaginairy)": {
        "dirs": {
            "GPT-4o": EXPERIMENTS_DIR / "numactors_gpt4o",
            "Gemini 2.5 Pro": EXPERIMENTS_DIR / "numactors_gemini",
            "Claude Sonnet 4.5": EXPERIMENTS_DIR / "numactors_claude",
        },
        "task_name": "Imaginairy",
        "cvss": 7.5,
        "order": 2,
    },
    "High (MLFlow0)": {
        "dirs": {
            "GPT-4o": EXPERIMENTS_DIR / "numactors_gpt4o_high",
            "Gemini 2.5 Pro": EXPERIMENTS_DIR / "numactors_gemini_high",
            "Claude Sonnet 4.5": EXPERIMENTS_DIR / "numactors_claude_high",
        },
        "task_name": "MLFlow0",
        "cvss": 10.0,
        "order": 3,
    },
}

MODELS = ["GPT-4o", "Gemini 2.5 Pro", "Claude Sonnet 4.5"]

# Probability directories (for comparison)
PROB_DIRS = {
    ("Claude Sonnet 4.5", "TA0002 (50%)"): EXPERIMENTS_DIR / "anova_probability",
    ("Claude Sonnet 4.5", "TA0007 (85%)"): EXPERIMENTS_DIR / "pilot_experiments" / "cross_step_TA0007_85pct",
    ("Claude Sonnet 4.5", "T1657 (30%)"): EXPERIMENTS_DIR / "pilot_experiments" / "cross_step_T1657_30pct",
    ("GPT-4o", "TA0002 (50%)"): EXPERIMENTS_DIR / "pilot_experiments" / "cross_model_gpt4o",
    ("GPT-4o", "TA0007 (85%)"): EXPERIMENTS_DIR / "pilot_experiments" / "cross_model_gpt4o_TA0007_85pct",
    ("GPT-4o", "T1657 (30%)"): EXPERIMENTS_DIR / "pilot_experiments" / "cross_model_gpt4o_T1657_30pct",
    ("Gemini 2.5 Pro", "TA0002 (50%)"): EXPERIMENTS_DIR / "pilot_experiments" / "cross_model_gemini_TA0002_50pct",
    ("Gemini 2.5 Pro", "TA0007 (85%)"): EXPERIMENTS_DIR / "pilot_experiments" / "cross_model_gemini_TA0007_85pct",
    ("Gemini 2.5 Pro", "T1657 (30%)"): EXPERIMENTS_DIR / "pilot_experiments" / "cross_model_gemini_T1657_30pct",
}

PROB_STEPS = ["TA0002 (50%)", "TA0007 (85%)", "T1657 (30%)"]


def _load_raw_quantity_estimates(data_dir: Path) -> pd.DataFrame:
    """Load point estimates from numactors CSVs."""
    rows = []
    if not data_dir.exists():
        return pd.DataFrame(columns=["estimate"])

    run_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    for run_dir in run_dirs:
        csv_path = run_dir / "detailed_estimates.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                if row.get("has_error", "").strip().lower() == "true":
                    continue
                try:
                    est = float(row["estimate"].strip())
                    rows.append({"estimate": est})
                except (ValueError, KeyError):
                    continue
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["estimate"])


def _load_raw_probability_estimates(data_dir: Path) -> pd.DataFrame:
    """Load point estimates from probability CSVs."""
    rows = []
    if not data_dir.exists():
        return pd.DataFrame(columns=["estimate"])

    run_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    for run_dir in run_dirs:
        csv_path = run_dir / "detailed_estimates.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                if row.get("has_error", "").strip().lower() == "true":
                    continue
                try:
                    est = float(row["most_likely_estimate"].strip())
                    rows.append({"estimate": est})
                except (ValueError, KeyError):
                    continue
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["estimate"])


def _compute_cv(values: np.ndarray):
    """Compute coefficient of variation = std / mean."""
    if len(values) < 2:
        return None
    mean = np.mean(values)
    if mean == 0.0:
        return None
    return float(np.std(values, ddof=1) / mean)


def build_cv_by_difficulty() -> pd.DataFrame:
    """Build DataFrame with CV for each (model, difficulty, node) combination."""
    records = []

    # Quantity nodes across difficulty levels
    for diff_name, diff_info in sorted(DIFFICULTY_LEVELS.items(), key=lambda x: x[1]["order"]):
        for model in MODELS:
            data_dir = diff_info["dirs"].get(model)
            if data_dir is None:
                continue

            df = _load_raw_quantity_estimates(data_dir)
            if df.empty or len(df) < 2:
                continue

            vals = df["estimate"].values
            cv = _compute_cv(vals)
            if cv is not None:
                records.append({
                    "model": model,
                    "difficulty": diff_name,
                    "cvss": diff_info["cvss"],
                    "difficulty_order": diff_info["order"],
                    "node": "# of actors",
                    "node_type": "quantity",
                    "mean": np.mean(vals),
                    "cv": cv,
                    "n": len(vals),
                })

    # Probability nodes (for comparison - use Medium difficulty task "Imaginairy")
    for step in PROB_STEPS:
        for model in MODELS:
            prob_dir = PROB_DIRS.get((model, step))
            if prob_dir is None:
                continue

            df = _load_raw_probability_estimates(prob_dir)
            if df.empty or len(df) < 2:
                continue

            vals = df["estimate"].values
            cv = _compute_cv(vals)
            if cv is not None:
                records.append({
                    "model": model,
                    "difficulty": "Medium (Imaginairy)",
                    "cvss": 7.5,
                    "difficulty_order": 2,
                    "node": step,
                    "node_type": "probability",
                    "mean": np.mean(vals),
                    "cv": cv,
                    "n": len(vals),
                })

    return pd.DataFrame(records)


def plot_cv_trend(df: pd.DataFrame, output_path: Path):
    """Plot CV trends across difficulty levels."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Quantity CV by difficulty
    qty_df = df[df["node_type"] == "quantity"].copy()

    if not qty_df.empty:
        for model in MODELS:
            model_df = qty_df[qty_df["model"] == model].sort_values("difficulty_order")
            if not model_df.empty:
                ax1.plot(model_df["difficulty_order"], model_df["cv"] * 100,
                        marker='o', label=model, linewidth=2, markersize=8)

        ax1.set_xlabel("Task Difficulty (CVSS score)", fontsize=11, weight='bold')
        ax1.set_ylabel("Coefficient of Variation (%)", fontsize=11, weight='bold')
        ax1.set_title("Quantity Node (# of actors): CV by Task Difficulty",
                     fontsize=12, weight='bold')
        ax1.set_xticks([1, 2, 3])
        ax1.set_xticklabels(["Low\n(CVSS 5.3)", "Medium\n(CVSS 7.5)", "High\n(CVSS 10)"])
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

    # Right plot: Probability vs Quantity (Medium difficulty)
    med_df = df[df["difficulty"] == "Medium (Imaginairy)"].copy()

    if not med_df.empty:
        # Group by model and node_type
        plot_data = []
        for model in MODELS:
            model_df = med_df[med_df["model"] == model]

            # Get quantity CV
            qty_cv = model_df[model_df["node_type"] == "quantity"]["cv"].values
            qty_cv = qty_cv[0] * 100 if len(qty_cv) > 0 else None

            # Get probability CVs
            prob_cvs = model_df[model_df["node_type"] == "probability"]["cv"].values * 100

            if qty_cv is not None:
                plot_data.append({
                    "model": model,
                    "qty_cv": qty_cv,
                    "prob_cvs": prob_cvs,
                })

        x_pos = np.arange(len(plot_data))
        width = 0.15

        for i, pd_entry in enumerate(plot_data):
            # Quantity bar
            ax2.bar(x_pos[i], pd_entry["qty_cv"], width*2,
                   label=f'{pd_entry["model"]} (Qty)' if i == 0 else "",
                   color=f'C{i}', alpha=0.8)

            # Probability bars (stacked next to it)
            for j, prob_cv in enumerate(pd_entry["prob_cvs"]):
                ax2.bar(x_pos[i] + (j+1)*width*2.2, prob_cv, width*1.5,
                       color=f'C{i}', alpha=0.3 + j*0.2)

        ax2.set_ylabel("Coefficient of Variation (%)", fontsize=11, weight='bold')
        ax2.set_title("Medium Difficulty: Quantity vs Probability Nodes",
                     fontsize=12, weight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([pd["model"] for pd in plot_data], rotation=15, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_delphi_table_with_difficulty(df: pd.DataFrame, output_path: Path):
    """Create a Delphi-style table with difficulty levels as columns."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    # Get sorted difficulties
    difficulties = sorted(df["difficulty"].unique(),
                         key=lambda x: df[df["difficulty"]==x]["difficulty_order"].iloc[0])

    # Build rows: Model × Node
    rows = []
    for model in MODELS:
        # Quantity row
        rows.append((model, "# of actors", "quantity"))
        # Probability rows (only for Medium difficulty)
        for step in PROB_STEPS:
            if any((df["model"] == model) & (df["node"] == step)):
                rows.append((model, step, "probability"))

    n_rows = len(rows)
    n_cols = len(difficulties)

    # Collect all CVs for color scaling
    all_cvs = df["cv"].values
    vmax = max(all_cvs) * 1.1 if len(all_cvs) > 0 else 0.2
    vmax = max(vmax, 0.05)

    cmap = plt.cm.RdYlGn_r

    # Build cell data
    cell_text = []
    cell_colors = []
    row_labels = []

    for model, node, node_type in rows:
        row_text = []
        row_colors = []

        for difficulty in difficulties:
            # Get data for this cell
            cell_df = df[(df["model"] == model) &
                        (df["node"] == node) &
                        (df["difficulty"] == difficulty)]

            if not cell_df.empty:
                row_data = cell_df.iloc[0]
                mean = row_data["mean"]
                cv = row_data["cv"]
                n = row_data["n"]

                # Format cell
                if node_type == "probability":
                    mean_str = f"{mean*100:.1f}%"
                else:
                    mean_str = f"{mean:.2f}"

                cell_str = f"{mean_str}\nCV {cv*100:.0f}%\nn={n}"
                row_text.append(cell_str)

                # Color by CV
                color_val = cv / vmax if vmax > 0 else 0.5
                row_colors.append(cmap(color_val))
            else:
                row_text.append("—")
                row_colors.append('white')

        cell_text.append(row_text)
        cell_colors.append(row_colors)
        row_labels.append(f"{model}\n{node}")

    # Create table
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=difficulties,
        cellColours=cell_colors,
        cellLoc='center',
        loc='center',
        colWidths=[0.25] * n_cols
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.0)

    # Style headers
    for i in range(n_cols):
        cell = table[(0, i)]
        cell.set_facecolor('#40466e')
        cell.set_text_props(weight='bold', color='white', fontsize=9)

    # Style row labels
    prev_model = None
    for i, (model, node, node_type) in enumerate(rows):
        cell = table[(i+1, -1)]
        if model != prev_model:
            cell.set_facecolor('#e8e8e8')
            cell.set_text_props(weight='bold', ha='left', fontsize=7)
        else:
            cell.set_facecolor('#f5f5f5')
            cell.set_text_props(ha='left', fontsize=7)
        prev_model = model

    plt.title("LLM Elicitation: CV by Difficulty Level × Node × Model",
              fontsize=13, weight='bold', pad=20)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    print("Building CV data across difficulty levels...")
    df = build_cv_by_difficulty()

    if df.empty:
        print("No data found!")
        return

    print(f"Found {len(df)} cells")

    # Save CSV
    csv_path = Path(__file__).parent / "cv_by_difficulty.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Create visualizations
    trend_path = Path(__file__).parent / "cv_trend_by_difficulty.png"
    plot_cv_trend(df, trend_path)
    print(f"Saved: {trend_path}")

    table_path = Path(__file__).parent / "cv_delphi_table_by_difficulty.png"
    create_delphi_table_with_difficulty(df, table_path)
    print(f"Saved: {table_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: CV TRENDS BY DIFFICULTY")
    print("=" * 80)

    qty_df = df[df["node_type"] == "quantity"]

    print("\nQuantity Node (# of actors) CV by Difficulty:")
    print("-" * 60)

    for diff in sorted(qty_df["difficulty"].unique(),
                      key=lambda x: qty_df[qty_df["difficulty"]==x]["difficulty_order"].iloc[0]):
        diff_df = qty_df[qty_df["difficulty"] == diff]
        cvss = diff_df["cvss"].iloc[0]
        cvs = diff_df["cv"].values * 100

        print(f"\n{diff} (CVSS {cvss}):")
        print(f"  CV range: {np.min(cvs):.1f}% - {np.max(cvs):.1f}%")
        print(f"  CV mean:  {np.mean(cvs):.1f}%")
        print(f"  Models:")
        for _, row in diff_df.iterrows():
            print(f"    {row['model']}: {row['cv']*100:.1f}% (n={row['n']})")

    # Check for trend
    print("\n" + "-" * 60)
    print("TREND ANALYSIS:")
    print("-" * 60)

    for model in MODELS:
        model_df = qty_df[qty_df["model"] == model].sort_values("difficulty_order")
        if len(model_df) >= 2:
            cvs = model_df["cv"].values * 100
            diffs = model_df["difficulty"].values

            print(f"\n{model}:")
            for i, (diff, cv) in enumerate(zip(diffs, cvs)):
                print(f"  {diff}: {cv:.1f}%")

            # Check if increasing
            is_increasing = all(cvs[i] <= cvs[i+1] for i in range(len(cvs)-1))
            is_decreasing = all(cvs[i] >= cvs[i+1] for i in range(len(cvs)-1))

            if is_increasing:
                print(f"  → Monotonically increasing ✓")
            elif is_decreasing:
                print(f"  → Monotonically decreasing")
            else:
                print(f"  → Non-monotonic (mixed trend)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
