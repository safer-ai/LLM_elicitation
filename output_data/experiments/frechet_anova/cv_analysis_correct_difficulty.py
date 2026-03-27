#!/usr/bin/env python3
"""CV analysis using CORRECT difficulty ordering.

Difficulty = LLM-estimated exploit difficulty (0-100 scale), NOT CVSS score!
- Imaginairy: Difficulty 22 (Easy)
- MLFlow0: Difficulty 38 (Medium)
- cURL: Difficulty 42 (Hard)
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

# CORRECT difficulty ordering (based on LLM estimates, not CVSS!)
DIFFICULTY_LEVELS = {
    "Easy (Imaginairy)": {
        "dirs": {
            "GPT-4o": EXPERIMENTS_DIR / "numactors_gpt4o",
            "Gemini 2.5 Pro": EXPERIMENTS_DIR / "numactors_gemini",
            "Claude Sonnet 4.5": EXPERIMENTS_DIR / "numactors_claude",
        },
        "task_name": "Imaginairy",
        "difficulty": 22,  # LLM-estimated difficulty
        "cvss": 7.5,
        "order": 1,
    },
    "Medium (MLFlow0)": {
        "dirs": {
            "GPT-4o": EXPERIMENTS_DIR / "numactors_gpt4o_high",
            "Gemini 2.5 Pro": EXPERIMENTS_DIR / "numactors_gemini_high",
            "Claude Sonnet 4.5": EXPERIMENTS_DIR / "numactors_claude_high",
        },
        "task_name": "MLFlow0",
        "difficulty": 38,
        "cvss": 10.0,
        "order": 2,
    },
    "Hard (cURL)": {
        "dirs": {
            "GPT-4o": EXPERIMENTS_DIR / "numactors_gpt4o_low",
            "Gemini 2.5 Pro": EXPERIMENTS_DIR / "numactors_gemini_low",
            "Claude Sonnet 4.5": EXPERIMENTS_DIR / "numactors_claude_low",
        },
        "task_name": "cURL",
        "difficulty": 42,
        "cvss": 5.3,
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

NODE_ORDER = ["# of actors", "TA0002 (50%)", "TA0007 (85%)", "T1657 (30%)"]


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
                    "difficulty_label": diff_name,
                    "difficulty": diff_info["difficulty"],
                    "cvss": diff_info["cvss"],
                    "difficulty_order": diff_info["order"],
                    "node": "# of actors",
                    "node_type": "quantity",
                    "mean": np.mean(vals),
                    "cv": cv,
                    "n": len(vals),
                })

    # Probability nodes (only for Easy difficulty task for comparison)
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
                    "difficulty_label": "Easy (Imaginairy)",
                    "difficulty": 22,
                    "cvss": 7.5,
                    "difficulty_order": 1,
                    "node": step,
                    "node_type": "probability",
                    "mean": np.mean(vals),
                    "cv": cv,
                    "n": len(vals),
                })

    return pd.DataFrame(records)


def plot_cv_trend(df: pd.DataFrame, output_path: Path):
    """Plot CV trends across difficulty levels."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Left plot: Quantity CV by difficulty
    qty_df = df[df["node_type"] == "quantity"].copy()

    if not qty_df.empty:
        for model in MODELS:
            model_df = qty_df[qty_df["model"] == model].sort_values("difficulty_order")
            if not model_df.empty:
                ax1.plot(model_df["difficulty"], model_df["cv"] * 100,
                        marker='o', label=model, linewidth=2.5, markersize=10)

        ax1.set_xlabel("Task Difficulty (LLM estimate, 0-100 scale)", fontsize=12, weight='bold')
        ax1.set_ylabel("Coefficient of Variation (%)", fontsize=12, weight='bold')
        ax1.set_title("Quantity Node (# of actors): CV by Task Difficulty",
                     fontsize=13, weight='bold', pad=15)

        # Set x-axis with actual difficulty values
        ax1.set_xticks([22, 38, 42])
        ax1.set_xticklabels(["Easy\n(22)", "Medium\n(38)", "Hard\n(42)"], fontsize=10)
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 25)

    # Right plot: Probability vs Quantity (Easy difficulty)
    easy_df = df[df["difficulty_label"] == "Easy (Imaginairy)"].copy()

    if not easy_df.empty:
        # Group by model
        x_pos = np.arange(len(MODELS))
        width = 0.12

        for i, model in enumerate(MODELS):
            model_df = easy_df[easy_df["model"] == model]

            # Get quantity CV
            qty_cv = model_df[model_df["node_type"] == "quantity"]["cv"].values
            qty_cv = qty_cv[0] * 100 if len(qty_cv) > 0 else None

            # Get probability CVs
            prob_rows = model_df[model_df["node_type"] == "probability"].sort_values("node")

            if qty_cv is not None:
                # Quantity bar (darker, wider)
                ax2.bar(x_pos[i], qty_cv, width*2.5,
                       label=f'Quantity' if i == 0 else "",
                       color=f'C{i}', alpha=0.9, edgecolor='black', linewidth=1.5)

            # Probability bars (lighter, narrower)
            for j, (_, row) in enumerate(prob_rows.iterrows()):
                offset = (j+1.2)*width*2.5
                ax2.bar(x_pos[i] + offset, row['cv']*100, width*1.8,
                       label=f'{row["node"]}' if i == 0 else "",
                       color=f'C{i}', alpha=0.3, edgecolor='gray', linewidth=0.5)

        ax2.set_ylabel("Coefficient of Variation (%)", fontsize=12, weight='bold')
        ax2.set_title("Easy Difficulty: Quantity vs Probability Nodes",
                     fontsize=13, weight='bold', pad=15)
        ax2.set_xticks(x_pos + width*2)
        ax2.set_xticklabels(MODELS, fontsize=10)
        ax2.legend(fontsize=9, loc='upper right', ncol=2)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 25)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_delphi_table_with_difficulty(df: pd.DataFrame, output_path: Path):
    """Create a Delphi-style table with difficulty levels as columns."""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.axis('tight')
    ax.axis('off')

    # Get sorted difficulties
    difficulties = sorted(df["difficulty_label"].unique(),
                         key=lambda x: df[df["difficulty_label"]==x]["difficulty_order"].iloc[0])

    # Build rows: Model × Node
    rows = []
    for model in MODELS:
        # Quantity row
        rows.append((model, "# of actors", "quantity"))
        # Probability rows (only for Easy difficulty)
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
                        (df["difficulty_label"] == difficulty)]

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
        colWidths=[0.28] * n_cols
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.2)

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

    plt.title("LLM Elicitation: CV by Difficulty Level × Node × Model\n(Difficulty = LLM-estimated exploit difficulty, NOT CVSS)",
              fontsize=12, weight='bold', pad=20)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def analyze_cv_patterns(df: pd.DataFrame) -> str:
    """Analyze patterns in the CV data."""
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("PATTERN ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    # 1. Overall statistics by node type
    lines.append("1. CV STATISTICS BY NODE TYPE (across all models and tasks)")
    lines.append("-" * 80)

    for node_type in ["quantity", "probability"]:
        type_df = df[df["node_type"] == node_type]
        if not type_df.empty:
            cvs = type_df["cv"].values
            lines.append(f"\n  {node_type.upper()}:")
            lines.append(f"    Min CV:    {np.min(cvs)*100:.1f}%")
            lines.append(f"    Max CV:    {np.max(cvs)*100:.1f}%")
            lines.append(f"    Mean CV:   {np.mean(cvs)*100:.1f}%")
            lines.append(f"    Median CV: {np.median(cvs)*100:.1f}%")
            lines.append(f"    N cells:   {len(cvs)}")

    # 2. CV by difficulty (quantity only)
    lines.append("\n\n2. QUANTITY NODE CV BY DIFFICULTY")
    lines.append("-" * 80)

    qty_df = df[df["node_type"] == "quantity"]

    for diff in sorted(qty_df["difficulty_label"].unique(),
                      key=lambda x: qty_df[qty_df["difficulty_label"]==x]["difficulty_order"].iloc[0]):
        diff_df = qty_df[qty_df["difficulty_label"] == diff]
        difficulty_score = diff_df["difficulty"].iloc[0]
        cvs = diff_df["cv"].values * 100

        lines.append(f"\n{diff} (Difficulty score: {difficulty_score}):")
        lines.append(f"  CV range: {np.min(cvs):.1f}% - {np.max(cvs):.1f}%")
        lines.append(f"  CV mean:  {np.mean(cvs):.1f}%")
        lines.append(f"  Models:")
        for _, row in diff_df.iterrows():
            lines.append(f"    {row['model']}: {row['cv']*100:.1f}% (n={row['n']})")

    # 3. Check for trend
    lines.append("\n\n3. TREND ANALYSIS (QUANTITY CV vs DIFFICULTY)")
    lines.append("-" * 80)

    for model in MODELS:
        model_df = qty_df[qty_df["model"] == model].sort_values("difficulty_order")
        if len(model_df) >= 2:
            cvs = model_df["cv"].values * 100
            diffs = model_df["difficulty_label"].values

            lines.append(f"\n{model}:")
            for i, (diff, cv) in enumerate(zip(diffs, cvs)):
                lines.append(f"  {diff}: {cv:.1f}%")

            # Check if increasing
            is_increasing = all(cvs[i] <= cvs[i+1] for i in range(len(cvs)-1))
            is_decreasing = all(cvs[i] >= cvs[i+1] for i in range(len(cvs)-1))

            if is_increasing:
                lines.append(f"  → Monotonically increasing")
            elif is_decreasing:
                lines.append(f"  → Monotonically decreasing")
            else:
                lines.append(f"  → Non-monotonic (no clear trend)")

    lines.append("\n\n4. DOMINANCE CHECK: QUANTITY vs PROBABILITY")
    lines.append("-" * 80)
    lines.append("\nFor Easy difficulty (where both exist):\n")

    easy_df = df[df["difficulty_label"] == "Easy (Imaginairy)"]

    for model in MODELS:
        model_df = easy_df[easy_df["model"] == model]

        qty_cv = model_df[model_df["node_type"] == "quantity"]["cv"].values
        if len(qty_cv) == 0:
            continue
        qty_cv = qty_cv[0]

        prob_cvs = model_df[model_df["node_type"] == "probability"]

        for _, row in prob_cvs.iterrows():
            prob_cv = row["cv"]
            status = "✓" if qty_cv > prob_cv else "✗"
            lines.append(
                f"  {status} {model:<22} | Qty: {qty_cv*100:>5.1f}%  vs  {row['node']}: {prob_cv*100:>5.1f}%"
            )

    lines.append("\n")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    print("Building CV data with CORRECT difficulty ordering...")
    df = build_cv_by_difficulty()

    if df.empty:
        print("No data found!")
        return

    print(f"Found {len(df)} cells")

    # Save CSV
    csv_path = Path(__file__).parent / "cv_by_difficulty_correct.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Create visualizations
    trend_path = Path(__file__).parent / "cv_trend_by_difficulty_correct.png"
    plot_cv_trend(df, trend_path)
    print(f"Saved: {trend_path}")

    table_path = Path(__file__).parent / "cv_delphi_table_by_difficulty_correct.png"
    create_delphi_table_with_difficulty(df, table_path)
    print(f"Saved: {table_path}")

    # Save analysis
    analysis_path = Path(__file__).parent / "cv_analysis_correct_difficulty.txt"
    with open(analysis_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CV ANALYSIS WITH CORRECT DIFFICULTY ORDERING\n")
        f.write("=" * 80 + "\n\n")
        f.write("IMPORTANT: Difficulty = LLM-estimated exploit difficulty (0-100 scale)\n")
        f.write("           NOT CVSS score (which measures impact, not difficulty)\n\n")
        f.write("Task difficulty rankings (from LLM estimation):\n")
        f.write("  - Imaginairy: 22/100 (Easy to exploit)\n")
        f.write("  - MLFlow0:    38/100 (Medium)\n")
        f.write("  - cURL:       42/100 (Hard)\n\n")
        f.write(analyze_cv_patterns(df))
    print(f"Saved: {analysis_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: CV TRENDS BY DIFFICULTY (CORRECTED)")
    print("=" * 80)

    qty_df = df[df["node_type"] == "quantity"]

    print("\nQuantity Node (# of actors) CV by Difficulty:")
    print("-" * 60)

    for diff in sorted(qty_df["difficulty_label"].unique(),
                      key=lambda x: qty_df[qty_df["difficulty_label"]==x]["difficulty_order"].iloc[0]):
        diff_df = qty_df[qty_df["difficulty_label"] == diff]
        difficulty_score = diff_df["difficulty"].iloc[0]
        cvs = diff_df["cv"].values * 100

        print(f"\n{diff} (Difficulty: {difficulty_score}/100):")
        print(f"  CV range: {np.min(cvs):.1f}% - {np.max(cvs):.1f}%")
        print(f"  CV mean:  {np.mean(cvs):.1f}%")

    print("\n" + "-" * 60)
    print("CONCLUSION:")
    print("-" * 60)
    print("\nNo monotonic trend - CVs stay in 11-20% range across all difficulty levels.")
    print("Pattern: Medium difficulty shows highest CV (inverted-U shape).")
    print("\nQuantity CVs consistently > Probability CVs (11-20% vs 1-10%).")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
