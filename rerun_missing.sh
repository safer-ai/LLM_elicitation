#!/bin/bash
# Rerun only the missing experiments to get exactly 10 complete runs per baseline/model

set -e

echo "=== Rerunning Missing Experiments ==="
echo ""

# Function to run one experiment
run_experiment() {
  config="$1"
  experiment="$2"
  run_num="$3"

  echo "----------------------------------------"
  echo "Running: $experiment (run $run_num)"
  echo "Config: $config"
  echo "----------------------------------------"

  ./run_one.sh "$config" "$experiment" "$run_num"

  echo ""
}

# Claude TA0007_85pct - need 3 more runs (currently has 7, need runs 11, 12, 13)
run_experiment "config_claude_TA0007.yaml" "percentile_claude_TA0007_85pct" 11
run_experiment "config_claude_TA0007.yaml" "percentile_claude_TA0007_85pct" 12
run_experiment "config_claude_TA0007.yaml" "percentile_claude_TA0007_85pct" 13

# Gemini TA0007_85pct - need 3 more runs (currently has 7, need runs 11, 12, 13)
run_experiment "config_gemini_TA0007.yaml" "percentile_gemini_TA0007_85pct" 11
run_experiment "config_gemini_TA0007.yaml" "percentile_gemini_TA0007_85pct" 12
run_experiment "config_gemini_TA0007.yaml" "percentile_gemini_TA0007_85pct" 13

# GPT4o TA0007_85pct - need 1 more run (currently has 9, need run 11)
run_experiment "config_gpt4o_TA0007.yaml" "percentile_gpt4o_TA0007_85pct" 11

echo "========================================"
echo "All missing experiments completed!"
echo "========================================"
