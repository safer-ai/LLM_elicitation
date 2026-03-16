#!/bin/bash
# Simple script to run ONE experiment run and organize output
# Usage: ./run_one.sh <config_file> <experiment_name> <run_number>
# Example: ./run_one.sh config_claude_TA0002.yaml percentile_claude_TA0002_50pct 1

set -e

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <config_file> <experiment_name> <run_number>"
    echo "Example: $0 config_claude_TA0002.yaml percentile_claude_TA0002_50pct 1"
    exit 1
fi

CONFIG_FILE="$1"
EXPERIMENT_NAME="$2"
RUN_NUMBER="$3"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

# Create experiment directory
EXPERIMENT_DIR="output_data/experiments/${EXPERIMENT_NAME}"
mkdir -p "$EXPERIMENT_DIR"

echo "========================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "Run Number: $RUN_NUMBER"
echo "Config: $CONFIG_FILE"
echo "========================================"

# Run the pipeline
python3 src/main.py -c "$CONFIG_FILE"

# Find the most recent run directory
LATEST_RUN=$(ls -t output_data/runs/ | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "Error: Could not find run output!"
    exit 1
fi

# Move and rename
TARGET_DIR="${EXPERIMENT_DIR}/run_${RUN_NUMBER}_${LATEST_RUN}"
mv "output_data/runs/${LATEST_RUN}" "$TARGET_DIR"

echo ""
echo "✓ SUCCESS!"
echo "Results saved to: $TARGET_DIR"
echo ""
echo "To check output:"
echo "  cat ${TARGET_DIR}/detailed_estimates.csv | head -15"
echo ""
