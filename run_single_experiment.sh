#!/bin/bash
# Script to run a single experiment 10 times
# Usage: ./run_single_experiment.sh <config_file> <experiment_name>
# Example: ./run_single_experiment.sh config_claude_TA0002.yaml percentile_claude_TA0002_50pct

set -e  # Exit on error

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <config_file> <experiment_name>"
    echo "Example: $0 config_claude_TA0002.yaml percentile_claude_TA0002_50pct"
    exit 1
fi

CONFIG_FILE="$1"
EXPERIMENT_NAME="$2"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

# Create experiment directory
EXPERIMENT_DIR="output_data/experiments/${EXPERIMENT_NAME}"
mkdir -p "$EXPERIMENT_DIR"

echo "========================================"
echo "Starting Experiment: $EXPERIMENT_NAME"
echo "Config: $CONFIG_FILE"
echo "Experiment Directory: $EXPERIMENT_DIR"
echo "========================================"
echo ""

# Run the experiment 10 times
for i in {1..10}; do
    echo "----------------------------------------"
    echo "Run $i/10 for $EXPERIMENT_NAME"
    echo "----------------------------------------"

    # Run the pipeline
    python3 src/main.py -c "$CONFIG_FILE"

    # Find the most recent run directory
    LATEST_RUN=$(ls -t output_data/runs/ | head -1)

    if [ -z "$LATEST_RUN" ]; then
        echo "Error: Could not find run output!"
        exit 1
    fi

    # Move and rename the run directory
    mv "output_data/runs/${LATEST_RUN}" "${EXPERIMENT_DIR}/run_${i}_${LATEST_RUN}"

    echo "✓ Run $i completed and saved to ${EXPERIMENT_DIR}/run_${i}_${LATEST_RUN}"
    echo ""

    # Small delay between runs to ensure unique timestamps
    sleep 2
done

echo "========================================"
echo "Experiment Complete: $EXPERIMENT_NAME"
echo "Total runs: 10"
echo "All results saved to: $EXPERIMENT_DIR"
echo "========================================"
