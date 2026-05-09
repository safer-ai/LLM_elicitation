#!/bin/bash
# Temperature-variance experiment.
# For each (model, step, temperature), runs 10 elicitations using a single
# fixed persona (Academic Security Researcher). Reuses run_single_experiment.sh
# to do the 10x execution and output organization.
#
# Usage:
#   ./run_temperature_sweep.sh                    # run everything
#   ./run_temperature_sweep.sh claude             # only Claude
#   ./run_temperature_sweep.sh gpt4o              # only GPT-4o
#   ./run_temperature_sweep.sh claude TA0002      # only Claude, TA0002

set -e

# Selectors (optional positional args)
MODEL_FILTER="${1:-all}"
STEP_FILTER="${2:-all}"

# Step name -> full label mapping (function for macOS bash 3.x compatibility)
step_full_name() {
    case "$1" in
        TA0002) echo "TA0002 - Execution" ;;
        TA0007) echo "TA0007 - Discovery" ;;
        T1657)  echo "T1657 - Impact: Financial Theft / Extortion" ;;
        *) echo "UNKNOWN" ;;
    esac
}

# Per-model temperature ranges (Anthropic API caps at 1.0).
CLAUDE_TEMPS=(0.0 0.25 0.5 0.75 1.0)
GPT4O_TEMPS=(0.0 0.5 1.0 1.5 2.0)

STEPS=("TA0002" "TA0007" "T1657")

run_one_combo() {
    local provider="$1"     # claude or gpt4o
    local step_short="$2"   # TA0002, TA0007, T1657
    local temp="$3"
    local config_file="config_${provider}_temperature.yaml"
    local step_full
    step_full=$(step_full_name "$step_short")
    local exp_name="temperature_${provider}_${step_short}_t${temp}"

    echo ""
    echo "##############################################"
    echo "# ${exp_name}"
    echo "# Step: ${step_full}"
    echo "# Temperature: ${temp}"
    echo "##############################################"

    # Patch temperature and scenario_steps in the config file in place.
    sed -i '' "s|^  temperature: .*|  temperature: ${temp}|" "$config_file"
    sed -i '' "s|^  scenario_steps: .*|  scenario_steps: [\"${step_full}\"]|" "$config_file"

    # Skip combos that are already complete (10 runs already saved).
    local exp_dir="output_data/experiments/${exp_name}"
    if [ -d "$exp_dir" ]; then
        local n_done
        n_done=$(find "$exp_dir" -maxdepth 1 -type d -name "run_*" | wc -l | tr -d ' ')
        if [ "$n_done" -ge 10 ]; then
            echo "  -> already complete (${n_done} runs), skipping"
            return
        fi
    fi

    ./run_single_experiment.sh "$config_file" "$exp_name"
}

dispatch_provider() {
    local provider="$1"
    local temps_var
    if [ "$provider" = "claude" ]; then
        temps_var=("${CLAUDE_TEMPS[@]}")
    else
        temps_var=("${GPT4O_TEMPS[@]}")
    fi

    for step in "${STEPS[@]}"; do
        if [ "$STEP_FILTER" != "all" ] && [ "$STEP_FILTER" != "$step" ]; then
            continue
        fi
        for temp in "${temps_var[@]}"; do
            run_one_combo "$provider" "$step" "$temp"
        done
    done
}

if [ "$MODEL_FILTER" = "all" ] || [ "$MODEL_FILTER" = "claude" ]; then
    dispatch_provider "claude"
fi
if [ "$MODEL_FILTER" = "all" ] || [ "$MODEL_FILTER" = "gpt4o" ]; then
    dispatch_provider "gpt4o"
fi

echo ""
echo "=========================================="
echo "  Temperature sweep complete."
echo "=========================================="
