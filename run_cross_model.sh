#!/bin/bash

MODELS=(
    "claude-sonnet-4-5"
    "claude-haiku-4-5"
    "claude-sonnet-4-6"
    "gpt-4o"
    "gpt-4o-mini"
    "o4-mini"
    "o3-mini"
    "gemini-2.5-flash"
    "gemini-2.5-pro"
    "gemini-2.5-flash-lite"
)

for model in "${MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "  Running model: $model"
    echo "=========================================="

    # Swap the model line in config.yaml
    sed -i '' "s|^  model: .*|  model: \"$model\"|" config.yaml

    python3 src/main.py -c config.yaml

    echo "  Done: $model"
    echo ""
done

# Reset back to default
sed -i '' 's|^  model: .*|  model: "claude-sonnet-4-5"|' config.yaml
echo "All models done. Config reset to claude-sonnet-4-5."
