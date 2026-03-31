# Quick Start: Inter-Benchmark Calibration

## Step 1: Prepare Data

### 1a. Prepare sorted benchmark files

For each benchmark (source and target), create a sorted YAML file with tasks in ascending difficulty. For benchmarks with per-task solve data, generate them from source data:

**For BountyBench (LLM-based difficulty estimation)**:
```bash
cd difficulty_estimation
pixi run python estimate_difficulty.py --config config/config.yaml
# Output: benchmark_tasks/bountybench_ordered.yaml
```

**For LiveBench (per-task solve rates from HuggingFace)**:
```bash
cd difficulty_estimation/benchmark_tasks
python livebench_processing.py
# Outputs:
#   - livebench_LCB_generation_ordered.yaml
#   - livebench_coding_completion_ordered.yaml
```

### 1b. Create per-benchmark score files

Create one JSON per benchmark in `input_data/scores/`:

```json
{
  "benchmark_name": "cybench",
  "models": [
    {"model": "GPT-4o", "score": 12.5},
    {"model": "Claude 3.5 Sonnet", "score": 17.5}
  ]
}
```

**Important**: Model name strings must match across files for the join to work.

### 1c. Compute ground truth

**Mode 1: Score-based (overall model performance)**:
```bash
python compute_ground_truth.py \
  --source-scores input_data/scores/cybench.json \
  --target-scores input_data/scores/swebench_verified.json \
  --source-bins "auto" --n-source-bins 4 \
  --target-percentiles "30,40,50,60" \
  --output input_data/ground_truth/cybench_to_swebench_verified_gt.json
```

**Mode 2: Per-task based (per-task solve rates)**:
```bash
python compute_ground_truth_per_task.py \
  --source-leaderboard ../difficulty_estimation/benchmark_tasks/livebench_LCB_generation_leaderboard.json \
  --target-solve-matrix ../difficulty_estimation/benchmark_tasks/livebench_coding_completion_solve_matrix.json \
  --target-ordered ../difficulty_estimation/benchmark_tasks/livebench_coding_completion_ordered.yaml \
  --source-bins "auto" --n-source-bins 2 \
  --target-percentiles "30" \
  --output input_data/ground_truth/livebench_lcb_to_cc_gt.json
```

For explicit source bins:
```bash
python compute_ground_truth_per_task.py \
  --source-leaderboard ... \
  --target-solve-matrix ... \
  --target-ordered ... \
  --source-bins "[[0.1,0.3],[0.3,0.5]]" \
  --target-percentiles "40,60"
```

## Step 2: Configure

Copy and edit the config to match your ground truth parameters:

```bash
cp config_example.yaml config.yaml
# Edit config.yaml:
# - Set API keys
# - Set benchmark file paths
# - IMPORTANT: source_bins and target_percentiles must match what was used in compute_ground_truth!
```

Example: If you computed ground truth with `--n-source-bins 2 --target-percentiles "30"`, then set:
```yaml
inter_benchmark_settings:
  source_bins: "auto"
  n_source_bins: 2
  
  target_benchmark:
    target_percentiles: [30]
```

**Critical**: Config parameters and ground truth must be aligned, or the workflow will fail with mismatched bin/percentile counts.

## Step 3: Run

```bash
python run_calibration.py -c config.yaml
# Debug mode:
python run_calibration.py -c config.yaml -d
```

Results are saved progressively to `results/<source>_to_<target>/<run_id>/`.

## Step 4: Plot

```bash
python plot_calibration.py -j results/<source>_to_<target>/<run_id>/<results_file>.json
# With interactive display:
python plot_calibration.py -j <path> -s
```

## Common Issues

- **"No overlapping models"**: Model names don't match across score files. Ensure consistent naming.
- **"No ground truth entries with sufficient sample"**: Too few models per source bin. Try fewer bins with `--n-source-bins` or lower `--min-sample-size`.
- **Empty source bins**: Some bins may have no models if the score distribution is uneven. Use explicit `--source-bins` to target the populated range.
- **"Mismatch between config and ground truth"**: The source_bins and target_percentiles in your config don't match the pre-computed ground truth file. Regenerate ground truth with matching parameters or update the config.

## Example: LiveBench LCB_generation → Coding_Completion

```bash
# 1. Generate data from HuggingFace
cd difficulty_estimation/benchmark_tasks
python livebench_processing.py
cd ../../inter_benchmark_calibration

# 2. Compute ground truth (2 source bins, 1 target percentile for testing)
python compute_ground_truth_per_task.py \
  --source-leaderboard ../difficulty_estimation/benchmark_tasks/livebench_LCB_generation_leaderboard.json \
  --target-solve-matrix ../difficulty_estimation/benchmark_tasks/livebench_coding_completion_solve_matrix.json \
  --target-ordered ../difficulty_estimation/benchmark_tasks/livebench_coding_completion_ordered.yaml \
  --source-bins "auto" --n-source-bins 2 \
  --target-percentiles "30" \
  --output input_data/ground_truth/livebench_lcb_to_cc_gt.json

# 3. Configure (edit config_livebench.yaml to match)

# 4. Run calibration
python run_calibration.py -c config_livebench.yaml

# 5. Plot results
python plot_calibration.py -j results/livebench_LCB_generation_to_livebench_coding_completion/*/estimates.json
```
