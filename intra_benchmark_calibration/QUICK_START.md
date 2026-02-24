# Quick Start Guide: Intra-Benchmark Calibration

## Prerequisites

1. **Ground truth data exists**: `input_data/benchmark/scores/{benchmark}_ground_truth_n{nbins}.json`
2. **Python environment set up**: `pip install -r requirements.txt`
3. **API keys configured**: Edit `config_intra_benchmark.yaml`

## 3-Step Setup

### Step 1: Preprocess Benchmark (One-Time)

```bash
python prepare_benchmark.py <input_file> [--metric METRIC]
```

**Expected Output**:

```
============================================================
SORTING SUMMARY
============================================================
Total tasks: 42
FST range:   0.5 to 1554
First 5 tasks (easiest):
  1. Open Sesame (FST: 0.5)
  2. LootStash (FST: 2)
  ...
```

**Result**: Creates `input_data/benchmark/{benchmark}_ascending_fst.yaml`

### Step 2: Configure API Keys

Edit `config_intra_benchmark.yaml`:

```yaml
# Add your API key
anthropic_api_key: "sk-ant-..."  # OR
openai_api_key: "sk-..."

# Verify settings
llm_settings:
  model: "claude-sonnet-4-5-20250929"  # or "gpt-4o"
  max_concurrent_calls: 5
  rate_limit_calls: 45
  rate_limit_period: 60

workflow_settings:
  num_experts: 5
  delphi_rounds: 3
  convergence_threshold: 0.05
```

### Step 3: Run Experiment

```bash
python src/main.py -c config_intra_benchmark.yaml
```

**Expected Output**:

```
--- Starting Intra-Benchmark Calibration Experiment ---
Detected mode: intra_benchmark
Loaded 42 benchmark tasks
Loaded 10 expert profiles
Processing prediction 1/5: Bin 0 → 1
  Round 1/3 for pair (0, 1) starting...
  ...
Intra-Benchmark Calibration Workflow Completed
  Run ID: 20250107_143022
  Predictions completed: 5/5
  Output path: output_data/intra_benchmark/{benchmark}/20250107_143022/
```

## Results Location

```
output_data/intra_benchmark/{benchmark}/{run_id}/
├── detailed_estimates.csv    # All expert estimates by round
└── full_results.json          # Complete Delphi deliberation history
```

## Quick Tests

### Test 1: Verify Mode Detection

```bash
python -c "from src.main import detect_mode; print(detect_mode('config_intra_benchmark.yaml'))"
```

**Expected**: `intra_benchmark`

### Test 2: Verify Sorted Benchmark

```bash
python -c "from src.data_loader import load_benchmark; \
           from pathlib import Path; \
           b = load_benchmark(Path('input_data/benchmark/{benchmark}_ascending_fst.yaml')); \
           print(f'{len(b.tasks)} tasks'); \
           print(f'FST range: {b.tasks[0].metrics[\"fst\"]} to {b.tasks[-1].metrics[\"fst\"]}')"
```

**Expected**:

```
42 tasks
FST range: 0.5 to 1554
```

### Test 3: Verify Ground Truth

```bash
python -c "from src.intra_benchmark.data_loader import load_ground_truth; \
           gt = load_ground_truth('{benchmark}', 4); \
           print(f'{len(gt[\"ground_truth\"])} pairs with sufficient sample')"
```

**Expected**: `5 pairs with sufficient sample`

### Test 4: Dry Run (Fast Test)

Edit `config_intra_benchmark.yaml`:

```yaml
workflow_settings:
  num_experts: 1      # Just 1 expert
  delphi_rounds: 1    # Just 1 round
```

Run:

```bash
python src/main.py -c config_intra_benchmark.yaml
```

## Common Issues

### "Ground truth file not found"

**Fix**: Check that `input_data/benchmark/scores/{benchmark}_ground_truth_n{nbins}.json` exists

### "Failed to load benchmark from ..."

**Fix**: Run preprocessing: `python prepare_benchmark.py <input_file> [--metric METRIC]`

### "Missing prompt template"

**Fix**: Verify all 3 files exist in `input_data/prompts/intra_benchmark/`:

- `task_relationship_analysis.txt`
- `initial_conditional_probability_estimation.txt`
- `subsequent_conditional_probability_estimation.txt`

### "API rate limit exceeded (429)"

**Fix**: Lower concurrency in `config_intra_benchmark.yaml`:

```yaml
llm_settings:
  max_concurrent_calls: 3
  rate_limit_calls: 30
```

### "ImportError: No module named 'intra_benchmark'"

**Fix**: Run from project root:

```bash
cd /path/to/LLM_elicitation
python src/main.py -c config_intra_benchmark.yaml
```

## Verify Original Mode Still Works

```bash
python src/main.py -c config.yaml
```

Should run risk scenario workflow without errors.

## API Call Estimation

**Calculation**:

- Round 1: 2 calls/expert (analysis + estimation), subsequent rounds: 1 call/expert (refinement)
- Total pairs for n bins: n(n-1)/2
- `delphi_rounds` = total rounds (e.g., `delphi_rounds: 3` means 1 initial + 2 refinements)

**Example** (5 experts, 3 rounds, n=4 bins):

- Per pair: 5×2 + 5×1 + 5×1 = 20 calls
- Total: 20 × 6 pairs = 120 calls per model assessed
- Runtime: 15-30 minutes (depends on rate limits)

## Next Steps

1. Run dry run to verify setup
2. Inspect CSV/JSON outputs
3. Run full experiment
4. Analyze calibration performance vs ground truth
5. (Optional) Plot predictions vs ground truth
6. (Optional) Tune prompts based on results
