# Intra-Benchmark Calibration

## Overview

The Intra-Benchmark Calibration tool enables LLM-based Delphi estimation of conditional probabilities between benchmark task difficulty bins. It answers the question: **"What is the probability that a model can solve tasks in bin j, given that it can solve all tasks in bin i?"**

This tool uses LLMs as expert personas to predict capability progression across difficulty levels, providing calibrated estimates that can be compared against ground truth data from actual model performance.

## Key Features

- **Conditional Probability Estimation**: Predicts P(j|i) for all bin pairs with sufficient ground truth data
- **Delphi Method**: Multiple expert personas provide independent estimates, then refine based on peer feedback
- **Ground Truth Comparison**: Results can be directly compared to empirical model performance data
- **Flexible Configuration**: Supports any benchmark with sorted tasks and ground truth distributions
- **Progressive Saving**: Results saved after each prediction for robustness and early inspection

## Quick Start

### Prerequisites

1. Python 3.8+ with dependencies from root `requirements.txt`
2. API key for Anthropic (Claude) or OpenAI (GPT models)
3. Ground truth data: `input_data/ground_truth/{benchmark}_ground_truth_n{nbins}.json`
4. Sorted benchmark file: `input_data/benchmarks/{benchmark}_ascending_{metric}.yaml`

### Installation

```bash
# From repository root
pip install -r requirements.txt
```

### Running an Experiment

```bash
# Navigate to intra_benchmark_calibration/
cd intra_benchmark_calibration

# Run with example config
python run_calibration.py -c config_example.yaml

# Or with your own config
cp config_example.yaml config.yaml
# Edit config.yaml with your settings
python run_calibration.py -c config.yaml

# Enable debug logging
python run_calibration.py -c config.yaml -d
```

## Configuration

Edit `config_example.yaml` (or create your own `config.yaml`):

### Key Settings

**API Keys:**

```yaml
anthropic_api_key: "YOUR_API_KEY_HERE"
openai_api_key: "YOUR_API_KEY_HERE"
```

**Benchmark Settings:**

```yaml
intra_benchmark_settings:
  benchmark_name: "cybench"  # Must match ground truth filename
  n_bins: 4                   # Must match ground truth n_bins
  benchmark_description: |    # Context provided to LLMs
    Description of what the benchmark measures...
```

**LLM Settings:**

```yaml
llm_settings:
  model: "claude-sonnet-4-5-20250929"  # or "gpt-4o"
  temperature: 0.8
  max_concurrent_calls: 5
  rate_limit_calls: 45
  rate_limit_period: 60
```

**Workflow Settings:**

```yaml
workflow_settings:
  num_experts: 5              # Number of expert personas to use
  delphi_rounds: 3            # Total rounds (1 initial + refinements)
  convergence_threshold: 0.05 # Early stopping if std dev < threshold
```

See `config_example.yaml` for full documentation.

## Workflow Steps

### 1. Preprocessing (One-Time)

Sort your benchmark tasks by a difficulty metric:

```bash
python prepare_benchmark.py ../input_data/benchmark/cybench.yaml --metric fst
```

This creates: `input_data/benchmarks/cybench_ascending_fst.yaml`

### 2. Ground Truth (One-Time)

Generate ground truth from model scores:

```bash
python compute_ground_truth.py
```

This creates: `input_data/ground_truth/{benchmark}_ground_truth_n{nbins}.json`

### 3. Run Calibration

```bash
python run_calibration.py -c config.yaml
```

## Output Structure

Results are saved to `results/{benchmark}/{run_id}/`:

```
results/
└── cybench/
    └── 20250109_143022/
        ├── detailed_estimates.csv  # All expert estimates by round
        └── full_results.json       # Complete Delphi history
```

### detailed_estimates.csv

Columns:

- `bin_i`, `bin_j`: Source and target bins
- `bin_i_range`, `bin_j_range`: Score percentile ranges
- `expert`: Expert profile name
- `round`: Delphi round number
- `percentile_25th`: Lower quartile (25th percentile) of the expert's probability distribution
- `percentile_50th`: Median (50th percentile)
- `percentile_75th`: Upper quartile (75th percentile)
- `rationale`: Expert's reasoning
- `ground_truth_p_j_given_i`: Actual P(j|i) from model data
- `sufficient_sample`: Whether ground truth is reliable

**Note on Uncertainty Quantification**: Expert estimates are elicited as three percentiles (25th, 50th, 75th) that characterise their subjective probability distribution. These values are intended for fitting a Beta distribution with support [0, 1], where the 50th percentile represents the median estimate, and the 25th and 75th percentiles encompass the inner 50% of probability mass.

### full_results.json

Complete structured data including:

- Run metadata (config, timestamp, model used)
- Per-prediction results with all rounds
- Task details for each bin
- Convergence information

## Analysis and Plotting

Visualize calibration performance:

```bash
python plot_calibration.py
```

This generates:

- Calibration plots (predicted vs actual)
- Heatmaps showing P(j|i) matrices
- Violin plots of expert distributions
- Statistical metrics (MAE, correlation, etc.)

## Example: n_bins=4

For CyBench with 4 bins:

**Bin Structure:**

- Bin 0: [5.0%, 9.4%) - Easiest tasks
- Bin 1: [9.4%, 13.8%)
- Bin 2: [13.8%, 18.1%)
- Bin 3: [18.1%, 22.5%) - Hardest tasks

**Predictions:**

- Total pairs: 6 = (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
- Typical sufficient_sample: 5 pairs (often excludes (2,3))

**API Calls:**

- Per pair: ~15-20 calls (5 experts × 2-stage process × rounds)
- Total: ~75-100 calls for complete experiment
- Cost estimate: $2-5 (Claude Sonnet) or $3-8 (GPT-4o)

**Runtime:**

- With max_concurrent_calls=5: ~15-30 minutes
- Sequential: ~30-60 minutes

## Troubleshooting

### "Ground truth file not found"

Ensure `input_data/ground_truth/{benchmark}_ground_truth_n{nbins}.json` exists.  
Run: `python compute_ground_truth.py`

### "Failed to load benchmark"

Ensure sorted benchmark exists: `input_data/benchmarks/{benchmark}_ascending_{metric}.yaml`  
Run: `python prepare_benchmark.py <input_file> --metric <metric>`

### "Missing prompt template"

Verify all 3 prompts exist in `prompts/`:

- `task_relationship_analysis.txt`
- `initial_conditional_probability_estimation.txt`
- `subsequent_conditional_probability_estimation.txt`

### "API rate limit exceeded (429)"

Lower concurrency in config:

```yaml
llm_settings:
  max_concurrent_calls: 3
  rate_limit_calls: 30
```

## Advanced Usage

### Custom Expert Profiles

Edit `input_data/expert_profiles/expert_profiles.yaml` to create custom expert personas with different backgrounds and biases.

### Different Benchmarks

1. Prepare your benchmark YAML with tasks sorted by difficulty
2. Generate ground truth from model performance data
3. Update `config.yaml` with benchmark name and n_bins
4. Run calibration

### Batch Experiments

Create multiple config files for different settings:

```bash
python run_calibration.py -c config_model_a.yaml
python run_calibration.py -c config_model_b.yaml
python run_calibration.py -c config_bins_6.yaml
```

## Documentation

- **Quick Start**: `QUICK_START.md` - Step-by-step getting started guide
- **Config Reference**: `config_example.yaml` - Annotated configuration file

## Support

For issues or questions:

1. Review `QUICK_START.md` for common setup issues
2. Inspect log output with `-d` debug flag

