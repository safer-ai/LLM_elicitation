# Inter-Benchmark Calibration

Predicts **P(model solves target task on benchmark B | model scores in bin X on source benchmark A)** using LLM-based Delphi estimation.

Unlike intra-benchmark calibration (which predicts within one benchmark), this tool predicts *across* benchmarks -- e.g. given a model's CyBench score, predict whether it can solve a specific SWE-bench task.

## Core Approach

**Task-level predictions**: For each (source_bin, target_percentile) pair, the system asks: "Given that a model scores in this range on the source benchmark, what's the probability it can solve this specific task on the target benchmark?"

This avoids the averaging problem of score-level predictions and maps directly to the P(tactic | benchmark score) framework used in capability elicitation research.

## Ground Truth Computation

The workflow requires **pre-computed ground truth** that specifies which models fall into each source bin and what the observed solve rates are for each target task. Ground truth is computed from:

- **Source leaderboard JSON**: Per-model scores on the source benchmark
- **Target solve matrix JSON** (for per-task data): Binary model×task solve outcomes
- **Target ordered YAML**: Tasks sorted by difficulty
- **Bin/percentile configuration**: Source bin boundaries and target percentiles to evaluate

### Two Ground Truth Computation Modes

#### Mode 1: Overall Score-Based (for benchmarks with only model scores)

```bash
python compute_ground_truth.py \
  --source-scores input_data/scores/cybench.json \
  --target-scores input_data/scores/swebench_verified.json \
  --source-bins "auto" --n-source-bins 4 \
  --target-percentiles "30,40,50,60" \
  --output input_data/ground_truth/cybench_to_swebench_verified_gt.json
```

Computes: P(target_score >= percentile | source_bin)

#### Mode 2: Per-Task Based (for benchmarks with per-task solve data)

```bash
python compute_ground_truth_per_task.py \
  --source-leaderboard input_data/scores/livebench_lcb_generation_leaderboard.json \
  --target-solve-matrix input_data/scores/livebench_coding_completion_solve_matrix.json \
  --target-ordered input_data/benchmarks/livebench_coding_completion_ordered.yaml \
  --source-bins "auto" --n-source-bins 4 \
  --target-percentiles "20,30,40,50,60,70,80" \
  --output input_data/ground_truth/livebench_lcb_to_cc_gt.json
```

Computes: P(model solves specific target task | source_bin)

**Important**: The ground truth file **fixes the source bins and target percentiles**. The config file must match the ground truth parameters, or predictions will be computed with mismatched structure. See QUICK_START.md for examples.

## Key Design Decisions

### Per-benchmark score files

Score data is stored in **one JSON per benchmark** (not per source-target pair). Files are stored in `input_data/scores/` with format:
```json
{
  "benchmark_name": "cybench",
  "models": [
    {"model": "GPT-4o", "score": 12.5},
    {"model": "Claude 3.5 Sonnet", "score": 17.5}
  ]
}
```

The compute ground truth scripts **join** score files by model name at runtime. **It is the user's responsibility to ensure model name strings match across files** (e.g. if one file lists `GPT-4o` and another lists `GPT 4o`, they won't be matched).

### Multi-source sparsity handling

When multiple source benchmarks are provided, models are binned by their score on the **primary (first) source benchmark only**. Other source scores are passed to the LLM as context but do not affect ground truth computation.

Rationale: with ~12 models and 4 bins per source, binning across 2 sources yields 16 cells -- most empty. Single-source binning gives 4 cells with ~3 models each.

### Configurable source bins

Source bins can be:
- **Explicit**: `[[10, 20], [20, 30], [30, 40]]` or (fractions) `[[0.1, 0.2], [0.2, 0.3]]` -- variable width is fine
- **Auto**: `n_source_bins` equal-width bins from min to max model score on the primary source

This is important because typically we don't want to elicit for 0-10% or 90-100% ranges.

### Expert profiles

Prior work showed expert profiles don't significantly affect diversity of opinion. Generic, domain-agnostic profiles are used (AI capabilities researcher, psychometrician, etc.) rather than domain-specific ones. The domain-specific CyBench and SWE-bench profiles from intra-benchmark are symlinked in for optional use.

## Configuration Reference

See `config_example.yaml` for a fully documented example. Key settings:

| Setting | Description |
|---------|-------------|
| `source_benchmarks` | List of source benchmarks (first = primary for binning) |
| `target_benchmark` | Target benchmark with `target_percentiles` |
| `source_bins` | Explicit `[[lo,hi],...]` or `"auto"` with `n_source_bins` |
| `target_percentiles` | Explicit `[30,40,50,60]` or `"auto"` with `n_target_percentiles` |
| `n_easier_tasks` | Number of representative source tasks per prediction |
| `ground_truth_file` | Path to pre-computed ground truth JSON |

## Cost Estimates

Per prediction: 2 API calls (analysis + estimation) per expert in round 1, then 1 call per expert per refinement round.

For 4 source bins x 4 target percentiles = 16 predictions, with 5 experts and 3 rounds:
- API calls: ~16 x 5 x (2 + 2) = ~320 calls
- Estimated cost with Claude Sonnet: ~$10-20

## Output Format

- **CSV**: One row per expert per round, with source_bin, target_percentile, percentile estimates, ground truth
- **JSON**: Full Delphi history with metadata, round-by-round expert responses, aggregated estimates
- **Plots**: Calibration scatter, transfer curves, heatmap, violin plots, statistics
