# Difficulty Estimation Pipeline

This pipeline uses Large Language Models (LLMs) to estimate the difficulty and exploitability of security vulnerabilities. It supports multiple estimation methodologies including First Solve Time (FST) prediction, arbitrary difficulty scale scoring, and iterative ranking approaches. The system is designed to enable research on LLM-based vulnerability assessment and comparison of different prompting strategies.

## Installation & Setup

### Prerequisites

- Python 3.14+
- [Pixi](https://pixi.sh/) package manager
- Anthropic API key

### Installation

1. Install dependencies using Pixi:
```bash
pixi install
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Quick Start

### Running an Experiment

Run difficulty estimation with the default configuration:

```bash
pixi run python estimate_difficulty.py --config config/config.yaml
```

### Available Estimation Modes

Run different estimation methods by specifying the appropriate config file:

```bash
# First Solve Time estimation (time in minutes to exploit)
pixi run python estimate_difficulty.py --config config/config_fst_nocode.yaml

# Difficulty scale (1-100) estimation
pixi run python estimate_difficulty.py --config config/config_difficulty_nocode.yaml

# Iterative ranking (easiest-first)
pixi run python estimate_difficulty.py --config config/config_fullrank_easy.yaml

# Iterative ranking (hardest-first)
pixi run python estimate_difficulty.py --config config/config_fullrank_hard.yaml
```

### What Happens During Execution

1. Loads benchmark tasks from `benchmark_tasks/bountybench.yaml`
2. Formats prompts based on the selected mode (with or without code)
3. Submits batch requests to Anthropic API
4. Polls for completion and retrieves responses
5. Saves raw outputs (including chain-of-thought reasoning) to `results/[benchmark]/[mode]/raw/`
6. Parses structured responses and generates ordered rankings
7. Outputs final YAML file to `results/[benchmark]/[mode]/ordered_tasks.yaml`

## Configuration

Configuration files are located in the `config/` directory. Each YAML file specifies:

### Key Parameters

- **`mode`**: Estimation methodology
  - `fst` - First Solve Time (parallel, all tasks at once)
  - `difficulty` - 1-100 difficulty scale (parallel)
  - `iterative_easiest` - Iteratively select easiest task
  - `iterative_hardest` - Iteratively select hardest task
  - `pairwise` - Pairwise task comparison - this is experimental and not fully developed currently

- **`prompt_template`**: Path to prompt file in `prompts/` directory

- **`include_code`**: Boolean, whether to include vulnerable code snippets in prompts. Note that it does not apply to the 'fullrank' ('iterative') mode, as including code for all benchmark tasks would overwhelm the estimator (and exceed the LLMs' context window).

- **`llm_settings`**: API configuration
  - `model_name`: Anthropic model to use (e.g., `claude-sonnet-4-20250514`)
  - `temperature`: Sampling temperature
  - `provider`: API provider (`anthropic`)

- **`tasks_path`**: Path to benchmark tasks YAML file

- **`output_file`**: Where to save parsed results

- **`raw_output_file`**: Where to save raw LLM outputs. Omit to not store raw outputs.

- **`ranking_file`**: (Optional) Previous ranking for pairwise mode

### Creating Custom Configurations

Copy an existing config file and modify parameters to experiment with different settings:

```bash
cp config/config.yaml config/my_experiment.yaml
# Edit my_experiment.yaml
pixi run python estimate_difficulty.py --config config/my_experiment.yaml
```

## Architecture Overview

### Core Components

```
estimate_difficulty.py    # Main orchestration script
├── formatting.py          # Prompt and task formatting utilities
├── utils/
│   ├── utils.py          # YAML loading, task ID extraction
│   ├── llm_api.py        # Anthropic batch API integration
│   └── code_retriever.py # Git-based code extraction and caching for bountybench.
├── prompts/              # LLM prompt templates by methodology
│   ├── fst/
│   ├── difficulty_scale/
│   ├── iterative_selection/
│   └── pairwise_ranking/
├── config/               # Experiment configurations
├── benchmark_tasks/      # Task definitions (BountyBench)
└── results/              # Experiment outputs
```

### Workflow

1. **Configuration Loading**: `load_yaml()` reads experiment config and benchmark tasks
2. **Prompt Formatting**:
   - Iterative modes: `format_task_list()` creates numbered task lists
   - Parallel modes: `format_single_task_prompt()` formats individual tasks
3. **Code Retrieval** (if `include_code=true`): `retrieve_and_cache_code()` extracts vulnerable/patched versions
4. **API Interaction**: `get_llm_response()` submits batch requests, `await_batch_list()` polls for completion
5. **Response Parsing**: `parse_llm_response()` extracts estimates and reasoning using regex
6. **Ranking Generation**: Tasks ordered by difficulty estimates or iterative selection
7. **Output**: Results saved as YAML with estimates and reasoning per task

## Results Visualization

### Using the Analysis Notebook

Launch marimo and open the plotting notebook:

```bash
pixi run marimo edit plot_difficulty_results.py
```

### Key Analysis Features

The notebook provides:

- **Kendall's W Concordance**: Measures agreement between different ranking methods
- **Correlation Matrices**: Pairwise comparison of estimation approaches
- **Borda Count Consensus**: Aggregates multiple rankings into consensus order
- **Metric Validation**: Correlates estimates with ground truth (CVSS, bounty amounts, solve rates)
- **Distribution Plots**: Visualizes FST and difficulty score distributions

### Comparing Results

Load and compare multiple ranking outputs:

```python
response_files = [
    "./results/bountybench/fst/ordered_tasks.yaml",
    "./results/bountybench/difficulty_scale/ordered_tasks.yaml",
]
responses = [{r["task_id"]:r for r in load_yaml(f)} for f in response_files]
```

## Benchmark Tasks

Tasks are defined in `benchmark_tasks/bountybench.yaml`. Each task includes:

- **name**: Vulnerability title
- **description**: Detailed writeup from bug bounty report
- **source_url**: Original report URL (Huntr, HackerOne, etc.)
- **metrics**: CVSS score, bounty amount, CWE, CVE
- **metadata**: Repository info, patch details, exploit chain

The BountyBench dataset contains real-world vulnerabilities across 30+ open source projects.

## Output Format

### Ordered Results YAML

Generated in `results/[benchmark]/[mode]/ordered_tasks.yaml`:

```yaml
- task_id: "486add92-275e-4a7b-92f9-42d84bc759da"
  estimate: 45.0  # FST in minutes or 1-100 difficulty score
  reasoning: "This requires understanding authentication flow..."
```

For iterative modes, `estimate` shows selection order (e.g., "SELECTION_INDEX / REMAINING_TASKS").

### Raw Outputs

Saved in `results/[benchmark]/[mode]/raw/` as text files with:
- Task separator headers
- Chain-of-thought reasoning (if extended thinking enabled)
- Final structured response

These files enable qualitative analysis of LLM reasoning patterns.
