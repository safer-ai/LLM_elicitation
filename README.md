# LLM Elicitation

This repository accompanies the paper [**Toward Quantitative Modeling of Cybersecurity Risks Due to AI Misuse**](https://arxiv.org/abs/2512.08864) (Barrett et al., 2025).

## Repository Structure

This repository contains **two independent tools** for LLM-based capability evaluation:

### 1. Risk Scenario Estimation (`src/`)

**Purpose:** Cyber risk scenario assessment using LLM-based Delphi method  
**Run:** `python src/main.py -c config.yaml`  
**Docs:** See sections below

The main tool simulates a Delphi-like estimation process using LLMs as expert personas. It assesses cyber risk scenarios by breaking them down into steps and evaluating probability of success against benchmark tasks.

### 2. Difficulty Estimation Pipeline (`difficulty_estimation/`)

**Purpose:** LLM-based vulnerability difficulty ranking using multiple methodologies  
**Run:** `cd difficulty_estimation && pixi run python estimate_difficulty.py --config config.yaml`  
**Docs:** See `difficulty_estimation/README.md`

Supports multiple ranking approaches including First Solve Time (FST) prediction, arbitrary difficulty scales, and iterative ranking for security vulnerabilities.

---

## Risk Scenario Estimation Tool

### Overview

The Risk Scenario Estimation tool (located in `src/`) is designed to simulate a Delphi-like estimation process using Large Language Models (LLMs) as expert personas. It assesses cyber risk scenarios by breaking them down into steps and, for each step, evaluating the probability of success or other metrics against a series of benchmark tasks. The tool also supports estimating scenario-level metrics such as the potential number of threat actors, number of attacks, or damage, by benchmarking LLM capabilities against dedicated tasks for these broader estimations.

The pipeline involves:

1. Defining a risk scenario, threat actors, and targets.
2. Specifying expert personas with different backgrounds and biases.
3. Using LLMs (e.g., Claude, GPT models) to:
    * Analyze benchmark tasks relevant to each scenario step or scenario-level metric.
    * Provide initial estimates (probability or other values).
    * Refine these estimates over multiple rounds based on aggregated feedback (simulating a Delphi panel).
4. Storing detailed results, including raw LLM responses, parsed estimates, and rationales.
5. Generating a summary CSV and plots for analysis.

## Setup

1. **Clone Repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Create Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    # or python3 -m venv venv
    ```

3. **Activate Virtual Environment:**
    * macOS/Linux: `source venv/bin/activate`
    * Windows: `.\venv\Scripts\activate`

4. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1. **Copy Example Config:**

    ```bash
    cp config_example.yaml config.yaml
    ```

2. **Edit `config.yaml`:**
    * **API Keys:** Add your API key(s) under `anthropic_api_key:` and/or `openai_api_key:`. Only the key corresponding to the provider you are using is required.
        **⚠️ Warning: Do NOT commit `config.yaml` with your API keys to version control!** Note that `config.yaml` is already in the `.gitignore` file.
    * **Input Paths:**
        * `prompts_dir`: Directory for prompt templates.
        * `expert_profiles_file`: Path to the YAML file defining expert personas.
        * `default_benchmark_file`: Path to the YAML file for the default benchmark set. This is used if a scenario step (for probability estimation) doesn't specify its own benchmark.
        * `scenario_file`: Path to the YAML file defining the risk scenario, its steps, and benchmark files for scenario-level metrics.
    * **LLM Settings (`llm_settings`):**
        * Set the specific `model` string (e.g., `"claude-3-5-sonnet-20240620"`, `"gpt-4o"`). The API provider (Anthropic/OpenAI) will be inferred from this.
        * Adjust `temperature` (0.0-2.0) for controlling randomness.
        * Configure `max_concurrent_calls` (how many API requests run in parallel).
        * Set `rate_limit_calls` and `rate_limit_period` (e.g., 45 calls per 60 seconds) based on your API plan's limits to avoid 429 errors.
    * **Workflow Settings (`workflow_settings`):**
        * `num_tasks`: Maximum number of tasks to process from *each* benchmark (for scenario steps *and* scenario-level metrics). Set to `null` to process all tasks.
        * `scenario_steps`: List of step names (from the scenario file) to process for probability estimation. Set to `null` to process all steps.
        * `num_experts`: Maximum number of expert personas to simulate. Set to `null` to use all experts.
        * `delphi_rounds`: Total number of iterative refinement rounds (minimum 1).
        * `convergence_threshold`: Standard deviation threshold for stopping Delphi rounds early (e.g., 0.05 for probabilities).
        * **Scenario-Level Metric Flags:**
            * `estimate_num_actors_per_task_benchmark`: (boolean) Set to `true` to enable estimation of the number of threat actors.
            * `estimate_num_attacks_per_task_benchmark`: (boolean) Set to `true` to enable estimation of the number of attacks.
            * `estimate_damage_per_task_benchmark`: (boolean) Set to `true` to enable estimation of damage.
            (These require corresponding entries in `scenario_file.scenario_level_metrics` and relevant prompt files.)

## Usage

1. Activate your virtual environment.
2. Navigate to the project root directory.
3. Execute the main script:

    ```bash
    python src/main.py
    ```

4. To use a different configuration file:

    ```bash
    python src/main.py -c path/to/your_config.yaml
    ```

5. To enable detailed debug logging:

    ```bash
    python src/main.py -d
    ```

6. Output logs to the console. Results are saved in `output_data/runs/{run_id}/` and registered in `output_data/run_registry.json`. A summary table is printed to the console upon completion.

**Note on API Costs:** Running this tool will make calls to the selected LLM API, which will incur costs based on the chosen model and the amount of input/output tokens processed. Monitor your usage.

## Generating Plots

After a run completes, a CSV file (`detailed_estimates.csv`) is generated in the run's output directory (`output_data/runs/{run_id}/`). You can use the `src/plot_fct/scatter_plots.py` script to visualize these results:

```bash
python src/plot_fct/scatter_plots.py
```

## Input Data Format

The pipeline relies on several YAML and text files for its configuration and input data. All paths specified in config.yaml or within other YAML files (like benchmark paths in a scenario file) should be relative to the project root directory or absolute paths.

### 1. Expert Profiles (expert_profiles_file)

**Format:** YAML list under the top-level key experts.  
**Structure:** Each item in the list is a dictionary representing one expert persona.  
**Required Keys per Expert:**

- `name`: (string) Unique identifier for the expert.
- `background`: (string) Description of the expert's professional background.
- `focus`: (string) The expert's area of specialization.
- `key_trait`: (string) A defining characteristic or skill.
- `bias`: (string) Known biases or perspectives the expert might have.
- `analytical_approach`: (string, optional) How the expert typically analyzes problems.

**Example Snippet** (`input_data/expert_profiles.yaml`):

```yaml
experts:
  - name: "Defensive Security Specialist"
    focus: "Network defense and intrusion detection systems"
    background: "10 years SOC experience, specialized in APT detection"
    bias: "Conservative estimates on attacker capabilities; emphasizes importance of defense-in-depth"
    key_trait: "Particular attention to detection evasion techniques and defense bypass methods"
    analytical_approach: "Analyzes from defender's perspective, focusing on detection points and security control effectiveness"
  # ... more experts
```

### 2. Scenario Definition (scenario_file)

**Format:** YAML dictionary.  
**Structure:** Defines the overall risk scenario, its components, and how different aspects are benchmarked.  
**Top-Level Keys:**

- `scenario_name`: (string) A descriptive title for the scenario.
- `scenario_description`: (string) A general overview of the scenario.
- `threat_actor`: (dictionary) Contains:
    - `name`: (string) Name/type of the threat actor (e.g., "Organized Crime Group OC5").
    - `description`: (string) Details about the threat actor's capabilities, motivations, etc.
- `target`: (dictionary) Contains:
    - `name`: (string) Name/type of the target (e.g., "Mid-sized Tech Company").
    - `description`: (string) Details about the target's profile, security posture, etc.
- `steps`: (list) A list of dictionaries, where each dictionary represents a specific step within the scenario for probability estimation.
    - `name`: (string) Unique name for this scenario step (e.g., "Initial Compromise via Spear-Phishing").
    - `description`: (string) Detailed description of what this step entails.
    - `assumptions`: (string, multiline allowed) Specific assumptions to consider when estimating this step's success probability.
    - `benchmark_file`: (string, optional) Path to a specific benchmark YAML file (e.g., `input_data/benchmark/phishing_benchmark.yaml`) to be used for this step. If omitted, the `default_benchmark_file` from `config.yaml` is used.
- `scenario_level_metrics`: (dictionary, optional) Maps scenario-level metric estimation types to their configuration. Each metric configuration includes:
    - `benchmark_file`: (string) Path to the benchmark YAML file for that metric (e.g., `input_data/benchmark/actor_analysis_benchmark.yaml`).
    - `assumptions`: (string, multiline allowed) Specific assumptions to consider when estimating this metric.
    - **Valid Keys:** `num_actors_estimation`, `num_attacks_estimation`, `damage_estimation`.

**Example Snippet** (`input_data/scenario/dummy_scenario.yaml`):

```yaml
scenario_name: "Targeted Ransomware Attack on SP500 Company"
scenario_description: "A multi-stage attack aiming to deploy ransomware..."

threat_actor:
  name: "FIN-X Group"
  description: "Highly skilled financially motivated group..."

target:
  name: "Global Manufacturing Corp (SP500)"
  description: "Large enterprise with mature but not infallible security."

scenario_level_metrics:
  num_actors_estimation:
    benchmark_file: "input_data/benchmark/cybench_actor_assessment.yaml"
    assumptions: |
      - Assume threat actors have moderate technical skills and resources
      - Consider that the SP500 company is a high-value target
      - Account for publicized successful attacks potentially deterring some actors
  damage_estimation:
    benchmark_file: "input_data/benchmark/cybench_impact_assessment.yaml"
    assumptions: |
      - Consider both direct and indirect costs
      - Assume cyber insurance with deductibles and coverage limits
      - Include regulatory fines and intellectual property theft

steps:
  - name: "Reconnaissance & OSINT"
    description: "Gathering information about the target..."
    assumptions: "Target has a significant public internet footprint."
    benchmark_file: "input_data/benchmark/osint_tasks.yaml"
  - name: "Exploit Public-Facing Application"
    description: "Exploiting a vulnerability in an internet-exposed service."
    assumptions: "A known CVE with a public exploit exists."
    # No benchmark_file here, so default_benchmark_file from config.yaml will be used.
  # ... more steps
```

### 3. Benchmark Definitions (default_benchmark_file and other *.yaml files)

**Format:** YAML dictionary.  
**Structure:** Defines a set of tasks against which LLM capabilities are measured for a particular scenario step or a scenario-level metric.  
**Top-Level Keys:**

- `benchmark_description`: (string) A description of what this benchmark represents and what its tasks generally involve.
- `metrics_to_use_for_estimation`: (list of strings) A list of metric keys (from the `metrics` dictionary within each task) that should be extracted and included in the prompts to the LLM when this benchmark is used. For example, if a task has `metrics: {fst: 7, difficulty: "hard"}` and this list is `["fst"]`, then "FST: 7" will be part of the prompt.
- `tasks`: (list) A list of dictionaries, where each dictionary is a specific task.
    - `name`: (string) Unique name for the task (e.g., "It Has Begun", "Network Tools").
    - `description`: (string, multiline allowed) A detailed description of what needs to be done to complete the task.
    - `metrics`: (dictionary) Key-value pairs providing specific data points or characteristics about the task. The keys here can be referenced in `metrics_to_use_for_estimation`. Example: `fst: 7`, `cvss_score: 9.8`, `category: "Reverse Engineering"`.

**Example Snippet** (`input_data/benchmark/cybench.yaml`):

```yaml
metrics_to_use_for_estimation: ["fst"]
benchmark_description: "This dataset is composed of Capture The Flag (CTF) tasks. 'fst' is the First Solve Time..."
tasks:
  - name: "It Has Begun"
    description: >-
      "This task involves analyzing a malicious bash script..."
    metrics:
      fst: 7
  - name: "Network Tools"
    description: >-
      "This task involves examining 'main.rs' and identifying a buffer overflow..."
    metrics:
      fst: 42
  # ... more tasks
```

### 4. Prompt Templates (prompts_dir/*.txt)

**Format:** Plain text files (.txt).  
**Structure:** These files contain the text of the prompts that will be sent to the LLM. They use placeholders in the format `{placeholder_name}` which are dynamically filled in by the script during runtime.  
**Key Placeholders for Scenario-Level Metrics:**

- `{scenario_level_metric_assumptions}`: The assumptions specific to the metric being estimated (from `scenario_level_metrics` in the scenario file).
- Other standard placeholders like `{scenario_name}`, `{threat_actor_name}`, `{target_name}`, etc.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{barrett2025quantitativemodelingcybersecurityrisks,
      title={Toward Quantitative Modeling of Cybersecurity Risks Due to AI Misuse}, 
      author={Steve Barrett and Malcolm Murray and Otter Quarks and Matthew Smith and Jakub Kryś and Siméon Campos and Alejandro Tlaie Boria and Chloé Touzet and Sevan Hayrapet and Fred Heiding and Omer Nevo and Adam Swanda and Jair Aguirre and Asher Brass Gershovich and Eric Clay and Ryan Fetterman and Mario Fritz and Marc Juarez and Vasilios Mavroudis and Henry Papadatos},
      year={2025},
      eprint={2512.08864},
      archivePrefix={arXiv},
      primaryClass={cs.CY},
      url={https://arxiv.org/abs/2512.08864}, 
}
```

## License

This project is licensed under the MIT License.
