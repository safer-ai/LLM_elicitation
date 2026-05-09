#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Workflow for task-level probability elicitation.

Elicits P(model passes target task Y | model passes anchor tasks A)
using the github_data ground truth.
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

import sys
# Add parent (root) and current (task_elicitation) to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import TaskData, assign_difficulty_bins, select_anchors, select_target_tasks, get_common_tasks

try:
    from shared.llm_client import initialize_client, make_api_call, LLMSettings
    from shared.parsing import parse_probability_response
except ImportError:
    initialize_client = None
    make_api_call = None
    LLMSettings = None
    parse_probability_response = None

logger = logging.getLogger(__name__)


@dataclass
class ElicitationConfig:
    """Configuration for elicitation experiment."""
    data_dir: Path = field(default_factory=lambda: Path("ground-truth/github_data"))
    output_dir: Path = field(default_factory=lambda: Path("output_data/task_elicitation"))

    # Which agents to evaluate
    target_agents: List[str] = field(default_factory=lambda: [
        "anthropic/claude-sonnet-4-6",
        "openai/gpt-4o-2024-08-06"
    ])

    # Elicitation model
    elicitation_model: str = "claude-sonnet-4-6-20241022"

    # Task selection
    n_anchors_per_bin: int = 2
    anchor_percentiles: List[float] = field(default_factory=lambda: [0.33, 0.67])
    n_target_tasks: int = 30
    difficulty_boundaries: List[float] = field(default_factory=lambda: [0, 5, 60, float('inf')])
    difficulty_labels: List[str] = field(default_factory=lambda: ['easy', 'medium', 'hard'])

    # API settings
    api_key_anthropic: Optional[str] = None
    api_key_openai: Optional[str] = None
    temperature: float = 0.7
    max_concurrent: int = 5

    seed: int = 42


def format_anchor_examples(task_data: TaskData, agent: str, anchor_ids: List[str]) -> str:
    """Format anchor tasks with model's pass/fail for prompt."""
    lines = []
    for task_id in anchor_ids:
        result = task_data.get_task_result(agent, task_id)
        if result is None:
            continue

        info = task_data.get_task_info(task_id)
        difficulty = info.get('difficulty_minutes', 0)
        family = info.get('task_family', '')
        desc = info.get('description') or 'No description available'

        # Truncate description if too long
        if desc and len(desc) > 300:
            desc = desc[:297] + "..."

        status = "PASSED" if result == 1 else "FAILED"
        lines.append(f"""
Task: {task_id}
Family: {family}
Difficulty: {difficulty:.1f} minutes
Description: {desc}
Result: {status}
""")

    return "\n---".join(lines)


def load_prompt_template() -> str:
    """Load the elicitation prompt template."""
    prompt_path = Path(__file__).parent / "prompts" / "elicitation.txt"
    if prompt_path.exists():
        return prompt_path.read_text()

    # Fallback inline template
    return """You are estimating the probability that an AI agent can solve a cybersecurity task.

<model>{agent_name}</model>

<anchor_tasks>
{anchor_examples}
</anchor_tasks>

<target_task>
Task ID: {target_task_id}
Family: {target_task_family}
Description: {target_description}
</target_task>

Estimate P(model solves target task). Provide percentiles for Beta distribution fitting:

**Percentile Estimates:**
- 25th percentile: [0.xx]
- 50th percentile (median): [0.xx]
- 75th percentile: [0.xx]

**Rationale:** [1-2 sentences]
"""


async def elicit_single(
    client,
    semaphore: asyncio.Semaphore,
    settings: LLMSettings,
    prompt: str
) -> Dict[str, Any]:
    """Make a single elicitation call and parse percentile estimates."""
    system_prompt = "You are an expert at estimating AI model capabilities on technical tasks."

    response = await make_api_call(
        client=client,
        semaphore=semaphore,
        settings=settings,
        system_prompt=system_prompt,
        user_prompt=prompt,
        max_tokens=800
    )

    # Use shared parsing for percentile extraction
    parsed = parse_probability_response(response)

    return {
        "response": response,
        "percentile_25th": parsed.get("percentile_25th"),
        "percentile_50th": parsed.get("percentile_50th"),
        "percentile_75th": parsed.get("percentile_75th"),
        "rationale": parsed.get("rationale", ""),
        "estimate": parsed.get("estimate")  # median as primary estimate
    }


async def run_elicitation(config: ElicitationConfig) -> Dict[str, Any]:
    """Run the full elicitation experiment."""
    logger.info("Starting task-level elicitation experiment")

    # Load data
    task_data = TaskData(config.data_dir)

    # Get tasks common to all target agents to ensure consistency
    logger.info(f"Finding tasks common to all {len(config.target_agents)} target agents...")
    common_tasks = get_common_tasks(task_data, config.target_agents)

    if len(common_tasks) == 0:
        raise ValueError("No tasks common to all target agents!")

    logger.info(f"Using {len(common_tasks)} tasks common to all agents")

    # Bin only the common tasks
    binned = assign_difficulty_bins(
        task_data,
        boundaries=config.difficulty_boundaries,
        labels=config.difficulty_labels,
        restrict_to_tasks=common_tasks
    )

    # Select global anchors from common tasks (same across all models)
    anchor_ids = select_anchors(
        task_data, binned,
        n_per_bin=config.n_anchors_per_bin,
        percentiles=config.anchor_percentiles,
        seed=config.seed
    )
    logger.info(f"Selected {len(anchor_ids)} global anchors: {anchor_ids}")

    # Verify all agents have all anchors
    for agent in config.target_agents:
        for anchor_id in anchor_ids:
            result = task_data.get_task_result(agent, anchor_id)
            if result is None:
                raise ValueError(f"Agent {agent} missing anchor {anchor_id}!")

    # Select target tasks from common tasks
    target_ids = select_target_tasks(
        task_data, binned, anchor_ids,
        n_targets=config.n_target_tasks,
        seed=config.seed
    )
    logger.info(f"Selected {len(target_ids)} target tasks")

    # Verify all agents have all targets
    for agent in config.target_agents:
        for target_id in target_ids:
            result = task_data.get_task_result(agent, target_id)
            if result is None:
                raise ValueError(f"Agent {agent} missing target {target_id}!")

    # Initialize LLM client
    client = initialize_client(
        config.api_key_anthropic,
        config.api_key_openai,
        config.elicitation_model
    )

    settings = LLMSettings(
        model=config.elicitation_model,
        temperature=config.temperature,
        max_concurrent_calls=config.max_concurrent
    )

    semaphore = asyncio.Semaphore(config.max_concurrent)
    prompt_template = load_prompt_template()

    # Collect results
    results = {
        "config": {
            "elicitation_model": config.elicitation_model,
            "target_agents": config.target_agents,
            "n_anchors_per_bin": config.n_anchors_per_bin,
            "n_target_tasks": config.n_target_tasks,
            "anchor_ids": anchor_ids,
            "target_ids": target_ids
        },
        "predictions": []
    }

    # Run elicitation for each agent and target
    for agent in config.target_agents:
        logger.info(f"\nEliciting for agent: {agent}")

        # Check agent has data
        agent_results = task_data.get_agent_results(agent)
        if len(agent_results) == 0:
            logger.warning(f"No results for agent {agent}, skipping")
            continue

        # Format anchors for this agent
        anchor_text = format_anchor_examples(task_data, agent, anchor_ids)

        for target_id in target_ids:
            target_info = task_data.get_task_info(target_id)
            ground_truth = task_data.get_task_result(agent, target_id)

            if ground_truth is None:
                logger.debug(f"No ground truth for {agent} on {target_id}")
                continue

            # Build prompt (NOTE: target difficulty is NOT included to prevent leakage)
            target_description = target_info.get('description') or 'No description available'

            prompt = prompt_template.format(
                agent_name=agent,
                anchor_examples=anchor_text,
                target_task_id=target_id,
                target_task_family=target_info.get('task_family', ''),
                target_description=target_description[:500]
            )

            try:
                result = await elicit_single(client, semaphore, settings, prompt)

                prediction = {
                    "agent": agent,
                    "target_task_id": target_id,
                    "target_difficulty_minutes": target_info.get('difficulty_minutes'),
                    "target_task_family": target_info.get('task_family', ''),
                    "ground_truth": ground_truth,
                    "percentile_25th": result["percentile_25th"],
                    "percentile_50th": result["percentile_50th"],
                    "percentile_75th": result["percentile_75th"],
                    "estimate": result["estimate"],
                    "rationale": result["rationale"][:300] if result["rationale"] else "",
                    "response": result["response"][:500]  # Truncate for storage
                }
                results["predictions"].append(prediction)

                p50 = result["estimate"]
                logger.info(
                    f"  {target_id}: P50={p50:.2f} [P25={result['percentile_25th']:.2f}, P75={result['percentile_75th']:.2f}] vs GT={ground_truth}"
                    if p50 is not None else f"  {target_id}: Parse failed"
                )

            except Exception as e:
                logger.error(f"Error eliciting for {target_id}: {e}")

    return results


def save_results(results: Dict[str, Any], output_dir: Path) -> Path:
    """Save results to output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save full results as JSON
    json_path = run_dir / "results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save predictions as CSV for easy analysis
    if results.get("predictions"):
        df = pd.DataFrame(results["predictions"])
        csv_path = run_dir / "predictions.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(df)} predictions to {csv_path}")

    logger.info(f"Results saved to {run_dir}")
    return run_dir


def compute_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """Compute calibration metrics from predictions."""
    predictions = results.get("predictions", [])
    if not predictions:
        return {}

    df = pd.DataFrame(predictions)
    # Use p50 (estimate) as the primary prediction
    df = df[df['estimate'].notna()]

    if len(df) == 0:
        return {}

    # Brier score (using median estimate)
    brier = ((df['estimate'] - df['ground_truth']) ** 2).mean()

    # Log loss (cross-entropy)
    eps = 1e-15
    probs = df['estimate'].clip(eps, 1 - eps).values
    gt = df['ground_truth'].values
    log_loss = -(gt * np.log(probs) + (1 - gt) * np.log(1 - probs)).mean()

    # Accuracy at 0.5 threshold
    predicted_binary = (df['estimate'] >= 0.5).astype(int)
    accuracy = (predicted_binary == df['ground_truth']).mean()

    # Uncertainty spread (IQR of estimates)
    df_with_iqr = df[df['percentile_25th'].notna() & df['percentile_75th'].notna()]
    mean_iqr = (df_with_iqr['percentile_75th'] - df_with_iqr['percentile_25th']).mean() if len(df_with_iqr) > 0 else None

    # Calibration by bin
    df['prob_bin'] = pd.cut(df['estimate'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    calibration_by_bin = df.groupby('prob_bin', observed=True).agg({
        'ground_truth': ['mean', 'count']
    }).to_dict()

    return {
        "n_predictions": len(df),
        "brier_score": float(brier),
        "log_loss": float(log_loss),
        "accuracy_at_0.5": float(accuracy),
        "mean_predicted_p50": float(df['estimate'].mean()),
        "mean_actual": float(df['ground_truth'].mean()),
        "mean_uncertainty_iqr": float(mean_iqr) if mean_iqr is not None else None
    }


async def main():
    """Main entry point."""
    import os

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load config
    config = ElicitationConfig(
        api_key_anthropic=os.environ.get("ANTHROPIC_API_KEY"),
        api_key_openai=os.environ.get("OPENAI_API_KEY"),
        target_agents=["anthropic/claude-sonnet-4-6", "openai/gpt-4o-2024-08-06"],
        n_target_tasks=30
    )

    results = await run_elicitation(config)

    # Compute metrics
    metrics = compute_metrics(results)
    results["metrics"] = metrics

    # Save
    output_dir = save_results(results, config.output_dir)

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Predictions: {metrics.get('n_predictions', 0)}")
    print(f"Brier Score: {metrics.get('brier_score', 'N/A'):.4f}")
    print(f"Accuracy @0.5: {metrics.get('accuracy_at_0.5', 'N/A'):.2%}")
    print(f"Mean Predicted: {metrics.get('mean_predicted', 'N/A'):.3f}")
    print(f"Mean Actual: {metrics.get('mean_actual', 'N/A'):.3f}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
