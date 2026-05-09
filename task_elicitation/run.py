#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run script for task-level elicitation experiment.

Usage:
    python task_elicitation/run.py
    python task_elicitation/run.py --dry-run  # Test without API calls
    python task_elicitation/run.py --agents "anthropic/claude-sonnet-4-6,openai/gpt-4o-2024-08-06"
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workflow import ElicitationConfig, run_elicitation, save_results, compute_metrics
from data_loader import TaskData, assign_difficulty_bins, select_anchors, select_target_tasks, get_common_tasks


def dry_run(config: ElicitationConfig):
    """Test data loading and task selection without API calls."""
    print("="*60)
    print("DRY RUN - Testing data loading and task selection")
    print("="*60)

    # Load data
    task_data = TaskData(config.data_dir)
    print(f"\nLoaded data for {len(task_data.get_agents())} agents, {len(task_data.get_task_ids())} tasks")

    # Show target agents and their coverage
    print(f"\nTarget agents ({len(config.target_agents)}):")
    for agent in config.target_agents:
        count = len(task_data.get_agent_results(agent))
        print(f"  {agent}: {count} task results")

    # Get common tasks
    print("\nFinding tasks common to all target agents...")
    common_tasks = get_common_tasks(task_data, config.target_agents)
    print(f"Common tasks: {len(common_tasks)} (ensures anchor/target consistency)")

    # Bin only common tasks
    binned = assign_difficulty_bins(
        task_data,
        boundaries=config.difficulty_boundaries,
        labels=config.difficulty_labels,
        restrict_to_tasks=common_tasks
    )
    print(f"\nBinned {len(binned)} tasks into {len(config.difficulty_labels)} bins")

    # Select anchors
    anchor_ids = select_anchors(
        task_data, binned,
        n_per_bin=config.n_anchors_per_bin,
        percentiles=config.anchor_percentiles,
        seed=config.seed
    )
    print(f"\nSelected {len(anchor_ids)} anchors:")
    for aid in anchor_ids:
        info = task_data.get_task_info(aid)
        print(f"  {aid}: {info.get('difficulty_minutes', 0):.1f} min, {info.get('task_family', '')}")

    # Verify all agents have all anchors
    print("\nVerifying anchor consistency across agents...")
    all_good = True
    for agent in config.target_agents:
        for aid in anchor_ids:
            if task_data.get_task_result(agent, aid) is None:
                print(f"  ERROR: {agent} missing anchor {aid}")
                all_good = False
    if all_good:
        print("  ✓ All agents have all anchors")

    # Select targets
    target_ids = select_target_tasks(
        task_data, binned, anchor_ids,
        n_targets=config.n_target_tasks,
        seed=config.seed
    )
    print(f"\nSelected {len(target_ids)} target tasks")

    # Verify all agents have all targets
    print("\nVerifying target consistency across agents...")
    all_good = True
    for agent in config.target_agents:
        for tid in target_ids:
            if task_data.get_task_result(agent, tid) is None:
                print(f"  ERROR: {agent} missing target {tid}")
                all_good = False
    if all_good:
        print("  ✓ All agents have all targets")

    # Show sample agent performance on anchors
    if config.target_agents:
        agent = config.target_agents[0]
        print(f"\nSample: {agent} results on anchors:")
        for aid in anchor_ids:
            result = task_data.get_task_result(agent, aid)
            status = "PASS" if result == 1 else "FAIL" if result == 0 else "N/A"
            print(f"  {aid}: {status}")

    print("\n" + "="*60)
    print("Dry run complete. Use --run to execute with API calls.")
    print("="*60)


async def full_run(config: ElicitationConfig):
    """Run the full elicitation experiment."""
    results = await run_elicitation(config)
    metrics = compute_metrics(results)
    results["metrics"] = metrics
    output_dir = save_results(results, config.output_dir)

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Predictions: {metrics.get('n_predictions', 0)}")
    print(f"Brier Score: {metrics.get('brier_score', 'N/A'):.4f}" if metrics.get('brier_score') else "Brier Score: N/A")
    print(f"Accuracy @0.5: {metrics.get('accuracy_at_0.5', 0):.2%}" if metrics.get('accuracy_at_0.5') else "Accuracy: N/A")
    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Task-level probability elicitation")
    parser.add_argument("--dry-run", action="store_true", help="Test without API calls")
    parser.add_argument("--run", action="store_true", help="Run full experiment")
    parser.add_argument("--agents", type=str, help="Comma-separated list of agents to evaluate")
    parser.add_argument("--n-targets", type=int, default=30, help="Number of target tasks")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6-20241022", help="Elicitation model")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Build config
    config = ElicitationConfig(
        api_key_anthropic=os.environ.get("ANTHROPIC_API_KEY"),
        api_key_openai=os.environ.get("OPENAI_API_KEY"),
        elicitation_model=args.model,
        n_target_tasks=args.n_targets
    )

    if args.agents:
        config.target_agents = [a.strip() for a in args.agents.split(",")]

    if args.dry_run or not args.run:
        dry_run(config)
    else:
        asyncio.run(full_run(config))


if __name__ == "__main__":
    main()
