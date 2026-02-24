#!/usr/bin/env python3
"""
Fill placeholder values in scenario YAML files with baseline data.

This script replaces {low_ci} and {high_ci} placeholders in scenario files
with actual values from corresponding baseline files, formatted as percentages.
"""

import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install it with: pip install pyyaml")
    sys.exit(1)


@dataclass
class ProcessingResult:
    """Tracks the results of processing a single scenario file."""
    scenario_name: str
    success: bool = False
    error_message: Optional[str] = None
    filled_placeholders: List[str] = field(default_factory=list)
    unfilled_placeholders: List[Tuple[str, str]] = field(default_factory=list)  # (name, reason)
    unused_baselines: List[str] = field(default_factory=list)
    baselines_available: Set[str] = field(default_factory=set)
    baselines_used: Set[str] = field(default_factory=set)


def load_yaml(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load and parse a YAML file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Error: Failed to read file {file_path}: {e}")
        return None


def save_yaml(data: Dict[str, Any], file_path: Path) -> bool:
    """Save data to a YAML file, preserving formatting where possible."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False,
                     allow_unicode=True, width=1000)
        return True
    except Exception as e:
        print(f"Error: Failed to write file {file_path}: {e}")
        return False


def extract_step_name(full_step_name: str) -> str:
    """
    Extract the step name from a full step identifier.

    Example: "TA0007 - Discovery" -> "Discovery"
             "T1565.001 - Impact: Stored Data Manipulation" -> "Impact: Stored Data Manipulation"
    """
    # Split on " - " and take everything after it
    if " - " in full_step_name:
        return full_step_name.split(" - ", 1)[1]
    return full_step_name


def format_large_number(value: float) -> str:
    """
    Format large numbers using k, m, b, t suffixes.

    Example: 30000 -> "30k"
             5000000 -> "5m"
             1500000 -> "1.5m"
             70000 -> "70k"
    """
    abs_value = abs(value)
    sign = "-" if value < 0 else ""

    if abs_value >= 1_000_000_000_000:  # trillion
        formatted = abs_value / 1_000_000_000_000
        suffix = "t"
    elif abs_value >= 1_000_000_000:  # billion
        formatted = abs_value / 1_000_000_000
        suffix = "b"
    elif abs_value >= 1_000_000:  # million
        formatted = abs_value / 1_000_000
        suffix = "m"
    elif abs_value >= 1_000:  # thousand
        formatted = abs_value / 1_000
        suffix = "k"
    else:
        # For small numbers, just return as-is
        if value == int(value):
            return f"{sign}{int(value)}"
        else:
            return f"{sign}{value:.1f}"

    # Format with appropriate decimal places
    if formatted == int(formatted):
        return f"{sign}{int(formatted)}{suffix}"
    else:
        return f"{sign}{formatted:.1f}{suffix}"


def decimal_to_percentage(value: float) -> str:
    """
    Convert a decimal probability to a percentage string.

    Example: 0.4 -> "40%"
             0.95 -> "95%"
    """
    # Handle both 0-1 range and already-scaled values
    if value <= 1.0:
        percentage = value * 100
    else:
        percentage = value

    # Format as integer if it's a whole number, otherwise with decimals
    if percentage == int(percentage):
        return f"{int(percentage)}%"
    else:
        return f"{percentage:.1f}%"


def fill_placeholders_in_step(step: Dict[str, Any], baselines: Dict[str, Dict[str, Any]]) -> bool:
    """
    Fill {low_ci} and {high_ci} placeholders in a step's assumptions.

    Returns True if any replacements were made, False otherwise.
    """
    if 'assumptions' not in step or not isinstance(step['assumptions'], str):
        return False

    assumptions = step['assumptions']

    # Check if there are placeholders to fill
    if '{low_ci}' not in assumptions and '{high_ci}' not in assumptions:
        return False

    # Extract the step name
    step_name = extract_step_name(step.get('name', ''))

    # Look up the baseline data
    if step_name not in baselines:
        print(f"Warning: No baseline data found for step '{step_name}'")
        return False

    baseline = baselines[step_name]

    # Get the low_ci and high_ci values
    if 'low_ci' not in baseline or 'high_ci' not in baseline:
        print(f"Warning: Missing low_ci or high_ci in baseline for '{step_name}'")
        return False

    low_ci = baseline['low_ci']
    high_ci = baseline['high_ci']

    # Determine format based on value range
    # If values are <= 1, treat as probabilities (percentages)
    # Otherwise, treat as large numbers (k/m/b/t format)
    if low_ci <= 1.0 and high_ci <= 1.0:
        low_ci_str = decimal_to_percentage(low_ci)
        high_ci_str = decimal_to_percentage(high_ci)
    else:
        low_ci_str = format_large_number(low_ci)
        high_ci_str = format_large_number(high_ci)

    # Replace placeholders
    assumptions = assumptions.replace('{low_ci}', low_ci_str)
    assumptions = assumptions.replace('{high_ci}', high_ci_str)

    # Update the step
    step['assumptions'] = assumptions

    print(f"  ✓ Filled placeholders in '{step_name}': low_ci={low_ci_str}, high_ci={high_ci_str}")
    return True


def fill_scenario_placeholders(scenario_path: Path, baseline_path: Path, output_path: Path) -> ProcessingResult:
    """
    Fill placeholders in a scenario file using baseline data.

    Args:
        scenario_path: Path to the scenario YAML file
        baseline_path: Path to the baseline YAML file
        output_path: Path where the filled scenario will be written

    Returns:
        ProcessingResult with detailed information about the processing
    """
    result = ProcessingResult(scenario_name=scenario_path.name)

    # Load baseline file to get the values
    print(f"Loading baseline file: {baseline_path}")
    baseline_data = load_yaml(baseline_path)
    if baseline_data is None:
        result.error_message = f"Failed to load baseline file: {baseline_path}"
        return result

    # Extract baselines dictionary
    if 'baselines' not in baseline_data:
        result.error_message = "Baseline file does not contain 'baselines' section"
        print(f"Error: {result.error_message}")
        return result

    baselines = baseline_data['baselines']
    result.baselines_available = set(baselines.keys())

    # Read the scenario file as raw text to preserve formatting
    print(f"Loading scenario file: {scenario_path}")
    try:
        with open(scenario_path, 'r', encoding='utf-8') as f:
            scenario_text = f.read()
    except Exception as e:
        result.error_message = f"Failed to read scenario file: {e}"
        print(f"Error: {result.error_message}")
        return result

    print(f"\nProcessing placeholders:")
    replacements_made = 0

    # Mapping from scenario_level_metrics keys to baseline keys
    metric_mapping = {
        'num_actors_estimation': 'NumActors',
        'num_attacks_estimation': 'NumAttacks',
        'damage_estimation': 'Damage'
    }

    # Process scenario_level_metrics section
    metric_pattern = re.compile(
        r'(  ([a-z_]+_estimation):\s*\n(?:(?!  [a-z_]+:).*\n?)*)',
        re.MULTILINE
    )

    for match in metric_pattern.finditer(scenario_text):
        full_metric_block = match.group(1)
        metric_key = match.group(2)

        # Check if this metric block has placeholders
        if '{low_ci}' not in full_metric_block and '{high_ci}' not in full_metric_block:
            continue

        # Map to baseline key
        baseline_key = metric_mapping.get(metric_key)
        if not baseline_key:
            result.unfilled_placeholders.append((metric_key, "No mapping defined for this metric"))
            continue

        # Look up baseline data
        if baseline_key not in baselines:
            print(f"  ⚠ Warning: No baseline data found for metric '{metric_key}' (baseline key: '{baseline_key}')")
            result.unfilled_placeholders.append((metric_key, f"No baseline data for key '{baseline_key}'"))
            continue

        baseline = baselines[baseline_key]
        result.baselines_used.add(baseline_key)

        if 'low_ci' not in baseline or 'high_ci' not in baseline:
            print(f"  ⚠ Warning: Missing low_ci or high_ci in baseline for '{baseline_key}'")
            result.unfilled_placeholders.append((metric_key, f"Missing low_ci or high_ci in baseline '{baseline_key}'"))
            continue

        low_ci = baseline['low_ci']
        high_ci = baseline['high_ci']

        # Determine format based on value range
        if low_ci <= 1.0 and high_ci <= 1.0:
            low_ci_str = decimal_to_percentage(low_ci)
            high_ci_str = decimal_to_percentage(high_ci)
        else:
            low_ci_str = format_large_number(low_ci)
            high_ci_str = format_large_number(high_ci)

        # Replace placeholders in this specific metric block
        updated_metric_block = full_metric_block.replace('{low_ci}', low_ci_str)
        updated_metric_block = updated_metric_block.replace('{high_ci}', high_ci_str)

        # Replace the metric block in the full text
        scenario_text = scenario_text.replace(full_metric_block, updated_metric_block)

        print(f"  ✓ Filled placeholders in '{metric_key}': low_ci={low_ci_str}, high_ci={high_ci_str}")
        result.filled_placeholders.append(metric_key)
        replacements_made += 1

    # Process steps section
    # Pattern to match entire step blocks: from "- name:" to the next "- name:" or end of steps section
    step_block_pattern = re.compile(
        r'(  - name: ["\']?([^"\'\n]+)["\']?\n(?:(?!  - name:).*\n?)*)',
        re.MULTILINE
    )

    # Find and process each step block
    for match in step_block_pattern.finditer(scenario_text):
        full_step_block = match.group(1)
        full_step_name = match.group(2)
        step_name = extract_step_name(full_step_name)

        # Check if this step has placeholders
        if '{low_ci}' not in full_step_block and '{high_ci}' not in full_step_block:
            continue

        # Look up baseline data for this step
        if step_name not in baselines:
            print(f"  ⚠ Warning: No baseline data found for step '{step_name}'")
            result.unfilled_placeholders.append((step_name, "No baseline data found"))
            continue

        baseline = baselines[step_name]
        result.baselines_used.add(step_name)

        if 'low_ci' not in baseline or 'high_ci' not in baseline:
            print(f"  ⚠ Warning: Missing low_ci or high_ci in baseline for '{step_name}'")
            result.unfilled_placeholders.append((step_name, "Missing low_ci or high_ci in baseline"))
            continue

        low_ci = baseline['low_ci']
        high_ci = baseline['high_ci']

        # Determine format based on value range
        if low_ci <= 1.0 and high_ci <= 1.0:
            low_ci_str = decimal_to_percentage(low_ci)
            high_ci_str = decimal_to_percentage(high_ci)
        else:
            low_ci_str = format_large_number(low_ci)
            high_ci_str = format_large_number(high_ci)

        # Replace placeholders in this specific step block
        updated_step_block = full_step_block.replace('{low_ci}', low_ci_str)
        updated_step_block = updated_step_block.replace('{high_ci}', high_ci_str)

        # Replace the step block in the full text
        scenario_text = scenario_text.replace(full_step_block, updated_step_block)

        print(f"  ✓ Filled placeholders in '{step_name}': low_ci={low_ci_str}, high_ci={high_ci_str}")
        result.filled_placeholders.append(step_name)
        replacements_made += 1

    # Calculate unused baselines
    result.unused_baselines = sorted(result.baselines_available - result.baselines_used)

    if replacements_made == 0:
        print("\nNo placeholders found to replace.")
        result.error_message = "No placeholders found to replace"
        return result

    # Save the updated scenario
    print(f"\nSaving filled scenario to: {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(scenario_text)
    except Exception as e:
        result.error_message = f"Failed to write file: {e}"
        print(f"Error: {result.error_message}")
        return result

    print(f"✓ Successfully filled placeholders")
    result.success = True
    return result


def print_detailed_summary(results: List[ProcessingResult]) -> None:
    """Print a detailed summary of all processing results."""
    print("\n" + "=" * 80)
    print("DETAILED SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print(f"\nTotal scenarios processed: {len(results)}")
    print(f"✓ Successful: {len(successful)}")
    print(f"✗ Failed: {len(failed)}")

    # Report failed scenarios
    if failed:
        print("\n" + "-" * 80)
        print("FAILED SCENARIOS")
        print("-" * 80)
        for r in failed:
            print(f"\n  ✗ {r.scenario_name}")
            if r.error_message:
                print(f"    Error: {r.error_message}")

    # Report unfilled placeholders across all scenarios
    scenarios_with_unfilled = [r for r in results if r.unfilled_placeholders]
    if scenarios_with_unfilled:
        print("\n" + "-" * 80)
        print("UNFILLED PLACEHOLDERS")
        print("-" * 80)
        for r in scenarios_with_unfilled:
            print(f"\n  {r.scenario_name}:")
            for name, reason in r.unfilled_placeholders:
                print(f"    ⚠ {name}: {reason}")

    # Report unused baseline data across all scenarios
    scenarios_with_unused = [r for r in results if r.unused_baselines]
    if scenarios_with_unused:
        print("\n" + "-" * 80)
        print("UNUSED BASELINE DATA")
        print("-" * 80)
        for r in scenarios_with_unused:
            print(f"\n  {r.scenario_name}:")
            print(f"    Baselines available: {len(r.baselines_available)}")
            print(f"    Baselines used: {len(r.baselines_used)}")
            print(f"    Unused baselines ({len(r.unused_baselines)}):")
            for baseline_key in r.unused_baselines:
                print(f"      - {baseline_key}")

    # Final status
    print("\n" + "=" * 80)
    has_issues = bool(failed or scenarios_with_unfilled or scenarios_with_unused)
    if has_issues:
        print("⚠ Processing completed with warnings/errors. Review the details above.")
    else:
        print("✓ All scenarios processed successfully with no issues.")
    print("=" * 80)


def main():
    """Main entry point for the script."""
    # Define mappings: (template_scenario, baseline, output_scenario)
    # Adjust these paths as needed based on your file naming conventions
    mappings: List[Tuple[str, str, str]] = [
        # OC1
        (
            "input_data/scenario/scenario_templates/OC1_phishing.yaml",
            "input_data/baselines/OC1_phishing.yaml",
            "input_data/scenario/OC1_phishing.yaml"
        ),
        # OC2
        (
            "input_data/scenario/scenario_templates/OC2_IAB.yaml",
            "input_data/baselines/OC2_iab.yaml",
            "input_data/scenario/OC2_IAB.yaml"
        ),
        (
            "input_data/scenario/scenario_templates/OC2_Data_Breach.yaml",
            "input_data/baselines/OC2_ransom.yaml",
            "input_data/scenario/OC2_Data_Breach.yaml"
        ),
        # OC3
        (
            "input_data/scenario/scenario_templates/OC3_DDoS.yaml",
            "input_data/baselines/OC3_ddos.yaml",
            "input_data/scenario/OC3_DDoS.yaml"
        ),
        (
            "input_data/scenario/scenario_templates/OC3_Ransomware_large.yaml",
            "input_data/baselines/OC3_lge_ransomware.yaml",
            "input_data/scenario/OC3_Ransomware_large.yaml"
        ),
        (
            "input_data/scenario/scenario_templates/OC3_Ransomware.yaml",
            "input_data/baselines/OC3_sme_ransomware.yaml",
            "input_data/scenario/OC3_Ransomware.yaml"
        ),
        # OC4
        (
            "input_data/scenario/scenario_templates/OC4_infrastructure_large.yaml",
            "input_data/baselines/OC4_infra_large.yaml",
            "input_data/scenario/OC4_infrastructure_large.yaml"
        ),
        (
            "input_data/scenario/scenario_templates/OC4_infrastructure.yaml",
            "input_data/baselines/OC4_infra_small.yaml",
            "input_data/scenario/OC4_infrastructure.yaml"
        ),
        # OC5
        (
            "input_data/scenario/scenario_templates/OC5_espionage.yaml",
            "input_data/baselines/OC5_espionage.yaml",
            "input_data/scenario/OC5_espionage.yaml"
        ),
    ]

    print("=" * 80)
    print("Filling placeholders in scenario templates")
    print("=" * 80)

    total_scenarios = len(mappings)
    results: List[ProcessingResult] = []

    for i, (template, baseline, output) in enumerate(mappings, 1):
        print(f"\n[{i}/{total_scenarios}] Processing: {Path(template).name}")
        print("-" * 80)

        template_path = Path(template)
        baseline_path = Path(baseline)
        output_path = Path(output)

        # Validate input files exist
        if not template_path.exists():
            print(f"✗ Error: Template file does not exist: {template_path}")
            result = ProcessingResult(
                scenario_name=template_path.name,
                error_message=f"Template file does not exist: {template_path}"
            )
            results.append(result)
            continue

        if not baseline_path.exists():
            print(f"✗ Error: Baseline file does not exist: {baseline_path}")
            result = ProcessingResult(
                scenario_name=template_path.name,
                error_message=f"Baseline file does not exist: {baseline_path}"
            )
            results.append(result)
            continue

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Run the fill operation
        result = fill_scenario_placeholders(template_path, baseline_path, output_path)
        results.append(result)

    # Print detailed summary
    print_detailed_summary(results)

    # Exit with appropriate code
    has_failures = any(not r.success for r in results)
    sys.exit(1 if has_failures else 0)


if __name__ == '__main__':
    main()
