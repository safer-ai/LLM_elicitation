import yaml
import os
import re
from main import parse_llm_response

def load_raw(raw_file: str) -> dict:
    """Loads the experiment configuration from a YAML file."""
    with open(raw_file, 'r') as f:
        return yaml.safe_load(f)
    
basepath = "./llm_estimator/results/"
target_file = "difficulty_scores"

if __name__ == "__main__":
    raw_file = os.path.join(basepath, f"{target_file}_raw.yaml")
    raw_data = load_raw(raw_file)
    parsed_results = {}
    for task_id, task_data in raw_data.items():
        parsed_results[task_id] = parse_llm_response(task_data)
        print(f"Parsed results for task {task_id}: {parsed_results[task_id]}")

    output_path = os.path.join(basepath, f"{target_file}.yaml")
    with open(output_path, 'w') as f:
        yaml.dump(parsed_results, f)

