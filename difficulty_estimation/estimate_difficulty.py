import yaml
import re
import json
from pathlib import Path

from dotenv import load_dotenv
from utils.llm_api import get_llm_response, await_batch_list
from utils.utils import load_yaml, load_prompt_template, load_tasks
from utils.benchmark_adapter import get_adapter
from formatting import format_task_list, format_single_task_prompt

ALL_MODES = ["iterative_hardest", "iterative_easiest", "difficulty", "fst"]

def parse_llm_response(response: str, mode: str ="iterative_hardest") -> dict:
    """Parses the FST and Reasoning from the LLM's response."""
    data_type = float
    if mode == "iterative_hardest":
        capture = "Hardest Task"
        data_type = int
    elif mode == "iterative_easiest":
        capture = "Easiest Task"
        data_type = int
    elif mode == "difficulty":
        capture = "Difficulty"
    elif mode == "fst":
        capture = "FST"
    elif mode == "pairwise":
        capture = "More challenging task"
        data_type = int
    else:
        raise ValueError(f"mode must be in {ALL_MODES}")
    
    estimate_match = re.search(capture + r": (\d+)", response.replace("**", ""))
    reasoning_match = re.search(r"Reasoning: (.*)", response.replace("**", ""))

    try:
        estimate = data_type(estimate_match.group(1)) if estimate_match else None
        reasoning = reasoning_match.group(1).strip() if reasoning_match else None
    except ValueError as e:
        print(f"ERROR: Failed to parse estimate from response. Pattern: '{capture}', Error: {e}")
        print(f"Response excerpt: {response[:500]}...")
        estimate = None
        reasoning = None

    return {
        "estimate": estimate,
        "reasoning": reasoning
    }

def get_rank_from_fst_difficulty(task_difficulties: dict, task: dict, adapter) -> float:
    task_id = adapter.get_task_id(task)
    fst_if_exists = task_difficulties[task_id].get('fst', 0)
    return task_difficulties[task_id].get('difficulty', fst_if_exists)



def main(config="config/config.yaml", retries=3):
    """Main function to run the estimation experiment."""
    config = load_yaml(config)
    tasks = load_tasks(config["tasks_path"])
    mode = config["mode"]
    prompt_template = load_prompt_template(f"{config['prompt_template']}")

    benchmark_type = config.get("benchmark_type", "bountybench")
    adapter = get_adapter(benchmark_type)

    parsed_results = {}

    while len(tasks) > 0:
        print(f"Processing {len(tasks)} tasks...")
        task_map = {}
        if mode.startswith("iterative_"):
            task_list, task_map = format_task_list(tasks, adapter)
            prompt = prompt_template.format(**{"task_description": task_list})
            requests = [dict(
                    custom_id=str(len(tasks)),
                    params=dict(
                        model=config["llm_settings"]['model_name'],
                        max_tokens=18000, # A reasonable default, can be adjusted if needed
                        thinking={"type": "enabled", "budget_tokens":10000},
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                )]
        else:
            start_ix = 0
            requests = []
            task_map = {}
  
            for i, task in enumerate(tasks):
                task_id = adapter.get_task_id(task)
                task_map[i] = task_id

            if mode == "pairwise":
                task_difficulties = load_yaml(f"{config['ranking_file']}")
                tasks.sort(key=lambda task: get_rank_from_fst_difficulty(task_difficulties, task, adapter))
                start_ix = 1
            for b in range(start_ix, len(tasks)):
                task = tasks[b]
                if mode == "pairwise":
                    prev_task = tasks[b-1]
                else:
                    prev_task = None
                
                code_cache_dir = config.get("code_cache_dir")
                prompt = format_single_task_prompt(prompt_template, task, adapter,
                                                  config['include_code'], code_cache_dir,
                                                  prev_task=prev_task)
                
                task_name = task.get('name', adapter.get_task_id(task))
                print(f"--- Processing task: {task_name} ---")

                task_id = adapter.get_task_id(task)
                if task_id:
                    requests.append(dict(
                        custom_id=task_id,
                        params=dict(
                            model=config["llm_settings"]['model_name'],
                            max_tokens=4000,
                            messages=[
                                {"role": "user", "content": prompt}
                            ]
                        )
                    ))
                else:
                    print(f"Could not get ID for task {task}")

        
        batch_id, client = get_llm_response(requests)
        batch_results = await_batch_list(batch_id, client)
        
        if config['raw_output_file']:
            if config['raw_output_file'].endswith(".txt"):
                config['raw_output_file'] = config['raw_output_file'][:-4]
            output_path = Path(config['raw_output_file'] + str(len(tasks)) + ".txt")
            output_path.parent.mkdir(exist_ok=True, parents=True)
            with open(output_path, 'w') as f:
                for task_id, response in batch_results.items():
                    f.write("=" * 30 + task_id + "=" * 30 + "\n")
                    thinking_text = response.get("thinking")
                    if thinking_text:
                        f.write("-" * 25 + " CoT " + "-" * 25 + "\n")
                        f.write(thinking_text)
                        f.write("\n")
                        f.write("-" * 25 + " Response " + "-" * 25 + "\n")
                    f.write(response["text"])

                    f.write("\n" + "=" * 70 + "\n\n")


        # Process results - note that batch_results keys are already task_ids (custom_ids) from await_batch_list
        for task_id, response in batch_results.items():
            parsed_response = parse_llm_response(response["text"], mode=mode)
            if parsed_response["estimate"] is None:
                print(f"WARNING: Estimation failed for task {task_id} - skipping this task")
                continue

            if mode.startswith("iterative_"):
                ix = int(parsed_response["estimate"])
                parsed_response["estimate"] = f"{ix} / {len(tasks)}"
                task_id = task_map[ix]
                tasks.pop(ix)

            parsed_results[task_id] = {**parsed_response, **{"task_id":task_id}}
            
        if not mode.startswith("iterative_"):
            tasks = []


    if mode == "iterative_hardest":
        # Reversed order (hardest first)
        parsed_results = list(reversed(parsed_results.values()))
    elif mode in ["iterative_easiest", "pairwise"]:
        parsed_results = list(parsed_results.values())
    else:
        # For difficulty and fst modes: sort by estimated difficulty (ascending)
        parsed_results = sorted(
                parsed_results.values(),
                key=lambda x: x["estimate"]
        )
    

    
    # Write to file
    output_path = Path(config['output_file'])
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, 'w') as f:
        yaml.dump(parsed_results, f)



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Estimate difficulty rankings from task descriptions")
    ap.add_argument("--config", type=str, default="config/config.yaml")
    args = ap.parse_args()
    load_dotenv(override=True)
    main(**vars(args))