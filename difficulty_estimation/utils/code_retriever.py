import yaml
import os
import subprocess
import shutil

def load_bounties(bounties_path: str) -> list:
    """Loads the bounty tasks from the main bounties.yaml file."""
    with open(bounties_path, 'r') as f:
        return yaml.safe_load(f).get('tasks', [])

def get_project_name_from_task(task: dict) -> str:
    """Extracts the project name from the task name."""
    name_parts = task.get('name', '').split(' in ')
    if len(name_parts) > 1:
        repo_name = name_parts[-1].split('/')[-1]
        return repo_name
    return None

def get_bounty_id(task: dict) -> str:
    """Generates a unique ID for the bounty from its source URL."""
    source_url = task.get('source_url', '')
    if not source_url:
        return None
    return source_url.split('/')[-1]

def retrieve_and_cache_code(bounties_path: str, cache_dir: str):
    """Retrieves and caches code snippets for all bounties."""
    bounties = load_bounties(bounties_path)

    for task in bounties:
        bounty_id = get_bounty_id(task)
        if not bounty_id:
            print(f"Skipping task with no source URL: {task.get('name')}")
            continue

        code_root = task.get('metadata', {}).get('code_root')
        if not code_root:
            print(f"Skipping task with no code_root: {task.get('name')}")
            continue

        project_name = get_project_name_from_task(task)
        if not project_name:
            print(f"Skipping task with no project name: {task.get('name')}")
            continue

        bounty_cache_dir = os.path.join(cache_dir, bounty_id)
        os.makedirs(bounty_cache_dir, exist_ok=True)

        

        # --- Git operations ---


        if task.get('metadata', {}).get('patch'):
            submodule_base = os.path.join(code_root, "../../codebase")
            submodule_path = os.path.relpath(submodule_base, './bountytasks')
    
            print(f"--- Processing {project_name} ---")
            print(f"Initializing submodule: {submodule_path}")
            subprocess.run(["git", "submodule", "init", submodule_path], check=True, cwd='./bountytasks')
            print(f"Updating submodule: {submodule_path}")
            subprocess.run(["git", "submodule", "update", "--depth=1", submodule_path], check=True, cwd='./bountytasks')

            for patched_file, unpatched_file in task['metadata']['patch'].items():
                patched_file_path = os.path.join(code_root, patched_file)
                unpatched_file_path = os.path.join(code_root,  "../..", unpatched_file)
                print(patched_file_path, unpatched_file_path)
                # Write patched code to cache
                try:
                    with open(patched_file_path, 'r') as f_in, open(os.path.join(bounty_cache_dir, 'patched_code.txt'), 'w') as f_out:
                        f_out.write(f_in.read())
                except FileNotFoundError:
                    with open(os.path.join(bounty_cache_dir, 'patched_code.txt'), 'w') as f_out:
                        f_out.write("# Patched file not found")

                # Write unpatched code to cache
                try:
                    with open(unpatched_file_path, 'r') as f_in, open(os.path.join(bounty_cache_dir, 'unpatched_code.txt'), 'w') as f_out:
                        f_out.write(f_in.read())
                except FileNotFoundError:
                    with open(os.path.join(bounty_cache_dir, 'unpatched_code.txt'), 'w') as f_out:
                        f_out.write("# Unpatched file not found")
            print(f"De-initializing submodule: {submodule_path}")
            subprocess.run(["git", "submodule", "deinit", "-f", submodule_path], check=True, cwd='./bountytasks')
            if os.path.exists(submodule_path):
                shutil.rmtree(submodule_base, ignore_errors=True)
            print(f"Cached code for bounty: {bounty_id}")
        else:
             with open(os.path.join(bounty_cache_dir, 'patched_code.txt'), 'w') as f_out:
                f_out.write("# No patch information found")
             with open(os.path.join(bounty_cache_dir, 'unpatched_code.txt'), 'w') as f_out:
                f_out.write("# No patch information found")


        # --- Cleanup operations ---


if __name__ == '__main__':
    bounties_yaml_path = 'bounties.yaml'
    code_cache_directory = 'llm_estimator/code_cache'
    retrieve_and_cache_code(bounties_yaml_path, code_cache_directory)
    print("\nCode retrieval and caching process complete.")