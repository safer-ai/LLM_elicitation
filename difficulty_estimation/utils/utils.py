import yaml

def load_yaml(config_path: str) -> dict:
    """Loads from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_tasks(tasks_path: str) -> list:
    """Loads tasks from the benchmark YAML file."""
    with open(tasks_path, 'r') as f:
        return yaml.safe_load(f).get('tasks', [])

def load_bounties(bounties_path: str) -> list:
    """Deprecated: Use load_tasks() instead. Kept for backward compatibility."""
    return load_tasks(bounties_path)

def load_prompt_template(template_path: str) -> str:
    """Loads the prompt template from a file."""
    with open(template_path, 'r') as f:
        return f.read()