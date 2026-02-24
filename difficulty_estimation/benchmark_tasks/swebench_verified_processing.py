import os
import yaml
import requests
from bs4 import BeautifulSoup
import json
from datasets import load_dataset
from huggingface_hub import login

BENCHMARK_DESCRIPTION = """A benchmark for evaluating large language models' (LLMs') abilities to solve real-world software issues sourced from GitHub. The benchmark involves giving agents a code repository and issue description, and challenging them to generate a patch that resolves the problem described by the issue. Resolving issues in SWE-bench frequently requires understanding and coordinating changes across multiple functions, classes, and even files simultaneously, calling for models to interact with execution environments,process extremely long contexts and perform complex reasoning that goes far beyond traditional code generation tasks."""

hf_token = os.getenv('HF_TOKEN')
login(token=hf_token)

ds = load_dataset("SWE-bench/SWE-bench_Verified")
ds = ds['test']
assert len(ds) == 500, 'SWE-bench verified has 500 tasks'

ds_out = ds.select_columns(['instance_id', 'problem_statement', 'patch', 'test_patch', 'difficulty'])
tasks_list = list(ds_out)

output = {
    'benchmark_description': BENCHMARK_DESCRIPTION,
    'tasks': tasks_list
}

# save to yaml
with open('swebench_verified.yaml', 'w') as f:
    yaml.safe_dump(output, f, sort_keys=False, default_flow_style=False)


# scraping results
print('-' * 100)
print('Scraping results...')

# Fetch the webpage
url = "https://www.swebench.com/"
response = requests.get(url)
response.raise_for_status()  # Raise an error for bad status codes

# Parse the HTML
soup = BeautifulSoup(response.text, 'html.parser')

# Find the script tag with the specific id
script_tag = soup.find('script', {'id': 'leaderboard-data', 'type': 'application/json'})

if not script_tag:
    print("Script tag not found")
else:
    swebench_results = json.loads(script_tag.string)
    swebench_verified_results = next((x for x in swebench_results if x['name'] == 'Verified'), None)
    if not swebench_verified_results:
        print("Verified results not found")
    else:
        swebench_verified_results = swebench_verified_results['results']
        parsed = [
            {'model': result['name'], 'score': result['resolved']}
            for result in swebench_verified_results
        ]
        # sort parsed according to score:
        parsed = sorted(parsed, key=lambda x: x['score'], reverse=True)
        print(f'Found {len(parsed)} Swe-Bench Verified results.')
        
        out = {
            'metadata': {'benchmark_name': 'swebench_verified'},
            'results': parsed
        }
        # save to json
        with open('swebench_verified_leaderboard.json', 'w') as f:
            json.dump(out, f, indent=4)
