"""Resolve LLM_elicitation repo root (directory containing src/main.py)."""
from pathlib import Path


def project_root() -> Path:
    p = Path(__file__).resolve()
    for ancestor in p.parents:
        if (ancestor / "src" / "main.py").exists():
            return ancestor
    raise RuntimeError("Could not find project root (src/main.py).")
