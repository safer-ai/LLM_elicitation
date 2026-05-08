#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API-key + .env resolution helpers shared across experiments.

Both `src/config.py` (the main risk-scenario workflow) and the calibration
sub-projects need the same logic: parse a `.env` file, resolve an API key
from a chain of search directories with explicit precedence, and log a
redacted prefix so the run output makes it obvious which key actually got
picked up.

The variadic `*search_dirs` form lets each caller specify its own search
order. `src/config.py` passes `(project_root,)`. The calibration projects
pass `(experiment_dir, repo_root)` so a project-local `.env` wins over the
repo-level one (which itself wins over the shell env), preventing a stale
shell-exported key from silently overriding a per-experiment `.env`.

API keys are **never** read from YAML configs. If you need to pin a key for
a specific run, drop it into the relevant `.env` file.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def redact(key: Optional[str]) -> str:
    """Return a short, safe-to-log prefix of an API key."""
    if not key:
        return "(empty)"
    return f"{key[:14]}…[len={len(key)}]"


def _parse_dotenv(path: Path) -> Dict[str, str]:
    """Minimal stdlib parser for KEY=VALUE lines in a .env file.

    - Ignores blank lines and lines starting with '#'.
    - Strips wrapping single or double quotes from the value.
    - Last assignment wins.
    - Does NOT do shell-style variable expansion.
    """
    out: Dict[str, str] = {}
    if not path.is_file():
        return out
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip()
            if (v.startswith("'") and v.endswith("'")) or (
                v.startswith('"') and v.endswith('"')
            ):
                v = v[1:-1]
            if k:
                out[k] = v
    except Exception as e:
        logger.warning(f"Failed to parse {path}: {e}")
    return out


def resolve_api_key(env_var: str, *search_dirs: Path) -> Optional[str]:
    """Resolve an API key with explicit precedence + clear logging.

    Precedence (highest first):
      1. KEY=VALUE in `<dir>/.env` for each `dir` in `search_dirs`, in order.
         (Project-local first, then repo root, etc.)
      2. Process environment variable `env_var`.

    The first non-empty source wins. Returns None if no source has a value.
    """
    for d in search_dirs:
        parsed = _parse_dotenv(d / ".env")
        if env_var in parsed and parsed[env_var]:
            v = parsed[env_var]
            logger.info(f"{env_var}: source = {d / '.env'}, key = {redact(v)}")
            return v

    env_val = os.environ.get(env_var)
    if env_val:
        logger.info(f"{env_var}: source = process env, key = {redact(env_val)}")
        return env_val

    logger.debug(f"{env_var}: no value found in .env or process env")
    return None


def parse_num_repeats(raw: Any) -> int:
    """Coerce the raw `workflow_settings.num_repeats` YAML value to a positive int.

    Rejects bools explicitly (otherwise YAML `true` would silently become 1 via
    `int(True)`), and emits a clearer error than the bare `int()` traceback for
    non-integer strings.
    """
    if isinstance(raw, bool):
        raise TypeError(
            "WorkflowSettings: 'num_repeats' must be a positive integer (got bool). "
            "Did you mean `num_repeats: 1` or `num_repeats: 20`?"
        )
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if raw.is_integer():
            return int(raw)
        raise TypeError(
            f"WorkflowSettings: 'num_repeats' must be a positive integer; got float {raw!r}."
        )
    try:
        coerced_str = str(raw).strip()
        return int(coerced_str)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"WorkflowSettings: 'num_repeats' must be a positive integer; got {raw!r}."
        ) from exc
