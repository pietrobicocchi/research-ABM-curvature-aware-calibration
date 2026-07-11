"""Run provenance metadata for reproducible experiment records.

Operationalizes the empty 05_RESULTS_LEDGER.md template WITHOUT writing to the
canonical vault: metadata is emitted only into a run directory under outputs/.
"""
from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import jax
import jax.numpy as jnp


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=5, check=False
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def short_commit() -> str:
    """Short git commit hash (or 'unknown')."""
    c = _git("rev-parse", "--short", "HEAD")
    return c or "unknown"


def run_metadata(experiment_id: str, config: dict, seeds: dict,
                 command: str | None = None) -> dict:
    """Assemble the provenance record. Call after enable_x64() so dtype is right.

    `command` is the human-facing executed command (e.g. the `uv run` invocation);
    the raw process argv is always recorded separately.
    """
    commit = _git("rev-parse", "HEAD") or "unknown"
    dirty = _git("status", "--porcelain")
    return {
        "experiment_id": experiment_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_short_commit": short_commit(),
        "git_dirty": (dirty != "") if dirty != "unknown" else "unknown",
        "executed_command": command if command is not None else " ".join(sys.argv),
        "argv": list(sys.argv),
        "jax_version": jax.__version__,
        "jaxlib_version": _pkg_version("jaxlib"),
        "numpy_version": _pkg_version("numpy"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "default_float_dtype": str(jnp.zeros(()).dtype),
        "seeds": seeds,
        "config": config,
        "plan": "EXP001_IMPLEMENTATION_PLAN.md",
    }


def write_json(path: str | Path, obj: dict) -> None:
    """Write a JSON file (sorted keys, indent 2), creating parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


# Back-compat alias used in the plan text.
def write_metadata(path: str | Path, metadata: dict) -> None:
    write_json(path, metadata)
