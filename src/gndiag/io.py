"""Saved results, figures and the command-line entry point shared by the experiments."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import jax
import numpy as np

from gndiag.config import ROOT

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def _environment() -> dict:
    return {"python": platform.python_version(), "jax": jax.__version__,
            "numpy": np.__version__, "float64": bool(jax.config.jax_enable_x64)}


def save_results(name: str, results: dict, config: dict, arrays: dict | None = None) -> None:
    """Write results/<name>.json (results, config, package versions) and, optionally,
    results/<name>.npz."""
    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {"results": results, "config": config, "environment": _environment()}
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(payload, indent=1))
    if arrays:
        np.savez_compressed(RESULTS_DIR / f"{name}.npz", **arrays)


def load_results(name: str) -> dict:
    return json.loads((RESULTS_DIR / f"{name}.json").read_text())["results"]


def load_arrays(name: str) -> dict:
    with np.load(RESULTS_DIR / f"{name}.npz") as f:
        return dict(f)


def save_figure(fig, name: str) -> Path:
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(path, metadata={"CreationDate": None})
    return path


def main(compute, plot) -> None:
    """`python -m experiments.<name> [compute|plot|all]` (default: all)."""
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step not in ("compute", "plot", "all"):
        raise SystemExit(f"unknown step {step!r}; use compute, plot or all")
    if step in ("compute", "all"):
        compute()
    if step in ("plot", "all"):
        plot()
