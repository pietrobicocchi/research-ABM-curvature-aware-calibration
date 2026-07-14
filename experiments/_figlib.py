"""Shared helpers for the manuscript figure-render scripts.

Each figNN_*.py LOADS a frozen experiment run record (never re-runs the
experiment) and renders a house-style figure into outputs/figures/. Provenance is
the source experiment's frozen run plus the committed render script.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

FIGDIR = "outputs/figures"


def latest_run(exp: str) -> str:
    dirs = sorted(glob.glob(f"outputs/{exp}/*/"))
    if not dirs:
        raise FileNotFoundError(f"no run record under outputs/{exp}/")
    return dirs[-1]


def load_metrics(exp: str):
    run = latest_run(exp)
    with open(os.path.join(run, "metrics.json")) as f:
        return json.load(f), run


def load_arrays(exp: str):
    return np.load(os.path.join(latest_run(exp), "arrays.npz"))


def savefig(fig, name: str) -> str:
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}")
    return path


def cov_ellipse(cov2, center=(0, 0), nsig=1.0, n=200):
    """(x,y) of the nsig covariance ellipse for a 2x2 covariance."""
    w, U = np.linalg.eigh(np.asarray(cov2))
    t = np.linspace(0, 2 * np.pi, n)
    circ = np.stack([np.cos(t), np.sin(t)])
    pts = (U @ (np.sqrt(np.maximum(w, 0))[:, None] * circ)) * nsig
    return pts[0] + center[0], pts[1] + center[1]
