"""Shared Brock-Hommes set-up: MMD mean embedding of the price trajectory with frozen
random Fourier features, and its Gauss-Newton matrix G = J^T J (W = I)."""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from bh_model import simulate
from gndiag.diagnostic.ggn import symmetrize
from gndiag.losses import rff
from gndiag.losses.mmd import median_heuristic

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"

T, R_GROSS, SIGMA = 100, 1.01, 0.04
PRIOR_SD = jnp.array([0.30, 0.10, 0.05, 0.10, 0.05])
POINTS = {"complex": (50.0, 0.9, 0.2, 0.9, -0.2),     # beta = 50
          "chaotic": (80.0, 0.9, 0.2, 0.9, -0.2)}     # beta = 80
WORKING_HORIZON = 20
N_FEATURES, FEATURE_SEED, N_SEEDS = 256, 7, 128
Z0 = jnp.zeros(5)
PARAM_LABELS = [r"$\beta$", r"$g_1$", r"$b_1$", r"$g_2$", r"$b_2$"]


def sim(theta0, grad_horizon):
    """z -> trajectory, with theta = theta0 + PRIOR_SD * z."""
    return lambda z, key: simulate(jnp.asarray(theta0) + PRIOR_SD * z, key,
                                   T=T, R=R_GROSS, sigma=SIGMA, grad_horizon=grad_horizon)


def embedding(theta0, features, keys, grad_horizon):
    """z -> seed-mean random-feature embedding (common random numbers)."""
    s = sim(theta0, grad_horizon)
    return lambda z: jnp.mean(rff.feature_map(features, jax.vmap(lambda k: s(z, k))(keys)), axis=0)


def ggn(embedding_fn, z):
    J = jax.jacfwd(embedding_fn)(z)
    return jnp.asarray(symmetrize(J.T @ J)), J


def setup(theta0):
    """Seeds and frozen features (median-heuristic bandwidth at theta0)."""
    keys = jax.random.split(jax.random.PRNGKey(0), N_SEEDS)
    X = jax.vmap(lambda k: sim(theta0, WORKING_HORIZON)(Z0, k))(keys)
    features = rff.frozen_rff(jax.random.PRNGKey(FEATURE_SEED), N_FEATURES, T,
                              float(median_heuristic(X)))
    return keys, features, X


def save(name, results):
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(results, indent=1))


def load(name):
    return json.loads((RESULTS_DIR / f"{name}.json").read_text())


def save_figure(fig, name):
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.pdf")


def to_list(x):
    return np.asarray(x).tolist()
