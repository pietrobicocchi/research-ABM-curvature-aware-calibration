"""Eigen-analysis of the OPG matrix: spectrum, eigenvectors, bootstrap CIs,
and principal angles between successive subspaces.

The OPG matrix F_hat = (1/M) sum_m g_m g_m^T is a stochastic Gauss-Newton
approximation of the MMD^2 Hessian (residual structure). Its eigenvectors are
the identifiable parameter combinations, ordered by how strongly the data
constrains them; its eigenvalues quantify the strength of each constraint.

Throughout, eigenvalues are returned in *descending* order.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class EigDecomp(NamedTuple):
    eigvals: jax.Array     # (P,) descending
    eigvecs: jax.Array     # (P, P), column k is eigenvector k


def eigendecompose(F: jax.Array) -> EigDecomp:
    """Symmetric eigendecomposition with eigenvalues sorted descending."""
    F_sym = 0.5 * (F + F.T)
    w, V = jnp.linalg.eigh(F_sym)
    order = jnp.argsort(-w)
    return EigDecomp(eigvals=w[order], eigvecs=V[:, order])


def opg_from_grads(per_seed_grads: jax.Array) -> jax.Array:
    """F_hat = (1/M) G^T G where G has shape (M, P)."""
    M = per_seed_grads.shape[0]
    return (per_seed_grads.T @ per_seed_grads) / M


def bootstrap_eigvals(
    per_seed_grads: jax.Array,
    n_boot: int = 500,
    key: jax.Array | None = None,
) -> jax.Array:
    """Bootstrap distribution of the OPG eigenvalues over resampling the M seeds.

    Returns (n_boot, P) array of descending eigenvalues per bootstrap replicate.
    Implemented in numpy to keep the resampling outside of JIT tracing.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    G = np.asarray(per_seed_grads)  # (M, P)
    M, P = G.shape
    seeds = jax.random.randint(key, (n_boot, M), 0, M)
    seeds = np.asarray(seeds)
    out = np.empty((n_boot, P))
    for b in range(n_boot):
        Gb = G[seeds[b]]
        Fb = (Gb.T @ Gb) / M
        Fb = 0.5 * (Fb + Fb.T)
        w = np.linalg.eigvalsh(Fb)
        out[b] = np.sort(w)[::-1]
    return jnp.asarray(out)


def principal_angles(V1: jax.Array, V2: jax.Array) -> jax.Array:
    """Principal angles (radians) between subspaces spanned by columns of V1, V2.

    V1, V2 must have the same number of columns. Returns one angle per column,
    sorted ascending (smallest principal angle first). Standard reference:
    Bjorck & Golub 1973; used in the sloppy-models literature for tracking
    eigenvector subspaces across resamples / horizon settings.
    """
    Q1, _ = jnp.linalg.qr(V1)
    Q2, _ = jnp.linalg.qr(V2)
    s = jnp.linalg.svd(Q1.T @ Q2, compute_uv=False)
    s = jnp.clip(s, min=-1.0, max=1.0)
    return jnp.arccos(s)
