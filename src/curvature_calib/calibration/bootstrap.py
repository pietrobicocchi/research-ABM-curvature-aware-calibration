"""Bootstrap confidence intervals for OPG eigenvalues and eigenvector subspaces.

All resampling is done in numpy for simplicity; inputs are JAX arrays but
outputs are JAX arrays for compatibility with downstream callers.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def bootstrap_eigvals(
    per_seed_grads: jax.Array,
    n_boot: int = 500,
    key: jax.Array | None = None,
) -> jax.Array:
    """Bootstrap distribution of OPG eigenvalues by resampling the M seeds.

    Returns (n_boot, P) array of descending eigenvalues per replicate.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    G = np.asarray(per_seed_grads)
    M, P = G.shape
    indices = np.asarray(jax.random.randint(key, (n_boot, M), 0, M))
    out = np.empty((n_boot, P))
    for b in range(n_boot):
        Gb = G[indices[b]]
        Fb = (Gb.T @ Gb) / M
        Fb = 0.5 * (Fb + Fb.T)
        w = np.linalg.eigvalsh(Fb)
        out[b] = np.sort(w)[::-1]
    return jnp.asarray(out)


def eigenvalue_cis(
    boot_eigvals: jax.Array,
    confidence: float = 0.95,
) -> jax.Array:
    """Percentile CIs for each eigenvalue from the bootstrap distribution.

    Returns (P, 2) where [:, 0] is lower bound, [:, 1] is upper bound.
    """
    alpha = (1.0 - confidence) / 2.0
    arr = np.asarray(boot_eigvals)
    lo = np.percentile(arr, 100 * alpha, axis=0)
    hi = np.percentile(arr, 100 * (1 - alpha), axis=0)
    return jnp.stack([jnp.asarray(lo), jnp.asarray(hi)], axis=1)


def noise_threshold(eigval_cis: jax.Array) -> float:
    """Upper CI bound of the smallest eigenvalue — the noise floor for d_eff."""
    return float(eigval_cis[-1, 1])


def bootstrap_subspace_cis(
    per_seed_grads: jax.Array,
    k: int,
    n_boot: int = 500,
    confidence: float = 0.95,
    key: jax.Array | None = None,
) -> float:
    """Upper percentile of max principal angle between bootstrap top-k subspaces
    and the full-data top-k subspace.

    Returns the (confidence * 100)-th percentile of the max-angle distribution.
    """
    from curvature_calib.calibration.diagnostic import eigendecompose, principal_angles

    if key is None:
        key = jax.random.PRNGKey(0)
    G = np.asarray(per_seed_grads)
    M = G.shape[0]
    F_full = jnp.asarray((G.T @ G) / M)
    V_full = eigendecompose(F_full).eigvecs[:, :k]

    indices = np.asarray(jax.random.randint(key, (n_boot, M), 0, M))
    max_angles = np.empty(n_boot)
    for b in range(n_boot):
        Gb = G[indices[b]]
        Fb = jnp.asarray((Gb.T @ Gb) / M)
        Vb = eigendecompose(Fb).eigvecs[:, :k]
        angles = np.asarray(principal_angles(V_full, Vb))
        max_angles[b] = float(np.max(angles))
    return float(np.percentile(max_angles, 100 * confidence))
