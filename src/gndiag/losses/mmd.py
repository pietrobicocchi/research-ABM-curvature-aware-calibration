"""Median-heuristic bandwidth for the Gaussian kernel of the MMD."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def _sq_dists(X: jax.Array, Y: jax.Array) -> jax.Array:
    xx = jnp.sum(X * X, axis=1, keepdims=True)
    yy = jnp.sum(Y * Y, axis=1, keepdims=True)
    return jnp.clip(xx + yy.T - 2.0 * X @ Y.T, min=0.0)


def median_heuristic(X: jax.Array) -> jax.Array:
    """sqrt(median pairwise squared distance / 2) over distinct rows of X."""
    d2 = _sq_dists(X, X)
    iu = jnp.triu_indices(d2.shape[0], k=1)
    return jnp.sqrt(jnp.median(d2[iu]) / 2.0 + 1e-12)
