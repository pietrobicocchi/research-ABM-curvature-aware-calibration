"""Maximum Mean Discrepancy with Gaussian RBF kernel.

We use the *unbiased* squared MMD estimator (Gretton et al. 2012):
    MMD^2(P, Q) = E_xx' k(x,x') + E_yy' k(y,y') - 2 E_xy k(x,y),
estimated by U-statistics excluding the diagonal.

Bandwidth via the median heuristic on pooled pairwise L2 distances.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _sq_dists(X: jax.Array, Y: jax.Array) -> jax.Array:
    # X: (M, D), Y: (N, D) -> (M, N) squared L2 distances.
    xx = jnp.sum(X * X, axis=1, keepdims=True)
    yy = jnp.sum(Y * Y, axis=1, keepdims=True)
    return jnp.clip(xx + yy.T - 2.0 * X @ Y.T, min=0.0)


def rbf_kernel(X: jax.Array, Y: jax.Array, sigma: jax.Array | float) -> jax.Array:
    """Gaussian RBF kernel matrix k(x, y) = exp(-||x-y||^2 / (2 sigma^2))."""
    return jnp.exp(-_sq_dists(X, Y) / (2.0 * sigma * sigma))


def median_heuristic(X: jax.Array, Y: jax.Array | None = None) -> jax.Array:
    """sigma = sqrt(median of pairwise sq-distances / 2) on the pooled sample.

    Standard scale-equivariant choice. If Y is None uses X-X distances only.
    Excludes self-distances.
    """
    Z = X if Y is None else jnp.concatenate([X, Y], axis=0)
    d2 = _sq_dists(Z, Z)
    n = d2.shape[0]
    # Mask out the diagonal (zeros), take median of the upper triangle.
    iu = jnp.triu_indices(n, k=1)
    return jnp.sqrt(jnp.median(d2[iu]) / 2.0 + 1e-12)


def mmd_sq_unbiased(X: jax.Array, Y: jax.Array, sigma: jax.Array | float) -> jax.Array:
    """Unbiased squared MMD with RBF kernel of bandwidth sigma."""
    M = X.shape[0]
    N = Y.shape[0]
    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)
    sum_xx = (jnp.sum(Kxx) - jnp.sum(jnp.diag(Kxx))) / (M * (M - 1))
    sum_yy = (jnp.sum(Kyy) - jnp.sum(jnp.diag(Kyy))) / (N * (N - 1))
    sum_xy = jnp.sum(Kxy) / (M * N)
    return sum_xx + sum_yy - 2.0 * sum_xy


def mmd_sq_with_median_bandwidth(X: jax.Array, Y: jax.Array) -> jax.Array:
    """Convenience: pick sigma from the pooled median heuristic, then compute MMD^2.

    The bandwidth is computed under `stop_gradient` so it does not carry
    gradient back through the median operator (which is non-differentiable at
    ties and would inject spurious second-order terms otherwise).
    """
    sigma = jax.lax.stop_gradient(median_heuristic(X, Y))
    return mmd_sq_unbiased(X, Y, sigma)
