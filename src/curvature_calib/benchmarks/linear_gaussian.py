"""Analytic linear-Gaussian benchmark for EXP-001.

Representation and loss (registry EXP-001; coordinates are z directly, T=Id):

    m(z) = A z,        L(z) = 1/2 || A z - y ||_W^2.

Because m is affine, the exact Hessian equals the GGN (R = H - G = 0), and both
equal the analytic matrix

    G = A^T W A.

`random_A` builds A via an SVD with a prescribed condition number of A^T A
(with W = I) and optional rank deficiency, so cond(A^T A) = `cond` exactly.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class LinearGaussian(NamedTuple):
    A: Array            # (K, P)
    y: Array            # (K,)
    W: Array | None     # (K,) diagonal, (K, K) dense, or None (identity)


def _apply_W(W: Array | None, r: Array) -> Array:
    if W is None:
        return r
    if W.ndim == 1:
        return W * r
    return W @ r


def make_representation(model: LinearGaussian) -> Callable[[Array], Array]:
    """z -> A z, shape (P,) -> (K,)."""
    A = model.A
    return lambda z: A @ z


def make_loss(model: LinearGaussian) -> Callable[[Array], Array]:
    """z -> 1/2 (A z - y)^T W (A z - y)."""
    A, y, W = model.A, model.y, model.W

    def loss(z: Array) -> Array:
        r = A @ z - y
        return 0.5 * jnp.dot(r, _apply_W(W, r))

    return loss


def analytic_ggn(model: LinearGaussian) -> Array:
    """G = A^T W A, shape (P, P), symmetrized."""
    A, W = model.A, model.W
    if W is None:
        G = A.T @ A
    elif W.ndim == 1:
        G = A.T @ (W[:, None] * A)
    else:
        G = A.T @ (W @ A)
    return 0.5 * (G + G.T)


def analytic_gradient(model: LinearGaussian, z: Array) -> Array:
    """grad L = A^T W (A z - y), shape (P,)."""
    A, y, W = model.A, model.y, model.W
    return A.T @ _apply_W(W, A @ z - y)


def _random_orthogonal(key: Array, n: int, dtype) -> Array:
    """Haar-ish orthogonal (n, n) via QR with deterministic sign fixing."""
    M = jax.random.normal(key, (n, n), dtype=dtype)
    Q, R = jnp.linalg.qr(M)
    d = jnp.sign(jnp.diag(R))
    d = jnp.where(d == 0, jnp.ones_like(d), d)
    return Q * d


def random_A(key: Array, K: int, P: int, cond: float | None = None,
             rank: int | None = None, dtype=jnp.float64) -> Array:
    """SVD-constructed A: (K, P). cond(A^T A) == `cond` (W=I); optional rank<P.

    Right singular vectors are the columns of V; the null space of a
    rank-deficient A is exactly V[:, rank:] (the zeroed singular values).
    """
    if rank is None and K < P:
        raise ValueError(f"full-rank A needs K >= P; got K={K}, P={P}")
    kU, kV = jax.random.split(key)
    U = _random_orthogonal(kU, K, dtype)[:, :P]   # (K, P)
    V = _random_orthogonal(kV, P, dtype)          # (P, P)

    if cond is None:
        sigma = jnp.ones((P,), dtype=dtype)
    elif P == 1:
        sigma = jnp.ones((1,), dtype=dtype)
    else:
        sqrt_cond = jnp.sqrt(jnp.asarray(cond, dtype=dtype))
        exps = jnp.linspace(0.0, jnp.log10(sqrt_cond), P, dtype=dtype)
        sigma = jnp.power(jnp.asarray(10.0, dtype=dtype), exps)

    if rank is not None and rank < P:
        mask = (jnp.arange(P) < rank).astype(dtype)
        sigma = sigma * mask

    A = U @ (sigma[:, None] * V.T)   # (K, P)
    return A


def right_singular_vectors(key: Array, K: int, P: int, dtype=jnp.float64) -> Array:
    """The V used by random_A for the same key — for null-space checks."""
    _, kV = jax.random.split(key)
    return _random_orthogonal(kV, P, dtype)
