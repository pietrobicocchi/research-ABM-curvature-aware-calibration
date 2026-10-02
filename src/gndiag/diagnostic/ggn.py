"""Eigen-analysis of the Gauss-Newton matrix G = J_m^T W J_m (Eq. 5)."""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class EigDecomp(NamedTuple):
    eigvals: Array  # (P,), descending
    eigvecs: Array  # (P, P), column k is the k-th eigenvector


def symmetrize(A: Array) -> Array:
    return 0.5 * (A + A.T)


def eigendecompose(F: Array) -> EigDecomp:
    """Symmetric eigendecomposition, eigenvalues in descending order (stiff to sloppy)."""
    w, V = jnp.linalg.eigh(0.5 * (F + F.T))
    order = jnp.argsort(-w)
    return EigDecomp(eigvals=w[order], eigvecs=V[:, order])


def principal_angles(V1: Array, V2: Array) -> Array:
    """Principal angles (radians, ascending) between the column spans of V1 and V2."""
    Q1, _ = jnp.linalg.qr(V1)
    Q2, _ = jnp.linalg.qr(V2)
    s = jnp.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return jnp.arccos(jnp.clip(s, min=-1.0, max=1.0))


def max_principal_angle(V1: Array, V2: Array) -> float:
    """Largest principal angle (radians) between the two subspaces."""
    return float(jnp.max(principal_angles(V1, V2)))


def rel_frobenius_error(A_est: Array, A_ref: Array) -> float:
    return float(jnp.linalg.norm(A_est - A_ref)) / max(float(jnp.linalg.norm(A_ref)), 1e-300)


def directional_loss_change(loss_fn: Callable[[Array], Array], z_hat: Array,
                            direction: Array, alphas: Array) -> Array:
    """L(z_hat + alpha d) - L(z_hat) for unit d along `direction`."""
    d = direction / jnp.linalg.norm(direction)
    L0 = loss_fn(z_hat)
    return jax.vmap(lambda a: loss_fn(z_hat + a * d) - L0)(alphas)
