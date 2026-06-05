"""Eigenanalysis of the OPG matrix: spectrum, eigenvectors, effective dimension.

Pure mathematical layer. Bootstrap confidence intervals live in bootstrap.py.
Throughout, eigenvalues are returned in descending order.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class EigDecomp(NamedTuple):
    eigvals: jax.Array  # (P,) descending
    eigvecs: jax.Array  # (P, P), column k is eigenvector k


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


def principal_angles(V1: jax.Array, V2: jax.Array) -> jax.Array:
    """Principal angles (radians) between subspaces spanned by columns of V1, V2.

    Both must have the same number of columns. Returns one angle per column,
    sorted ascending (smallest first). Reference: Bjorck & Golub 1973.
    """
    Q1, _ = jnp.linalg.qr(V1)
    Q2, _ = jnp.linalg.qr(V2)
    s = jnp.linalg.svd(Q1.T @ Q2, compute_uv=False)
    s = jnp.clip(s, min=-1.0, max=1.0)
    return jnp.arccos(s)


def effective_dimension(eigvals: jax.Array, noise_floor: float) -> int:
    """Number of eigenvalues strictly above noise_floor."""
    return int(jnp.sum(eigvals > noise_floor))


def d_eff_from_bootstrap(eigval_cis: jax.Array) -> int:
    """Number of eigenvalues whose bootstrap CI lower bound is strictly positive.

    eigval_cis: (P, 2) where [:, 0] is lower bound, [:, 1] is upper bound.
    """
    return int(jnp.sum(eigval_cis[:, 0] > 0))
