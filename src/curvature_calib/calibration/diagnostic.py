"""Eigenanalysis utilities for symmetric matrices: spectrum, eigenvectors,
effective dimension, principal angles.

Pure mathematical layer, agnostic to which matrix it is handed. Bootstrap
confidence intervals live in bootstrap.py. Throughout, eigenvalues are returned
in descending order.
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


def scalar_gradient_opg(per_seed_grads: jax.Array) -> jax.Array:
    """F_OPG = (1/M) Σ_m g_m g_mᵀ = (1/M) GᵀG, G shape (M, P).

    The **uncentered second moment** of per-seed *scalar-loss* gradient
    contributions g_m (rows of G). This is NOT the generalized Gauss–Newton
    matrix, Fisher information, Hessian, or a curvature/identifiability matrix
    in general (DEC-001, Math-Spec §14): for a residual loss g = Jᵀr, so
    g gᵀ = Jᵀ r rᵀ J ≠ JᵀJ, and at an exact fit g = 0 while the GGN JᵀJ ≠ 0.
    It is retained only as a labeled comparison object.
    """
    M = per_seed_grads.shape[0]
    return (per_seed_grads.T @ per_seed_grads) / M


def scalar_gradient_covariance(per_seed_grads: jax.Array) -> jax.Array:
    """C_g = (1/M) Σ_m (g_m − ḡ)(g_m − ḡ)ᵀ, the **centered** covariance of the
    per-seed scalar-loss gradients. Related to the uncentered second moment by
    F_OPG = C_g + ḡ ḡᵀ. Also NOT the GGN.
    """
    M = per_seed_grads.shape[0]
    centered = per_seed_grads - jnp.mean(per_seed_grads, axis=0, keepdims=True)
    return (centered.T @ centered) / M


# Deprecated alias — prefer scalar_gradient_opg. Retained for backward
# compatibility; the numerical object is unchanged.
opg_from_grads = scalar_gradient_opg


def principal_angles(V1: jax.Array, V2: jax.Array) -> jax.Array:
    """Principal angles (radians) between subspaces spanned by columns of V1, V2.

    Both must have the same number of columns. Returns one angle per column,
    sorted ascending (smallest first). Reference: Bjorck & Golub 1973.
    """
    assert V1.shape[1] == V2.shape[1], (
        f"V1 and V2 must have the same number of columns, got {V1.shape[1]} vs {V2.shape[1]}"
    )
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
