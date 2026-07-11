"""MMD-GGN estimators from per-seed feature Jacobians (Math-Spec §13).

Per-seed feature Jacobian A_m = D_z ψ(X_z(ε_m)) ∈ R^{D×P}, with E[A_m] = J_η.

    ggn_plugin(A)        -> Ĝ_V = Ĵ_ηᵀ Ĵ_η,      Ĵ_η = mean_m A_m      (PSD; biased)
    ggn_cross_seed(A)    -> Ĝ_U = (SᵀS − Q)/(M(M−1)),  S=Σ A_m, Q=Σ A_mᵀA_m  (unbiased)
    plugin_bias_covariance(A, J_eta) -> Ĉ = mean_m (A_m−J_η)ᵀ(A_m−J_η)
        so predicted plug-in bias = Ĉ / M   (E[Ĝ_V] = G_ref + (1/M)C).

The scalar-loss OPGs (population-residual and empirical-loss) are LABELED
comparison objects, NOT curvature estimates (DEC-001).
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from curvature_calib.geometry.ggn import symmetrize

Array = jax.Array


def feature_jacobians(feature_of_z: Callable[[Array, Array], Array], z: Array,
                      eps_batch: Array) -> Array:
    """Stack A_m = D_z ψ(X_z(ε_m)) over M seeds. eps_batch: (M, n) -> (M, D, P)."""
    jac1 = jax.jacfwd(lambda zz, e: feature_of_z(zz, e), argnums=0)
    return jax.vmap(lambda e: jac1(z, e))(eps_batch)


def ggn_plugin(A: Array) -> Array:
    """Ĝ_V = Ĵ_ηᵀ Ĵ_η with Ĵ_η = mean_m A_m. A: (M, D, P) -> (P, P). PSD."""
    J_hat = jnp.mean(A, axis=0)          # (D, P)
    return symmetrize(J_hat.T @ J_hat)


def ggn_plugin_matvec(A: Array, v: Array) -> Array:
    """Matrix-free Ĝ_V v = Ĵ_ηᵀ(Ĵ_η v). A: (M,D,P), v: (P,) -> (P,)."""
    J_hat = jnp.mean(A, axis=0)
    return J_hat.T @ (J_hat @ v)


def ggn_cross_seed(A: Array) -> Array:
    """Ĝ_U = (SᵀS − Q)/(M(M−1)), S=Σ A_m, Q=Σ A_mᵀA_m. Unbiased; symmetrized."""
    M = A.shape[0]
    S = jnp.sum(A, axis=0)                        # (D, P)
    Q = jnp.einsum("mdp,mdq->pq", A, A)           # Σ_m A_mᵀ A_m, (P, P)
    G = (S.T @ S - Q) / (M * (M - 1))
    return symmetrize(G)


def plugin_bias_covariance(A: Array, J_eta: Array) -> Array:
    """Ĉ = (1/(M−1)) Σ_m (A_m − J_η)ᵀ(A_m − J_η). Predicted plug-in bias = Ĉ/M."""
    M = A.shape[0]
    dA = A - J_eta[None]                          # (M, D, P)
    C = jnp.einsum("mdp,mdq->pq", dA, dA) / (M - 1)
    return symmetrize(C)


def population_residual_opg(A: Array, r: Array) -> Array:
    """LABELED comparison object (DEC-001), NOT curvature.

    g_m = A_mᵀ r (fixed population residual r = η(z★) − η_y);
    F_OPG_pop = (1/M) Σ_m g_m g_mᵀ. Zero when r=0 while G_ref may be nonzero.
    """
    g = jnp.einsum("mdp,d->mp", A, r)             # (M, P)
    M = A.shape[0]
    return symmetrize(g.T @ g / M)


def empirical_loss_opg(A: Array, feats: Array, eta_y: Array) -> Array:
    """LABELED comparison object (DEC-001), NOT curvature.

    r̂ = η̂ − η_y with η̂ = mean_m ψ(X_m); g_m = A_mᵀ r̂; (1/M) Σ g_m g_mᵀ.
    feats: (M, D) per-seed features.
    """
    r_hat = jnp.mean(feats, axis=0) - eta_y       # (D,)
    return population_residual_opg(A, r_hat)


def trace_normalized(G: Array, eps: float = 1e-300) -> Array:
    """G / max(tr G, eps) — shape comparison independent of overall scale."""
    return G / jnp.maximum(jnp.trace(G), eps)
