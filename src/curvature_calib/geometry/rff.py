"""Frozen random Fourier features (RBF) and the analytic MMD-GGN reference.

Feature map (frozen w, b):
    ψ(X) = sqrt(2/D) · [cos(w_jᵀ X + b_j)]_{j=1..D} ∈ R^D.

Feature mean embedding η(z) = E_ε[ψ(X_z)]. For Gaussian X_z ~ N(a(z), Σ(z)):
    η_j(z) = sqrt(2/D) · exp(−½ w_jᵀ Σ(z) w_j) · cos(w_jᵀ a(z) + b_j)   (closed form).

Reference geometry (Math-Spec §12–13): J_η = D_z η,  G_ref = J_ηᵀ J_η.

Secondary: the exact Gaussian-RBF cross-kernel gives the population RBF GGN via
the mixed second derivative ∂²K(z,z')/∂z_i∂z'_j |_{z'=z}.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from curvature_calib.benchmarks import gaussian_location_scale as gls

Array = jax.Array


class RFF(NamedTuple):
    w: Array      # (D, n) frequencies
    b: Array      # (D,) phases
    gamma: float  # bandwidth


def frozen_rff(key: Array, D: int, n: int, gamma: float, dtype=jnp.float64) -> RFF:
    """Draw and FREEZE {w_j ~ N(0, γ⁻² I), b_j ~ U[0, 2π]}."""
    kw, kb = jax.random.split(key)
    w = jax.random.normal(kw, (D, n), dtype=dtype) / gamma
    b = jax.random.uniform(kb, (D,), dtype=dtype) * (2.0 * jnp.pi)
    return RFF(w=w, b=b, gamma=gamma)


def feature_map(rff: RFF, X: Array) -> Array:
    """ψ(X). X: (n,) -> (D,)  or  (M, n) -> (M, D)."""
    D = rff.w.shape[0]
    scale = jnp.sqrt(2.0 / D)
    proj = X @ rff.w.T + rff.b        # (D,) or (M, D)
    return scale * jnp.cos(proj)


def feature_mean_closed_form(rff: RFF, a: Array, Sigma: Array) -> Array:
    """η given (a, Σ) in closed form. -> (D,)."""
    D = rff.w.shape[0]
    scale = jnp.sqrt(2.0 / D)
    quad = 0.5 * jnp.sum((rff.w @ Sigma) * rff.w, axis=1)   # ½ w_jᵀ Σ w_j, (D,)
    phase = rff.w @ a + rff.b                               # (D,)
    return scale * jnp.exp(-quad) * jnp.cos(phase)


def feature_mean(cfg: gls.GLSConfig, rff: RFF, z: Array) -> Array:
    """η(z) via the closed form at (a(z), Σ(z))."""
    return feature_mean_closed_form(rff, gls.gaussian_mean(cfg, z), gls.covariance(cfg, z))


def feature_mean_jacobian(cfg: gls.GLSConfig, rff: RFF, z: Array) -> Array:
    """J_η = D_z η(z) ∈ R^{D×P} (AD of the closed form)."""
    return jax.jacfwd(lambda zz: feature_mean(cfg, rff, zz))(z)


def ggn_reference(cfg: gls.GLSConfig, rff: RFF, z: Array) -> Array:
    """G_ref = J_ηᵀ J_η ∈ R^{P×P} (exact for the frozen finite-D features)."""
    J = feature_mean_jacobian(cfg, rff, z)
    G = J.T @ J
    return 0.5 * (G + G.T)


def feature_of_z(cfg: gls.GLSConfig, rff: RFF, z: Array, eps: Array) -> Array:
    """ψ(X_z(ε)) for one reparameterized draw. -> (D,)."""
    return feature_map(rff, gls.simulate(cfg, z, eps))


# --------------------------------------------------------------------------- #
# Secondary: exact Gaussian-RBF population GGN (bandwidth γ)
# --------------------------------------------------------------------------- #
def rbf_cross_kernel(cfg: gls.GLSConfig, gamma: float, z: Array, zp: Array) -> Array:
    """K(z,z') = <μ_z, μ_z'> for the Gaussian-RBF mean embedding (closed form)."""
    a1, a2 = gls.gaussian_mean(cfg, z), gls.gaussian_mean(cfg, zp)
    S = gls.covariance(cfg, z) + gls.covariance(cfg, zp)
    g2 = gamma * gamma
    M = jnp.eye(2, dtype=z.dtype) + S / g2
    da = a1 - a2
    quad = da @ jnp.linalg.solve(g2 * jnp.eye(2, dtype=z.dtype) + S, da)
    return jnp.linalg.det(M) ** (-0.5) * jnp.exp(-0.5 * quad)


def ggn_rbf(cfg: gls.GLSConfig, gamma: float, z: Array) -> Array:
    """Population RBF GGN: [G_RBF]_ij = ∂²K/∂z_i∂z'_j |_{z'=z}."""
    mixed = jax.jacfwd(jax.jacrev(lambda a, b: rbf_cross_kernel(cfg, gamma, a, b), argnums=1),
                       argnums=0)
    G = mixed(z, z)
    return 0.5 * (G + G.T)
