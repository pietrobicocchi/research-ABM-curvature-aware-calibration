"""Controlled Gaussian location-scale simulator for EXP-003.

    X_z = a(z) + L(z) ε,   ε ~ N(0, I_n),   n = 2,

with a documented prior-scaled transform θ = T(z), z ~ N(0, I_P), P = 5.
Parameters affect BOTH the mean a(z) and the covariance Σ(z)=L(z)L(z)ᵀ.

Notation (revision: no collision with the feature mean embedding η):
    a(z)      — Gaussian simulator mean          (this module)
    Σ(z)      — Gaussian covariance
    η(z)      — feature mean embedding E[ψ(X_z)]  (geometry/rff.py)
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class GLSConfig(NamedTuple):
    """Transform constants for θ = T(z). All fixed and recorded in run config."""
    m0: float = 0.0        # mean offset
    sigma_m: float = 1.0   # mean scale (z1, z2)
    rho0: float = 0.0      # log-scale offset  (s = exp(rho0 + sigma_s z))
    sigma_s: float = 0.30  # log-scale slope   (z3, z4)
    sigma_l: float = 0.30  # off-diagonal Cholesky scale (z5)


def gaussian_mean(cfg: GLSConfig, z: Array) -> Array:
    """a(z) ∈ R², affine in (z1, z2)."""
    return jnp.array([cfg.m0 + cfg.sigma_m * z[0], cfg.m0 + cfg.sigma_m * z[1]], dtype=z.dtype)


def cholesky(cfg: GLSConfig, z: Array) -> Array:
    """L(z), lower-triangular; log-parameterized diagonal keeps s>0."""
    s1 = jnp.exp(cfg.rho0 + cfg.sigma_s * z[2])
    s2 = jnp.exp(cfg.rho0 + cfg.sigma_s * z[3])
    ell = cfg.sigma_l * z[4]
    return jnp.array([[s1, 0.0], [ell, s2]], dtype=z.dtype)


def covariance(cfg: GLSConfig, z: Array) -> Array:
    """Σ(z) = L(z) L(z)ᵀ."""
    L = cholesky(cfg, z)
    return L @ L.T


def simulate(cfg: GLSConfig, z: Array, eps: Array) -> Array:
    """One reparameterized draw X_z(ε) = a(z) + L(z) ε. eps: (2,) -> (2,)."""
    return gaussian_mean(cfg, z) + cholesky(cfg, z) @ eps
