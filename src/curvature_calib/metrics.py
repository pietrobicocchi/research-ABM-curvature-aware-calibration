"""Matrix / subspace comparison metrics for geometry validation.

Scale-aware relative errors (EXP001_IMPLEMENTATION_PLAN §3). Eigen- and
subspace machinery is reused from calibration.diagnostic (not reimplemented).
All ratios are computed in Python float so the guard epsilon behaves across
float32 and float64.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from curvature_calib.calibration.diagnostic import principal_angles

_EPS = 1e-300  # denominator guard (Python float; dtype-agnostic)


def rel_frobenius_error(A_est: jax.Array, A_ref: jax.Array) -> float:
    """||A_est - A_ref||_F / max(||A_ref||_F, eps)."""
    num = float(jnp.linalg.norm(A_est - A_ref))
    den = max(float(jnp.linalg.norm(A_ref)), _EPS)
    return num / den


def rel_l2_error(v_est: jax.Array, v_ref: jax.Array) -> float:
    """||v_est - v_ref||_2 / max(||v_ref||_2, eps)."""
    num = float(jnp.linalg.norm(v_est - v_ref))
    den = max(float(jnp.linalg.norm(v_ref)), _EPS)
    return num / den


def eigenvalue_rel_error(w_est: jax.Array, w_ref: jax.Array) -> jax.Array:
    """Per-eigenvalue relative error |w_est - w_ref| / max(|w_ref|, eps). Shape (P,)."""
    return jnp.abs(w_est - w_ref) / jnp.maximum(jnp.abs(w_ref), _EPS)


def subspace_principal_angles(V1: jax.Array, V2: jax.Array) -> jax.Array:
    """Principal angles (radians) between column spans of V1 and V2."""
    return principal_angles(V1, V2)


def max_principal_angle(V1: jax.Array, V2: jax.Array) -> float:
    """Largest principal angle (radians) between the two subspaces."""
    return float(jnp.max(principal_angles(V1, V2)))


def numerical_rank(w: jax.Array, rtol: float = 1e-10, atol: float = 0.0) -> int:
    """Number of |eigenvalues| above rtol * max(|w|) + atol."""
    aw = jnp.abs(w)
    thresh = rtol * float(jnp.max(aw)) + atol
    return int(jnp.sum(aw > thresh))
