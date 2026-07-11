"""Local quadratic-prediction and empirical validity radius (Math-Spec §16).

For a stationary point ẑ and a unit direction v_k with GGN eigenvalue λ_k:

    ΔL_k(α) = L(ẑ + α v_k) − L(ẑ)      (actual local loss change)
    Q_k(α)  = ½ α² λ_k                  (GGN quadratic prediction)

The validity radius is the largest contiguous-from-zero extent of a PRESPECIFIED
signed α-grid on which the relative prediction error stays within a PRESPECIFIED
tolerance. The α-grid and tolerance are inputs — never chosen from the results.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array
_EPS = 1e-300


def curvature_bias(H: Array, G: Array, v: Array) -> float:
    """Infinitesimal GGN curvature bias along direction v (Math-Spec §7):

        e_curv(v) = |vᵀ(H − G)v| / max(|vᵀHv|, ε).

    This is the residual-curvature bias that can remain NONZERO as α→0 — a
    distinct failure mode from the nonlinear validity radius (which measures the
    additional breakdown as the perturbation grows). Reusable across experiments.
    """
    vn = v / jnp.linalg.norm(v)
    num = jnp.abs(vn @ (H - G) @ vn)
    den = jnp.maximum(jnp.abs(vn @ H @ vn), _EPS)
    return float(num / den)


def directional_loss_change(loss_fn: Callable[[Array], Array], z_hat: Array,
                            direction: Array, alphas: Array) -> Array:
    """ΔL(α) = L(ẑ + α·d̂) − L(ẑ) over the α-grid. d is normalized. Shape (len(A),)."""
    d = direction / jnp.linalg.norm(direction)
    L0 = loss_fn(z_hat)
    return jax.vmap(lambda a: loss_fn(z_hat + a * d) - L0)(alphas)


def quadratic_prediction(lambda_k: float | Array, alphas: Array) -> Array:
    """Q(α) = ½ α² λ_k. Shape (len(A),)."""
    return 0.5 * alphas ** 2 * lambda_k


def quadratic_model_error(loss_fn: Callable[[Array], Array], z_hat: Array,
                          v_k: Array, lambda_k: float | Array, alphas: Array) -> Array:
    """Relative prediction error e(α) = |ΔL − Q| / max(|ΔL|, ε). Shape (len(A),)."""
    dL = directional_loss_change(loss_fn, z_hat, v_k, alphas)
    Q = quadratic_prediction(lambda_k, alphas)
    return jnp.abs(dL - Q) / jnp.maximum(jnp.abs(dL), _EPS)


def validity_radius(loss_fn: Callable[[Array], Array], z_hat: Array, v_k: Array,
                    lambda_k: float | Array, alphas: Array, tol: float) -> dict:
    """Largest contiguous-from-zero α (per sign) with e(α) ≤ tol.

    Returns {'rho_plus', 'rho_minus', 'rho'} with rho = min(rho_plus, rho_minus).
    `alphas` and `tol` are prespecified inputs.
    """
    e = np.asarray(quadratic_model_error(loss_fn, z_hat, v_k, lambda_k, alphas))
    a = np.asarray(alphas)

    def contiguous(mags_errs):
        r = 0.0
        for mag, err in mags_errs:  # ascending magnitude
            if err <= tol:
                r = mag
            else:
                break
        return r

    pos = sorted(((float(av), float(ev)) for av, ev in zip(a, e) if av > 0))
    neg = sorted(((float(-av), float(ev)) for av, ev in zip(a, e) if av < 0))
    rho_plus = contiguous(pos)
    rho_minus = contiguous(neg)
    return {"rho_plus": rho_plus, "rho_minus": rho_minus, "rho": min(rho_plus, rho_minus)}
