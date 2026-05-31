"""Damped preconditioned step using the OPG matrix.

Step:
    step = (F_hat + lambda * I)^{-1} g

with `lambda` an adaptive Tikhonov-style damping (Levenberg-Marquardt /
Martens-Grosse 2015 sec. 6.4): increase when the realised loss reduction
falls below the quadratic-model prediction, decrease when the prediction is
accurate.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class LMState(NamedTuple):
    damping: float


def damped_step(F_hat: jax.Array, g: jax.Array, damping: float) -> jax.Array:
    """Return -(F_hat + damping * I)^{-1} g (the descent direction)."""
    P = F_hat.shape[0]
    A = F_hat + damping * jnp.eye(P, dtype=F_hat.dtype)
    # Cholesky solve. Falls back to lstsq if Cholesky fails numerically.
    try:
        L = jnp.linalg.cholesky(A)
        y = jax.scipy.linalg.solve_triangular(L, g, lower=True)
        delta = jax.scipy.linalg.solve_triangular(L.T, y, lower=False)
    except Exception:
        delta, *_ = jnp.linalg.lstsq(A, g, rcond=None)
    return -delta


def quadratic_model_reduction(
    F_hat: jax.Array, g: jax.Array, step: jax.Array, damping: float
) -> float:
    """Predicted loss decrease from the quadratic + damping model:

        m(0) - m(step) = - g^T step - 0.5 step^T (F + lam I) step.
    """
    P = F_hat.shape[0]
    A = F_hat + damping * jnp.eye(P, dtype=F_hat.dtype)
    return float(-jnp.dot(g, step) - 0.5 * step @ A @ step)


def update_damping(
    damping: float,
    realised_reduction: float,
    predicted_reduction: float,
    increase_factor: float = 10.0 ** (1.0 / 2.0),
    decrease_factor: float = 10.0 ** (-1.0 / 2.0),
    min_damping: float = 1e-6,
    max_damping: float = 1e6,
) -> float:
    """Levenberg-Marquardt damping update by the reduction ratio rho.

    rho > 0.75 -> trust the quadratic model -> decrease damping
    rho < 0.25 -> distrust -> increase damping
    otherwise -> keep
    """
    if predicted_reduction <= 0:
        return min(damping * increase_factor, max_damping)
    rho = realised_reduction / predicted_reduction
    if rho > 0.75:
        return max(damping * decrease_factor, min_damping)
    if rho < 0.25:
        return min(damping * increase_factor, max_damping)
    return damping
