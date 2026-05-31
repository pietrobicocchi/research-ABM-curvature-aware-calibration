"""Brock & Hommes (1998) heterogeneous-agent asset pricing model.

Deviation form. State: x_t = p_t - p_t* (price deviation from fundamental).
H trader types with forecasting rule f_{h,t} = g_h x_{t-1} + b_h. Type 0 is the
fundamentalist (g_0 = b_0 = 0). Fractions evolve by discrete choice over
realised profits with intensity-of-choice beta.

Canonical setup: H = 3 (fundamentalist + 2 free types), theta in R^5:
    theta = (beta, g_1, b_1, g_2, b_2).

Dynamics (R = gross risk-free rate, sigma = noise scale):
    n_t        = softmax(beta * U_{t-1})                       # fractions
    x_t        = (1/R) sum_h n_{h,t} (g_h x_{t-1} + b_h) + eps_t
    U_{h,t}    = (x_t - R x_{t-1}) (g_h x_{t-2} + b_h - R x_{t-1})

Ref: Brock & Hommes, "Heterogeneous beliefs and routes to chaos in a simple
asset pricing model," JEDC 22(8), 1998.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class BHParams(NamedTuple):
    beta: jax.Array  # scalar
    g: jax.Array     # (H,)
    b: jax.Array     # (H,)


def pack_canonical(theta: jax.Array) -> BHParams:
    """theta = (beta, g_1, b_1, g_2, b_2) -> BHParams with fundamentalist type 0."""
    zero = jnp.zeros((), dtype=theta.dtype)
    g = jnp.stack([zero, theta[1], theta[3]])
    b = jnp.stack([zero, theta[2], theta[4]])
    return BHParams(beta=theta[0], g=g, b=b)


def _step(state, eps, params: BHParams, R: float):
    x_prev, x_prev2, U = state
    n = jax.nn.softmax(params.beta * U)
    forecasts = params.g * x_prev + params.b
    x_t = jnp.sum(n * forecasts) / R + eps
    excess = x_t - R * x_prev
    U_new = excess * (params.g * x_prev2 + params.b - R * x_prev)
    return (x_t, x_prev, U_new), x_t


def simulate(
    theta: jax.Array,
    key: jax.Array,
    T: int = 500,
    R: float = 1.01,
    sigma: float = 0.05,
    H: int = 3,
    x_init: float = 0.0,
) -> jax.Array:
    """Simulate one BH trajectory of length T. Returns x_{1..T}.

    x_init perturbs both lagged states (x_{-1}, x_{-2}); useful for inspecting
    the bifurcation in the noiseless limit (sigma = 0), where the fundamental
    steady state x = 0 is a fixed point.
    """
    params = pack_canonical(theta)
    eps = sigma * jax.random.normal(key, (T,), dtype=theta.dtype)
    x0 = jnp.asarray(x_init, dtype=theta.dtype)
    init = (x0, x0, jnp.zeros((H,), dtype=theta.dtype))
    _, xs = jax.lax.scan(lambda s, e: _step(s, e, params, R), init, eps)
    return xs
