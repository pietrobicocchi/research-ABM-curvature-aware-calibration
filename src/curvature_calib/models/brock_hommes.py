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
    grad_horizon: int | None = None,
) -> jax.Array:
    """Simulate one BH trajectory of length T. Returns x_{1..T}.

    Args:
        theta, key, T, R, sigma, H, x_init: as before. (`H` is the number of
            trader types; unrelated to `grad_horizon`.)
        grad_horizon: if None or >= T, gradient flows through the entire
            trajectory (default). Otherwise, gradient flows only through the
            last `grad_horizon` steps; the first `T - grad_horizon` steps are
            wrapped in `stop_gradient`. This is the standard
            gradient-truncation trick from Quera-Bofarull et al. 2023
            ("Some Challenges of Calibrating Differentiable ABMs"), used in
            Phase 1's horizon-bias killswitch experiment to test whether the
            OPG eigenstructure is robust to truncation bias.

    Note: forward (primal) pass is identical regardless of `grad_horizon`;
    only the backward (tangent) pass differs.

    x_init perturbs both lagged states (x_{-1}, x_{-2}); useful for inspecting
    the bifurcation in the noiseless limit (sigma = 0), where the fundamental
    steady state x = 0 is a fixed point.
    """
    params = pack_canonical(theta)
    eps = sigma * jax.random.normal(key, (T,), dtype=theta.dtype)
    x0 = jnp.asarray(x_init, dtype=theta.dtype)
    init = (x0, x0, jnp.zeros((H,), dtype=theta.dtype))

    step_fn = lambda s, e: _step(s, e, params, R)

    if grad_horizon is None or grad_horizon >= T:
        _, xs = jax.lax.scan(step_fn, init, eps)
        return xs

    n_pre = T - grad_horizon
    # Primal pass for the first n_pre steps; cut gradient at the boundary.
    state_pre, xs_pre = jax.lax.scan(step_fn, init, eps[:n_pre])
    state_pre = jax.tree.map(jax.lax.stop_gradient, state_pre)
    xs_pre = jax.lax.stop_gradient(xs_pre)
    # Last `grad_horizon` steps with gradient flowing.
    _, xs_post = jax.lax.scan(step_fn, state_pre, eps[n_pre:])
    return jnp.concatenate([xs_pre, xs_post])
