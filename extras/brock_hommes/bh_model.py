"""Brock & Hommes (1998) heterogeneous-beliefs asset pricing model, deviation form.

Three trader types: a fundamentalist (g_0 = b_0 = 0) and two with forecasts
g_h x_{t-1} + b_h. theta = (beta, g_1, b_1, g_2, b_2), beta the intensity of choice.

    n_t = softmax(beta U_{t-1})
    x_t = (1/R) sum_h n_{h,t} (g_h x_{t-1} + b_h) + eps_t
    U_{h,t} = (x_t - R x_{t-1}) (g_h x_{t-2} + b_h - R x_{t-1})

Complex and chaotic dynamics require large beta (about 50 to 100 here).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def _step(state, eps, beta, g, b, R):
    x_prev, x_prev2, U = state
    n = jax.nn.softmax(beta * U)
    x_t = jnp.sum(n * (g * x_prev + b)) / R + eps
    U_new = (x_t - R * x_prev) * (g * x_prev2 + b - R * x_prev)
    return (x_t, x_prev, U_new), x_t


def simulate(theta, key, T=500, R=1.01, sigma=0.05, x_init=0.0, grad_horizon=None):
    """Trajectory x_1..x_T. With grad_horizon < T, derivatives flow only through the
    last grad_horizon steps; the forward pass is unchanged."""
    zero = jnp.zeros((), dtype=theta.dtype)
    g = jnp.stack([zero, theta[1], theta[3]])
    b = jnp.stack([zero, theta[2], theta[4]])
    eps = sigma * jax.random.normal(key, (T,), dtype=theta.dtype)
    x0 = jnp.asarray(x_init, dtype=theta.dtype)
    init = (x0, x0, jnp.zeros((3,), dtype=theta.dtype))
    step = lambda s, e: _step(s, e, theta[0], g, b, R)

    if grad_horizon is None or grad_horizon >= T:
        return jax.lax.scan(step, init, eps)[1]
    n_pre = T - grad_horizon
    state_pre, xs_pre = jax.lax.scan(step, init, eps[:n_pre])
    state_pre = jax.tree.map(jax.lax.stop_gradient, state_pre)
    xs_pre = jax.lax.stop_gradient(xs_pre)
    _, xs_post = jax.lax.scan(step, state_pre, eps[n_pre:])
    return jnp.concatenate([xs_pre, xs_post])
