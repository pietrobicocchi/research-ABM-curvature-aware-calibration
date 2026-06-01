"""Mean-field SIR with smooth lockdown intervention (Tier A second model).

Discretised SIR dynamics in a closed population of size N:

    S_{t+1} = S_t - dt * beta_eff(t) * S_t * I_t / N
    I_{t+1} = I_t + dt * (beta_eff(t) * S_t * I_t / N - gamma * I_t)
    R_{t+1} = R_t + dt * gamma * I_t

Smooth time-varying transmission rate (sigmoid-modulated lockdown):

    beta_eff(t) = beta * (1 - (1 - f_lock) * sigmoid(k * (t/T - t_lock_norm)))

where k controls the lockdown sharpness (large k = sharp policy
introduction; small k = gradual). The sigmoid is differentiable
everywhere in theta, unlike a step.

Observation: daily incidence with Gaussian noise:

    obs_t = dt * beta_eff(t) * S_t * I_t / N + N(0, sigma_obs^2).

theta = (beta, gamma, I0_frac, t_lock_norm, f_lock), P = 5.

Used as the project's *second* model (alongside Brock-Hommes) for the
"diagnostic generalises" claim. Fully smooth -- no surrogate gradients
needed -- so it isolates the curvature methodology from the
Gumbel-Softmax / straight-through bias question. The harder
network-with-discrete-transitions version is deferred.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class SIRParams(NamedTuple):
    beta: jax.Array         # transmission rate per day
    gamma: jax.Array        # recovery rate per day
    I0_frac: jax.Array      # initial infected fraction of N
    t_lock_norm: jax.Array  # lockdown timing as fraction of T in [0, 1]
    f_lock: jax.Array       # post-lockdown beta multiplier in (0, 1]


def pack(theta: jax.Array) -> SIRParams:
    return SIRParams(
        beta=theta[0],
        gamma=theta[1],
        I0_frac=theta[2],
        t_lock_norm=theta[3],
        f_lock=theta[4],
    )


def _step(state, t_eps, params: SIRParams, N: float, dt: float,
          sigma_obs: float, k_sig: float):
    S, I, R = state
    t_norm, eps_t = t_eps
    beta_eff = params.beta * (
        1.0 - (1.0 - params.f_lock) *
        jax.nn.sigmoid(k_sig * (t_norm - params.t_lock_norm))
    )
    # Clip the *flux* (not the state) so derivatives stay well-defined
    # while preserving non-negativity.
    incidence = beta_eff * jnp.clip(S, min=0.0) * jnp.clip(I, min=0.0) / N
    new_S = S - dt * incidence
    new_I = I + dt * (incidence - params.gamma * I)
    new_R = R + dt * params.gamma * I
    obs = dt * incidence + sigma_obs * eps_t
    return (new_S, new_I, new_R), obs


def simulate(
    theta: jax.Array,
    key: jax.Array,
    T: int = 200,
    N: float = 1e5,
    dt: float = 1.0,
    sigma_obs: float = 10.0,
    k_sig: float = 20.0,
    grad_horizon: int | None = None,
) -> jax.Array:
    """Return the (T,) trajectory of daily reported incidence.

    `grad_horizon` mirrors the Brock-Hommes API: gradients flow only
    through the last `grad_horizon` steps; the rest are wrapped in
    stop_gradient. Primal pass is unchanged.
    """
    params = pack(theta)
    eps = jax.random.normal(key, (T,), dtype=theta.dtype)
    ts = jnp.arange(T, dtype=theta.dtype) / float(T)

    I0 = N * params.I0_frac
    init = (
        N - I0,
        I0,
        jnp.zeros((), dtype=theta.dtype),
    )

    def step_fn(state, t_eps_pair):
        return _step(state, t_eps_pair, params, N, dt, sigma_obs, k_sig)

    if grad_horizon is None or grad_horizon >= T:
        _, xs = jax.lax.scan(step_fn, init, (ts, eps))
        return xs

    n_pre = T - grad_horizon
    state_pre, xs_pre = jax.lax.scan(step_fn, init, (ts[:n_pre], eps[:n_pre]))
    state_pre = jax.tree.map(jax.lax.stop_gradient, state_pre)
    xs_pre = jax.lax.stop_gradient(xs_pre)
    _, xs_post = jax.lax.scan(step_fn, state_pre, (ts[n_pre:], eps[n_pre:]))
    return jnp.concatenate([xs_pre, xs_post])
