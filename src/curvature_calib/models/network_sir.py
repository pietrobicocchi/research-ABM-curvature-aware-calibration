# src/curvature_calib/models/network_sir.py
"""Network-SIR with configurable surrogate gradients.

Discrete-state SIR on a fixed contact graph. Surrogate for discrete Bernoulli
transitions is selectable: "gumbel" (Gumbel-Sigmoid, default) or
"straight_through" (hard sample, gradient passes through).

theta = (beta, gamma, I0_frac, t_lock_norm, f_lock), P = 5.
"""
from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from curvature_calib.models.surrogates import gumbel_sigmoid, straight_through_bernoulli


class NetSIRParams(NamedTuple):
    beta: jax.Array
    gamma: jax.Array
    I0_frac: jax.Array
    t_lock_norm: jax.Array
    f_lock: jax.Array


def pack(theta: jax.Array) -> NetSIRParams:
    return NetSIRParams(
        beta=theta[0], gamma=theta[1], I0_frac=theta[2],
        t_lock_norm=theta[3], f_lock=theta[4],
    )


def build_er_graph(N: int, mean_degree: float, key: jax.Array) -> jax.Array:
    """Erdos-Renyi (N, N) adjacency, no self-loops, symmetric."""
    p_edge = mean_degree / max(N - 1, 1)
    upper = jax.random.bernoulli(key, p=p_edge, shape=(N, N)).astype(jnp.float32)
    upper = jnp.triu(upper, k=1)
    return upper + upper.T


def _init_state(I0_frac: jax.Array, N: int, k_init: float,
                node_priorities: jax.Array) -> jax.Array:
    I0 = jax.nn.sigmoid(k_init * (I0_frac - node_priorities))
    S0 = 1.0 - I0
    R0 = jnp.zeros_like(I0)
    return jnp.stack([S0, I0, R0], axis=1)


def _step(state, t_norm_eps, params, A, dt, k_sig, surrogate_fn):
    t_norm, key = t_norm_eps
    S = state[:, 0]
    I = state[:, 1]
    R = state[:, 2]

    beta_eff = params.beta * (
        1.0 - (1.0 - params.f_lock) *
        jax.nn.sigmoid(k_sig * (t_norm - params.t_lock_norm))
    )
    foi = A @ I
    p_infect  = 1.0 - jnp.exp(-beta_eff * foi * dt)
    p_recover = 1.0 - jnp.exp(-params.gamma * dt)

    k_inf, k_rec = jax.random.split(key)
    inf_draw = surrogate_fn(p_infect, k_inf)
    rec_draw = surrogate_fn(p_recover * jnp.ones_like(I), k_rec)

    transition_S_to_I = S * inf_draw
    transition_I_to_R = I * rec_draw

    new_S = S - transition_S_to_I
    new_I = I + transition_S_to_I - transition_I_to_R
    new_R = R + transition_I_to_R

    obs = jnp.sum(transition_S_to_I)
    return jnp.stack([new_S, new_I, new_R], axis=1), obs


def simulate(
    theta: jax.Array,
    key: jax.Array,
    T: int = 200,
    N: int = 300,
    mean_degree: float = 6.0,
    dt: float = 1.0,
    k_sig: float = 20.0,
    k_init: float = 50.0,
    gumbel_tau: float = 0.5,
    surrogate: str = "gumbel",
    graph_seed: int = 17,
    grad_horizon: int | None = None,
) -> jax.Array:
    """Run one network-SIR trajectory of length T.

    surrogate: "gumbel" uses Gumbel-Sigmoid at temperature gumbel_tau (default);
               "straight_through" uses hard Bernoulli with straight-through gradient.
    graph_seed is fixed across simulation seeds — the contact graph is shared.
    """
    params = pack(theta)
    A = build_er_graph(N, mean_degree, jax.random.PRNGKey(graph_seed))

    if surrogate == "straight_through":
        surrogate_fn = straight_through_bernoulli
    else:
        surrogate_fn = partial(gumbel_sigmoid, tau=gumbel_tau)

    k_priors, k_steps = jax.random.split(key)
    node_priorities = jax.random.uniform(k_priors, (N,), dtype=theta.dtype)
    init = _init_state(params.I0_frac, N, k_init, node_priorities)
    ts = jnp.arange(T, dtype=theta.dtype) / float(T)
    step_keys = jax.random.split(k_steps, T)

    step_fn = lambda s, t_k: _step(s, t_k, params, A, dt, k_sig, surrogate_fn)

    if grad_horizon is None or grad_horizon >= T:
        _, xs = jax.lax.scan(step_fn, init, (ts, step_keys))
        return xs

    n_pre = T - grad_horizon
    state_pre, xs_pre = jax.lax.scan(step_fn, init, (ts[:n_pre], step_keys[:n_pre]))
    state_pre = jax.tree.map(jax.lax.stop_gradient, state_pre)
    xs_pre = jax.lax.stop_gradient(xs_pre)
    _, xs_post = jax.lax.scan(step_fn, state_pre, (ts[n_pre:], step_keys[n_pre:]))
    return jnp.concatenate([xs_pre, xs_post])
