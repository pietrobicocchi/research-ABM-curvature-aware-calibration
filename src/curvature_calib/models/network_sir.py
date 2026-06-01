"""Network-SIR with Gumbel-Sigmoid surrogate gradients (Phase 3 Tier B).

Discrete-state SIR on a fixed contact graph. Each node is in state
S (0), I (1), or R (2). At each timestep:

    p_infect(i)  = 1 - exp(-beta_eff(t) * (sum_j A[i,j] * I[j]) * dt)   for S nodes
    p_recover(i) = 1 - exp(-gamma * dt)                                 for I nodes

We approximate the per-node Bernoulli transitions by a *Gumbel-Sigmoid*
draw, the binary special case of Gumbel-Softmax:

    Bernoulli(p) ~= sigmoid((logit(p) + G_1 - G_0) / tau)

where G_0, G_1 are Gumbel(0, 1) noises and tau is the temperature. This is
differentiable in p (hence in theta) but **biased** in the Bernoulli mean
for tau > 0. At tau -> 0 it converges to a true Bernoulli (zero bias,
high gradient variance).

theta = (beta, gamma, I0_frac, t_lock_norm, f_lock), P = 5 -- matches the
mean-field SIR API exactly so all downstream OPG / falsification scripts
work without changes.

The contact graph (Erdos-Renyi with fixed mean degree) is **fixed across
seeds**; randomness comes from initial-infection assignment + Gumbel noise.

The point of this model is to exercise the surrogate-gradient bias regime
that the Phase 3 plan flags as the harder generalisation question.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


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
    # upper-triangular Bernoulli, then symmetrise
    upper = jax.random.bernoulli(key, p=p_edge, shape=(N, N)).astype(jnp.float32)
    upper = jnp.triu(upper, k=1)
    A = upper + upper.T
    return A


def _gumbel_sigmoid(p: jax.Array, key: jax.Array,
                    tau: float, eps: float = 1e-6) -> jax.Array:
    """Differentiable Bernoulli draw via Gumbel-Sigmoid at temperature tau.

    Reduces to sample ~ Bernoulli(p) as tau -> 0; smooth in p for tau > 0.
    """
    k1, k2 = jax.random.split(key)
    u1 = jnp.clip(jax.random.uniform(k1, p.shape, dtype=p.dtype), eps, 1.0 - eps)
    u2 = jnp.clip(jax.random.uniform(k2, p.shape, dtype=p.dtype), eps, 1.0 - eps)
    g1 = -jnp.log(-jnp.log(u1))
    g0 = -jnp.log(-jnp.log(u2))
    p_clipped = jnp.clip(p, eps, 1.0 - eps)
    logit = jnp.log(p_clipped) - jnp.log(1.0 - p_clipped)
    return jax.nn.sigmoid((logit + g1 - g0) / tau)


def _init_state(I0_frac: jax.Array, N: int, k_init: float,
                node_priorities: jax.Array) -> jax.Array:
    """Differentiable initial state, smooth in I0_frac.

    Each node has a fixed random priority u in [0, 1]; node i is initially
    infected with smooth probability sigmoid(k_init * (I0_frac - u_i)).
    """
    I0 = jax.nn.sigmoid(k_init * (I0_frac - node_priorities))   # (N,)
    S0 = 1.0 - I0
    R0 = jnp.zeros_like(I0)
    return jnp.stack([S0, I0, R0], axis=1)                       # (N, 3)


def _step(state, t_norm_eps, params, A, dt, k_sig, gumbel_tau):
    """One discrete timestep on the network.

    state: (N, 3) soft one-hot per node.
    Returns updated state and the daily new-infection count (scalar).
    """
    t_norm, key = t_norm_eps
    S = state[:, 0]
    I = state[:, 1]
    R = state[:, 2]

    beta_eff = params.beta * (
        1.0 - (1.0 - params.f_lock) *
        jax.nn.sigmoid(k_sig * (t_norm - params.t_lock_norm))
    )
    foi = A @ I                                                   # force of infection per node
    p_infect = 1.0 - jnp.exp(-beta_eff * foi * dt)                # per-node
    p_recover = 1.0 - jnp.exp(-params.gamma * dt)                 # scalar

    k_inf, k_rec = jax.random.split(key)
    inf_draw = _gumbel_sigmoid(p_infect, k_inf, gumbel_tau)
    rec_draw = _gumbel_sigmoid(p_recover * jnp.ones_like(I), k_rec, gumbel_tau)

    transition_S_to_I = S * inf_draw
    transition_I_to_R = I * rec_draw

    new_S = S - transition_S_to_I
    new_I = I + transition_S_to_I - transition_I_to_R
    new_R = R + transition_I_to_R

    obs = jnp.sum(transition_S_to_I)                              # daily new infections
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
    graph_seed: int = 17,
    grad_horizon: int | None = None,
) -> jax.Array:
    """Run one network-SIR trajectory of length T (daily new-infection counts).

    `graph_seed` is fixed across seeds -- the contact graph is the same for
    all per-seed evaluations within a calibration. Randomness across seeds
    comes from `key`, which drives the initial-infection priorities and the
    per-step Gumbel noise.

    `gumbel_tau` controls the surrogate-gradient bias: smaller tau -> closer
    to a true Bernoulli (low bias, high variance); larger tau -> smoother
    (high bias, low variance). Default tau=0.5 is the typical Gumbel-Softmax
    literature value.

    `grad_horizon` mirrors the BH and SIR APIs.
    """
    params = pack(theta)

    # Build the contact graph once (deterministic).
    A = build_er_graph(N, mean_degree, jax.random.PRNGKey(graph_seed))

    # Per-simulation noise streams.
    k_priors, k_steps = jax.random.split(key)
    node_priorities = jax.random.uniform(k_priors, (N,), dtype=theta.dtype)

    init = _init_state(params.I0_frac, N, k_init, node_priorities)
    ts = jnp.arange(T, dtype=theta.dtype) / float(T)
    step_keys = jax.random.split(k_steps, T)

    step_fn = lambda s, t_k: _step(s, t_k, params, A, dt, k_sig, gumbel_tau)

    if grad_horizon is None or grad_horizon >= T:
        _, xs = jax.lax.scan(step_fn, init, (ts, step_keys))
        return xs

    n_pre = T - grad_horizon
    state_pre, xs_pre = jax.lax.scan(
        step_fn, init, (ts[:n_pre], step_keys[:n_pre])
    )
    state_pre = jax.tree.map(jax.lax.stop_gradient, state_pre)
    xs_pre = jax.lax.stop_gradient(xs_pre)
    _, xs_post = jax.lax.scan(
        step_fn, state_pre, (ts[n_pre:], step_keys[n_pre:])
    )
    return jnp.concatenate([xs_pre, xs_post])
