"""Network SIR on a fixed Erdos-Renyi contact graph (Sec. 6, App. B.2).

Each step, a susceptible agent with infected-neighbour mass n is infected with
probability 1 - exp(-beta(t) n dt) and an infected agent recovers with probability
1 - exp(-gamma dt); beta(t) carries the same smooth lockdown as the mean-field model.
The Bernoulli transitions are made differentiable by a Gumbel-sigmoid relaxation or
a straight-through estimator.

theta = (beta, gamma, I0, t_lock, f_lock).
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

PARAM_NAMES = ("beta", "gamma", "I0", "t_lock", "f_lock")


def gumbel_sigmoid(p, key, tau=0.5, eps=1e-6):
    """Relaxed Bernoulli(p) draw at temperature tau."""
    k1, k2 = jr.split(key)
    u1 = jnp.clip(jr.uniform(k1, p.shape, dtype=p.dtype), eps, 1.0 - eps)
    u2 = jnp.clip(jr.uniform(k2, p.shape, dtype=p.dtype), eps, 1.0 - eps)
    g1 = -jnp.log(-jnp.log(u1))
    g0 = -jnp.log(-jnp.log(u2))
    p_clipped = jnp.clip(p, eps, 1.0 - eps)
    logit = jnp.log(p_clipped) - jnp.log(1.0 - p_clipped)
    return jax.nn.sigmoid((logit + g1 - g0) / tau)


@jax.custom_jvp
def straight_through_bernoulli(p, key):
    """Hard Bernoulli(p) draw whose derivative is that of p."""
    return jr.bernoulli(key, p).astype(p.dtype)


@straight_through_bernoulli.defjvp
def _straight_through_jvp(primals, tangents):
    p, key = primals
    p_dot, _ = tangents
    return straight_through_bernoulli(p, key), p_dot


def build_er_graph(N, mean_degree, key):
    """Symmetric Erdos-Renyi adjacency without self-loops."""
    p_edge = mean_degree / max(N - 1, 1)
    upper = jax.random.bernoulli(key, p=p_edge, shape=(N, N)).astype(jnp.float32)
    upper = jnp.triu(upper, k=1)
    return upper + upper.T


def _step(state, t_key, params, A, dt, kappa, draw):
    t_norm, key = t_key
    S, I, R = state[:, 0], state[:, 1], state[:, 2]
    beta, gamma, _, t_lock, f_lock = params
    beta_eff = beta * (1.0 - (1.0 - f_lock) * jax.nn.sigmoid(kappa * (t_norm - t_lock)))
    p_infect = 1.0 - jnp.exp(-beta_eff * (A @ I) * dt)
    p_recover = 1.0 - jnp.exp(-gamma * dt)
    k_inf, k_rec = jax.random.split(key)
    s_to_i = S * draw(p_infect, k_inf)
    i_to_r = I * draw(p_recover * jnp.ones_like(I), k_rec)
    new_state = jnp.stack([S - s_to_i, I + s_to_i - i_to_r, R + i_to_r], axis=1)
    return new_state, jnp.sum(s_to_i)


def simulate(theta, key, T, N, mean_degree, dt, kappa, kappa_init, gumbel_tau,
             surrogate, graph_seed, grad_horizon=None):
    """Incidence trajectory (T,). The graph depends only on graph_seed; `key` drives
    the initial infections and the transitions. With grad_horizon < T, derivatives
    flow only through the last grad_horizon steps."""
    params = tuple(theta[i] for i in range(5))
    A = build_er_graph(N, mean_degree, jax.random.PRNGKey(graph_seed))
    if surrogate == "gumbel":
        draw = partial(gumbel_sigmoid, tau=gumbel_tau)
    elif surrogate == "straight_through":
        draw = straight_through_bernoulli
    else:
        raise ValueError(f"unknown surrogate {surrogate!r}")

    k_priorities, k_steps = jax.random.split(key)
    priorities = jax.random.uniform(k_priorities, (N,), dtype=theta.dtype)
    I_init = jax.nn.sigmoid(kappa_init * (params[2] - priorities))
    init = jnp.stack([1.0 - I_init, I_init, jnp.zeros_like(I_init)], axis=1)
    ts = jnp.arange(T, dtype=theta.dtype) / float(T)
    step_keys = jax.random.split(k_steps, T)
    step = lambda s, tk: _step(s, tk, params, A, dt, kappa, draw)

    if grad_horizon is None or grad_horizon >= T:
        return jax.lax.scan(step, init, (ts, step_keys))[1]
    n_pre = T - grad_horizon
    state_pre, xs_pre = jax.lax.scan(step, init, (ts[:n_pre], step_keys[:n_pre]))
    state_pre = jax.lax.stop_gradient(state_pre)
    xs_pre = jax.lax.stop_gradient(xs_pre)
    _, xs_post = jax.lax.scan(step, state_pre, (ts[n_pre:], step_keys[n_pre:]))
    return jnp.concatenate([xs_pre, xs_post])


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


class NetworkSIR:
    """The network-SIR set-up of Table 1 at one operating point."""

    def __init__(self, cfg: dict, point: str, horizon: int | None = None):
        op = cfg["operating_points"][point]
        self.point = point
        self.theta_star = tuple(float(op[p]) for p in PARAM_NAMES)
        self.mean_degree = float(op["mean_degree"])
        self.graph_seed = int(op["graph_seed"])
        self.T = int(horizon if horizon is not None else op["horizon"])
        self.N = int(cfg["population"])
        self.dt = float(cfg["dt"])
        self.kappa = float(cfg["kappa"])
        self.kappa_init = float(cfg["kappa_init"])
        self.gumbel_tau = float(cfg["gumbel_tau"])
        self.prior_sd = jnp.array([float(cfg["prior_sd"][p]) for p in PARAM_NAMES])
        self.n_seeds = int(cfg["seeds"])
        self.base_seed = int(cfg["base_seed"])
        self.n_features = int(cfg["random_features"])

    def theta(self, z):
        b, g, i0, tl, fl = self.theta_star
        s = self.prior_sd
        return jnp.stack([
            b * jnp.exp(s[0] * z[0]),
            g * jnp.exp(s[1] * z[1]),
            i0 * jnp.exp(s[2] * z[2]),
            jax.nn.sigmoid(_logit(tl) + s[3] * z[3]),
            jax.nn.sigmoid(_logit(fl) + s[4] * z[4]),
        ])

    def seed_keys(self):
        return jax.random.split(jax.random.PRNGKey(self.base_seed), self.n_seeds)

    def rff_key(self):
        return jax.random.PRNGKey(self.base_seed + 1)

    def incidence(self, theta, key, surrogate="gumbel", grad_horizon=None):
        return simulate(theta, key, T=self.T, N=self.N, mean_degree=self.mean_degree,
                        dt=self.dt, kappa=self.kappa, kappa_init=self.kappa_init,
                        gumbel_tau=self.gumbel_tau, surrogate=surrogate,
                        graph_seed=self.graph_seed, grad_horizon=grad_horizon)
