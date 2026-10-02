"""Mean-field SIR with a smooth lockdown (App. B.1).

    beta(t) = beta [1 - (1 - f_lock) sigmoid(kappa (t/T - t_lock))]
    incidence_t = beta(t) S_t I_t / N,   S <- S - dt inc,   I <- I + dt (inc - gamma I)

theta = (beta, gamma, I0, t_lock, f_lock), with t_lock as a fraction of the horizon.
Prior-whitened coordinates z ~ N(0, I) map to theta by a log transform for beta,
gamma and a logit transform for I0, t_lock, f_lock, centred at theta*.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

PARAM_NAMES = ("beta", "gamma", "I0", "t_lock", "f_lock")


def _logit(p):
    return jnp.log(p / (1 - p))


def simulate(theta, key, T, N, dt, sigma_obs, kappa):
    """Daily incidence (T,), plus Gaussian observation noise of s.d. sigma_obs."""
    beta, gamma, I0_frac, t_lock, f_lock = theta[0], theta[1], theta[2], theta[3], theta[4]
    eps = jax.random.normal(key, (T,), dtype=theta.dtype)
    ts = jnp.arange(T, dtype=theta.dtype) / float(T)
    I0 = N * I0_frac
    init = (N - I0, I0, jnp.zeros((), dtype=theta.dtype))

    def step(state, t_eps):
        S, I, R = state
        t_norm, eps_t = t_eps
        beta_eff = beta * (1.0 - (1.0 - f_lock) * jax.nn.sigmoid(kappa * (t_norm - t_lock)))
        inc = beta_eff * jnp.clip(S, min=0.0) * jnp.clip(I, min=0.0) / N
        new_state = (S - dt * inc, I + dt * (inc - gamma * I), R + dt * gamma * I)
        return new_state, dt * inc + sigma_obs * eps_t

    return jax.lax.scan(step, init, (ts, eps))[1]


class MeanFieldSIR:
    """The mean-field SIR set-up of Table 1, as functions of z."""

    def __init__(self, cfg: dict):
        self.N = float(cfg["population"])
        self.T = int(cfg["horizon"])
        self.dt = float(cfg["dt"])
        self.kappa = float(cfg["kappa"])
        self.sigma_obs = float(cfg["sigma_obs"])
        self.w = float(cfg["temperature"])
        ts = cfg["theta_star"]
        self.theta_star = tuple(float(ts[p]) for p in PARAM_NAMES)
        self.prior_log_sd = float(cfg["prior_log_sd"])
        self.prior_logit_sd = float(cfg["prior_logit_sd"])
        self.key = jax.random.PRNGKey(cfg["simulation_key"])

    def theta(self, z):
        b, g, i0, tl, fl = self.theta_star
        sl, sg = self.prior_log_sd, self.prior_logit_sd
        return jnp.array([
            jnp.exp(jnp.log(b) + sl * z[0]),
            jnp.exp(jnp.log(g) + sl * z[1]),
            jax.nn.sigmoid(_logit(i0) + sg * z[2]),
            jax.nn.sigmoid(_logit(tl) + sg * z[3]),
            jax.nn.sigmoid(_logit(fl) + sg * z[4]),
        ])

    def incidence(self, z):
        """Noise-free incidence trajectory m(z) (Eq. 6 with psi the identity)."""
        return simulate(self.theta(z), self.key, self.T, self.N, self.dt, 0.0, self.kappa)

    def loss(self, z, y):
        """Gaussian loss 1/2 ||m(z) - y||^2 / sigma^2 (Eq. 7 with W = sigma^-2 I)."""
        return 0.5 * jnp.sum((self.incidence(z) - y) ** 2) / self.sigma_obs ** 2

    def trajectory(self, z):
        """(incidence, prevalence, beta(t)), each (T,), from the same dynamics."""
        beta, gamma, I0f, tl, fl = self.theta(z)
        N, dt, kappa = self.N, self.dt, self.kappa
        I0 = N * I0f
        ts = jnp.arange(self.T, dtype=jnp.float64) / float(self.T)

        def step(state, t_norm):
            S, I = state
            beta_eff = beta * (1.0 - (1.0 - fl) * jax.nn.sigmoid(kappa * (t_norm - tl)))
            inc = beta_eff * jnp.clip(S, min=0.0) * jnp.clip(I, min=0.0) / N
            return (S - dt * inc, I + dt * (inc - gamma * I)), (dt * inc, I, beta_eff)

        _, (inc, prev, beta_eff) = jax.lax.scan(step, (N - I0, I0), ts)
        return inc, prev, beta_eff

    def total_without_lockdown(self, z):
        """C(z°): total infections with f_lock = 1 (Eq. 17)."""
        beta, gamma, I0f, _, _ = self.theta(z)
        N, dt = self.N, self.dt
        I0 = N * I0f
        ts = jnp.arange(self.T, dtype=jnp.float64) / float(self.T)

        def step(state, _t):
            S, I = state
            inc = beta * jnp.clip(S, min=0.0) * jnp.clip(I, min=0.0) / N
            return (S - dt * inc, I + dt * (inc - gamma * I)), dt * inc

        return jnp.sum(jax.lax.scan(step, (N - I0, I0), ts)[1])
