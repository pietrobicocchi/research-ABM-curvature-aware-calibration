"""Surrogate gradient implementations for discrete ABM transitions.

Two variants:
  - gumbel_sigmoid: differentiable Bernoulli via Gumbel-Sigmoid at temperature
    tau (Jang et al. 2017). Smooth in p for any tau > 0; biased for tau > 0;
    recovers hard Bernoulli as tau -> 0.
  - straight_through_bernoulli: hard Bernoulli sample with gradient passed
    straight through as if the output were p (Bengio et al. 2013). Unbiased
    forward; biased surrogate gradient.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr


def gumbel_sigmoid(
    p: jax.Array,
    key: jax.Array,
    tau: float = 0.5,
    eps: float = 1e-6,
) -> jax.Array:
    """Differentiable Bernoulli draw via Gumbel-Sigmoid at temperature tau."""
    k1, k2 = jr.split(key)
    u1 = jnp.clip(jr.uniform(k1, p.shape, dtype=p.dtype), eps, 1.0 - eps)
    u2 = jnp.clip(jr.uniform(k2, p.shape, dtype=p.dtype), eps, 1.0 - eps)
    g1 = -jnp.log(-jnp.log(u1))
    g0 = -jnp.log(-jnp.log(u2))
    p_clipped = jnp.clip(p, eps, 1.0 - eps)
    logit = jnp.log(p_clipped) - jnp.log(1.0 - p_clipped)
    return jax.nn.sigmoid((logit + g1 - g0) / tau)


@jax.custom_jvp
def straight_through_bernoulli(p: jax.Array, key: jax.Array) -> jax.Array:
    """Hard Bernoulli sample; gradient passes through as if output were p."""
    return jr.bernoulli(key, p).astype(p.dtype)


@straight_through_bernoulli.defjvp
def _st_jvp(primals, tangents):
    p, key = primals
    p_dot, _ = tangents
    return straight_through_bernoulli(p, key), p_dot
