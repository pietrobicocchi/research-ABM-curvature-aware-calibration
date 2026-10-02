"""Frozen random Fourier features for the kernel mean embedding (Eq. 11)."""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class RFF(NamedTuple):
    w: Array      # (D, n) frequencies
    b: Array      # (D,) phases
    gamma: float  # bandwidth


def frozen_rff(key: Array, D: int, n: int, gamma: float, dtype=jnp.float64) -> RFF:
    """Draw w_j ~ N(0, gamma^-2 I), b_j ~ U[0, 2 pi] once; they stay fixed."""
    kw, kb = jax.random.split(key)
    w = jax.random.normal(kw, (D, n), dtype=dtype) / gamma
    b = jax.random.uniform(kb, (D,), dtype=dtype) * (2.0 * jnp.pi)
    return RFF(w=w, b=b, gamma=gamma)


def feature_map(rff: RFF, X: Array) -> Array:
    """phi(X) = sqrt(2/D) cos(w X + b); X: (n,) -> (D,) or (M, n) -> (M, D)."""
    D = rff.w.shape[0]
    return jnp.sqrt(2.0 / D) * jnp.cos(X @ rff.w.T + rff.b)
