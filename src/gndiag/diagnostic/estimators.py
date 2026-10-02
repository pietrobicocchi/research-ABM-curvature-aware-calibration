"""Estimators of G = (E A)^T (E A) from per-seed feature Jacobians A_j (Sec. 3.4, Eq. 12)."""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from gndiag.diagnostic.ggn import symmetrize

Array = jax.Array


def feature_jacobians(feature_of_z: Callable[[Array, Array], Array], z: Array,
                      keys: Array) -> Array:
    """A_j = D_z psi(X(z, xi_j)) by forward mode, stacked over seeds: (M, D, P)."""
    jac = jax.jacfwd(lambda zz, k: feature_of_z(zz, k), argnums=0)
    return jax.vmap(lambda k: jac(z, k))(keys)


def ggn_plugin(A: Array) -> Array:
    """Plug-in G_V = J^T J with J the seed-mean Jacobian (Eq. 14)."""
    J = jnp.mean(A, axis=0)
    return symmetrize(J.T @ J)


def ggn_cross_seed(A: Array) -> Array:
    """Cross-seed G_U = sum_{j != l} A_j^T A_l / (M (M - 1)) (Eq. 16)."""
    M = A.shape[0]
    S = jnp.sum(A, axis=0)
    Q = jnp.einsum("mdp,mdq->pq", A, A)
    return symmetrize((S.T @ S - Q) / (M * (M - 1)))
