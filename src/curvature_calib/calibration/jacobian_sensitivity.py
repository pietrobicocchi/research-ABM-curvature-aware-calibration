"""Per-parameter first-order sensitivity analysis (Quera-Bofarull et al. 2025 §5.4).

The standard sensitivity analysis in differentiable ABMs is the per-parameter
Jacobian of the simulator observable:

    S_k(theta) = || d Phi(x_m) / d theta_k ||   (some norm, averaged across seeds)

It tells you which *individual* parameter has the largest impact on the
simulator output. By construction it is coordinate-aligned: parameter 1 vs
parameter 2 vs ... .

This contrasts with the OPG eigendecomposition we compute, which surfaces
parameter *combinations* — eigenvectors that are off-axis mixtures of the
original parameters. The headline of our comparison is that the most
identifiable structure in MMD-calibrated Brock-Hommes is the symmetric bias
combination b_1 + b_2, which no coordinate-aligned analysis can surface.

This module provides:
    - `per_param_jacobian_sensitivity(simulate_fn, theta, keys)`
       (Frobenius-norm sensitivity of d trajectory / d theta_k)
    - `opg_correlation_matrix(F_hat)`
       (off-diagonal correlations are what the eigendecomposition exploits)
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def per_param_jacobian_sensitivity(
    simulate_fn: Callable,
    theta: jax.Array,
    keys: jax.Array,
) -> jax.Array:
    """Mean across seeds of || d x_m / d theta_k ||_2 for each parameter k.

    Concretely: for each seed m we compute the (T, P) Jacobian J_m = d x_m / d theta;
    the column norm ||J_m[:, k]||_2 is the per-parameter "sensitivity strength"
    for seed m. We average it across the M seeds.

    Returns shape (P,).
    """
    P = theta.shape[0]

    def one_seed(key: jax.Array) -> jax.Array:
        # Forward-mode Jacobian (P columns, each one extra simulation).
        J = jax.jacfwd(lambda t: simulate_fn(t, key))(theta)  # (T, P)
        return jnp.linalg.norm(J, axis=0)                     # (P,)

    sens = jax.vmap(one_seed)(keys)                            # (M, P)
    return sens.mean(axis=0)


def opg_diagonal_sensitivity(opg: jax.Array) -> jax.Array:
    """Per-parameter sensitivity from the OPG diagonal.

    sqrt(F_hat_kk) = standard deviation of the per-seed gradient
    component k. This is what a "diagonal-only" reading of the OPG matrix
    gives, ignoring the off-diagonal coupling that the eigendecomposition
    exploits.
    """
    return jnp.sqrt(jnp.clip(jnp.diag(opg), min=0.0))


def opg_correlation_matrix(opg: jax.Array) -> jax.Array:
    """Pearson-style correlation matrix derived from F_hat.

        rho_kl = F_hat_kl / sqrt(F_hat_kk * F_hat_ll)

    The off-diagonal entries (|rho_kl| close to 1) are what the
    eigendecomposition discovers and the per-parameter view ignores.
    """
    d = jnp.sqrt(jnp.clip(jnp.diag(opg), min=1e-30))
    return opg / (d[:, None] * d[None, :])
