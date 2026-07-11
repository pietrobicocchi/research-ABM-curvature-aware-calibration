"""Per-seed scalar-loss gradients and their uncentered second moment (F_OPG).

The MMD^2 loss is a U-statistic that couples samples; the "per-seed gradient" is
the contribution to nabla_theta MMD^2 from one simulator seed via the chain rule

    g_m = M * (dMMD^2 / dx_m) . (dx_m / dtheta)

The factor M is chosen so that the *mean* of the per-seed gradients equals the
total gradient,

    mean_g = (1/M) sum_m g_m = nabla_theta MMD^2.

From these the module forms the **uncentered second moment**

    F_OPG = (1/M) sum_m g_m g_m^T.

IMPORTANT (DEC-001, Math-Spec §14): F_OPG is a *labeled comparison object*, NOT
the generalized Gauss–Newton matrix, Fisher information, Hessian, or a
curvature/identifiability matrix. For a residual loss g = J^T r, so
g g^T = J^T r r^T J != J^T J, and at an exact fit g = 0 while the true GGN
J^T J != 0. The true MMD GGN is J_eta^T J_eta (see geometry.rff / geometry.ggn).
The centered covariance C_g is available via
diagnostic.scalar_gradient_covariance.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from curvature_calib.losses.mmd import mmd_sq_with_median_bandwidth


class CalibStats(NamedTuple):
    loss: jax.Array            # scalar, unbiased MMD^2
    mean_grad: jax.Array       # (P,) mean per-seed gradient = nabla_theta MMD^2
    per_seed_grads: jax.Array  # (M, P)
    # (P, P) F_OPG = (1/M) sum_m g_m g_m^T: uncentered second moment of per-seed
    # scalar-loss gradients. NOT the GGN/Fisher/Hessian/curvature (DEC-001).
    opg: jax.Array


def vmap_simulate(simulate_fn: Callable, theta: jax.Array, keys: jax.Array) -> jax.Array:
    """Run M seeds in parallel. Returns (M, ...) trajectory tensor."""
    return jax.vmap(lambda k: simulate_fn(theta, k))(keys)


def per_seed_loss_and_grads(
    simulate_fn: Callable,
    theta: jax.Array,
    keys: jax.Array,
    Y_ref: jax.Array,
) -> CalibStats:
    """Compute MMD^2, per-seed gradients, mean gradient and OPG matrix.

    Strategy:
        1. Forward-simulate M trajectories X via vmap.
        2. VJP through the MMD^2(X, Y_ref) scalar wrt X to get (dL/dx_m)_m.
        3. For each seed m, VJP through simulate(theta; key_m) with cotangent
           dL/dx_m to obtain (dL/dx_m) . (dx_m/dtheta). Multiply by M.
        4. Stack -> per_seed_grads (M, P); mean -> mean_grad; outer product mean -> OPG.
    """
    M = keys.shape[0]

    # Step 1: forward
    X = vmap_simulate(simulate_fn, theta, keys)  # (M, T)

    # Step 2: VJP of MMD^2 wrt X
    L, vjp_X = jax.vjp(lambda x: mmd_sq_with_median_bandwidth(x, Y_ref), X)
    (dL_dX,) = vjp_X(jnp.ones((), dtype=L.dtype))  # (M, T)

    # Step 3: per-seed VJPs through simulate
    def one_seed_grad(key: jax.Array, cotangent: jax.Array) -> jax.Array:
        _, vjp_t = jax.vjp(lambda t: simulate_fn(t, key), theta)
        (g,) = vjp_t(cotangent)
        return M * g

    per_seed = jax.vmap(one_seed_grad)(keys, dL_dX)  # (M, P)

    mean_grad = jnp.mean(per_seed, axis=0)            # (P,)
    opg = (per_seed.T @ per_seed) / M                  # F_OPG (comparison object)
    return CalibStats(loss=L, mean_grad=mean_grad,
                      per_seed_grads=per_seed, opg=opg)


def loss_only(
    simulate_fn: Callable,
    theta: jax.Array,
    keys: jax.Array,
    Y_ref: jax.Array,
) -> jax.Array:
    """MMD² for a given theta without computing gradients."""
    X = vmap_simulate(simulate_fn, theta, keys)
    return mmd_sq_with_median_bandwidth(X, Y_ref)
