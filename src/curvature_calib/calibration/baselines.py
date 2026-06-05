"""First-order baseline optimizers for comparison with the OPG preconditioner.

Plain SGD and Adam, sharing the per-seed-gradient backbone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp

from curvature_calib.calibration.per_seed_grads import (
    loss_only,
    per_seed_loss_and_grads,
    vmap_simulate,
)

@dataclass
class FirstOrderLog:
    thetas: list = field(default_factory=list)
    losses: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)
    mean_grads: list = field(default_factory=list)

    def as_arrays(self):
        import numpy as np
        return {
            "thetas": np.asarray(self.thetas),
            "losses": np.asarray(self.losses),
            "val_losses": np.asarray(self.val_losses),
            "mean_grads": np.asarray(self.mean_grads),
        }


def sgd(simulate_fn: Callable, theta0, Y_ref, M=64, n_iter=80,
        lr=1e-2, seed_base=0, verbose=False,
        val_M=None, val_seed=999_999) -> FirstOrderLog:
    theta = theta0
    log = FirstOrderLog(thetas=[theta], losses=[], val_losses=[], mean_grads=[])
    master = jax.random.PRNGKey(seed_base)
    val_M = val_M or M
    val_keys = jax.random.split(jax.random.PRNGKey(val_seed), val_M)
    for t in range(n_iter):
        keys = jax.random.split(jax.random.fold_in(master, t), M)
        stats = per_seed_loss_and_grads(simulate_fn, theta, keys, Y_ref)
        log.losses.append(float(stats.loss))
        log.val_losses.append(float(loss_only(simulate_fn, theta, val_keys, Y_ref)))
        log.mean_grads.append(stats.mean_grad)
        theta = theta - lr * stats.mean_grad
        log.thetas.append(theta)
        if verbose and t % 10 == 0:
            print(f"  SGD iter {t}  L={float(stats.loss):.4e}  val={log.val_losses[-1]:+.3e}")
    keys = jax.random.split(jax.random.fold_in(master, n_iter), M)
    log.losses.append(float(loss_only(simulate_fn, theta, keys, Y_ref)))
    log.val_losses.append(float(loss_only(simulate_fn, theta, val_keys, Y_ref)))
    return log


def adam(simulate_fn: Callable, theta0, Y_ref, M=64, n_iter=80,
         lr=1e-2, b1=0.9, b2=0.999, eps=1e-8, seed_base=0,
         verbose=False, val_M=None, val_seed=999_999) -> FirstOrderLog:
    theta = theta0
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    log = FirstOrderLog(thetas=[theta], losses=[], val_losses=[], mean_grads=[])
    master = jax.random.PRNGKey(seed_base)
    val_M = val_M or M
    val_keys = jax.random.split(jax.random.PRNGKey(val_seed), val_M)
    for t in range(1, n_iter + 1):
        keys = jax.random.split(jax.random.fold_in(master, t), M)
        stats = per_seed_loss_and_grads(simulate_fn, theta, keys, Y_ref)
        log.losses.append(float(stats.loss))
        log.val_losses.append(float(loss_only(simulate_fn, theta, val_keys, Y_ref)))
        log.mean_grads.append(stats.mean_grad)
        g = stats.mean_grad
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        theta = theta - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        log.thetas.append(theta)
        if verbose and t % 10 == 0:
            print(f"  Adam iter {t}  L={float(stats.loss):.4e}  val={log.val_losses[-1]:+.3e}")
    keys = jax.random.split(jax.random.fold_in(master, n_iter + 1), M)
    log.losses.append(float(loss_only(simulate_fn, theta, keys, Y_ref)))
    log.val_losses.append(float(loss_only(simulate_fn, theta, val_keys, Y_ref)))
    return log
