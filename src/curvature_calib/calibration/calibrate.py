"""End-to-end OPG-preconditioned calibration of a differentiable ABM under MMD.

Each iterate logs:
    - theta_t
    - loss (MMD^2)
    - mean gradient
    - per-seed gradients (full M x P matrix)
    - OPG matrix and its eigendecomposition
    - damping
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import jax
import jax.numpy as jnp

from curvature_calib.calibration.opg import eigendecompose
from curvature_calib.calibration.per_seed_grads import (
    per_seed_loss_and_grads,
    vmap_simulate,
)
from curvature_calib.calibration.preconditioner import (
    damped_step,
    quadratic_model_reduction,
    update_damping,
)
from curvature_calib.losses.mmd import mmd_sq_with_median_bandwidth


@dataclass
class CalibLog:
    thetas: list           # iterates (n_iter+1, P)
    losses: list           # (n_iter+1,)
    mean_grads: list       # (n_iter, P)
    opgs: list             # (n_iter, P, P)
    eigvals: list          # (n_iter, P) descending
    eigvecs: list          # (n_iter, P, P)
    per_seed_grads: list   # (n_iter, M, P)
    dampings: list         # (n_iter,)

    def as_arrays(self):
        import numpy as np
        return {
            "thetas": np.asarray(self.thetas),
            "losses": np.asarray(self.losses),
            "mean_grads": np.asarray(self.mean_grads),
            "opgs": np.asarray(self.opgs),
            "eigvals": np.asarray(self.eigvals),
            "eigvecs": np.asarray(self.eigvecs),
            "per_seed_grads": np.asarray(self.per_seed_grads),
            "dampings": np.asarray(self.dampings),
        }


def _loss_only(simulate_fn, theta, keys, Y_ref):
    X = vmap_simulate(simulate_fn, theta, keys)
    return mmd_sq_with_median_bandwidth(X, Y_ref)


def calibrate(
    simulate_fn: Callable,
    theta0: jax.Array,
    Y_ref: jax.Array,
    M: int = 64,
    n_iter: int = 80,
    learning_rate: float = 1.0,
    init_damping: float = 1.0,
    seed_base: int = 0,
    verbose: bool = True,
) -> CalibLog:
    """Run OPG-preconditioned Levenberg-Marquardt calibration."""
    theta = theta0
    damping = init_damping

    log = CalibLog(
        thetas=[theta], losses=[], mean_grads=[], opgs=[], eigvals=[], eigvecs=[],
        per_seed_grads=[], dampings=[],
    )

    # Seed plan: each iterate gets its own set of M independent keys (fresh
    # noise per evaluation, preventing the optimiser from overfitting one set).
    master = jax.random.PRNGKey(seed_base)

    for t in range(n_iter):
        keys = jax.random.split(jax.random.fold_in(master, t), M)
        stats = per_seed_loss_and_grads(simulate_fn, theta, keys, Y_ref)
        L_curr = float(stats.loss)

        # Eigendecomposition for the diagnostic.
        eig = eigendecompose(stats.opg)

        # Propose damped step.
        step = learning_rate * damped_step(stats.opg, stats.mean_grad, damping)
        theta_proposed = theta + step

        # Realised loss at the proposed iterate, using the SAME keys.
        L_prop = float(_loss_only(simulate_fn, theta_proposed, keys, Y_ref))

        # LM ratio.
        pred = quadratic_model_reduction(stats.opg, stats.mean_grad, step, damping)
        realised = L_curr - L_prop

        # Accept or reject.
        accept = realised > 0.0
        if accept:
            theta = theta_proposed
            L_new = L_prop
        else:
            L_new = L_curr

        # Log BEFORE updating damping (so the log reflects what was used).
        log.losses.append(L_curr)
        log.mean_grads.append(stats.mean_grad)
        log.opgs.append(stats.opg)
        log.eigvals.append(eig.eigvals)
        log.eigvecs.append(eig.eigvecs)
        log.per_seed_grads.append(stats.per_seed_grads)
        log.dampings.append(damping)
        log.thetas.append(theta)

        damping = update_damping(damping, realised, pred)

        if verbose and (t % 5 == 0 or t == n_iter - 1):
            grad_norm = float(jnp.linalg.norm(stats.mean_grad))
            cond = float(eig.eigvals[0] / jnp.clip(eig.eigvals[-1], min=1e-30))
            print(
                f"  iter {t:3d}  L={L_curr:.4e}  |g|={grad_norm:.3e}  "
                f"lam={damping:.2e}  cond(F)={cond:.2e}  accept={accept}"
            )

    # Final loss at last theta.
    keys = jax.random.split(jax.random.fold_in(master, n_iter), M)
    L_final = float(_loss_only(simulate_fn, theta, keys, Y_ref))
    log.losses.append(L_final)
    return log
