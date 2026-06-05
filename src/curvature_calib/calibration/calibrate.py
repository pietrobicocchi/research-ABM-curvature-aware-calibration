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
    loss_only,
    per_seed_loss_and_grads,
    vmap_simulate,
)
from curvature_calib.calibration.preconditioner import (
    damped_step,
    quadratic_model_reduction,
    update_damping,
)

@dataclass
class CalibLog:
    thetas: list           # iterates (n_iter+1, P)
    losses: list           # (n_iter+1,) -- noisy training-time MMD^2 (fresh seeds each iter)
    val_losses: list       # (n_iter+1,) -- clean validation MMD^2 (FIXED seed set across iters)
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
            "val_losses": np.asarray(self.val_losses),
            "mean_grads": np.asarray(self.mean_grads),
            "opgs": np.asarray(self.opgs),
            "eigvals": np.asarray(self.eigvals),
            "eigvecs": np.asarray(self.eigvecs),
            "per_seed_grads": np.asarray(self.per_seed_grads),
            "dampings": np.asarray(self.dampings),
        }


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
    val_M: int | None = None,
    val_seed: int = 999_999,
) -> CalibLog:
    """Run OPG-preconditioned Levenberg-Marquardt calibration.

    Two loss tracks are recorded:
        - `losses`: the noisy MMD^2 estimated on the same fresh-per-iter seeds
          the optimizer actually used. Reflects the optimizer's view.
        - `val_losses`: a clean MMD^2 estimated on a FIXED held-out seed set
          (`val_seed`, val_M seeds). Monotone-decreasing tracking; the right
          quantity to plot for "is the optimizer making progress".
    """
    theta = theta0
    damping = init_damping

    log = CalibLog(
        thetas=[theta], losses=[], val_losses=[], mean_grads=[], opgs=[],
        eigvals=[], eigvecs=[], per_seed_grads=[], dampings=[],
    )

    master = jax.random.PRNGKey(seed_base)
    val_M = val_M or M
    val_keys = jax.random.split(jax.random.PRNGKey(val_seed), val_M)  # FIXED

    for t in range(n_iter):
        keys = jax.random.split(jax.random.fold_in(master, t), M)
        stats = per_seed_loss_and_grads(simulate_fn, theta, keys, Y_ref)
        L_curr = float(stats.loss)
        L_val = float(loss_only(simulate_fn, theta, val_keys, Y_ref))

        eig = eigendecompose(stats.opg)

        step = learning_rate * damped_step(stats.opg, stats.mean_grad, damping)
        theta_proposed = theta + step
        L_prop = float(loss_only(simulate_fn, theta_proposed, keys, Y_ref))

        pred = quadratic_model_reduction(stats.opg, stats.mean_grad, step, damping)

        # Step explosion guard: a too-large step at low damping can push
        # theta into the unstable BH regime (g/R > ~1.5) where simulator
        # output goes inf/nan and MMD returns NaN. Treat that as a hard
        # rejection AND force damping to step up so the next attempt is smaller.
        import math
        if not math.isfinite(L_prop):
            realised = -math.inf
            damping = min(damping * 10.0, 1e8)
            accept = False
        else:
            realised = L_curr - L_prop
            accept = realised > 0.0
        if accept:
            theta = theta_proposed

        log.losses.append(L_curr)
        log.val_losses.append(L_val)
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
                f"  iter {t:3d}  L={L_curr:+.3e} (val {L_val:+.3e})  "
                f"|g|={grad_norm:.3e}  lam={damping:.2e}  cond(F)={cond:.2e}  accept={accept}"
            )

    keys = jax.random.split(jax.random.fold_in(master, n_iter), M)
    log.losses.append(float(loss_only(simulate_fn, theta, keys, Y_ref)))
    log.val_losses.append(float(loss_only(simulate_fn, theta, val_keys, Y_ref)))
    return log
