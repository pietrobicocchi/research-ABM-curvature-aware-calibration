"""Adam-lr sweep: is the wandering structural or just bad hyperparameters?

Phase 2 result was that Adam at lr=1e-2 actively moves AWAY from truth.
We claimed this reflects noise adaptation amplifying sloppy directions
(structural, Kunstner et al. 2019 reading). The honest sanity check is to
sweep lr and see whether a smaller lr would have kept Adam at theta_0.

Protocol:
    For one medium-difficulty pair, run Adam at lr in
        {1e-1, 1e-2, 1e-3, 1e-4, 1e-5}
    for 80 iterations, M=64. Compare to OPG and SGD on the same problem.

Hypothesis (structural):
    Adam diverges from theta* at ALL nontrivial lrs because its per-
    coordinate adaptation amplifies the gradient component along sloppy
    directions, where the true MMD gradient is weak.

Alternative (hyperparameter artefact):
    At some small lr, Adam stays at theta_0 like OPG/SGD.

Run: uv run python scripts/12_adam_lr_sweep.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from curvature_calib.calibration.baselines import adam, sgd
from curvature_calib.calibration.calibrate import calibrate
from curvature_calib.calibration.per_seed_grads import vmap_simulate
from curvature_calib.models.brock_hommes import simulate
from curvature_calib.viz.style import QUAL, apply_style


T = 200
SIGMA = 0.05
R = 1.1
THETA_STAR = jnp.array([3.0, 1.2, 0.2, 1.2, -0.2])
N_ITER = 80
M = 64
LR_GRID = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]


def _sim(theta, key):
    return simulate(theta, key, T=T, sigma=SIGMA, R=R, x_init=0.0,
                    grad_horizon=None)


def main() -> None:
    apply_style()
    out = Path("outputs")

    # Reference at theta*.
    M_ref = 128
    ref_keys = jax.random.split(jax.random.PRNGKey(0), M_ref)
    Y_ref = vmap_simulate(_sim, THETA_STAR, ref_keys)

    # Pick a fixed, reproducible theta_0 at the medium-difficulty distance.
    # We sample a random unit direction (PRNGKey deterministic) and scale to
    # 0.15 in parameter space, with safety guard against the explosion
    # threshold g_h / R > 1.36.
    key = jax.random.PRNGKey(1234)
    while True:
        u = jax.random.normal(key, (5,))
        u = u / jnp.linalg.norm(u)
        candidate = THETA_STAR + 0.15 * u
        g1, g2 = float(candidate[1]), float(candidate[3])
        if g1 < 1.4 and g2 < 1.4 and g1 > 0 and g2 > 0:
            theta_0 = candidate
            break
        key, _ = jax.random.split(key)
    print(f"theta_0 = {np.asarray(theta_0)}")
    print(f"||theta_0 - theta*|| = {float(jnp.linalg.norm(theta_0 - THETA_STAR)):.4f}")

    # Sweep Adam over lrs.
    print("Adam sweep ...")
    adam_results = {}
    for lr in LR_GRID:
        log = adam(_sim, theta_0, Y_ref, M=M, n_iter=N_ITER, lr=lr)
        a = log.as_arrays()
        adam_results[lr] = {
            "thetas": a["thetas"],
            "losses": a["val_losses"],  # clean track for plots
            "err": np.linalg.norm(a["thetas"] - np.asarray(THETA_STAR), axis=1),
        }
        print(f"  lr={lr:.0e}: err_end={adam_results[lr]['err'][-1]:.3f}  "
              f"val_loss_end={adam_results[lr]['losses'][-1]:+.2e}")

    # OPG and SGD for reference.
    print("OPG ...")
    opg_log = calibrate(_sim, theta_0, Y_ref, M=M, n_iter=N_ITER, verbose=False)
    opg_a = opg_log.as_arrays()
    opg_a["losses"] = opg_a["val_losses"]  # use clean track for plots below
    opg_err = np.linalg.norm(opg_a["thetas"] - np.asarray(THETA_STAR), axis=1)

    print("SGD ...")
    sgd_log = sgd(_sim, theta_0, Y_ref, M=M, n_iter=N_ITER, lr=1e-3)
    sgd_a = sgd_log.as_arrays()
    sgd_a["losses"] = sgd_a["val_losses"]
    sgd_err = np.linalg.norm(sgd_a["thetas"] - np.asarray(THETA_STAR), axis=1)

    # =============================== FIGURE
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # A. Adam parameter-recovery error vs iteration, one line per lr.
    ax = axes[0]
    cmap = plt.cm.plasma
    for i, lr in enumerate(LR_GRID):
        color = cmap(i / max(len(LR_GRID) - 1, 1))
        ax.plot(adam_results[lr]["err"], color=color, lw=1.8,
                label=f"Adam lr={lr:.0e}")
    ax.plot(opg_err, color="black", lw=2.0, ls="--", label="OPG (LM)")
    ax.plot(sgd_err, color="grey", lw=2.0, ls=":", label="SGD lr=1e-3")
    ax.axhline(float(jnp.linalg.norm(theta_0 - THETA_STAR)),
               color="grey", lw=1, alpha=0.5,
               label=r"$\|\theta_0 - \theta^*\|$")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\|\theta_t - \theta^*\|_2$")
    ax.set_title("A. Adam-lr effect on parameter recovery")
    ax.legend(fontsize=8)

    # B. Final error vs lr.
    ax = axes[1]
    finals = [adam_results[lr]["err"][-1] for lr in LR_GRID]
    ax.semilogx(LR_GRID, finals, "o-", color=QUAL[1], lw=2, markersize=10,
                markerfacecolor=QUAL[1], markeredgecolor="white",
                markeredgewidth=1.4, label="Adam final err")
    ax.axhline(float(opg_err[-1]), color=QUAL[0], lw=1.5, ls="--",
               label=f"OPG final = {opg_err[-1]:.3f}")
    ax.axhline(float(sgd_err[-1]), color=QUAL[2], lw=1.5, ls=":",
               label=f"SGD final = {sgd_err[-1]:.3f}")
    ax.axhline(float(jnp.linalg.norm(theta_0 - THETA_STAR)),
               color="grey", lw=1, alpha=0.5,
               label=fr"$\|\theta_0\| = ${float(jnp.linalg.norm(theta_0 - THETA_STAR)):.3f}")
    ax.set_xlabel("Adam learning rate")
    ax.set_ylabel(r"final $\|\theta_T - \theta^*\|_2$")
    ax.set_title("B. Does any lr save Adam?")
    ax.legend(fontsize=8)

    # C. Adam loss curves per lr.
    ax = axes[2]
    for i, lr in enumerate(LR_GRID):
        color = cmap(i / max(len(LR_GRID) - 1, 1))
        ax.semilogy(np.clip(adam_results[lr]["losses"], 1e-6, None),
                    color=color, lw=1.6, label=f"Adam lr={lr:.0e}")
    ax.semilogy(np.clip(opg_a["losses"], 1e-6, None),
                color="black", lw=2.0, ls="--", label="OPG (LM)")
    ax.semilogy(np.clip(sgd_a["losses"], 1e-6, None),
                color="grey", lw=2.0, ls=":", label="SGD")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"MMD$^2$ (clipped at $10^{-6}$)")
    ax.set_title("C. Adam-lr effect on loss trajectory")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Adam-lr sweep: is the wandering structural or hyperparameter artefact?",
        fontsize=14, fontweight="bold", y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    p = out / "12_adam_lr_sweep.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {p}")

    # ===================================================== verdict
    print("\n" + "=" * 76)
    print("ADAM-LR VERDICT")
    print("=" * 76)
    d0 = float(jnp.linalg.norm(theta_0 - THETA_STAR))
    print(f"\n||theta_0 - theta*|| = {d0:.4f}")
    print(f"\nAdam final err by lr:")
    for lr in LR_GRID:
        e = adam_results[lr]["err"][-1]
        worse = "WORSE THAN START" if e > d0 * 1.05 else (
                "stay at start" if abs(e - d0) < 0.02 else "moved")
        print(f"  lr={lr:.0e}: err_end={e:.4f}  ({worse})")
    print(f"\nOPG final err = {opg_err[-1]:.4f}")
    print(f"SGD final err = {sgd_err[-1]:.4f}")


if __name__ == "__main__":
    main()
