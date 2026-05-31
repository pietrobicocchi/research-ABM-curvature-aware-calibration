"""Compare OPG-preconditioned vs Adam vs SGD on the same calibration problem."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from curvature_calib.calibration.baselines import adam, sgd
from curvature_calib.calibration.calibrate import calibrate
from curvature_calib.calibration.opg import eigendecompose
from curvature_calib.calibration.per_seed_grads import vmap_simulate
from curvature_calib.models.brock_hommes import simulate
from curvature_calib.viz.style import QUAL, apply_style


T = 200
SIGMA = 0.05
R = 1.1
THETA_STAR = jnp.array([3.0, 1.2, 0.2, 1.2, -0.2])
PARAM_NAMES = [r"$\beta$", r"$g_1$", r"$b_1$", r"$g_2$", r"$b_2$"]


def _sim(theta, key):
    return simulate(theta, key, T=T, sigma=SIGMA, R=R, x_init=0.0)


def main() -> None:
    apply_style()

    M_ref = 128
    ref_keys = jax.random.split(jax.random.PRNGKey(0), M_ref)
    Y_ref = vmap_simulate(_sim, THETA_STAR, ref_keys)

    # Init in the STIFF direction (symmetric bias shift). Generic large
    # perturbations often land along sloppy directions where MMD^2 is
    # already at the noise floor; perturbing b_1 and b_2 in the same
    # direction puts the loss meaningfully above the floor.
    theta0 = THETA_STAR + jnp.array([0.0, 0.0, 0.1, 0.0, 0.1])
    n_iter = 50
    M = 64

    print("OPG-preconditioned...")
    opg_log = calibrate(_sim, theta0, Y_ref, M=M, n_iter=n_iter,
                        init_damping=100.0, verbose=False)
    print("Adam...")
    adam_log = adam(_sim, theta0, Y_ref, M=M, n_iter=n_iter, lr=1e-2)
    print("SGD...")
    sgd_log = sgd(_sim, theta0, Y_ref, M=M, n_iter=n_iter, lr=1e-3)

    o = opg_log.as_arrays()
    a = adam_log.as_arrays()
    s = sgd_log.as_arrays()

    def err(thetas):
        return np.linalg.norm(thetas - np.asarray(THETA_STAR), axis=1)

    err_o = err(o["thetas"])
    err_a = err(a["thetas"])
    err_s = err(s["thetas"])

    eig = eigendecompose(jnp.asarray(o["opgs"][-1]))
    V = np.asarray(eig.eigvecs)
    proj_o = (o["thetas"] - np.asarray(THETA_STAR)) @ V
    proj_a = (a["thetas"] - np.asarray(THETA_STAR)) @ V
    proj_s = (s["thetas"] - np.asarray(THETA_STAR)) @ V

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # A: best-so-far val losses (standard optim convention, monotone)
    ax = axes[0, 0]
    for name, arr, color in [("OPG", o, QUAL[0]), ("Adam", a, QUAL[1]), ("SGD", s, QUAL[2])]:
        best = np.minimum.accumulate(arr["val_losses"])
        ax.semilogy(np.clip(best, 1e-12, None), color=color, lw=2.2, label=name)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"best-so-far val MMD$^2$ (clipped)")
    ax.set_title("A. Loss vs iteration (best val so far)")
    ax.legend(fontsize=9)

    # B: parameter distance
    ax = axes[0, 1]
    ax.semilogy(np.clip(err_o, 1e-6, None), color=QUAL[0], lw=1.8, label="OPG")
    ax.semilogy(np.clip(err_a, 1e-6, None), color=QUAL[1], lw=1.8, label="Adam")
    ax.semilogy(np.clip(err_s, 1e-6, None), color=QUAL[2], lw=1.8, label="SGD")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\|\theta_t - \theta^*\|_2$")
    ax.set_title("B. Distance to truth (log)")
    ax.legend(fontsize=9)

    # C: final eigenvalue spectrum from the OPG run
    ax = axes[0, 2]
    final_eigs = np.clip(o["eigvals"][-1], 1e-30, None)
    xs = np.arange(len(final_eigs))
    ax.semilogy(xs, final_eigs, "o-", color=QUAL[0], markersize=11,
                markerfacecolor=QUAL[0], markeredgecolor="white",
                markeredgewidth=1.4, lw=1.6)
    ax.set_xlabel("eigendirection")
    ax.set_ylabel(r"$\lambda_k$")
    ax.set_title("C. OPG spectrum at final iterate (sloppy)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$v_{k+1}$" for k in xs])

    # D: trajectories in eigenbasis (stiff vs sloppy)
    ax = axes[1, 0]
    for label, proj, color in [("OPG", proj_o, QUAL[0]),
                               ("Adam", proj_a, QUAL[1]),
                               ("SGD",  proj_s, QUAL[2])]:
        ax.plot(proj[:, 0], proj[:, -1], "o-", color=color, lw=1.2,
                markersize=3.5, label=label, alpha=0.85)
        ax.scatter(proj[0, 0], proj[0, -1], color=color, s=90,
                   marker="o", edgecolor="black", linewidth=1.2, zorder=5)
        ax.scatter(proj[-1, 0], proj[-1, -1], color=color, s=140,
                   marker="*", edgecolor="black", linewidth=1.0, zorder=5)
    ax.scatter([0], [0], s=260, marker="*", color="#ffea00",
               edgecolor="black", linewidth=1.2, label=r"$\theta^*$", zorder=10)
    ax.set_xlabel(r"projection on $v_1$ (stiff)")
    ax.set_ylabel(r"projection on $v_P$ (sloppy)")
    ax.set_title("D. Trajectories in OPG eigenbasis")
    ax.legend(fontsize=8)

    # E: per-parameter recovery (linear y)
    ax = axes[1, 1]
    width = 0.25
    xs_p = np.arange(len(PARAM_NAMES))
    for i, (label, arr, color) in enumerate([("OPG", o, QUAL[0]),
                                              ("Adam", a, QUAL[1]),
                                              ("SGD", s, QUAL[2])]):
        err_per = np.abs(arr["thetas"][-1] - np.asarray(THETA_STAR))
        ax.bar(xs_p + (i - 1) * width, err_per, width,
               color=color, edgecolor="white", label=label)
    ax.set_xticks(xs_p)
    ax.set_xticklabels(PARAM_NAMES)
    ax.set_xlabel("parameter")
    ax.set_ylabel(r"$|\theta^{(T)}_k - \theta^*_k|$")
    ax.set_title("E. Per-parameter recovery at final iterate")
    ax.legend(fontsize=8)

    # F: condition number over OPG run
    ax = axes[1, 2]
    eigs_traj = np.clip(o["eigvals"], 1e-30, None)
    cond = eigs_traj[:, 0] / eigs_traj[:, -1]
    ax.semilogy(cond, color=QUAL[0], lw=1.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\kappa(\hat F)$")
    ax.set_title(r"F. Condition number of $\hat F$ over OPG run")

    fig.suptitle("Optimiser comparison on Brock-Hommes calibration",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = Path("outputs") / "07_optimizer_comparison.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
