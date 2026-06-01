"""Script 22: paper Figure 3 — the diagnostic predicting Adam.

Paper-polished version of script 11. Collapses script 11's
3-difficulty x 3-channel grid into a single representative two-panel
figure: medium difficulty, two panels.

    (a) recovery error per eigendirection of F_hat(theta*)
        — OPG / SGD / Adam side-by-side, log scale
    (b) lambda_k vs median squared error decomposition
        — the "stiff = constrained, sloppy = unconstrained" reading

Adam's recovery error grows in lockstep with shrinking lambda_k while
OPG's stays flat: the visual statement that "the diagnostic predicts
where Adam fails".

Polish vs script 11:
    * float64 throughout
    * SELF-CONTAINED: runs N_PAIRS calibrations inline at medium distance,
      no dependence on outputs/brock_hommes/10_phase2_convergence.npz
    * two panels instead of nine
    * panel titles (a)/(b); paper-style suptitle
    * stiff/sloppy ratio annotation in panel (a)
    * writes to outputs/paper/figures/fig3_predicts_adam.png

Run: uv run python scripts/22_fig3_predicts_adam.py
"""

from __future__ import annotations

from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from curvature_calib.calibration.baselines import adam, sgd
from curvature_calib.calibration.calibrate import calibrate
from curvature_calib.calibration.opg import eigendecompose
from curvature_calib.calibration.per_seed_grads import (
    per_seed_loss_and_grads, vmap_simulate,
)
from curvature_calib.losses.mmd import mmd_sq_with_median_bandwidth
from curvature_calib.models.brock_hommes import simulate
from curvature_calib.viz.style import QUAL, apply_style, save


T = 200
SIGMA = 0.05
R = 1.1
THETA_STAR = jnp.array([3.0, 1.2, 0.2, 1.2, -0.2], dtype=jnp.float64)

# Embedded slim Phase-2 medium-difficulty run.
DISTANCE = 0.15
N_PAIRS = 5
N_ITER = 60
M = 64
OPTIMIZERS = ["opg", "adam", "sgd"]
OPT_COLOR = {"opg": QUAL[0], "adam": QUAL[1], "sgd": QUAL[2]}
OPT_LABEL = {"opg": "OPG (LM)", "adam": "Adam", "sgd": "SGD"}


def _sim(theta, key):
    return simulate(theta, key, T=T, sigma=SIGMA, R=R, x_init=0.0,
                    grad_horizon=None)


def _is_safe(theta_np):
    beta, g1, b1, g2, b2 = theta_np
    if beta <= 0 or beta > 50:
        return False
    if g1 > 1.45 or g2 > 1.45 or g1 < 0 or g2 < 0:
        return False
    if abs(b1) > 0.6 or abs(b2) > 0.6:
        return False
    return True


def _finite_loss(theta, ref, key):
    keys = jax.random.split(key, 16)
    X = vmap_simulate(_sim, theta, keys)
    if not bool(jnp.all(jnp.isfinite(X))):
        return False
    return np.isfinite(float(mmd_sq_with_median_bandwidth(X, ref)))


def _sample_theta0(distance, key, ref, max_attempts=60):
    for a in range(max_attempts):
        k = jax.random.fold_in(key, a)
        u = jax.random.normal(k, (5,))
        u = u / jnp.linalg.norm(u)
        cand = THETA_STAR + distance * u
        if _is_safe(np.asarray(cand)) and _finite_loss(cand, ref, k):
            return cand
    raise RuntimeError(f"No safe theta_0 at distance {distance}")


def _run_one(opt_name, theta_0, Y_ref):
    if opt_name == "opg":
        log = calibrate(_sim, theta_0, Y_ref, M=M, n_iter=N_ITER, verbose=False)
    elif opt_name == "adam":
        log = adam(_sim, theta_0, Y_ref, M=M, n_iter=N_ITER, lr=1e-2)
    elif opt_name == "sgd":
        log = sgd(_sim, theta_0, Y_ref, M=M, n_iter=N_ITER, lr=1e-3)
    arrs = log.as_arrays()
    return arrs["thetas"]   # (n_iter+1, P)


def main() -> None:
    apply_style()

    # OPG eigenbasis at theta*.
    print("Computing F_hat(theta*) in float64...")
    M_ref = 128
    ref_keys = jax.random.split(jax.random.PRNGKey(0), M_ref)
    Y_ref = vmap_simulate(_sim, THETA_STAR, ref_keys)
    eig_keys = jax.random.split(jax.random.PRNGKey(1), 96)
    stats = per_seed_loss_and_grads(_sim, THETA_STAR, eig_keys, Y_ref)
    eig = eigendecompose(stats.opg)
    eigvals = np.asarray(eig.eigvals)
    V_star = np.asarray(eig.eigvecs)
    P = eigvals.size
    print(f"  eigenvalues: {eigvals}")
    print(f"  condition:   {eigvals[0]/max(eigvals[-1],1e-30):.2e}")

    # Slim embedded Phase-2 medium runs.
    print(f"\nRunning slim Phase-2 (medium d={DISTANCE}, "
          f"{N_PAIRS} pairs x {len(OPTIMIZERS)} opts)...")
    components = {opt: np.full((N_PAIRS, P), np.nan) for opt in OPTIMIZERS}
    for p in range(N_PAIRS):
        theta_0 = _sample_theta0(DISTANCE,
                                 jax.random.PRNGKey(2000 + p), Y_ref)
        print(f"  pair {p}: theta_0 dist = "
              f"{float(jnp.linalg.norm(theta_0 - THETA_STAR)):.3f}")
        for opt in OPTIMIZERS:
            thetas = _run_one(opt, theta_0, Y_ref)
            if not np.all(np.isfinite(thetas[-1])):
                continue
            delta = np.asarray(thetas[-1]) - np.asarray(THETA_STAR)
            components[opt][p] = V_star.T @ delta
            err = float(np.linalg.norm(delta))
            print(f"    {OPT_LABEL[opt]:9s} final err = {err:.4f}")

    # ============================================================ FIGURE
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # --- Panel (a): recovery error per eigendirection ----------------------
    ax = axes[0]
    width = 0.27
    xs = np.arange(P)
    for i, opt in enumerate(OPTIMIZERS):
        arr = components[opt] ** 2
        med = np.nanmedian(arr, axis=0)
        q25 = np.nanpercentile(arr, 25, axis=0)
        q75 = np.nanpercentile(arr, 75, axis=0)
        ax.bar(xs + (i - 1) * width, np.clip(med, 1e-9, None), width,
               color=OPT_COLOR[opt], edgecolor="white",
               label=OPT_LABEL[opt],
               yerr=[np.clip(med - q25, 0, None),
                     np.clip(q75 - med, 0, None)],
               capsize=2, error_kw=dict(lw=0.8))
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$v_{k+1}$" for k in xs])
    ax.set_xlabel(r"eigendirection of $\hat F(\theta^*)$ (stiff $\to$ sloppy)")
    ax.set_ylabel(r"$(v_k^\top (\theta_T - \theta^*))^2$  (median, IQR)")
    ax.set_title("(a) recovery error per direction",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)

    # Annotation: Adam sloppy error vs OPG stiff error (the cross-optimizer
    # headline — what makes Adam "wander along the sloppy direction").
    opg_med = np.nanmedian(components["opg"] ** 2, axis=0)
    adam_med = np.nanmedian(components["adam"] ** 2, axis=0)
    sgd_med = np.nanmedian(components["sgd"] ** 2, axis=0)
    adam_vs_opg_stiff = adam_med[0] / max(opg_med[0], 1e-30)
    adam_vs_opg_sloppy = adam_med[-1] / max(opg_med[-1], 1e-30)
    adam_sloppy_vs_opg_stiff = adam_med[-1] / max(opg_med[0], 1e-30)
    ax.text(0.98, 0.04,
            "cross-optimiser comparison:\n"
            f"  Adam $v_1$ / OPG $v_1$:    {adam_vs_opg_stiff:6.1e}\n"
            f"  Adam $v_P$ / OPG $v_P$:    {adam_vs_opg_sloppy:6.1e}\n"
            f"  Adam $v_P$ / OPG $v_1$:   {adam_sloppy_vs_opg_stiff:6.1e}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, family="monospace",
            bbox=dict(facecolor="white", edgecolor="grey",
                      alpha=0.85, boxstyle="round,pad=0.3"))

    # --- Panel (b): lambda_k vs error ----------------------------------------
    ax = axes[1]
    for opt in OPTIMIZERS:
        arr = components[opt] ** 2
        med_err = np.nanmedian(arr, axis=0)
        ax.scatter(eigvals, np.clip(med_err, 1e-9, None),
                   s=110, color=OPT_COLOR[opt],
                   edgecolor="white", linewidth=1.2,
                   label=OPT_LABEL[opt], alpha=0.85)
        order = np.argsort(eigvals)
        ax.plot(eigvals[order],
                np.clip(med_err[order], 1e-9, None),
                color=OPT_COLOR[opt], lw=1.0, alpha=0.5)
    for k in range(P):
        ax.annotate(f"$v_{k+1}$", (eigvals[k], 3e-9),
                    fontsize=8, ha="center", color="grey")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda_k$  (eigenvalue of $\hat F(\theta^*)$)")
    ax.set_ylabel(r"median $(v_k^\top \Delta\theta)^2$")
    ax.set_title("(b) small-$\\lambda$ directions = large errors",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    # invert so stiff is on the right (high-lambda) and sloppy on the left
    ax.invert_xaxis()

    fig.suptitle(
        r"The diagnostic predicts where Adam fails: recovery error along $\hat F(\theta^*)$",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    out_dir = "outputs/paper/figures"
    p = save(fig, "fig3_predicts_adam.png", out_dir=out_dir)
    print(f"saved {p}")

    np.savez_compressed(
        f"{out_dir}/fig3_predicts_adam.npz",
        eigvals=eigvals, V_star=V_star,
        opg_components=components["opg"],
        adam_components=components["adam"],
        sgd_components=components["sgd"],
        distance=DISTANCE,
        n_pairs=N_PAIRS,
        n_iter=N_ITER,
    )

    # Console verdict
    print(f"\n=== MEDIUM d={DISTANCE} ratio: error(v_P) / error(v_1) ===")
    for opt in OPTIMIZERS:
        arr = components[opt] ** 2
        med = np.nanmedian(arr, axis=0)
        ratio = med[-1] / max(med[0], 1e-30)
        print(f"  {OPT_LABEL[opt]:9s}: {ratio:.2e}")


if __name__ == "__main__":
    main()
