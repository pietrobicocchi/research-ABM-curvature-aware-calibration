"""SIR calibration race — OPG vs Adam vs SGD, mirroring the BH analysis.

Follows scripts/06 (single-init paper-quality), 14 (multi-seed robustness),
and Cell 14 of the BH paper notebook (trajectory in stiff/sloppy plane),
but on the mean-field SIR model from curvature_calib.models.sir.

The diagnostic question: F_hat(theta*) eigenstructure predicted that
    v_1 (stiff)  is dominated by I_0
    v_P (sloppy) is dominated by f_lock + 0.11 t_lock
Therefore the prediction for SIR calibration is:
    - All optimisers should recover the stiff direction (I_0).
    - Adam should wander along the sloppy direction (f_lock).
    - OPG and SGD should not wander, and parameter-recovery distance
      should be much smaller than Adam's.

Outputs: figure outputs/sir/17_sir_calibration_race.png and the npz.

Run: uv run python scripts/17_sir_calibration_race.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from curvature_calib.calibration.baselines import adam, sgd
from curvature_calib.calibration.calibrate import calibrate
from curvature_calib.calibration.opg import eigendecompose
from curvature_calib.calibration.per_seed_grads import (
    per_seed_loss_and_grads,
    vmap_simulate,
)
from curvature_calib.losses.mmd import mmd_sq_with_median_bandwidth
from curvature_calib.models.sir import simulate
from curvature_calib.viz.style import QUAL, apply_style


T = 200
N_POP = 1e5
SIGMA_OBS = 10.0
THETA_STAR = jnp.array([0.40, 0.10, 1e-3, 0.40, 0.50])
PARAM_NAMES = [r"$\beta$", r"$\gamma$", r"$I_0$",
               r"$t_{\mathrm{lock}}$", r"$f_{\mathrm{lock}}$"]

N_ITER = 60
M = 64
N_SEEDS_MULTI = 10
ALPHA_STIFF = 1.5e-3   # picked so that MMD^2(theta_0) is meaningfully above noise floor


def _sim(theta, key):
    return simulate(theta, key, T=T, N=N_POP, sigma_obs=SIGMA_OBS,
                    grad_horizon=None)


def is_safe(theta_np: np.ndarray) -> bool:
    beta, gamma, I0_frac, t_lock_norm, f_lock = theta_np
    if beta <= 0 or beta > 2.0: return False
    if gamma <= 0 or gamma > 1.0: return False
    if I0_frac <= 0 or I0_frac > 0.1: return False
    if t_lock_norm <= 0 or t_lock_norm >= 1: return False
    if f_lock <= 0.05 or f_lock > 1.5: return False
    return True


def main() -> None:
    apply_style()
    out_dir = Path("outputs/sir")
    out_dir.mkdir(exist_ok=True, parents=True)

    # Reference + eigenbasis at theta*.
    print("Reference + eigenbasis at theta*...")
    ref_keys = jax.random.split(jax.random.PRNGKey(0), 128)
    Y_ref = vmap_simulate(_sim, THETA_STAR, ref_keys)

    eig_keys = jax.random.split(jax.random.PRNGKey(1), 96)
    # Eigendecomposition slightly off theta* (mean grad ~ 0 at the truth).
    theta_eval = THETA_STAR + jnp.array([0.01, -0.005, 1e-4, 0.02, -0.02])
    stats_star = per_seed_loss_and_grads(_sim, theta_eval, eig_keys, Y_ref)
    eig = eigendecompose(stats_star.opg)
    eigvals = np.asarray(eig.eigvals)
    V_star = np.asarray(eig.eigvecs)
    v1 = V_star[:, 0]
    vp = V_star[:, -1]
    print(f"  eigenvalues: {eigvals}")
    print(f"  v_1 (stiff) : {v1}")
    print(f"  v_P (sloppy): {vp}")

    # ============================================================ single init
    print("\n=== Single-init race (theta_0 = theta* + alpha * v_1) ===")
    theta0_single = THETA_STAR + ALPHA_STIFF * jnp.asarray(v1)
    print(f"  theta_0 = {np.asarray(theta0_single)}")
    print(f"  ||theta_0 - theta*|| = {float(jnp.linalg.norm(theta0_single - THETA_STAR)):.4f}")

    check_keys = jax.random.split(jax.random.PRNGKey(99), 64)
    X0 = vmap_simulate(_sim, theta0_single, check_keys)
    init_mmd = float(mmd_sq_with_median_bandwidth(X0, Y_ref))
    print(f"  init MMD^2 = {init_mmd:+.4e}")

    print("  OPG (LM) ...")
    opg_log = calibrate(_sim, theta0_single, Y_ref, M=M, n_iter=N_ITER,
                        init_damping=100.0, verbose=False)
    opg_a = opg_log.as_arrays()
    err_opg = np.linalg.norm(opg_a["thetas"] - np.asarray(THETA_STAR), axis=1)

    print("  Adam (lr=1e-2) ...")
    adam_log = adam(_sim, theta0_single, Y_ref, M=M, n_iter=N_ITER, lr=1e-2)
    adam_a = adam_log.as_arrays()
    err_adam = np.linalg.norm(adam_a["thetas"] - np.asarray(THETA_STAR), axis=1)

    print("  SGD (lr=1e-3) ...")
    sgd_log = sgd(_sim, theta0_single, Y_ref, M=M, n_iter=N_ITER, lr=1e-3)
    sgd_a = sgd_log.as_arrays()
    err_sgd = np.linalg.norm(sgd_a["thetas"] - np.asarray(THETA_STAR), axis=1)

    print(f"\n  Final ||theta_T - theta*||:")
    print(f"    OPG  : {err_opg[-1]:.4f}")
    print(f"    Adam : {err_adam[-1]:.4f}")
    print(f"    SGD  : {err_sgd[-1]:.4f}")

    # Project trajectories onto V_star eigenbasis.
    def project_traj(thetas):
        delta = thetas - np.asarray(THETA_STAR)
        return delta @ V_star
    traj_opg = project_traj(opg_a["thetas"])
    traj_adam = project_traj(adam_a["thetas"])
    traj_sgd = project_traj(sgd_a["thetas"])

    # Final-iterate per-direction components.
    comp = {
        "OPG":  V_star.T @ (opg_a["thetas"][-1] - np.asarray(THETA_STAR)),
        "Adam": V_star.T @ (adam_a["thetas"][-1] - np.asarray(THETA_STAR)),
        "SGD":  V_star.T @ (sgd_a["thetas"][-1] - np.asarray(THETA_STAR)),
    }

    # ============================================================ multi-seed
    print(f"\n=== Multi-seed race ({N_SEEDS_MULTI} random inits at distance {ALPHA_STIFF}) ===")
    # Sample N random unit vectors u, set theta_0 = theta* + alpha * V_star @ u.
    # This ensures each init has the same distance in "rescaled eigenvector"
    # space and tests robustness to direction choice.
    multi_results = []
    key_master = jax.random.PRNGKey(2026)
    attempt = 0
    while len(multi_results) < N_SEEDS_MULTI and attempt < 80:
        k = jax.random.fold_in(key_master, attempt)
        u = jax.random.normal(k, (5,))
        u = u / jnp.linalg.norm(u)
        # Map into parameter space via eigenbasis.
        delta = jnp.asarray(V_star) @ u * ALPHA_STIFF
        theta0_i = THETA_STAR + delta
        if not is_safe(np.asarray(theta0_i)):
            attempt += 1
            continue
        # Calibrate.
        opg_l = calibrate(_sim, theta0_i, Y_ref, M=M, n_iter=N_ITER,
                          init_damping=100.0, seed_base=int(attempt), verbose=False)
        opg_arr = opg_l.as_arrays()
        ad_l = adam(_sim, theta0_i, Y_ref, M=M, n_iter=N_ITER, lr=1e-2,
                    seed_base=int(attempt))
        ad_arr = ad_l.as_arrays()
        sg_l = sgd(_sim, theta0_i, Y_ref, M=M, n_iter=N_ITER, lr=1e-3,
                   seed_base=int(attempt))
        sg_arr = sg_l.as_arrays()
        multi_results.append({
            "theta_0": np.asarray(theta0_i),
            "opg_err":  np.linalg.norm(opg_arr["thetas"] - np.asarray(THETA_STAR), axis=1),
            "adam_err": np.linalg.norm(ad_arr["thetas"] - np.asarray(THETA_STAR), axis=1),
            "sgd_err":  np.linalg.norm(sg_arr["thetas"] - np.asarray(THETA_STAR), axis=1),
            "opg_comp":  V_star.T @ (opg_arr["thetas"][-1] - np.asarray(THETA_STAR)),
            "adam_comp": V_star.T @ (ad_arr["thetas"][-1] - np.asarray(THETA_STAR)),
            "sgd_comp":  V_star.T @ (sg_arr["thetas"][-1] - np.asarray(THETA_STAR)),
        })
        print(f"  seed {len(multi_results)-1:2d}: "
              f"OPG {multi_results[-1]['opg_err'][-1]:.4f}  "
              f"Adam {multi_results[-1]['adam_err'][-1]:.4f}  "
              f"SGD {multi_results[-1]['sgd_err'][-1]:.4f}")
        attempt += 1

    # Stack for stats.
    err_opg_m  = np.stack([r["opg_err"]  for r in multi_results])
    err_adam_m = np.stack([r["adam_err"] for r in multi_results])
    err_sgd_m  = np.stack([r["sgd_err"]  for r in multi_results])

    # ============================================================ FIGURE
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35)

    opt_colors = {"OPG": QUAL[0], "Adam": QUAL[1], "SGD": QUAL[2]}

    # A. Best-so-far val loss (single init)
    ax = fig.add_subplot(gs[0, 0])
    runs = [("OPG", opg_a, opt_colors["OPG"]),
            ("Adam", adam_a, opt_colors["Adam"]),
            ("SGD", sgd_a, opt_colors["SGD"])]
    for label, arr, color in runs:
        best = np.minimum.accumulate(arr["val_losses"])
        ax.semilogy(np.clip(best, 1e-2, None), color=color, lw=2.2, label=label)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"best-so-far val MMD$^2$ (clipped at noise floor)")
    ax.set_title(rf"A. Loss trajectory, single init at $\theta^* + \alpha v_1$ ($\alpha={ALPHA_STIFF}$)")
    ax.legend(fontsize=10)

    # B. Parameter recovery (single init)
    ax = fig.add_subplot(gs[0, 1])
    d0_single = float(jnp.linalg.norm(theta0_single - THETA_STAR))
    ax.plot(err_opg,  color=opt_colors["OPG"],  lw=2.2, label="OPG")
    ax.plot(err_adam, color=opt_colors["Adam"], lw=2.2, label="Adam")
    ax.plot(err_sgd,  color=opt_colors["SGD"],  lw=2.2, label="SGD")
    ax.axhline(d0_single, color="grey", lw=1, ls="--",
               label=fr"$d_0={d0_single:.4f}$")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\|\theta_t - \theta^*\|_2$")
    ax.set_title(r"B. Parameter recovery (single init)")
    ax.legend(fontsize=10)

    # C. Trajectory in (v_1, v_P) plane — the smoking-gun panel
    ax = fig.add_subplot(gs[0, 2])
    for label, t, color in [("OPG",  traj_opg,  opt_colors["OPG"]),
                            ("Adam", traj_adam, opt_colors["Adam"]),
                            ("SGD",  traj_sgd,  opt_colors["SGD"])]:
        ax.plot(t[:, 0], t[:, -1], "o-", color=color, lw=1.2,
                markersize=3, label=label, alpha=0.85)
        ax.scatter(t[0, 0], t[0, -1], color=color, s=110,
                   marker="o", edgecolor="black", linewidth=1.4, zorder=10)
        ax.scatter(t[-1, 0], t[-1, -1], color=color, s=180,
                   marker="*", edgecolor="black", linewidth=1.0, zorder=10)
    ax.scatter([0], [0], s=350, marker="*", color="#f1c40f",
               edgecolor="black", linewidth=1.4, label=r"$\theta^*$", zorder=15)
    ax.set_xlabel(rf"projection on $v_1$ (stiff, $\lambda$={eigvals[0]:.1e})")
    ax.set_ylabel(rf"projection on $v_P$ (sloppy, $\lambda$={eigvals[-1]:.1e})")
    ax.set_title(r"C. Trajectory in (stiff, sloppy) plane")
    ax.legend(fontsize=9)
    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.axvline(0, color="grey", lw=0.5, ls=":")

    # D. Per-eigendirection error at final iterate (single init)
    ax = fig.add_subplot(gs[1, 0])
    xs = np.arange(5)
    width = 0.27
    for i, (label, color) in enumerate([("OPG",  opt_colors["OPG"]),
                                         ("Adam", opt_colors["Adam"]),
                                         ("SGD",  opt_colors["SGD"])]):
        sq = comp[label] ** 2
        ax.bar(xs + (i - 1) * width, np.clip(sq, 1e-12, None), width,
               color=color, edgecolor="white", label=label)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$v_{k+1}$" + "\n" + rf"$\lambda$={eigvals[k]:.1e}"
                        for k in xs], fontsize=8)
    ax.set_xlabel("eigendirection (stiff $\\to$ sloppy)")
    ax.set_ylabel(r"$(v_k^\top (\theta_T - \theta^*))^2$")
    ax.set_title("D. Recovery error per eigendirection (single init)")
    ax.legend(fontsize=9)

    # E. Multi-seed param recovery (median + IQR)
    ax = fig.add_subplot(gs[1, 1])
    its = np.arange(err_opg_m.shape[1])
    for label, mat, color in [("OPG",  err_opg_m,  opt_colors["OPG"]),
                              ("Adam", err_adam_m, opt_colors["Adam"]),
                              ("SGD",  err_sgd_m,  opt_colors["SGD"])]:
        med = np.median(mat, axis=0)
        q25 = np.percentile(mat, 25, axis=0)
        q75 = np.percentile(mat, 75, axis=0)
        ax.plot(its, med, color=color, lw=2.2, label=label)
        ax.fill_between(its, q25, q75, color=color, alpha=0.18)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\|\theta_t - \theta^*\|_2$  (median + IQR)")
    ax.set_title(f"E. Multi-seed recovery ({N_SEEDS_MULTI} random inits)")
    ax.legend(fontsize=10)

    # F. Multi-seed final-error boxplot
    ax = fig.add_subplot(gs[1, 2])
    data = [err_opg_m[:, -1], err_adam_m[:, -1], err_sgd_m[:, -1]]
    bp = ax.boxplot(data, tick_labels=["OPG", "Adam", "SGD"],
                    patch_artist=True, widths=0.55, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="black",
                                   markeredgecolor="white", markersize=6))
    for patch, opt in zip(bp["boxes"], ["OPG", "Adam", "SGD"]):
        patch.set_facecolor(opt_colors[opt])
        patch.set_alpha(0.7)
    ax.set_ylabel(r"final $\|\theta_T - \theta^*\|_2$")
    ax.set_title(f"F. Multi-seed final error ({N_SEEDS_MULTI} runs / opt)")
    ax.set_yscale("log")

    fig.suptitle(
        rf"SIR calibration race — Phase 3 Tier A optimizer comparison",
        fontsize=14, fontweight="bold", y=0.995,
    )
    p = out_dir / "17_sir_calibration_race.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {p}")

    # ============================================================ verdict
    print("\n" + "=" * 70)
    print("SIR CALIBRATION VERDICT")
    print("=" * 70)

    print(f"\nSingle init (theta_0 = theta* + {ALPHA_STIFF} * v_1, d_0 = {d0_single:.4f}):")
    for opt, c in comp.items():
        e = {"OPG": err_opg, "Adam": err_adam, "SGD": err_sgd}[opt][-1]
        stiff = c[0] ** 2
        sloppy = c[-1] ** 2
        print(f"  {opt:<4s}: err_end={e:.4f}  stiff_err^2={stiff:.2e}  "
              f"sloppy_err^2={sloppy:.2e}  ratio={sloppy/max(stiff, 1e-30):.2e}")

    print(f"\nMulti-seed final ||theta_T - theta*|| (N={N_SEEDS_MULTI}, d_0=~{ALPHA_STIFF}):")
    for opt, mat in [("OPG", err_opg_m), ("Adam", err_adam_m), ("SGD", err_sgd_m)]:
        final = mat[:, -1]
        print(f"  {opt:<4s}: median = {np.median(final):.4f}  "
              f"IQR = [{np.percentile(final, 25):.4f}, "
              f"{np.percentile(final, 75):.4f}]  max = {np.max(final):.4f}")

    print(f"\nPairwise wins:")
    for a_name, a_mat in [("OPG", err_opg_m), ("Adam", err_adam_m), ("SGD", err_sgd_m)]:
        for b_name, b_mat in [("OPG", err_opg_m), ("Adam", err_adam_m), ("SGD", err_sgd_m)]:
            if a_name == b_name:
                continue
            wins = int(np.sum(a_mat[:, -1] < b_mat[:, -1]))
            print(f"  {a_name} < {b_name} on {wins}/{N_SEEDS_MULTI} seeds")

    np.savez(out_dir / "17_sir_calibration_race.npz",
             theta_star=np.asarray(THETA_STAR), V_star=V_star, eigvals=eigvals,
             theta0_single=np.asarray(theta0_single),
             traj_opg=traj_opg, traj_adam=traj_adam, traj_sgd=traj_sgd,
             err_opg=err_opg, err_adam=err_adam, err_sgd=err_sgd,
             multi_err_opg=err_opg_m, multi_err_adam=err_adam_m,
             multi_err_sgd=err_sgd_m)
    print(f"saved {out_dir / '17_sir_calibration_race.npz'}")


if __name__ == "__main__":
    main()
