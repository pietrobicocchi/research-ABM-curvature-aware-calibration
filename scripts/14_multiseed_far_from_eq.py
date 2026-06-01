"""Multi-seed far-from-equilibrium calibration race.

Addresses reviewer concern #1 from the figure critique: the headline
single-init result is one realisation of a noisy process. This script
repeats it across N_SEEDS random unit-vector initialisations at fixed
distance d_0 = 0.14, reports median +/- IQR per optimizer.

Each (seed, optimizer) is one full calibration with the now-standard
infrastructure:
    - val_losses on a fixed held-out seed set
    - best-so-far display
    - init_damping=100 for OPG

Run: uv run python scripts/14_multiseed_far_from_eq.py
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
from curvature_calib.models.brock_hommes import simulate
from curvature_calib.viz.style import QUAL, apply_style


T = 200
SIGMA = 0.05
R = 1.1
THETA_STAR = jnp.array([3.0, 1.2, 0.2, 1.2, -0.2])
PARAM_NAMES = [r"$\beta$", r"$g_1$", r"$b_1$", r"$g_2$", r"$b_2$"]

D0 = 0.14
N_SEEDS = 15
N_ITER = 60
M = 64


def _sim(theta, key):
    return simulate(theta, key, T=T, sigma=SIGMA, R=R, x_init=0.0,
                    grad_horizon=None)


def is_safe(theta_np: np.ndarray) -> bool:
    beta, g1, b1, g2, b2 = theta_np
    if beta <= 0 or beta > 50: return False
    if g1 > 1.4 or g2 > 1.4: return False
    if g1 < 0 or g2 < 0: return False
    if abs(b1) > 0.6 or abs(b2) > 0.6: return False
    return True


def sample_safe_inits(theta_star: jax.Array, distance: float,
                      n_inits: int, key: jax.Array,
                      Y_ref: jax.Array, max_attempts: int = 200):
    """Random unit-vector inits on sphere of radius `distance`, with safety filtering."""
    inits = []
    attempts = 0
    while len(inits) < n_inits and attempts < max_attempts:
        k = jax.random.fold_in(key, attempts)
        u = jax.random.normal(k, (5,))
        u = u / jnp.linalg.norm(u)
        cand = theta_star + distance * u
        if not is_safe(np.asarray(cand)):
            attempts += 1
            continue
        # Also require finite loss.
        check_keys = jax.random.split(k, 16)
        X = vmap_simulate(_sim, cand, check_keys)
        v = float(mmd_sq_with_median_bandwidth(X, Y_ref))
        if not np.isfinite(v):
            attempts += 1
            continue
        inits.append((np.asarray(cand), v))
        attempts += 1
    if len(inits) < n_inits:
        raise RuntimeError(f"Only found {len(inits)} safe inits in {max_attempts} attempts.")
    return inits


def run_three(theta0: jax.Array, Y_ref: jax.Array, seed_idx: int) -> dict:
    out = {}
    log_opg = calibrate(_sim, theta0, Y_ref, M=M, n_iter=N_ITER,
                        init_damping=100.0, seed_base=seed_idx, verbose=False)
    a = log_opg.as_arrays()
    err = np.linalg.norm(a["thetas"] - np.asarray(THETA_STAR), axis=1)
    out["opg"] = {"val": np.asarray(a["val_losses"]),
                  "err": err,
                  "theta_T": np.asarray(a["thetas"][-1]),
                  "opg_T": np.asarray(a["opgs"][-1])}

    log_adam = adam(_sim, theta0, Y_ref, M=M, n_iter=N_ITER, lr=1e-2,
                    seed_base=seed_idx)
    a = log_adam.as_arrays()
    err = np.linalg.norm(a["thetas"] - np.asarray(THETA_STAR), axis=1)
    out["adam"] = {"val": np.asarray(a["val_losses"]),
                   "err": err,
                   "theta_T": np.asarray(a["thetas"][-1])}

    log_sgd = sgd(_sim, theta0, Y_ref, M=M, n_iter=N_ITER, lr=1e-3,
                  seed_base=seed_idx)
    a = log_sgd.as_arrays()
    err = np.linalg.norm(a["thetas"] - np.asarray(THETA_STAR), axis=1)
    out["sgd"] = {"val": np.asarray(a["val_losses"]),
                  "err": err,
                  "theta_T": np.asarray(a["thetas"][-1])}
    return out


def main() -> None:
    apply_style()
    out_dir = Path("outputs/brock_hommes")
    out_dir.mkdir(exist_ok=True)

    # Reference.
    print(f"Building reference at theta* (M_ref=128)...")
    ref_keys = jax.random.split(jax.random.PRNGKey(0), 128)
    Y_ref = vmap_simulate(_sim, THETA_STAR, ref_keys)

    # OPG eigenbasis at theta* (for per-direction decomposition).
    print("Computing F_hat(theta*) eigenbasis...")
    eig_keys = jax.random.split(jax.random.PRNGKey(1), 96)
    stats_star = per_seed_loss_and_grads(_sim, THETA_STAR, eig_keys, Y_ref)
    eig = eigendecompose(stats_star.opg)
    V_star = np.asarray(eig.eigvecs)
    eigvals = np.asarray(eig.eigvals)
    print(f"  eigenvalues: {eigvals}")

    # Sample initialisations.
    print(f"Sampling {N_SEEDS} safe initialisations at d_0={D0}...")
    inits = sample_safe_inits(THETA_STAR, D0, N_SEEDS,
                              jax.random.PRNGKey(2026), Y_ref)
    print(f"  initial MMD^2 values: {[f'{mmd:+.2e}' for _, mmd in inits[:5]]}...")

    # Run.
    print(f"Running {N_SEEDS} x 3 optimizers x {N_ITER} iter ...")
    results = []
    for i, (theta0_np, init_mmd) in enumerate(inits):
        theta0 = jnp.asarray(theta0_np)
        r = run_three(theta0, Y_ref, seed_idx=i)
        r["theta_0"] = theta0_np
        r["init_mmd"] = init_mmd
        results.append(r)
        print(f"  seed {i:2d}: init_mmd={init_mmd:+.2e}  "
              f"OPG err {r['opg']['err'][-1]:.3f}  "
              f"Adam err {r['adam']['err'][-1]:.3f}  "
              f"SGD err {r['sgd']['err'][-1]:.3f}")

    # ============================================================ aggregate
    def stack(field, opt):
        return np.stack([np.asarray(r[opt][field]) for r in results])

    val_opg  = stack("val", "opg");   err_opg  = stack("err", "opg")
    val_adam = stack("val", "adam");  err_adam = stack("err", "adam")
    val_sgd  = stack("val", "sgd");   err_sgd  = stack("err", "sgd")

    # Best-so-far per seed.
    best_opg  = np.minimum.accumulate(val_opg,  axis=1)
    best_adam = np.minimum.accumulate(val_adam, axis=1)
    best_sgd  = np.minimum.accumulate(val_sgd,  axis=1)

    # Per-direction decomposition of final theta.
    def decomp(opt):
        deltas = np.stack([results[i][opt]["theta_T"] - np.asarray(THETA_STAR)
                            for i in range(N_SEEDS)])
        return deltas @ V_star  # (N_SEEDS, P) components

    comp_opg  = decomp("opg")
    comp_adam = decomp("adam")
    comp_sgd  = decomp("sgd")

    # ============================================================ figure
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

    colors = {"opg": QUAL[0], "adam": QUAL[1], "sgd": QUAL[2]}
    labels = {"opg": "OPG (LM)", "adam": "Adam", "sgd": "SGD"}

    # A. Loss (best-so-far) median + IQR
    ax = fig.add_subplot(gs[0, 0])
    its = np.arange(best_opg.shape[1])
    for opt, mat in [("opg", best_opg), ("adam", best_adam), ("sgd", best_sgd)]:
        med = np.median(mat, axis=0)
        q25 = np.percentile(mat, 25, axis=0)
        q75 = np.percentile(mat, 75, axis=0)
        ax.semilogy(its, np.clip(med, 1e-5, None), color=colors[opt],
                    lw=2.2, label=labels[opt])
        ax.fill_between(its, np.clip(q25, 1e-5, None), np.clip(q75, 1e-5, None),
                        color=colors[opt], alpha=0.18)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"best-so-far val MMD$^2$ (clipped at $10^{-5}$)")
    ax.set_title(f"A. Loss: median + IQR across {N_SEEDS} random inits")
    ax.legend(fontsize=10)

    # B. Parameter recovery median + IQR
    ax = fig.add_subplot(gs[0, 1])
    its_e = np.arange(err_opg.shape[1])
    for opt, mat in [("opg", err_opg), ("adam", err_adam), ("sgd", err_sgd)]:
        med = np.median(mat, axis=0)
        q25 = np.percentile(mat, 25, axis=0)
        q75 = np.percentile(mat, 75, axis=0)
        ax.plot(its_e, med, color=colors[opt], lw=2.2, label=labels[opt])
        ax.fill_between(its_e, q25, q75, color=colors[opt], alpha=0.18)
    ax.axhline(D0, color="grey", ls="--", lw=1, label=fr"$d_0={D0}$")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\|\theta_t - \theta^*\|_2$ (median + IQR)")
    ax.set_title("B. Parameter recovery across seeds")
    ax.legend(fontsize=10)

    # C. Final-error distribution (boxplot)
    ax = fig.add_subplot(gs[0, 2])
    data = [err_opg[:, -1], err_adam[:, -1], err_sgd[:, -1]]
    bp = ax.boxplot(data, tick_labels=[labels[o] for o in ["opg", "adam", "sgd"]],
                    patch_artist=True, widths=0.55, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="black",
                                   markeredgecolor="white", markersize=6))
    for patch, opt in zip(bp["boxes"], ["opg", "adam", "sgd"]):
        patch.set_facecolor(colors[opt])
        patch.set_alpha(0.7)
    ax.set_ylabel(r"final $\|\theta_T - \theta^*\|_2$")
    ax.set_title(f"C. Final error ({N_SEEDS} runs / opt)")
    ax.set_yscale("log")

    # D. Stiff-direction error across seeds
    ax = fig.add_subplot(gs[1, 0])
    for opt, comp in [("opg", comp_opg), ("adam", comp_adam), ("sgd", comp_sgd)]:
        stiff = np.abs(comp[:, 0])
        ax.scatter(np.full_like(stiff, ["opg", "adam", "sgd"].index(opt), dtype=float),
                   stiff, s=80, color=colors[opt], edgecolor="white",
                   linewidth=1.2, alpha=0.85)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([labels[o] for o in ["opg", "adam", "sgd"]])
    ax.set_yscale("log")
    ax.set_ylabel(r"$|v_1^\top (\theta_T - \theta^*)|$  (stiff error)")
    ax.set_title("D. Stiff-direction error per seed")

    # E. Sloppy-direction error across seeds
    ax = fig.add_subplot(gs[1, 1])
    for opt, comp in [("opg", comp_opg), ("adam", comp_adam), ("sgd", comp_sgd)]:
        sloppy = np.abs(comp[:, -1])
        ax.scatter(np.full_like(sloppy, ["opg", "adam", "sgd"].index(opt), dtype=float),
                   sloppy, s=80, color=colors[opt], edgecolor="white",
                   linewidth=1.2, alpha=0.85)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([labels[o] for o in ["opg", "adam", "sgd"]])
    ax.set_yscale("log")
    ax.set_ylabel(r"$|v_P^\top (\theta_T - \theta^*)|$  (sloppy error)")
    ax.set_title("E. Sloppy-direction error per seed")

    # F. Summary table-like panel: median final err per opt + stiff/sloppy split.
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    rows = []
    for opt in ["opg", "adam", "sgd"]:
        comp = {"opg": comp_opg, "adam": comp_adam, "sgd": comp_sgd}[opt]
        final = {"opg": err_opg, "adam": err_adam, "sgd": err_sgd}[opt][:, -1]
        rows.append([
            labels[opt],
            f"{np.median(final):.3f}",
            f"[{np.percentile(final, 25):.3f}, {np.percentile(final, 75):.3f}]",
            f"{np.median(np.abs(comp[:, 0])):.2e}",
            f"{np.median(np.abs(comp[:, -1])):.2e}",
        ])
    table = ax.table(cellText=rows,
                     colLabels=["optimizer", "median err",
                                "IQR err",
                                r"med $|v_1^\top\Delta|$",
                                r"med $|v_P^\top\Delta|$"],
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    ax.set_title(f"F. Summary table ({N_SEEDS} seeds)")

    fig.suptitle(
        f"Multi-seed far-from-equilibrium calibration ($N$={N_SEEDS}, "
        rf"$d_0$={D0}, random direction)",
        fontsize=14, fontweight="bold", y=0.995,
    )
    p = out_dir / "14_multiseed_far_from_eq.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {p}")

    # ============================================================ console
    print("\n" + "=" * 70)
    print(f"MULTI-SEED SUMMARY (N={N_SEEDS}, d_0={D0})")
    print("=" * 70)
    for opt in ["opg", "adam", "sgd"]:
        mat = {"opg": err_opg, "adam": err_adam, "sgd": err_sgd}[opt]
        final = mat[:, -1]
        print(f"\n  {labels[opt]:<10s}: median err = {np.median(final):.4f}  "
              f"IQR = [{np.percentile(final, 25):.4f}, "
              f"{np.percentile(final, 75):.4f}]  "
              f"max = {np.max(final):.4f}")
    print()
    # Pairwise win counts.
    for a_name, a_mat in [("OPG", err_opg), ("Adam", err_adam), ("SGD", err_sgd)]:
        for b_name, b_mat in [("OPG", err_opg), ("Adam", err_adam), ("SGD", err_sgd)]:
            if a_name == b_name:
                continue
            wins = int(np.sum(a_mat[:, -1] < b_mat[:, -1]))
            print(f"  {a_name} < {b_name} on  {wins}/{N_SEEDS} seeds")

    # Persist raw.
    np.savez(out_dir / "14_multiseed_far_from_eq.npz",
             N_SEEDS=N_SEEDS, D0=D0, N_ITER=N_ITER,
             V_star=V_star, eigvals=eigvals,
             val_opg=val_opg, val_adam=val_adam, val_sgd=val_sgd,
             err_opg=err_opg, err_adam=err_adam, err_sgd=err_sgd,
             comp_opg=comp_opg, comp_adam=comp_adam, comp_sgd=comp_sgd)
    print(f"\nsaved {out_dir / '14_multiseed_far_from_eq.npz'}")


if __name__ == "__main__":
    main()
