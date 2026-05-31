"""The "diagnostic predicts itself" visualization.

Decomposes each Phase-2 optimizer's parameter-recovery error
    delta_T = theta_T - theta*
along the OPG eigenbasis computed at theta* (the ground-truth basis).

Hypothesis:
    The OPG spectrum at theta* predicts which directions the MMD loss
    constrains. Therefore the recovery error projected onto eigendirection k
    should be:
        - SMALL when lambda_k is large (data sees this combination clearly)
        - LARGE when lambda_k is small (data does not see this combination)

If true, this plot makes the diagnostic empirically self-validating: the
sloppy directions identified at theta* are *exactly* the directions in
which optimizers fail to recover the truth.

Run: uv run python scripts/11_stiff_sloppy_decomposition.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from curvature_calib.calibration.opg import eigendecompose
from curvature_calib.calibration.per_seed_grads import (
    per_seed_loss_and_grads,
    vmap_simulate,
)
from curvature_calib.models.brock_hommes import simulate
from curvature_calib.viz.style import QUAL, apply_style


T = 200
SIGMA = 0.05
R = 1.1
THETA_STAR = jnp.array([3.0, 1.2, 0.2, 1.2, -0.2])
PARAM_NAMES = [r"$\beta$", r"$g_1$", r"$b_1$", r"$g_2$", r"$b_2$"]


def _sim(theta, key):
    return simulate(theta, key, T=T, sigma=SIGMA, R=R, x_init=0.0,
                    grad_horizon=None)


def main() -> None:
    apply_style()
    out = Path("outputs")

    # Load Phase 2 raw results.
    npz_path = out / "10_phase2_convergence.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Run scripts/10_phase2_convergence.py first to produce {npz_path}.")
    raw = np.load(npz_path)

    # Recompute the OPG eigenbasis at theta*. This is the "ground-truth"
    # identifiability geometry against which we judge each optimizer.
    print("Computing F_hat at theta* ...")
    M_ref = 128
    ref_keys = jax.random.split(jax.random.PRNGKey(0), M_ref)
    Y_ref = vmap_simulate(_sim, THETA_STAR, ref_keys)
    keys_eig = jax.random.split(jax.random.PRNGKey(1), 96)
    stats_star = per_seed_loss_and_grads(_sim, THETA_STAR, keys_eig, Y_ref)
    eig = eigendecompose(stats_star.opg)
    eigvals = np.asarray(eig.eigvals)           # (P,) descending
    V_star = np.asarray(eig.eigvecs)            # (P, P)
    P = eigvals.size
    print(f"  eigenvalues: {eigvals}")
    print(f"  condition  : {eigvals[0] / max(eigvals[-1], 1e-30):.2e}")

    # For each (difficulty, pair, optimizer), compute the eigenbasis
    # components of the recovery error.
    difficulties = ["easy", "medium", "hard"]
    optimizers = ["opg", "adam", "sgd"]
    opt_colors = {"opg": QUAL[0], "adam": QUAL[1], "sgd": QUAL[2]}
    opt_labels = {"opg": "OPG (LM)", "adam": "Adam", "sgd": "SGD"}
    N_PAIRS = 5

    # components[(difficulty, opt)] = ndarray (N_PAIRS, P)
    components = {}
    for d in difficulties:
        for opt in optimizers:
            arr = np.zeros((N_PAIRS, P))
            for p in range(N_PAIRS):
                key = f"{d}_pair{p}_{opt}_thetas"
                theta_traj = raw[key]               # (n_iter+1, P)
                if not np.all(np.isfinite(theta_traj[-1])):
                    arr[p] = np.nan
                    continue
                delta = theta_traj[-1] - np.asarray(THETA_STAR)
                arr[p] = V_star.T @ delta            # eigenbasis components
            components[(d, opt)] = arr

    # ============================================================ FIGURE
    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35,
                          height_ratios=[1.0, 1.0, 1.0])

    # Rows: difficulty. Cols: bar chart per eigendirection.
    for row, d in enumerate(difficulties):
        # A. Squared components per eigendirection (median + IQR across pairs).
        ax = fig.add_subplot(gs[row, 0])
        width = 0.27
        xs = np.arange(P)
        for i, opt in enumerate(optimizers):
            arr = components[(d, opt)] ** 2          # (N_PAIRS, P)
            # Mask NaN (Adam in some hard runs)
            med = np.nanmedian(arr, axis=0)
            q25 = np.nanpercentile(arr, 25, axis=0)
            q75 = np.nanpercentile(arr, 75, axis=0)
            ax.bar(xs + (i - 1) * width, np.clip(med, 1e-9, None), width,
                   color=opt_colors[opt], edgecolor="white",
                   label=opt_labels[opt] if row == 0 else None,
                   yerr=[np.clip(med - q25, 0, None),
                         np.clip(q75 - med, 0, None)],
                   capsize=2, error_kw=dict(lw=0.8))
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"$v_{k+1}$" for k in xs])
        ax.set_xlabel(r"eigendirection of $\hat F(\theta^*)$ (stiff $\to$ sloppy)")
        ax.set_ylabel(r"$(v_k^\top (\theta_T - \theta^*))^2$")
        ax.set_title(f"{d.capitalize()}: recovery error per eigendirection")
        if row == 0:
            ax.legend(fontsize=8)

        # B. The diagnostic scatter: lambda_k (x) vs median error (y).
        ax = fig.add_subplot(gs[row, 1])
        for opt in optimizers:
            arr = components[(d, opt)] ** 2
            med_err = np.nanmedian(arr, axis=0)
            ax.scatter(eigvals, np.clip(med_err, 1e-9, None),
                       s=110, color=opt_colors[opt],
                       edgecolor="white", linewidth=1.2,
                       label=opt_labels[opt], alpha=0.85)
            # Connect with a line ordered by lambda for clarity.
            order = np.argsort(eigvals)
            ax.plot(eigvals[order],
                    np.clip(med_err[order], 1e-9, None),
                    color=opt_colors[opt], lw=1.0, alpha=0.5)
        # Annotate eigendirections.
        for k in range(P):
            ax.annotate(f"$v_{k+1}$", (eigvals[k], 1e-8),
                        fontsize=7, ha="center", color="grey")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\lambda_k$  (eigenvalue of $\hat F(\theta^*)$)")
        ax.set_ylabel(r"median $(v_k^\top \Delta\theta)^2$")
        ax.set_title(f"{d.capitalize()}: small-lambda directions = large errors")
        if row == 0:
            ax.legend(fontsize=8)

        # C. Ratio: error / lambda (a dimensional check).
        # Within a sloppy basin, error in direction k should scale as 1/sqrt(lambda_k).
        ax = fig.add_subplot(gs[row, 2])
        for opt in optimizers:
            arr = components[(d, opt)] ** 2
            med_err = np.nanmedian(arr, axis=0)
            # Predicted scaling: err_k ~ noise_floor / lambda_k (heuristic).
            # Plot err_k * lambda_k -- should be approximately constant if the
            # diagnostic captures the right scaling.
            ratio = med_err * eigvals
            ax.plot(np.arange(P), np.clip(ratio, 1e-12, None), "o-",
                    color=opt_colors[opt], lw=1.5, markersize=8,
                    markerfacecolor=opt_colors[opt], markeredgecolor="white",
                    markeredgewidth=1.1,
                    label=opt_labels[opt])
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"$v_{k+1}$" for k in xs])
        ax.set_xlabel("eigendirection (stiff $\\to$ sloppy)")
        ax.set_ylabel(r"$\lambda_k \cdot (v_k^\top \Delta\theta)^2$")
        ax.set_title(f"{d.capitalize()}: signal energy in each direction")
        if row == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        "The diagnostic predicting itself — recovery error decomposed along "
        r"the OPG eigenbasis of $\hat F(\theta^*)$",
        fontsize=14, fontweight="bold", y=0.995,
    )

    p = out / "11_stiff_sloppy_decomposition.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {p}")

    # ===================================================== console verdict
    print("\n" + "=" * 76)
    print("STIFF-SLOPPY DECOMPOSITION VERDICT")
    print("=" * 76)
    print(f"\nEigenvalues at theta*:")
    for k in range(P):
        print(f"  lambda_{k+1} = {eigvals[k]:.3e}")

    print("\nMedian |v_k^T (theta_T - theta*)|^2 by difficulty / optimizer / direction:")
    for d in difficulties:
        print(f"\n  {d.upper()}:")
        header = "    direction  " + "  ".join(f"{opt_labels[opt]:>12s}"
                                                for opt in optimizers)
        print(header)
        for k in range(P):
            row_str = f"    v_{k+1} (lam={eigvals[k]:.1e}) "
            for opt in optimizers:
                arr = components[(d, opt)] ** 2
                med_err = float(np.nanmedian(arr[:, k]))
                row_str += f"  {med_err:>10.3e}  "
            print(row_str)

    # Key claim: ratio of stiff-direction error to sloppy-direction error.
    print("\n\nKEY CLAIM: error_sloppy / error_stiff should be LARGE,")
    print("           reflecting that data sees stiff but not sloppy.")
    for d in difficulties:
        print(f"\n  {d}:")
        for opt in optimizers:
            arr = components[(d, opt)] ** 2
            med_err = np.nanmedian(arr, axis=0)
            stiff_err = med_err[0]
            sloppy_err = med_err[-1]
            ratio = sloppy_err / max(stiff_err, 1e-30)
            print(f"    {opt_labels[opt]:<10s}: stiff={stiff_err:.2e}  "
                  f"sloppy={sloppy_err:.2e}  ratio={ratio:.1e}")


if __name__ == "__main__":
    main()
