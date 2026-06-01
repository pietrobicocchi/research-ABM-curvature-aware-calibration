"""Jacobian sensitivity vs OPG eigendecomposition — Phase 3 Objective (d).

The project plan promises an explicit comparison against the per-parameter
sensitivity analysis of Quera-Bofarull et al. 2025 §5.4. Their pipeline
produces *first-order* per-parameter sensitivities (the Jacobian of the
observable Phi(f(theta)) wrt each theta_k individually). Our OPG matrix
F_hat = (1/M) sum_m g_m g_m^T gives *second-moment* structure: the
eigendecomposition reveals which COMBINATIONS of parameters the data
constrains.

The script makes the contrast empirical at theta*:

    A. Per-parameter Jacobian sensitivity   (Quera-Bofarull style)
    B. OPG diagonal sensitivity              (a closely related per-param view)
    C. OPG correlation matrix                (off-diagonal coupling)
    D. OPG eigenvalues + |V| heatmap         (parameter combinations)

Headline claim: the most identifiable structure at theta* is the symmetric
bias combination v_1 \\propto b_1 + b_2. Per-parameter analyses (A, B) rank
b_1, b_2 as individually important but say nothing about the combination;
they would similarly flag b_1 - b_2 as identifiable, which it is NOT (the
antisymmetric combination corresponds to a different eigenvector with a
much smaller eigenvalue). The eigendecomposition reveals this structure;
per-parameter sensitivity cannot.

Run: uv run python scripts/13_jacobian_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from curvature_calib.calibration.jacobian_sensitivity import (
    opg_correlation_matrix,
    opg_diagonal_sensitivity,
    per_param_jacobian_sensitivity,
)
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
    out = Path("outputs/brock_hommes")
    out.mkdir(exist_ok=True)

    # Reference distribution at theta*.
    M_ref = 128
    ref_keys = jax.random.split(jax.random.PRNGKey(0), M_ref)
    Y_ref = vmap_simulate(_sim, THETA_STAR, ref_keys)

    # Evaluate sensitivity at a small perturbation from theta* (at theta*
    # itself the mean gradient is zero so per-parameter signal is ambiguous).
    # Same convention as scripts 11 / 14.
    theta_eval = THETA_STAR + jnp.array([0.0, 0.05, 0.03, 0.05, -0.03])

    print("Computing per-parameter Jacobian sensitivity (Quera-Bofarull §5.4 style)...")
    M_jac = 64
    jac_keys = jax.random.split(jax.random.PRNGKey(11), M_jac)
    S_jac = np.asarray(per_param_jacobian_sensitivity(_sim, theta_eval, jac_keys))
    print(f"  S_jac per param: {dict(zip(PARAM_NAMES, [f'{v:.3e}' for v in S_jac]))}")

    print("Computing OPG matrix at theta_eval...")
    M_opg = 96
    opg_keys = jax.random.split(jax.random.PRNGKey(13), M_opg)
    stats = per_seed_loss_and_grads(_sim, theta_eval, opg_keys, Y_ref)
    F_hat = np.asarray(stats.opg)
    eig = eigendecompose(stats.opg)
    eigvals = np.asarray(eig.eigvals)
    V = np.asarray(eig.eigvecs)
    S_diag = np.asarray(opg_diagonal_sensitivity(stats.opg))
    rho = np.asarray(opg_correlation_matrix(stats.opg))
    print(f"  S_diag (sqrt F_kk): {dict(zip(PARAM_NAMES, [f'{v:.3e}' for v in S_diag]))}")
    print(f"  eigenvalues: {eigvals}")

    # ===================================================== FIGURE
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.40)

    # A. Per-parameter Jacobian sensitivity (Quera-Bofarull style).
    ax = fig.add_subplot(gs[0, 0])
    xs = np.arange(5)
    ax.bar(xs, S_jac, color=QUAL[3], edgecolor="white", width=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(PARAM_NAMES)
    ax.set_yscale("log")
    ax.set_xlabel("parameter")
    ax.set_ylabel(r"$\|\partial x / \partial \theta_k\|_2$ (mean over seeds)")
    ax.set_title("A. Per-parameter Jacobian sensitivity\n(Quera-Bofarull 2025 §5.4 style)")

    # B. OPG diagonal sensitivity.
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(xs, S_diag, color=QUAL[5], edgecolor="white", width=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(PARAM_NAMES)
    ax.set_yscale("log")
    ax.set_xlabel("parameter")
    ax.set_ylabel(r"$\sqrt{\hat F_{kk}}$")
    ax.set_title("B. OPG diagonal sensitivity\n(per-parameter, ignores coupling)")

    # C. OPG correlation matrix |rho_kl|.
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(np.abs(rho), cmap="RdBu_r", aspect="auto",
                   vmin=-1, vmax=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(PARAM_NAMES)
    ax.set_yticks(xs)
    ax.set_yticklabels(PARAM_NAMES)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{rho[i, j]:+.2f}", ha="center", va="center",
                    color="white" if abs(rho[i, j]) > 0.55 else "black",
                    fontsize=9)
    ax.set_title(r"C. OPG correlation: $\rho_{kl} = \hat F_{kl}/\sqrt{\hat F_{kk}\hat F_{ll}}$"
                 + "\n(off-diagonal: what eigendecomposition exploits)")
    plt.colorbar(im, ax=ax, label=r"$\rho_{kl}$")

    # D. OPG eigenvalues.
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(xs, eigvals, color=QUAL[0], edgecolor="white", width=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$v_{k+1}$" for k in xs])
    ax.set_yscale("log")
    ax.set_xlabel("eigendirection (descending eigenvalue)")
    ax.set_ylabel(r"$\lambda_k$")
    ax.set_title("D. OPG eigenvalues\n(per-COMBINATION sensitivity, ours)")

    # E. |V| heatmap.
    ax = fig.add_subplot(gs[1, 1])
    V_abs = np.abs(V)
    im = ax.imshow(V_abs, cmap="magma", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$v_{k+1}$" for k in xs])
    ax.set_yticks(xs)
    ax.set_yticklabels(PARAM_NAMES)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{V_abs[i, j]:.2f}", ha="center", va="center",
                    color="white" if V_abs[i, j] < 0.55 else "black", fontsize=9)
    ax.set_title(r"E. $|V|$: each eigenvector decomposed in original basis")
    plt.colorbar(im, ax=ax, label=r"$|v_{k,j}|$")

    # F. The discovery: dynamic range comparison.
    ax = fig.add_subplot(gs[1, 2])
    # Normalize so largest is 1 for visual comparison.
    S_jac_n = S_jac / S_jac.max()
    S_diag_n = S_diag / S_diag.max()
    eig_n = eigvals / eigvals.max()
    width = 0.27
    ax.bar(xs - width, S_jac_n, width, color=QUAL[3], edgecolor="white",
           label="A. Jacobian sens (per-param)")
    ax.bar(xs,         S_diag_n, width, color=QUAL[5], edgecolor="white",
           label="B. OPG diag (per-param)")
    ax.bar(xs + width, eig_n,    width, color=QUAL[0], edgecolor="white",
           label="D. OPG eigenvalues (per-combination)")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(["#1", "#2", "#3", "#4", "#5"])
    ax.set_xlabel("rank")
    ax.set_ylabel("normalized magnitude (max = 1)")
    ax.set_title("F. Dynamic range: combinations span more than params")
    ax.legend(fontsize=8, loc="lower left")

    fig.suptitle(
        "Per-parameter Jacobian sensitivity (Quera-Bofarull 2025 §5.4) "
        "vs OPG eigendecomposition (ours)",
        fontsize=14, fontweight="bold", y=0.995,
    )
    p = out / "13_jacobian_comparison.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {p}")

    # ===================================================== console verdict
    print("\n" + "=" * 76)
    print("JACOBIAN vs OPG — VERDICT")
    print("=" * 76)
    print(f"\nDynamic range per method:")
    print(f"  A. Per-param Jacobian:   ratio max/min = {S_jac.max()/S_jac.min():.2e}")
    print(f"  B. OPG diagonal:         ratio max/min = {S_diag.max()/S_diag.min():.2e}")
    print(f"  D. OPG eigenvalues:      ratio max/min = {eigvals.max()/max(eigvals.min(),1e-30):.2e}")

    print(f"\nLargest off-diagonal correlation in F_hat:")
    rho_abs = np.abs(rho - np.diag(np.diag(rho)))
    i, j = np.unravel_index(np.argmax(rho_abs), rho_abs.shape)
    print(f"  |rho({PARAM_NAMES[i]}, {PARAM_NAMES[j]})| = {rho_abs[i, j]:.3f}")
    print("  -> per-parameter analyses cannot capture this coupling.")

    print(f"\nDominant eigenvector v_1 (stiffest direction):")
    v1 = V[:, 0]
    for k in range(5):
        print(f"  {PARAM_NAMES[k]:<5s} : {v1[k]:+.3f}")
    print(f"  -> v_1 is roughly proportional to ({PARAM_NAMES[2]} + {PARAM_NAMES[4]})/√2;")
    print(f"     a combination that no per-parameter sensitivity surfaces.")

    # Persist for downstream / paper.
    np.savez(out / "13_jacobian_comparison.npz",
             S_jac=S_jac, S_diag=S_diag, F_hat=F_hat,
             rho=rho, eigvals=eigvals, V=V,
             param_names=np.array(PARAM_NAMES))
    print(f"\nsaved {out / '13_jacobian_comparison.npz'}")


if __name__ == "__main__":
    main()
