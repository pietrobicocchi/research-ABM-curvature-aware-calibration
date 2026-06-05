"""Jacobian-based curvature vs OPG matrix — mean-field SIR (Phase 3 Tier A).

Mirrors scripts/13_jacobian_comparison.py (Brock-Hommes) for the SIR model.

Two curvature estimates at theta*:

    J^T J / M   — raw output-sensitivity Hessian-surrogate:
                  average over M seeds of J_m^T J_m where J_m = dX_m/dtheta
                  is the (T, P) Jacobian of the T-dimensional simulation
                  output wrt the P=5 parameters.

    F_hat       — OPG matrix: (1/M) sum_m g_m g_m^T where g_m is the
                  per-seed gradient of MMD^2.  The OPG uses the MMD
                  discrepancy as a "loss" so it captures curvature
                  *relative to the reference distribution*; J^T J captures
                  raw simulation sensitivity irrespective of the reference.

The comparison is diagnostic: we expect them to broadly agree on the
stiff/sloppy hierarchy while differing in magnitude due to the MMD
bandwidth factor embedded in F_hat.

Run: uv run python scripts/27_jacobian_comparison_sir.py
"""

from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from curvature_calib.calibration.diagnostic import eigendecompose, principal_angles
from curvature_calib.calibration.per_seed_grads import per_seed_loss_and_grads, vmap_simulate
from curvature_calib.models.sir import simulate
from curvature_calib.viz.style import QUAL, apply_style, save


T = 200
M_JAC = 64     # Jacobian computation is expensive (full (T, P) matrix per seed)
M_OPG = 128    # OPG can use more seeds cheaply

THETA_STAR = jnp.array([0.40, 0.10, 1e-3, 0.40, 0.50], dtype=jnp.float64)
PARAM_NAMES = [r"$\beta$", r"$\gamma$", r"$I_0$", r"$t_\mathrm{lock}$",
               r"$f_\mathrm{lock}$"]
P = len(THETA_STAR)


def _sim(theta, key):
    return simulate(theta, key, T=T, N=1e5, sigma_obs=10.0, grad_horizon=None)


def main() -> None:
    apply_style()
    out = Path("outputs/sir")
    out.mkdir(parents=True, exist_ok=True)

    # ── Jacobian computation ─────────────────────────────────────────────────
    print(f"Computing Jacobian J = dX/dtheta at THETA_STAR "
          f"(M={M_JAC} seeds, T={T}, P={P})...")
    jac_keys = jax.random.split(jax.random.PRNGKey(42), M_JAC)

    # jac[m] has shape (T, P): partial derivatives of each output time step
    # wrt each parameter.  We use jax.jacobian which returns d(output)/d(input)
    # with output shape leading.
    jac = jax.vmap(
        lambda k: jax.jacobian(lambda t: _sim(t, k))(THETA_STAR)
    )(jac_keys)                    # (M_JAC, T, P)

    jac_np = np.asarray(jac)

    # Curvature matrix: (1/M) sum_m J_m^T J_m  — shape (P, P)
    JtJ = np.einsum("mti,mtj->ij", jac_np, jac_np) / M_JAC
    print(f"  JtJ diagonal: {np.diag(JtJ).round(4)}")

    eig_jac = eigendecompose(jnp.array(JtJ))
    eigvals_jac = np.asarray(eig_jac.eigvals)
    V_jac = np.asarray(eig_jac.eigvecs)
    print(f"  JtJ eigenvalues: {eigvals_jac.round(4)}")
    print(f"  JtJ dynamic range: {eigvals_jac.max()/max(eigvals_jac.min(), 1e-30):.2e}")

    # ── OPG computation ──────────────────────────────────────────────────────
    print(f"\nComputing OPG matrix F_hat at THETA_STAR (M={M_OPG} seeds)...")
    ref_keys = jax.random.split(jax.random.PRNGKey(0), M_OPG)
    Y_ref = vmap_simulate(_sim, THETA_STAR, ref_keys)

    opg_keys = jax.random.split(jax.random.PRNGKey(7), M_OPG)
    stats = per_seed_loss_and_grads(_sim, THETA_STAR, opg_keys, Y_ref)
    F_hat = np.asarray(stats.opg)

    eig_opg = eigendecompose(stats.opg)
    eigvals_opg = np.asarray(eig_opg.eigvals)
    V_opg = np.asarray(eig_opg.eigvecs)
    print(f"  F_hat eigenvalues: {eigvals_opg.round(6)}")
    print(f"  F_hat dynamic range: {eigvals_opg.max()/max(eigvals_opg.min(), 1e-30):.2e}")

    # ── Principal angles between top-k subspaces ─────────────────────────────
    print("\nPrincipal angles between JtJ and F_hat top-k subspaces:")
    max_angles = []
    for k in range(1, P):
        pa = principal_angles(
            eig_jac.eigvecs[:, :k],
            eig_opg.eigvecs[:, :k],
        )
        max_ang_deg = float(np.degrees(np.max(np.asarray(pa))))
        max_angles.append(max_ang_deg)
        print(f"  top-{k}: max principal angle = {max_ang_deg:.1f} deg")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel (a): eigenvalue spectra comparison
    ax = axes[0]
    xs = np.arange(P)
    ax.semilogy(xs, eigvals_jac, "o-", color=QUAL[0], lw=2.0,
                label=r"$J^\top J / M$ (Jacobian)")
    ax.semilogy(xs, eigvals_opg, "s--", color=QUAL[1], lw=2.0,
                label=r"$\hat{F}$ (OPG / MMD gradient)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$v_{k+1}$" for k in xs])
    ax.set_ylabel(r"$\lambda_k$ (log scale)")
    ax.set_title("(a) Eigenvalue spectra: $J^\\top J/M$ vs $\\hat{F}$",
                 fontweight="bold")
    ax.legend(fontsize=9)

    # Panel (b): max principal angles per top-k
    ax = axes[1]
    ks = np.arange(1, P)
    ax.plot(ks, max_angles, "o-", color=QUAL[2], lw=2.0)
    ax.set_xlabel("top-$k$ subspace dimension")
    ax.set_ylabel("max principal angle (degrees)")
    ax.set_ylim(0, 95)
    ax.axhline(45, ls=":", color="grey", lw=1.0, label="45° (partial overlap)")
    ax.axhline(90, ls=":", color="red", lw=1.0, alpha=0.5, label="90° (orthogonal)")
    ax.legend(fontsize=8)
    ax.set_title("(b) Max principal angles between top-$k$ subspaces",
                 fontweight="bold")

    fig.suptitle(
        r"Jacobian-based curvature ($J^\top J/M$) vs OPG matrix ($\hat{F}$)"
        " — mean-field SIR",
        fontweight="bold", fontsize=13,
    )
    fig.tight_layout()

    p = save(fig, "27_jacobian_comparison_sir.png", out_dir=str(out))
    print(f"\nSaved {p}")

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("JACOBIAN vs OPG — SIR VERDICT")
    print("=" * 72)
    print(f"\nEigenvalue ranges:")
    print(f"  J^T J / M : [{eigvals_jac.min():.3e}, {eigvals_jac.max():.3e}]  "
          f"ratio = {eigvals_jac.max()/max(eigvals_jac.min(),1e-30):.2e}")
    print(f"  F_hat     : [{eigvals_opg.min():.3e}, {eigvals_opg.max():.3e}]  "
          f"ratio = {eigvals_opg.max()/max(eigvals_opg.min(),1e-30):.2e}")
    print(f"\nMax principal angle (top-1): {max_angles[0]:.1f} deg")
    if P > 2:
        print(f"Max principal angle (top-{P-1}): {max_angles[-1]:.1f} deg")
    print(f"\nDominant eigenvector v_1 in JtJ vs F_hat:")
    for k in range(P):
        print(f"  {PARAM_NAMES[k]:<16s}  JtJ: {V_jac[k,0]:+.3f}   F_hat: {V_opg[k,0]:+.3f}")

    # ── Persist ──────────────────────────────────────────────────────────────
    np.savez_compressed(
        out / "27_jacobian_comparison_sir.npz",
        JtJ=JtJ,
        F_hat=F_hat,
        eigvals_jac=eigvals_jac,
        V_jac=V_jac,
        eigvals_opg=eigvals_opg,
        V_opg=V_opg,
        max_principal_angles_deg=np.array(max_angles),
        param_names=np.array(PARAM_NAMES),
    )
    print(f"Saved {out / '27_jacobian_comparison_sir.npz'}")


if __name__ == "__main__":
    main()
