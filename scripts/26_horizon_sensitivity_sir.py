"""Script 26: Gradient-horizon sensitivity for mean-field SIR (Appendix B).

Mirrors script 09 (Brock-Hommes) for the SIR model. At a fixed reference
theta, computes F_hat with gradient truncated to horizons H in
{5, 10, 20, 50, T_full=200}. Compares eigenvalue spectra and principal
angles between top-k subspaces at each H vs H=T_full.

Run: uv run python scripts/26_horizon_sensitivity_sir.py
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from curvature_calib.calibration.diagnostic import eigendecompose, principal_angles
from curvature_calib.calibration.per_seed_grads import (
    per_seed_loss_and_grads,
    vmap_simulate,
)
from curvature_calib.models.sir import simulate
from curvature_calib.viz.style import QUAL, apply_style, save

T = 200
M_REF = 96
M_EVAL = 128
HORIZONS = [5, 10, 20, 50, T]

THETA_STAR = jnp.array([0.40, 0.10, 1e-3, 0.40, 0.50], dtype=jnp.float64)
THETA_EVAL = THETA_STAR + jnp.array([0.01, -0.005, 1e-4, 0.02, -0.02],
                                     dtype=jnp.float64)
PARAM_NAMES = [r"$\beta$", r"$\gamma$", r"$I_0$", r"$t_\mathrm{lock}$",
               r"$f_\mathrm{lock}$"]


def make_sim(H):
    def sim(theta, key):
        return simulate(theta, key, T=T, N=1e5, sigma_obs=10.0,
                        grad_horizon=(None if H >= T else H))
    return sim


def main():
    apply_style()

    ref_keys = jax.random.split(jax.random.PRNGKey(0), M_REF)
    Y_ref = vmap_simulate(make_sim(T), THETA_STAR, ref_keys)
    eval_keys = jax.random.split(jax.random.PRNGKey(1), M_EVAL)

    eig_by_H = {}
    for H in HORIZONS:
        sim_fn = make_sim(H)
        stats = per_seed_loss_and_grads(sim_fn, THETA_EVAL, eval_keys, Y_ref)
        eig_by_H[H] = eigendecompose(stats.opg)
        ev = np.asarray(eig_by_H[H].eigvals)
        print(f"  H={H:4d}  eigvals={ev.round(4)}")

    eig_full = eig_by_H[T]
    P = len(np.asarray(eig_full.eigvals))

    # Max principal angle top-k vs full for each H < T
    angles = {}
    for H in HORIZONS[:-1]:
        angles[H] = [
            float(np.max(np.asarray(principal_angles(
                eig_full.eigvecs[:, :k], eig_by_H[H].eigvecs[:, :k]))))
            for k in range(1, P)
        ]

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for i, H in enumerate(HORIZONS):
        ev = np.asarray(eig_by_H[H].eigvals)
        style = "-" if H == T else "--"
        lw = 2.0 if H == T else 1.2
        ax.semilogy(np.arange(P), ev, style, color=QUAL[i % len(QUAL)],
                    lw=lw, label=f"H={H}" if H < T else f"H={T} (full)")
    ax.set_xticks(np.arange(P))
    ax.set_xticklabels([f"$v_{k+1}$" for k in range(P)])
    ax.set_ylabel(r"$\lambda_k$")
    ax.set_title("(a) Eigenvalue spectra across horizons", fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[1]
    for i, H in enumerate(HORIZONS[:-1]):
        ax.plot(np.arange(1, P), np.degrees(angles[H]),
                "o-", color=QUAL[i], label=f"H={H}")
    ax.set_xlabel("top-k subspace")
    ax.set_ylabel("max principal angle vs full-horizon (degrees)")
    ax.set_title("(b) Subspace drift vs full-horizon", fontweight="bold")
    ax.legend(fontsize=8)

    fig.suptitle("Gradient-horizon sensitivity — mean-field SIR",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()

    out_dir = "outputs/sir"
    p = save(fig, "26_horizon_sensitivity_sir.png", out_dir=out_dir)
    np.savez_compressed(f"{out_dir}/26_horizon_sensitivity_sir.npz",
                        horizons=np.array(HORIZONS),
                        **{f"eigvals_H{H}": np.asarray(eig_by_H[H].eigvals)
                           for H in HORIZONS})
    print(f"Saved {p}")


if __name__ == "__main__":
    main()
