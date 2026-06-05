"""Script 24: Gumbel-Sigmoid vs Straight-Through OPG comparison on Network-SIR.

Paper §4.2 'critical additional result': if the two surrogates yield the same
eigenstructure, the diagnostic is robust to surrogate choice.

Three panels:
    A. Eigenvalue spectra overlaid (log scale, both surrogates)
    B. Principal angles between top-k subspaces (k=1..P)
    C. Eigenvector content |V| side by side

Run: uv run python scripts/24_surrogate_comparison.py
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
from curvature_calib.models.network_sir import simulate
from curvature_calib.viz.style import QUAL, apply_style, save

# ── Model config (matches script 18 / fig4 for comparability) ──────────────
T = 200
N = 250
MEAN_DEG = 6.0
THETA_STAR = jnp.array([0.30, 0.10, 0.05, 0.40, 0.50], dtype=jnp.float64)
THETA_EVAL = THETA_STAR + jnp.array([0.005, -0.003, 0.005, 0.02, -0.02],
                                     dtype=jnp.float64)
PARAM_NAMES = [r"$\beta$", r"$\gamma$", r"$I_0$", r"$t_\mathrm{lock}$",
               r"$f_\mathrm{lock}$"]
M_REF = 96
M_EVAL = 128


def make_sim_fn(surrogate: str):
    def sim(theta, key):
        return simulate(theta, key, T=T, N=N, mean_degree=MEAN_DEG,
                        surrogate=surrogate, gumbel_tau=0.5)
    return sim


def compute_eig(surrogate: str):
    sim = make_sim_fn(surrogate)
    ref_keys = jax.random.split(jax.random.PRNGKey(0), M_REF)
    Y_ref = vmap_simulate(sim, THETA_STAR, ref_keys)
    eval_keys = jax.random.split(jax.random.PRNGKey(1), M_EVAL)
    stats = per_seed_loss_and_grads(sim, THETA_EVAL, eval_keys, Y_ref)
    return eigendecompose(stats.opg), np.asarray(stats.per_seed_grads)


def main():
    apply_style()
    print("Computing Gumbel-Sigmoid OPG...")
    eig_g, G_g = compute_eig("gumbel")
    print("Computing Straight-Through OPG...")
    eig_st, G_st = compute_eig("straight_through")

    ev_g  = np.asarray(eig_g.eigvals)
    ev_st = np.asarray(eig_st.eigvals)
    V_g   = np.asarray(eig_g.eigvecs)
    V_st  = np.asarray(eig_st.eigvecs)
    P = len(ev_g)

    # Principal angles between top-k subspaces for k = 1..P-1
    max_angles = np.array([
        float(np.max(np.asarray(principal_angles(
            jnp.asarray(V_g[:, :k]), jnp.asarray(V_st[:, :k])))))
        for k in range(1, P)
    ])

    print(f"Gumbel eigenvalues:  {ev_g}")
    print(f"StraightThru eigvals:{ev_st}")
    print(f"Max principal angles (top-k): {np.degrees(max_angles).round(1)}")

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # A — spectra
    ax = axes[0]
    xs = np.arange(P)
    ax.semilogy(xs, ev_g,  "o-", color=QUAL[0], label="Gumbel-Sigmoid")
    ax.semilogy(xs, ev_st, "s--", color=QUAL[1], label="Straight-Through")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$v_{k+1}$" for k in xs])
    ax.set_ylabel(r"$\lambda_k$")
    ax.set_title("(a) Eigenvalue spectra", fontweight="bold")
    ax.legend()

    # B — principal angles
    ax = axes[1]
    ax.bar(np.arange(1, P), np.degrees(max_angles), color=QUAL[2])
    ax.set_xlabel("top-k subspace")
    ax.set_ylabel("max principal angle (degrees)")
    ax.set_title("(b) Subspace alignment", fontweight="bold")
    ax.axhline(10, color="gray", ls="--", lw=1, label="10°")
    ax.legend(fontsize=8)

    # C — |V| side by side
    ax = axes[2]
    combined = np.hstack([np.abs(V_g), np.abs(V_st)])
    im = ax.imshow(combined, cmap="magma", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(2 * P))
    ax.set_xticklabels(
        [f"G$v_{k+1}$" for k in range(P)] + [f"ST$v_{k+1}$" for k in range(P)],
        fontsize=7,
    )
    ax.set_yticks(np.arange(P))
    ax.set_yticklabels(PARAM_NAMES)
    ax.set_title("(c) Eigenvector content $|V|$", fontweight="bold")
    ax.axvline(P - 0.5, color="white", lw=2)
    plt.colorbar(im, ax=ax, label=r"$|v_{k,j}|$")

    fig.suptitle("Gumbel-Sigmoid vs Straight-Through: Network-SIR OPG",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()

    out_dir = "outputs/sir"
    p = save(fig, "24_surrogate_comparison.png", out_dir=out_dir)
    np.savez_compressed(f"{out_dir}/24_surrogate_comparison.npz",
                        eigvals_gumbel=ev_g, eigvals_st=ev_st,
                        V_gumbel=V_g, V_st=V_st,
                        max_principal_angles=max_angles)
    print(f"Saved {p}")


if __name__ == "__main__":
    main()
