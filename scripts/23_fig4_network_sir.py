"""Script 23: paper Figure 4 — the network-SIR closer.

Paper-polished version of script 18 (the network-SIR diagnostic). Four
panels in a 2x2 grid:
    (a) Daily new infections under theta* vs the no-lockdown counterfactual
    (b) OPG spectrum + 95% bootstrap CIs (network-SIR, Gumbel-Sigmoid)
    (c) |V|: parameter content of each eigendirection
    (d) Network-SIR spectrum vs the mean-field SIR baseline (the
        "generalisation" reading: same identifiability structure under
        biased surrogate gradients)

Falsification ratios are NOT shown here — they live in Figure 2 (script 20)
where they appear alongside Brock-Hommes and mean-field SIR.

Polish vs script 18:
    * float64 throughout
    * dropped panel D (falsification) -- redundant with Fig 2
    * paper-style suptitle
    * span / condition annotation in panel (b)
    * writes to outputs/paper/figures/fig4_network_sir.png

Run: uv run python scripts/23_fig4_network_sir.py
"""

from __future__ import annotations

from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from curvature_calib.calibration.opg import bootstrap_eigvals, eigendecompose
from curvature_calib.calibration.per_seed_grads import (
    per_seed_loss_and_grads, vmap_simulate,
)
from curvature_calib.models.network_sir import simulate as net_simulate
from curvature_calib.models.sir import simulate as mf_simulate
from curvature_calib.viz.style import QUAL, apply_style, save


T = 200
N_NODES = 250
MEAN_DEG = 6.0
GUMBEL_TAU = 0.5
THETA_STAR = jnp.array([0.30, 0.10, 0.05, 0.40, 0.50], dtype=jnp.float64)
THETA_EVAL = THETA_STAR + jnp.array([0.005, -0.003, 0.005, 0.02, -0.02],
                                    dtype=jnp.float64)
PARAM_NAMES = [r"$\beta$", r"$\gamma$", r"$I_0$",
               r"$t_{\mathrm{lock}}$", r"$f_{\mathrm{lock}}$"]

MF_THETA_STAR = jnp.array([0.40, 0.10, 1e-3, 0.40, 0.50], dtype=jnp.float64)
MF_THETA_EVAL = MF_THETA_STAR + jnp.array([0.01, -0.005, 1e-4, 0.02, -0.02],
                                          dtype=jnp.float64)


def _net_sim(theta, key):
    return net_simulate(theta, key, T=T, N=N_NODES, mean_degree=MEAN_DEG,
                        gumbel_tau=GUMBEL_TAU, grad_horizon=None)


def _mf_sim(theta, key):
    return mf_simulate(theta, key, T=T, N=1e5, sigma_obs=10.0,
                       grad_horizon=None)


def main() -> None:
    apply_style()
    print(f"Network-SIR: N={N_NODES}, mean_degree={MEAN_DEG}, "
          f"gumbel_tau={GUMBEL_TAU}, T={T}")

    # === Network-SIR spectrum and eigenvectors ===
    M_ref = 96
    ref_keys = jax.random.split(jax.random.PRNGKey(0), M_ref)
    Y_ref = vmap_simulate(_net_sim, THETA_STAR, ref_keys)
    eig_keys = jax.random.split(jax.random.PRNGKey(1), 96)
    stats = per_seed_loss_and_grads(_net_sim, THETA_EVAL, eig_keys, Y_ref)
    eig = eigendecompose(stats.opg)
    eigvals = np.asarray(eig.eigvals)
    V = np.asarray(eig.eigvecs)
    print(f"  net-SIR eigvals: {eigvals}")
    print(f"  net-SIR condition: {eigvals[0]/max(eigvals[-1],1e-30):.2e}")

    boot = np.asarray(bootstrap_eigvals(stats.per_seed_grads, n_boot=300,
                                        key=jax.random.PRNGKey(7)))
    boot_lo = np.percentile(boot, 2.5, axis=0)
    boot_hi = np.percentile(boot, 97.5, axis=0)

    # === Sample trajectories ===
    M_show = 32
    show_keys = jax.random.split(jax.random.PRNGKey(11), M_show)
    X_star = np.asarray(vmap_simulate(_net_sim, THETA_STAR, show_keys))
    theta_no_lock = THETA_STAR.at[4].set(1.0)
    X_no_lock = np.asarray(vmap_simulate(_net_sim, theta_no_lock, show_keys))

    # === Mean-field SIR spectrum (for comparison) ===
    mf_ref_keys = jax.random.split(jax.random.PRNGKey(100), 96)
    Y_mf = vmap_simulate(_mf_sim, MF_THETA_STAR, mf_ref_keys)
    mf_eig_keys = jax.random.split(jax.random.PRNGKey(101), 96)
    mf_stats = per_seed_loss_and_grads(_mf_sim, MF_THETA_EVAL, mf_eig_keys, Y_mf)
    mf_eig = eigendecompose(mf_stats.opg)
    mf_eigvals = np.asarray(mf_eig.eigvals)
    mf_V = np.asarray(mf_eig.eigvecs)
    print(f"  mean-field eigvals: {mf_eigvals}")

    # ============================================================ FIGURE
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) trajectories
    ax = axes[0, 0]
    for m in range(min(M_show, 12)):
        ax.plot(X_star[m], color=QUAL[0], lw=0.6, alpha=0.4)
        ax.plot(X_no_lock[m], color=QUAL[1], lw=0.6, alpha=0.4)
    ax.plot(X_star.mean(0), color=QUAL[0], lw=2.4,
            label=r"$\theta^*$ (with lockdown)")
    ax.plot(X_no_lock.mean(0), color=QUAL[1], lw=2.4,
            label=r"no lockdown ($f_{\mathrm{lock}}=1$)")
    ax.set_xlabel("day")
    ax.set_ylabel("daily new infections")
    ax.set_title("(a) network-SIR trajectories", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")

    # (b) OPG spectrum
    ax = axes[0, 1]
    P = eigvals.size
    xs = np.arange(P)
    le = np.clip(eigvals - boot_lo, a_min=0.0, a_max=None)
    ue = np.clip(boot_hi - eigvals, a_min=0.0, a_max=None)
    ax.errorbar(xs, eigvals, yerr=[le, ue], fmt="o",
                color=QUAL[0], capsize=4, markersize=12,
                markerfacecolor=QUAL[0], markeredgecolor="white",
                markeredgewidth=1.6, lw=1.8)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$v_{k+1}$" for k in xs])
    ax.set_xlabel("eigendirection")
    ax.set_ylabel(r"$\lambda_k$")
    span_oom = np.log10(eigvals[0] / max(eigvals[-1], 1e-30))
    ax.set_title("(b) OPG spectrum + 95% bootstrap CI",
                 fontsize=11, fontweight="bold")
    ax.text(0.04, 0.04,
            f"span: {span_oom:.1f} OOM\n"
            rf"$\lambda_1/\lambda_P$: {eigvals[0]/max(eigvals[-1],1e-30):.1e}",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(facecolor="white", edgecolor="grey",
                      alpha=0.85, boxstyle="round,pad=0.3"))

    # (c) |V| heatmap
    ax = axes[1, 0]
    V_abs = np.abs(V)
    im = ax.imshow(V_abs, cmap="magma", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$v_{k+1}$" for k in xs])
    ax.set_yticks(np.arange(P))
    ax.set_yticklabels(PARAM_NAMES)
    for i in range(P):
        for j in range(P):
            ax.text(j, i, f"{V_abs[i, j]:.2f}", ha="center", va="center",
                    color="white" if V_abs[i, j] < 0.55 else "black", fontsize=9)
    ax.set_title("(c) parameter content $|V|$",
                 fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label=r"$|v_{k,j}|$")

    # (d) mean-field vs network-SIR spectrum
    ax = axes[1, 1]
    ax.semilogy(xs, mf_eigvals, "o-", color=QUAL[3], markersize=10, lw=1.8,
                markerfacecolor=QUAL[3], markeredgecolor="white",
                markeredgewidth=1.2, label="mean-field SIR")
    ax.semilogy(xs, eigvals, "s-", color=QUAL[0], markersize=10, lw=1.8,
                markerfacecolor=QUAL[0], markeredgecolor="white",
                markeredgewidth=1.2, label="network-SIR (Gumbel)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$v_{k+1}$" for k in xs])
    ax.set_xlabel("eigendirection")
    ax.set_ylabel(r"$\lambda_k$")
    ax.set_title("(d) mean-field vs network-SIR spectra",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")

    fig.suptitle(
        r"Network-SIR diagnostic with Gumbel-Sigmoid surrogate gradients",
        fontsize=13, fontweight="bold", y=1.00,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir = "outputs/paper/figures"
    p = save(fig, "fig4_network_sir.png", out_dir=out_dir)
    print(f"saved {p}")

    np.savez_compressed(
        f"{out_dir}/fig4_network_sir.npz",
        eigvals=eigvals, V=V, boot=boot,
        mf_eigvals=mf_eigvals, mf_V=mf_V,
    )

    # console summary
    print(f"\nStiff direction v_1 content:")
    for k in range(P):
        print(f"  {PARAM_NAMES[k]:<10s}: {V[k, 0]:+.3f}")
    print(f"Sloppy direction v_P content:")
    for k in range(P):
        print(f"  {PARAM_NAMES[k]:<10s}: {V[k, -1]:+.3f}")


if __name__ == "__main__":
    main()
