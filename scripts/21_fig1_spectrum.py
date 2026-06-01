"""Script 21: paper Figure 1 — OPG geometry of the Brock-Hommes model.

Paper-polished version of script 05. Same three-panel layout:
    A. Per-seed gradient cloud projected onto the top-2 OPG eigenaxes
       (with the 1-sigma OPG ellipse overlaid)
    B. Eigenvalue spectrum (log scale) with 95% bootstrap CIs
    C. Eigenvector content heatmap |V| (which parameter combinations the
       data constrains)

Polish vs script 05:
    * float64 (avoids float32 noise at the bottom of a ~10^7-condition spectrum)
    * paper-style suptitle
    * simpler legend, no double sigma contours
    * writes to outputs/paper/figures/fig1_spectrum.png

Run: uv run python scripts/21_fig1_spectrum.py
"""

from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from curvature_calib.calibration.opg import bootstrap_eigvals, eigendecompose
from curvature_calib.calibration.per_seed_grads import (
    per_seed_loss_and_grads, vmap_simulate,
)
from curvature_calib.models.brock_hommes import simulate
from curvature_calib.viz.style import QUAL, apply_style, save


T = 200
SIGMA = 0.05
R = 1.1
THETA_STAR = jnp.array([3.0, 1.2, 0.2, 1.2, -0.2], dtype=jnp.float64)
THETA_EVAL = THETA_STAR + jnp.array([0.0, 0.05, 0.03, 0.05, -0.03],
                                    dtype=jnp.float64)
PARAM_NAMES = [r"$\beta$", r"$g_1$", r"$b_1$", r"$g_2$", r"$b_2$"]


def _sim(theta, key):
    return simulate(theta, key, T=T, sigma=SIGMA, R=R, x_init=0.0)


def confidence_ellipse(F, ax, n_std=1.0, **kwargs):
    w, V = np.linalg.eigh(F)
    order = np.argsort(-w)
    w, V = w[order], V[:, order]
    angle = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(w, 0))
    e = mpatches.Ellipse(xy=(0, 0), width=width, height=height, angle=angle,
                         fill=False, **kwargs)
    ax.add_patch(e)


def main() -> None:
    apply_style()

    M_ref = 128
    M_eval = 200
    ref_keys = jax.random.split(jax.random.PRNGKey(0), M_ref)
    Y_ref = vmap_simulate(_sim, THETA_STAR, ref_keys)
    keys = jax.random.split(jax.random.PRNGKey(11), M_eval)
    stats = per_seed_loss_and_grads(_sim, THETA_EVAL, keys, Y_ref)

    G = np.asarray(stats.per_seed_grads)
    F = np.asarray(stats.opg)
    eig = eigendecompose(jnp.asarray(F))
    eigvals = np.asarray(eig.eigvals)
    V = np.asarray(eig.eigvecs)

    print(f"eigenvalues: {eigvals}")
    print(f"condition  : {eigvals[0] / max(eigvals[-1], 1e-30):.2e}")

    g_centered = G - G.mean(axis=0)
    proj = g_centered @ V[:, :2]
    F_2d = V[:, :2].T @ F @ V[:, :2]

    boot = np.asarray(bootstrap_eigvals(stats.per_seed_grads, n_boot=500,
                                        key=jax.random.PRNGKey(2)))

    # ============================================================ FIGURE
    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.2], wspace=0.32)

    # A — gradient cloud.
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(proj[:, 0], proj[:, 1], s=14, alpha=0.55,
               color=QUAL[0], edgecolor="white", linewidth=0.3,
               label=fr"per-seed gradients ($M={M_eval}$)")
    ax.scatter([0], [0], marker="x", color="black", s=80, lw=2,
               label="mean (centred)")
    confidence_ellipse(F_2d, ax, n_std=1.0,
                       edgecolor=QUAL[1], lw=2,
                       label=r"$1\sigma$ OPG ellipse")
    ax.set_xlabel(r"$v_1^\top g_m$ (stiff direction)")
    ax.set_ylabel(r"$v_2^\top g_m$")
    ax.set_title("(a) per-seed gradient cloud", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")

    # B — spectrum.
    ax2 = fig.add_subplot(gs[0, 1])
    P = len(eigvals)
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    xs = np.arange(P)
    lower_err = np.clip(eigvals - lo, a_min=0.0, a_max=None)
    upper_err = np.clip(hi - eigvals, a_min=0.0, a_max=None)
    ax2.errorbar(xs, eigvals, yerr=[lower_err, upper_err],
                 fmt="o", color=QUAL[0], markersize=10, capsize=4,
                 markerfacecolor=QUAL[0], markeredgecolor="white",
                 markeredgewidth=1.3)
    ax2.set_yscale("log")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"$v_{k+1}$" for k in xs])
    ax2.set_xlabel("eigenvector (descending)")
    ax2.set_ylabel(r"$\lambda_k$")
    ax2.set_title("(b) OPG spectrum + 95% bootstrap CI",
                  fontsize=11, fontweight="bold")
    # Headline annotation: condition number / span.
    span_oom = np.log10(eigvals[0] / max(eigvals[-1], 1e-30))
    ax2.text(0.04, 0.04,
             f"span: {span_oom:.1f} OOM\n"
             rf"$\lambda_1/\lambda_P$: {eigvals[0]/max(eigvals[-1],1e-30):.1e}",
             transform=ax2.transAxes, fontsize=9, va="bottom",
             bbox=dict(facecolor="white", edgecolor="grey",
                       alpha=0.85, boxstyle="round,pad=0.3"))

    # C — eigenvector content.
    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(np.abs(V), cmap="magma", aspect="auto", vmin=0, vmax=1)
    ax3.set_xticks(xs)
    ax3.set_xticklabels([f"$v_{k+1}$" for k in xs])
    ax3.set_yticks(np.arange(P))
    ax3.set_yticklabels(PARAM_NAMES)
    ax3.set_title("(c) parameter content $|V|$", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax3, label=r"$|v_{k,j}|$")
    for i in range(P):
        for j in range(P):
            val = np.abs(V[i, j])
            ax3.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color="white" if val < 0.55 else "black", fontsize=8)

    fig.suptitle(
        r"OPG geometry of the Brock-Hommes calibration loss",
        fontsize=13, fontweight="bold", y=1.04,
    )

    out_dir = "outputs/paper/figures"
    p = save(fig, "fig1_spectrum.png", out_dir=out_dir)
    print(f"saved {p}")

    np.savez_compressed(
        f"{out_dir}/fig1_spectrum.npz",
        eigvals=eigvals, V=V, boot=boot,
        theta_star=np.asarray(THETA_STAR), theta_eval=np.asarray(THETA_EVAL),
    )


if __name__ == "__main__":
    main()
