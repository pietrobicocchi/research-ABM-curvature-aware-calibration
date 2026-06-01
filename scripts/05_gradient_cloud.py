"""The cloud of per-seed gradients and what F_hat says about it.

For a single theta we visualise the M per-seed gradient vectors {g_m} and the
ellipse defined by the OPG matrix F_hat = (1/M) sum_m g_m g_m^T:
    - left:   gradients projected onto their top-2 PCA axes; the OPG ellipse
              is the 1-sigma contour of the second-moment matrix.
    - middle: the full eigenvalue spectrum (log scale) with bootstrap CIs.
    - right:  the eigenvector heatmap (parameter content of each direction).

This is the diagnostic in its raw form: the elongation of the cloud is what
licenses calling some directions stiff and others sloppy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from curvature_calib.calibration.opg import bootstrap_eigvals, eigendecompose
from curvature_calib.calibration.per_seed_grads import (
    per_seed_loss_and_grads,
    vmap_simulate,
)
from curvature_calib.models.brock_hommes import simulate
from curvature_calib.viz.style import QUAL, apply_style, save


T = 200
SIGMA = 0.05
R = 1.1
THETA_STAR = jnp.array([3.0, 1.2, 0.2, 1.2, -0.2])
PARAM_NAMES = [r"$\beta$", r"$g_1$", r"$b_1$", r"$g_2$", r"$b_2$"]


def _sim(theta, key):
    return simulate(theta, key, T=T, sigma=SIGMA, R=R, x_init=0.0)


def confidence_ellipse(F, ax, n_std=1.0, **kwargs):
    """Plot the n_std covariance ellipse of a 2x2 PSD matrix F at the origin."""
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

    # Reference and evaluation samples at a point just off truth (so MMD > 0
    # and there is non-trivial gradient signal).
    M_ref = 128
    M_eval = 200
    ref_keys = jax.random.split(jax.random.PRNGKey(0), M_ref)
    Y_ref = vmap_simulate(_sim, THETA_STAR, ref_keys)

    theta0 = THETA_STAR + jnp.array([0.0, 0.05, 0.03, 0.05, -0.03])
    keys = jax.random.split(jax.random.PRNGKey(11), M_eval)
    stats = per_seed_loss_and_grads(_sim, theta0, keys, Y_ref)

    G = np.asarray(stats.per_seed_grads)        # (M, P)
    F = np.asarray(stats.opg)                    # (P, P)
    eig = eigendecompose(jnp.asarray(F))
    eigvals = np.asarray(eig.eigvals)
    V = np.asarray(eig.eigvecs)

    # PCA = OPG eigendecomposition (since cov of g is ~ F when mean ~ 0,
    # which holds far from optimum but slightly biased otherwise -- still
    # the right visual basis).
    g_centered = G - G.mean(axis=0)
    # Project onto top-2 eigenvectors of F for visualisation.
    proj = g_centered @ V[:, :2]                 # (M, 2)
    F_2d = V[:, :2].T @ F @ V[:, :2]             # 2x2 projected OPG

    # Bootstrap eigenvalues.
    boot = np.asarray(bootstrap_eigvals(stats.per_seed_grads, n_boot=300,
                                        key=jax.random.PRNGKey(2)))

    fig = plt.figure(figsize=(16, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.2], wspace=0.35)

    # Left: gradient cloud in top-2 OPG directions.
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(proj[:, 0], proj[:, 1], s=14, alpha=0.55,
               color=QUAL[0], edgecolor="white", linewidth=0.3,
               label=fr"per-seed gradients ($M={M_eval}$)")
    # Mean gradient (should be at origin after centering).
    ax.scatter([0], [0], marker="x", color="black", s=80, lw=2,
               label="mean (after centering)")
    confidence_ellipse(F_2d, ax, n_std=1.0,
                       edgecolor=QUAL[1], lw=2, label=r"$1\sigma$ OPG ellipse")
    confidence_ellipse(F_2d, ax, n_std=2.0,
                       edgecolor=QUAL[1], lw=1, linestyle="--", alpha=0.6)
    ax.set_xlabel(r"$v_1^\top g_m$  (stiffest direction)")
    ax.set_ylabel(r"$v_2^\top g_m$")
    ax.set_title("Gradient cloud in top-2 OPG axes")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")

    # Middle: eigenvalue spectrum with bootstrap CIs.
    ax2 = fig.add_subplot(gs[0, 1])
    P = len(eigvals)
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    xs = np.arange(P)
    # Bootstrap can flip past the point estimate for small eigenvalues; clip
    # the half-widths to be non-negative for plotting.
    lower_err = np.clip(eigvals - lo, a_min=0.0, a_max=None)
    upper_err = np.clip(hi - eigvals, a_min=0.0, a_max=None)
    ax2.errorbar(xs, eigvals, yerr=[lower_err, upper_err],
                 fmt="o", color=QUAL[0], markersize=10, capsize=4,
                 markerfacecolor=QUAL[0], markeredgecolor="white",
                 markeredgewidth=1.3)
    ax2.set_yscale("log")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"$v_{k+1}$" for k in xs])
    ax2.set_xlabel("eigenvector (ordered by eigenvalue)")
    ax2.set_ylabel(r"$\lambda_k$ (log scale)")
    ax2.set_title("OPG spectrum + 95% bootstrap CIs")

    # Right: eigenvector heatmap (|V|).
    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(np.abs(V), cmap="magma", aspect="auto", vmin=0, vmax=1)
    ax3.set_xticks(xs)
    ax3.set_xticklabels([f"$v_{k+1}$" for k in xs])
    ax3.set_yticks(np.arange(P))
    ax3.set_yticklabels(PARAM_NAMES)
    ax3.set_title(r"$|V|$: parameter content per direction")
    plt.colorbar(im, ax=ax3, label=r"$|v_{k,j}|$")
    # Annotate cells.
    for i in range(P):
        for j in range(P):
            val = np.abs(V[i, j])
            txt = ax3.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color="white" if val < 0.55 else "black",
                            fontsize=8)

    fig.suptitle(
        r"OPG geometry at one $\theta$: the cloud, the spectrum, the content",
        fontsize=13, fontweight="bold", y=1.04,
    )
    p = save(fig, "05_gradient_cloud.png", out_dir="outputs/brock_hommes")
    print(f"saved {p}")


if __name__ == "__main__":
    main()
