"""Booklet 2, Figure 7: BH per-seed gradient cloud + OPG spectrum (result)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curvature_calib.calibration.opg import bootstrap_eigvals, eigendecompose  # noqa: E402
from curvature_calib.calibration.per_seed_grads import (  # noqa: E402
    per_seed_loss_and_grads,
    vmap_simulate,
)
from curvature_calib.models.brock_hommes import simulate  # noqa: E402
from curvature_calib.viz.booklet_annotate import SLOPPY_COLOR, STIFF_COLOR  # noqa: E402
from curvature_calib.viz.booklet_style import QUAL, apply_booklet_style, save_vector  # noqa: E402
from scripts.booklets._figbase import out_dir  # noqa: E402

OUT_AREA = "methodology"
OUT_NAME = "fig_07_gradient_cloud"

T = 200
SIGMA = 0.05
R = 1.1


def _sim(theta, key):
    return simulate(theta, key, T=T, sigma=SIGMA, R=R, x_init=0.0)


def confidence_ellipse(F, ax, n_std=1.0, **kwargs):
    """Draw a 1-sigma (or n_std-sigma) confidence ellipse for a 2-D covariance F."""
    w, V = np.linalg.eigh(F)
    order = np.argsort(-w)
    w, V = w[order], V[:, order]
    angle = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(w, 0))
    e = mpatches.Ellipse(xy=(0, 0), width=width, height=height, angle=angle,
                         fill=False, **kwargs)
    ax.add_patch(e)


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    theta_star = jnp.array([3.0, 1.2, 0.2, 1.2, -0.2], dtype=jnp.float64)
    theta_eval = theta_star + jnp.array([0.0, 0.05, 0.03, 0.05, -0.03],
                                        dtype=jnp.float64)
    apply_booklet_style()

    # ── computation (verbatim from script 21) ────────────────────────────────
    M_ref = 128
    M_eval = 200
    ref_keys = jax.random.split(jax.random.PRNGKey(0), M_ref)
    Y_ref = vmap_simulate(_sim, theta_star, ref_keys)
    keys = jax.random.split(jax.random.PRNGKey(11), M_eval)
    stats = per_seed_loss_and_grads(_sim, theta_eval, keys, Y_ref)

    G = np.asarray(stats.per_seed_grads)
    F = np.asarray(stats.opg)
    eig = eigendecompose(jnp.asarray(F))
    eigvals = np.asarray(eig.eigvals)
    V = np.asarray(eig.eigvecs)

    g_centered = G - G.mean(axis=0)
    proj = g_centered @ V[:, :2]
    F_2d = V[:, :2].T @ F @ V[:, :2]

    boot = np.asarray(bootstrap_eigvals(stats.per_seed_grads, n_boot=500,
                                        key=jax.random.PRNGKey(2)))

    # ── sanity print ─────────────────────────────────────────────────────────
    span_oom = np.log10(eigvals[0] / max(eigvals[-1], 1e-30))
    print(f"eigenvalues : {eigvals}")
    print(f"OOM span    : {span_oom:.2f}")
    print(f"λ₁/λ_P      : {eigvals[0] / max(eigvals[-1], 1e-30):.2e}")

    # ── figure ───────────────────────────────────────────────────────────────
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Panel (a) — gradient cloud
    ax.scatter(proj[:, 0], proj[:, 1], s=14, alpha=0.45,
               color=QUAL[0], edgecolor="white", linewidth=0.3,
               label=fr"per-seed gradients ($M={M_eval}$)")
    ax.scatter([0], [0], marker="x", color="black", s=80, lw=2,
               label="mean (centred)")
    confidence_ellipse(F_2d, ax, n_std=1.0,
                       edgecolor=QUAL[1], lw=2,
                       label=r"$1\sigma$ OPG ellipse")
    ax.set_xlabel(r"$v_1^\top g_m$ (stiff)")
    ax.set_ylabel(r"$v_2^\top g_m$")
    ax.set_title("(a) per-seed gradient cloud", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")

    # Panel (b) — eigenvalue spectrum
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
    ax2.set_ylabel(r"$\lambda_k$")
    ax2.set_title("(b) OPG spectrum (95% bootstrap CI)", fontweight="bold")

    # span annotation box
    ax2.text(0.04, 0.04,
             f"span: {span_oom:.1f} OOM\n"
             rf"$\lambda_1/\lambda_P$: {eigvals[0]/max(eigvals[-1],1e-30):.1e}",
             transform=ax2.transAxes, fontsize=9, va="bottom",
             bbox=dict(facecolor="white", edgecolor="grey",
                       alpha=0.85, boxstyle="round,pad=0.3"))

    # stiff / sloppy text labels (plain text; avoids messy arrows on log axis)
    ax2.text(0, eigvals[0] * 1.8, "stiff", ha="center", va="bottom",
             color=STIFF_COLOR, fontsize=9, fontweight="bold")
    ax2.text(P - 1, eigvals[-1] * 0.45, "sloppy", ha="center", va="top",
             color=SLOPPY_COLOR, fontsize=9, fontweight="bold")

    fig.suptitle(
        "Brock–Hommes calibration loss has a strongly sloppy OPG spectrum",
        fontweight="bold",
    )
    fig.tight_layout()

    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)


if __name__ == "__main__":
    main()
