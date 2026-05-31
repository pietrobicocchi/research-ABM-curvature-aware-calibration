"""MMD^2 landscape over a 2D slice of theta, with the true theta* marked.

We sweep (g_1, g_2) on a grid and compute the unbiased MMD^2 between the
simulator distribution at that (g_1, g_2) and the reference (held at theta*).
This is the loss surface the calibrator climbs against. The slice exposes:
    - sharpness in the well-identified direction (color contrast around theta*)
    - flatness in poorly identified directions (broad plateaus / valleys)

The sloppy-direction visualisation that follows in script 05 is this same
landscape's local curvature, surfaced cheaply through gradient geometry.

Run: uv run python scripts/04_mmd_landscape.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from curvature_calib.calibration.per_seed_grads import vmap_simulate
from curvature_calib.losses.mmd import mmd_sq_with_median_bandwidth
from curvature_calib.models.brock_hommes import simulate
from curvature_calib.viz.style import apply_style, save


T = 200
SIGMA = 0.05
R = 1.1

THETA_STAR = jnp.array([3.0, 1.2, 0.2, 1.2, -0.2])


def _sim(theta, key):
    return simulate(theta, key, T=T, sigma=SIGMA, R=R, x_init=0.0)


def main() -> None:
    apply_style()

    M_ref = 128
    M_eval = 96
    n_grid = 25  # 25 x 25 = 625 evaluations

    # Reference cloud at theta*.
    ref_keys = jax.random.split(jax.random.PRNGKey(0), M_ref)
    Y_ref = vmap_simulate(_sim, THETA_STAR, ref_keys)

    eval_keys = jax.random.split(jax.random.PRNGKey(1), M_eval)

    g1_grid = np.linspace(THETA_STAR[1] - 0.20, THETA_STAR[1] + 0.20, n_grid)
    g2_grid = np.linspace(THETA_STAR[3] - 0.20, THETA_STAR[3] + 0.20, n_grid)

    Z = np.zeros((n_grid, n_grid))
    for i, g1 in enumerate(g1_grid):
        for j, g2 in enumerate(g2_grid):
            theta = THETA_STAR.at[1].set(g1).at[3].set(g2)
            X = vmap_simulate(_sim, theta, eval_keys)
            Z[i, j] = float(mmd_sq_with_median_bandwidth(X, Y_ref))
        if i % 5 == 0:
            print(f"  row {i+1}/{n_grid}")
    # Clip tiny negatives (U-statistic can yield slightly < 0 at the truth).
    Z = np.maximum(Z, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Linear heatmap.
    ax = axes[0]
    G1, G2 = np.meshgrid(g1_grid, g2_grid, indexing="ij")
    im = ax.pcolormesh(G1, G2, Z, cmap="magma", shading="auto")
    cs = ax.contour(G1, G2, Z, levels=10, colors="white",
                    linewidths=0.5, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=7)
    ax.scatter([THETA_STAR[1]], [THETA_STAR[3]], s=200, marker="*",
               edgecolor="white", facecolor="#ffea00", linewidth=1.6,
               zorder=10, label=r"$\theta^*$")
    ax.set_xlabel(r"$g_1$ (trend coeff, type 1)")
    ax.set_ylabel(r"$g_2$ (trend coeff, type 2)")
    ax.set_title(r"$\widehat{\mathrm{MMD}}^2(\mathbb{P}_\theta, \mathbb{P}_{\theta^*})$ over a 2D $\theta$ slice")
    ax.legend(loc="upper right")
    plt.colorbar(im, ax=ax, label=r"MMD$^2$")

    # Log heatmap with zoom on basin.
    ax = axes[1]
    Z_log = np.log10(Z + 1e-6)
    im2 = ax.pcolormesh(G1, G2, Z_log, cmap="viridis", shading="auto")
    cs2 = ax.contour(G1, G2, Z_log, levels=8, colors="white",
                     linewidths=0.5, alpha=0.6)
    ax.clabel(cs2, inline=True, fontsize=7, fmt="%.1f")
    ax.scatter([THETA_STAR[1]], [THETA_STAR[3]], s=200, marker="*",
               edgecolor="white", facecolor="#ffea00", linewidth=1.6,
               zorder=10, label=r"$\theta^*$")
    ax.set_xlabel(r"$g_1$")
    ax.set_ylabel(r"$g_2$")
    ax.set_title(r"$\log_{10}\widehat{\mathrm{MMD}}^2$ — basin geometry")
    ax.legend(loc="upper right")
    plt.colorbar(im2, ax=ax, label=r"$\log_{10}$ MMD$^2$")

    fig.suptitle(r"MMD$^2$ loss landscape: slice over $(g_1, g_2)$, other params at truth",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    p = save(fig, "04_mmd_landscape.png")
    print(f"saved {p}")


if __name__ == "__main__":
    main()
