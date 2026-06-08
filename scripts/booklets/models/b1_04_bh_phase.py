"""Booklet 1, Figure 4: Brock-Hommes lag-plot geometry of the three regimes.

State-space (lag-plot) view that complements fig_03. Because the deterministic
deviation dynamics relax to the fixed point at these beta values (see
bh_regimes_are_stochastic memory), we show the *stochastic* lag plots
x_t vs x_{t-1}: persistence (high beta) elongates the cloud along the y=x
diagonal, while the fast-reverting fundamental regime fills a rounder blob.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curvature_calib.models.brock_hommes import simulate  # noqa: E402
from curvature_calib.viz.booklet_style import apply_booklet_style, save_vector  # noqa: E402
from curvature_calib.viz.style import REGIME  # noqa: E402
from scripts.booklets._figbase import out_dir  # noqa: E402

OUT_AREA = "models"
OUT_NAME = "fig_04_bh_phase"

_THETA = {
    "fundamental": jnp.array([1.0,  0.5,  0.0,  0.5,  0.0], dtype=jnp.float64),
    "periodic":    jnp.array([5.0,  1.2,  0.0, -0.5,  0.0], dtype=jnp.float64),
    "chaotic":     jnp.array([10.0, 1.2,  0.2,  1.2, -0.2], dtype=jnp.float64),
}
_LABELS = {
    "fundamental": r"Fundamental ($\beta=1$)",
    "periodic":    r"Periodic ($\beta=5$)",
    "chaotic":     r"Chaotic ($\beta=10$)",
}

T = 4000   # long run for a dense, smooth lag cloud
R = 1.1
SIGMA = 0.05
SEED = 0


def main() -> None:
    apply_booklet_style()

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))

    lim = 0.0
    series = {}
    for regime, theta in _THETA.items():
        xs = np.asarray(simulate(theta, jax.random.PRNGKey(SEED),
                                 T=T, R=R, sigma=SIGMA))[200:]
        series[regime] = xs
        lim = max(lim, np.abs(xs).max())
    lim *= 1.05

    for ax, regime in zip(axes, _THETA, strict=True):
        xs = series[regime]
        color = REGIME[regime]
        x_prev, x_curr = xs[:-1], xs[1:]
        rho1 = np.corrcoef(x_prev, x_curr)[0, 1]

        ax.plot([-lim, lim], [-lim, lim], color="#bbbbbb", lw=0.8,
                zorder=0, label=r"$x_t=x_{t-1}$")
        ax.scatter(x_prev, x_curr, s=3, alpha=0.25, color=color,
                   edgecolor="none", zorder=2)
        ax.set_title(_LABELS[regime], color=color, fontweight="bold")
        ax.set_xlabel(r"$x_{t-1}$")
        if ax is axes[0]:
            ax.set_ylabel(r"$x_t$")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.text(0.05, 0.92, fr"$\rho_1={rho1:.2f}$", transform=ax.transAxes,
                fontsize=9, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=color, lw=0.6, alpha=0.9))

    axes[0].legend(loc="lower right", fontsize=7, frameon=False)
    fig.suptitle(
        r"Brock–Hommes lag-plot geometry: persistence elongates the cloud "
        r"along $x_t=x_{t-1}$",
        fontweight="bold",
    )
    fig.tight_layout()

    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)


if __name__ == "__main__":
    main()
