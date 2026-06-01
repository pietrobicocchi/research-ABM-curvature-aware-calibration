"""Gallery: sample trajectories + returns + variance scaling across regimes.

Three rows:
    1. Sample x_t trajectories at five beta values, in time.
    2. The corresponding return distributions (kde overlay).
    3. Variance vs. beta sweep, showing the regime transition.

Run: uv run python scripts/02_simulation_gallery.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from curvature_calib.models.brock_hommes import simulate
from curvature_calib.viz.style import QUAL, apply_style, save


def main() -> None:
    apply_style()

    g, b, R, sigma = 1.2, 0.2, 1.1, 0.05
    T = 1500
    betas = [0.0, 5.0, 20.0, 50.0, 80.0]
    labels = [r"$\beta=0$ (stable)",
              r"$\beta=5$ (near-bifurcation)",
              r"$\beta=20$",
              r"$\beta=50$ (chaotic)",
              r"$\beta=80$ (strongly chaotic)"]
    key = jax.random.PRNGKey(7)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 5, hspace=0.45, wspace=0.30)

    trajs = []
    for j, beta in enumerate(betas):
        theta = jnp.array([beta, g, b, g, -b])
        xs = simulate(theta, key, T=T, sigma=sigma, R=R, x_init=0.0)
        trajs.append(np.asarray(xs))

        ax = fig.add_subplot(gs[0, j])
        ax.plot(xs[-400:], color=QUAL[j], lw=1.0)
        ax.set_title(labels[j])
        ax.set_xlabel("t")
        if j == 0:
            ax.set_ylabel(r"$x_t$ (deviation)")
        ax.tick_params(labelsize=8)

        ax2 = fig.add_subplot(gs[1, j])
        rets = np.diff(xs)
        ax2.hist(rets, bins=60, density=True, color=QUAL[j], alpha=0.85,
                 edgecolor="white", linewidth=0.4)
        ax2.set_xlabel(r"$\Delta x_t$")
        if j == 0:
            ax2.set_ylabel("density")
        ax2.set_yscale("log")
        ax2.tick_params(labelsize=8)

    # Bottom: variance vs beta over a fine grid.
    ax3 = fig.add_subplot(gs[2, :])
    betas_fine = np.linspace(0.0, 100.0, 100)
    variances = []
    for beta in betas_fine:
        theta = jnp.array([beta, g, b, g, -b])
        xs = simulate(theta, key, T=T, sigma=sigma, R=R, x_init=0.0)
        variances.append(float(jnp.var(xs[-500:])))
    ax3.plot(betas_fine, variances, color="#2c3e50", lw=1.4)
    ax3.fill_between(betas_fine, variances, alpha=0.15, color="#2c3e50")
    for j, beta in enumerate(betas):
        idx = int(np.argmin(np.abs(betas_fine - beta)))
        ax3.scatter([beta], [variances[idx]], color=QUAL[j], s=80, zorder=5,
                    edgecolor="white", linewidth=1.4, label=labels[j])
    ax3.set_xlabel(r"$\beta$ (intensity of choice)")
    ax3.set_ylabel(r"Var$(x_t)$ over last 500 steps")
    ax3.set_yscale("log")
    ax3.set_title("Bifurcation signature in the calibration-relevant statistic")
    ax3.legend(loc="upper left", ncol=2, framealpha=0.95)

    fig.suptitle("Brock–Hommes simulator — regime gallery",
                 fontsize=14, fontweight="bold", y=0.995)
    p = save(fig, "02_simulation_gallery.png", out_dir="outputs/brock_hommes")
    print(f"saved {p}")


if __name__ == "__main__":
    main()
