"""Phase-space portraits of the deterministic Brock-Hommes map.

For each of three beta values we plot:
    (a) the trajectory in time (transient + steady state)
    (b) the lag-plot x_{t+1} vs x_t with a colour-coded time axis
    (c) the return map cobweb on the 1-d marginal

This visually exposes the fixed-point -> limit-cycle -> chaos transition,
which is the underlying structure the OPG eigenvalues will surface during
calibration.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from curvature_calib.models.brock_hommes import simulate
from curvature_calib.viz.style import REGIME, apply_style, save


def main() -> None:
    apply_style()
    g, b, R = 1.2, 0.2, 1.1
    T = 4000
    key = jax.random.PRNGKey(3)
    cases = [
        (3.0,  "fundamental", "stable"),
        (15.0, "periodic",    "limit cycle / quasi-periodic"),
        (60.0, "chaotic",     "chaotic"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))

    for i, (beta, regime_name, label) in enumerate(cases):
        color = REGIME[regime_name]
        theta = jnp.array([beta, g, b, g, -b])
        xs = simulate(theta, key, T=T, sigma=0.0, R=R, x_init=0.5)
        xs = np.asarray(xs)

        ax0 = axes[i, 0]
        ax0.plot(xs[:400], color=color, lw=0.9)
        ax0.set_title(rf"$\beta={beta:.0f}$ ({label}): time series")
        ax0.set_xlabel("t")
        ax0.set_ylabel(r"$x_t$")

        ax1 = axes[i, 1]
        tail = xs[-2000:]
        sc = ax1.scatter(tail[:-1], tail[1:],
                         c=np.arange(tail.size - 1),
                         cmap="viridis", s=3, alpha=0.6)
        ax1.set_title("lag plot: $x_{t+1}$ vs $x_t$ (steady state)")
        ax1.set_xlabel(r"$x_t$")
        ax1.set_ylabel(r"$x_{t+1}$")
        cb = plt.colorbar(sc, ax=ax1, shrink=0.7)
        cb.set_label("time step", fontsize=8)

        ax2 = axes[i, 2]
        rets = np.diff(xs[-2000:])
        ax2.hist(rets, bins=80, density=True, color=color,
                 alpha=0.75, edgecolor="white", linewidth=0.4)
        ax2.set_title("return distribution (steady state)")
        ax2.set_xlabel(r"$\Delta x_t$")
        ax2.set_ylabel("density")
        ax2.set_yscale("log")

    fig.suptitle("Brock–Hommes deterministic dynamics across regimes",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = save(fig, "03_phase_portraits.png", out_dir="outputs/brock_hommes")
    print(f"saved {p}")


if __name__ == "__main__":
    main()
