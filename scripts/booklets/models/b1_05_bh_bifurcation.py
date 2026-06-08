"""Booklet 1, Figure 5: Brock-Hommes deterministic bifurcation diagram.

Sweeps the intensity of choice beta with noise off and plots the long-run
(post-transient) price deviation. The fundamental fixed point x=0 is globally
attracting up to beta ~ 36; beyond it the deviation dynamics undergo a sharp
transition to sustained, chaotic motion -- the route to chaos of Brock & Hommes
(1998). The three calibration regimes (beta = 1, 5, 10) all sit in the stable
zone: at those values the regime differences are noise-driven (see fig_03/04),
which is why the booklet labels them as stochastic regimes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curvature_calib.models.brock_hommes import simulate  # noqa: E402
from curvature_calib.viz.booklet_style import apply_booklet_style, save_vector  # noqa: E402
from curvature_calib.viz.style import REGIME  # noqa: E402
from scripts.booklets._figbase import out_dir  # noqa: E402

OUT_AREA = "models"
OUT_NAME = "fig_05_bh_bifurcation"

R = 1.1
T = 800
N_TRANSIENT = 650          # keep the last T - N_TRANSIENT points
BETA_GRID = np.linspace(0.0, 60.0, 600)
X_INIT = 0.3
KEY = jax.random.PRNGKey(0)

_REGIME_BETAS = {"fundamental": 1.0, "periodic": 5.0, "chaotic": 10.0}


def _tail(beta: float) -> np.ndarray:
    _gb = jnp.array([1.2, 0.2, 1.2, -0.2], dtype=jnp.float64)
    theta = jnp.concatenate([jnp.array([beta], dtype=jnp.float64), _gb])
    xs = simulate(theta, KEY, T=T, R=R, sigma=0.0, x_init=X_INIT)
    return np.asarray(xs[N_TRANSIENT:])


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    apply_booklet_style()

    betas_plot, xs_plot = [], []
    for beta in BETA_GRID:
        tail = _tail(float(beta))
        betas_plot.append(np.full_like(tail, beta))
        xs_plot.append(tail)
    betas_plot = np.concatenate(betas_plot)
    xs_plot = np.concatenate(xs_plot)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.scatter(betas_plot, xs_plot, s=0.6, alpha=0.35, color="#1f3a93",
               edgecolor="none", rasterized=True)

    # Mark the three calibration regimes (all in the stable zone).
    for regime, beta in _REGIME_BETAS.items():
        ax.axvline(beta, color=REGIME[regime], lw=1.4, ls="--", alpha=0.9)
        ax.text(beta, ax.get_ylim()[1] * 0.92, fr" $\beta={beta:.0f}$",
                color=REGIME[regime], fontsize=8.5, fontweight="bold",
                rotation=90, va="top", ha="left")

    ax.set_xlabel(r"intensity of choice $\beta$")
    ax.set_ylabel(r"long-run price deviation $x_t$")
    ax.set_xlim(0, 60)
    ax.set_title(
        "Brock–Hommes route to chaos: deterministic bifurcation in $\\beta$",
        fontweight="bold",
    )

    # Annotate the stable zone and the chaotic onset.
    ax.annotate("stable fixed point\n(calibration regimes live here)",
                xy=(10, 0.0), xytext=(15, 0.10),
                fontsize=8.5, color="#444444", ha="left",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=1.0))
    ax.annotate("onset of chaos", xy=(40, 0.12), xytext=(44, 0.16),
                fontsize=8.5, color="#c0392b", fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.0))

    fig.tight_layout()
    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)


if __name__ == "__main__":
    main()
