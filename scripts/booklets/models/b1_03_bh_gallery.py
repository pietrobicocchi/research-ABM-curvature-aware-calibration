"""Booklet 1, Figure 3: Brock-Hommes simulation gallery (three regimes).

The three regimes (fundamental / periodic / chaotic) are the canonical
beta-defined regimes used throughout the project (cf. scripts/25). At the
calibration beta values they are *stochastic* regimes: the deterministic
deviation dynamics relax to the fundamental fixed point, and the qualitative
differences live in the noise-driven persistence. The autocorrelation column
(right) makes that distinction legible -- the chaotic regime (high beta) carries
strong momentum (slowly decaying ACF) while the fundamental regime reverts fast.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curvature_calib.models.brock_hommes import simulate  # noqa: E402
from curvature_calib.viz.booklet_style import apply_booklet_style, save_vector  # noqa: E402
from curvature_calib.viz.style import REGIME  # noqa: E402
from scripts.booklets._figbase import out_dir  # noqa: E402

OUT_AREA = "models"
OUT_NAME = "fig_03_bh_gallery"
_LABELS = {
    "fundamental": r"Fundamental ($\beta=1$)",
    "periodic":    r"Periodic ($\beta=5$)",
    "chaotic":     r"Chaotic ($\beta=10$)",
}

T = 200
R = 1.1
SIGMA = 0.05
N_LAGS = 20
SEED = 0  # shared across regimes: differences are dynamics, not noise draw


def _acf(x: np.ndarray, n_lags: int) -> np.ndarray:
    """Sample autocorrelation for lags 0..n_lags."""
    x = x - x.mean()
    denom = x @ x
    return np.array([(x[: len(x) - k] @ x[k:]) / denom for k in range(n_lags + 1)])


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    _THETA = {
        "fundamental": jnp.array([1.0,  0.5,  0.0,  0.5,  0.0], dtype=jnp.float64),
        "periodic":    jnp.array([5.0,  1.2,  0.0, -0.5,  0.0], dtype=jnp.float64),
        "chaotic":     jnp.array([10.0, 1.2,  0.2,  1.2, -0.2], dtype=jnp.float64),
    }
    apply_booklet_style()

    fig, axes = plt.subplots(3, 2, figsize=(10, 7),
                             gridspec_kw={"width_ratios": [2.4, 1.0]})

    for row, (regime, theta) in enumerate(_THETA.items()):
        color = REGIME[regime]
        xs = np.asarray(simulate(theta, jax.random.PRNGKey(SEED),
                                 T=T, R=R, sigma=SIGMA))
        xs_post = xs[50:]  # drop transient for the ACF estimate
        acf = _acf(xs_post, N_LAGS)

        ax_ts, ax_ac = axes[row]

        # Time series.
        ax_ts.plot(xs, color=color, lw=1.1)
        ax_ts.axhline(0.0, color="#bbbbbb", lw=0.7, zorder=0)
        ax_ts.set_title(_LABELS[regime], loc="left", fontweight="bold", color=color)
        ax_ts.set_ylabel(r"$x_t$")

        # Autocorrelation.
        lags = np.arange(N_LAGS + 1)
        ax_ac.vlines(lags, 0.0, acf, color=color, lw=1.6)
        ax_ac.plot(lags, acf, "o", color=color, ms=2.5)
        ax_ac.axhline(0.0, color="#bbbbbb", lw=0.7)
        ax_ac.set_ylim(-0.35, 1.02)
        ax_ac.set_ylabel("ACF")
        # Annotate lag-1 autocorrelation as the headline discriminator.
        ax_ac.annotate(fr"$\rho_1={acf[1]:.2f}$", xy=(1, acf[1]),
                       xytext=(0.55, 0.80), textcoords="axes fraction",
                       fontsize=8.5, color=color, fontweight="bold")

    axes[-1, 0].set_xlabel("time $t$")
    axes[-1, 1].set_xlabel("lag")

    fig.suptitle(
        r"Brock–Hommes regimes: intensity of choice $\beta$ shapes persistence",
        fontweight="bold",
    )
    fig.tight_layout()

    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)


if __name__ == "__main__":
    main()
