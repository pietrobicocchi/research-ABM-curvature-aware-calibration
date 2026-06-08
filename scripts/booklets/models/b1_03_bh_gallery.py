"""Booklet 1, Figure 3: Brock-Hommes simulation gallery (three regimes)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curvature_calib.models.brock_hommes import simulate  # noqa: E402
from curvature_calib.viz.booklet_style import apply_booklet_style, save_vector  # noqa: E402
from curvature_calib.viz.style import REGIME  # noqa: E402
from scripts.booklets._figbase import out_dir  # noqa: E402

OUT_AREA = "models"
OUT_NAME = "fig_03_bh_gallery"

# ── Regime parameter vectors (beta, g1, b1, g2, b2) ──────────────────────────
# Taken verbatim from scripts/25_eigenvalue_trajectory.py REGIMES dict.
_THETA = {
    "fundamental": jnp.array([1.0,  0.5,  0.0,  0.5,  0.0], dtype=jnp.float64),
    "periodic":    jnp.array([5.0,  1.2,  0.0, -0.5,  0.0], dtype=jnp.float64),
    "chaotic":     jnp.array([10.0, 1.2,  0.2,  1.2, -0.2], dtype=jnp.float64),
}

T = 200
R = 1.1
SIGMA = 0.05

# Human-readable labels (regime name + β value)
_LABELS = {
    "fundamental": r"Fundamental regime ($\beta = 1.0$)",
    "periodic":    r"Periodic regime ($\beta = 5.0$)",
    "chaotic":     r"Chaotic regime ($\beta = 10.0$)",
}


def main() -> None:
    apply_booklet_style()

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=False)

    for ax, (regime, theta) in zip(axes, _THETA.items(), strict=True):
        key = jax.random.PRNGKey(0)
        xs = np.asarray(simulate(theta, key, T=T, R=R, sigma=SIGMA))

        ax.plot(xs, color=REGIME[regime], lw=1.2)
        ax.set_title(_LABELS[regime], loc="left", fontweight="bold")
        ax.set_ylabel(r"$x_t$")

    # Only the bottom row gets x-label
    axes[-1].set_xlabel("time $t$")

    fig.suptitle("Brock–Hommes: three dynamical regimes", fontweight="bold")
    fig.tight_layout()

    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)


if __name__ == "__main__":
    main()
