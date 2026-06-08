"""Booklet 1, Figure 10: mean-field vs network SIR -- same class, different dynamics.

Matched at R0 ~ 2.7 (no lockdown). The mean-field model (N=1e5) is smooth and
slow; the network model (N=250, Erdos-Renyi, mean degree 6) burns through its
finite contact structure faster and carries finite-size stochasticity from the
discrete per-node transitions -- the transitions that require the Gumbel-sigmoid
surrogate gradient. To compare *shape* rather than scale, both curves are
normalised to unit peak height and to their own peak time. This motivates the
surrogate-gradient regime treated in the methodology booklet.
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
from curvature_calib.models.network_sir import simulate as net_sim  # noqa: E402
from curvature_calib.models.sir import simulate as mf_sim  # noqa: E402
from curvature_calib.viz.booklet_style import QUAL, apply_booklet_style, save_vector  # noqa: E402
from scripts.booklets._figbase import out_dir  # noqa: E402

OUT_AREA = "models"
OUT_NAME = "fig_10_mf_vs_network"

T = 200
GAMMA = 0.10
R0 = 2.7
MEAN_DEG = 6.0
M_SEEDS = 24


def _norm(curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (t/t_peak, curve/peak) restricted to a sensible window."""
    pk = curve.max()
    t_pk = max(int(curve.argmax()), 1)
    t_norm = np.arange(len(curve)) / t_pk
    return t_norm, curve / pk


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    apply_booklet_style()

    # Match R0: mean-field beta = R0*gamma; network beta = R0*gamma/mean_degree.
    mf_beta = R0 * GAMMA
    net_beta = R0 * GAMMA / MEAN_DEG

    mf_curve = np.asarray(mf_sim(jnp.array([mf_beta, GAMMA, 1e-3, 0.5, 1.0]),
                                 jax.random.PRNGKey(0), T=T, N=1e5))
    keys = jax.random.split(jax.random.PRNGKey(0), M_SEEDS)
    net_curves = np.stack([
        np.asarray(net_sim(jnp.array([net_beta, GAMMA, 0.02, 0.5, 1.0]),
                           k, T=T, N=250, mean_degree=MEAN_DEG))
        for k in keys
    ])
    net_mean = net_curves.mean(0)

    # Normalise both to unit peak and to their own peak time.
    mf_t, mf_y = _norm(mf_curve)
    pk_day = max(int(net_mean.argmax()), 1)
    net_t = np.arange(T) / pk_day
    net_pk = net_mean.max()
    net_y = net_mean / net_pk
    net_lo = (net_mean - net_curves.std(0)) / net_pk
    net_hi = (net_mean + net_curves.std(0)) / net_pk

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(mf_t, mf_y, color=QUAL[0], lw=2.4, label="Mean-field SIR  ($N=10^5$)")
    ax.fill_between(net_t, net_lo, net_hi, color=QUAL[1], alpha=0.15, lw=0)
    ax.plot(net_t, net_y, color=QUAL[1], lw=2.0,
            label="Network SIR  ($N=250$, surrogate)")

    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 1.18)
    ax.axvline(1.0, color="#bbbbbb", ls=":", lw=1.0, zorder=0)
    ax.set_xlabel(r"time (relative to each model's peak, $t/t_{\rm peak}$)")
    ax.set_ylabel("incidence (relative to peak)")
    ax.set_title("Mean-field vs network SIR: same class, different dynamics",
                 fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.annotate("network: fluctuates &\nheavier tail (finite $N$,\ndiscrete transitions)",
                xy=(1.7, np.interp(1.7, net_t, net_y)), xytext=(2.0, 0.72),
                fontsize=8, color=QUAL[1], ha="left",
                arrowprops=dict(arrowstyle="->", color=QUAL[1], lw=1.0))
    ax.annotate("mean-field:\nsmooth, narrow,\ndeterministic", xy=(1.2, np.interp(1.2, mf_t, mf_y)),
                xytext=(0.12, 0.40), fontsize=8, color=QUAL[0], ha="left",
                arrowprops=dict(arrowstyle="->", color=QUAL[0], lw=1.0))

    fig.tight_layout()
    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)


if __name__ == "__main__":
    main()
