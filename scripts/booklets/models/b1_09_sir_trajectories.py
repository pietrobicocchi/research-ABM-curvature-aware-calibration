"""Booklet 1, Figure 9: network-SIR epidemic curve, lockdown on vs off.

IMPORTANT modelling note. At the *calibration* operating point of scripts/18
(theta* = [0.30, 0.10, 0.05, 0.40, 0.50]) the network epidemic has R0 ~ 18: it
peaks on day ~2 and is extinct by day ~9, whereas t_lock_norm = 0.40 places the
lockdown on day 80. The lockdown therefore fires long after the epidemic is over
and has *no* effect on the dynamics -- which is exactly why lockdown strength /
timing is the sloppy (near-unidentifiable) direction the OPG diagnostic flags
(see sir_generalization memory). To *illustrate the lockdown mechanism* in the
model booklet we use a slower, well-timed illustrative regime (R0 ~ 2.7, lockdown
engaging just before the peak). This is labelled as illustrative in the figure.
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
from curvature_calib.models.network_sir import simulate as net_simulate  # noqa: E402
from curvature_calib.viz.booklet_style import QUAL, apply_booklet_style, save_vector  # noqa: E402
from scripts.booklets._figbase import out_dir  # noqa: E402

OUT_AREA = "models"
OUT_NAME = "fig_09_sir_trajectories"

# Network settings match scripts/18 (T, N, mean_degree).
T = 200
N = 250
MEAN_DEG = 6.0
M_SEEDS = 16
DAYS_SHOWN = 50

# Illustrative regime (R0 ~ 2.7) so the lockdown, timed just before the peak,
# visibly flattens the curve. theta = (beta, gamma, I0_frac, t_lock_norm, f_lock).
BETA, GAMMA, I0 = 0.045, 0.10, 0.02
T_LOCK = 0.04          # day 8 of 200
F_LOCK_ON = 0.20       # strong lockdown
THETA_OFF = jnp.array([BETA, GAMMA, I0, T_LOCK, 1.0])   # no lockdown
THETA_ON = jnp.array([BETA, GAMMA, I0, T_LOCK, F_LOCK_ON])


def _mean_std(theta):
    keys = jax.random.split(jax.random.PRNGKey(0), M_SEEDS)
    xs = np.stack([np.asarray(net_simulate(theta, k, T=T, N=N, mean_degree=MEAN_DEG))
                   for k in keys])
    return xs.mean(0), xs.std(0)


def main() -> None:
    apply_booklet_style()

    off_mean, off_std = _mean_std(THETA_OFF)
    on_mean, on_std = _mean_std(THETA_ON)
    t = np.arange(T)
    lock_day = T_LOCK * T

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.fill_between(t, off_mean - off_std, off_mean + off_std,
                    color=QUAL[1], alpha=0.13, lw=0)
    ax.fill_between(t, on_mean - on_std, on_mean + on_std,
                    color=QUAL[0], alpha=0.13, lw=0)
    ax.plot(t, off_mean, color=QUAL[1], lw=2.2, label="No lockdown")
    ax.plot(t, on_mean, color=QUAL[0], lw=2.2,
            label=f"With lockdown ($f_{{\\rm lock}}={F_LOCK_ON}$)")

    ax.axvline(lock_day, color="#888888", ls="--", lw=1.2, zorder=0)
    ax.text(lock_day + 0.6, ax.get_ylim()[1] * 0.97, "lockdown\nengages",
            fontsize=8.5, color="#555555", va="top", ha="left")

    # Peak-reduction annotation: vertical double arrow between the two peaks.
    pk_off = off_mean.max()
    pk_on = on_mean.max()
    x_pk = int(off_mean.argmax())
    ax.annotate("", xy=(x_pk, pk_off), xytext=(x_pk, pk_on),
                arrowprops=dict(arrowstyle="<->", color="#2c3e50", lw=1.4))
    ax.text(x_pk + 1.5, (pk_off + pk_on) / 2,
            f"peak −{100 * (1 - pk_on / pk_off):.0f}%",
            fontsize=9, fontweight="bold", color="#2c3e50", va="center")

    ax.set_xlim(0, DAYS_SHOWN)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("time (days)")
    ax.set_ylabel("daily incidence")
    ax.set_title("Network-SIR: a well-timed lockdown flattens the epidemic curve",
                 fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.text(0.99, 0.60,
            r"illustrative regime ($R_0\approx2.7$);"
            "\nlockdown engages before the peak",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color="#777777", style="italic")

    fig.tight_layout()
    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)


if __name__ == "__main__":
    main()
