"""Booklet 2, Figure 3: surrogate gradient for discrete events (concept)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curvature_calib.viz.booklet_annotate import callout  # noqa: E402
from curvature_calib.viz.booklet_style import QUAL, apply_booklet_style, save_vector  # noqa: E402
from scripts.booklets._figbase import out_dir  # noqa: E402

OUT_AREA = "methodology"
OUT_NAME = "fig_03_surrogate"

_DARK = "#2c3e50"

# Tau values: decreasing tau -> sharper (closer to step)
_TAUS = [1.5, 0.7, 0.3]
# Color shades: progressively darker blues from QUAL[0] (#1f3a93)
_TAU_COLORS = ["#5d8fdc", "#2e6abf", "#1f3a93"]


def sig(x: np.ndarray, tau: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x / tau))


def main() -> None:
    apply_booklet_style()

    x = np.linspace(-6, 6, 600)
    step = (x >= 0).astype(float)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4.2))

    # ── LEFT: the event and its relaxations ──────────────────────────────────
    ax_l.plot(x, step, color=_DARK, lw=2.2, label=r"hard event $H(x)$", zorder=5)
    for tau, color in zip(_TAUS, _TAU_COLORS, strict=True):
        ax_l.plot(
            x, sig(x, tau), color=color, lw=1.8,
            label=fr"surrogate $\sigma(x/\tau)$, $\tau={tau}$",
        )

    ax_l.set_xlabel(r"input $x$ (e.g. $t - t_{\rm lock}$)")
    ax_l.set_ylabel("fire probability")
    ax_l.set_title("(a) discrete event → smooth surrogate")
    ax_l.set_xlim(-6, 6)
    ax_l.set_ylim(-0.08, 1.18)
    ax_l.legend(loc="upper left", framealpha=0.92, fontsize=7.5)

    # ── RIGHT: where gradient flows ───────────────────────────────────────────
    # Hard step derivative: zero a.e.
    ax_r.axhline(0, color=_DARK, lw=2.2, label=r"$H'(x)=0$ a.e. (no gradient)", zorder=5)

    # Degenerate spike at x=0 for the hard step
    ax_r.annotate(
        "", xy=(0, 0.30), xytext=(0, 0.04),
        arrowprops=dict(arrowstyle="-|>", color=_DARK, lw=1.4),
    )
    ax_r.text(0.15, 0.32, "Dirac spike\n(not a useful gradient)",
              fontsize=7.5, color=_DARK, va="bottom", ha="left")

    # Sigmoid derivatives: bell curves
    for tau, color in zip(_TAUS, _TAU_COLORS, strict=True):
        s = sig(x, tau)
        dsig = (1.0 / tau) * s * (1.0 - s)
        ax_r.plot(
            x, dsig, color=color, lw=1.8,
            label=fr"$d\sigma/dx$, $\tau={tau}$",
        )

    # Callout pointing at the peak of the widest (most visible) bell, tau=1.5
    s15 = sig(0.0, 1.5)
    peak_height = (1.0 / 1.5) * s15 * (1.0 - s15)
    callout(
        ax_r,
        xy=(0.0, peak_height),
        text="non-zero gradient\n=> backprop through the event",
        xytext=(2.8, peak_height + 0.06),
        color=QUAL[1],
        fontsize=8.0,
    )

    ax_r.set_xlabel(r"input $x$")
    ax_r.set_ylabel("gradient w.r.t. $x$")
    ax_r.set_title("(b) surrogate restores a usable gradient")
    ax_r.set_xlim(-6, 6)
    ax_r.set_ylim(-0.04, 0.55)
    ax_r.legend(loc="upper right", framealpha=0.92, fontsize=7.5)

    fig.suptitle(
        "Surrogate gradients: relaxing discrete agent events so the simulator is differentiable",
        fontweight="bold",
    )
    fig.tight_layout()

    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)


if __name__ == "__main__":
    main()
