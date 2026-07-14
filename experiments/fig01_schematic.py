"""FIG-01 — concept schematic.

The whole idea in one picture: a differentiable ABM exposes the Jacobian of its
calibrated representation; its Gauss–Newton form is a local ellipse in parameter
space whose short (sloppy) axes are the combinations the data leaves free — and a
policy counterfactual can lie along one of them. Illustration only (no data).

Run:  uv run python -m experiments.fig01_schematic
"""
from __future__ import annotations

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Ellipse, Circle, FancyArrowPatch  # noqa: E402

from curvature_calib.viz import style as STY  # noqa: E402
from experiments import _figlib as FL  # noqa: E402


def _box(ax, x, y, w, h, text, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.6, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5,
            color=STY.ROLE["truth"], zorder=4)


def _arrow(ax, p0, p1, label=None):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                                 lw=1.6, color=STY.ROLE["truth"], zorder=2))
    if label:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 0.16, label, ha="center",
                va="bottom", fontsize=7.5, color=STY.ROLE["truth"], style="italic")


def run():
    STY.apply_style()
    R = STY.ROLE
    fig, ax = plt.subplots(figsize=STY.figsize("double", 0.44))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.5); ax.axis("off")

    # pipeline of stages
    y, h = 2.75, 1.05
    _box(ax, 0.2, y, 2.2, h, "differentiable ABM\n$X(\\theta,\\xi)$", "#eef1f6", R["ggn"])
    _box(ax, 3.0, y, 2.5, h, "calibrated rep.\n$m(z)=\\mathbb{E}_\\xi[\\psi(X)]$", "#eef1f6", R["ggn"])
    _box(ax, 6.3, y, 2.3, h, "local GGN\n$G=Dm^{\\top}WDm$", "#e9ecf7", R["ggn"])
    _arrow(ax, (2.4, y + h / 2), (3.0, y + h / 2))
    _arrow(ax, (5.5, y + h / 2), (6.3, y + h / 2), "$Dm$  (AD)")
    _arrow(ax, (8.6, y + 0.28), (9.95, 2.5))

    # the GGN uncertainty ellipse in prior-scaled parameter space:
    # narrow along stiff (data pins it, well inside the prior), ≈prior along sloppy.
    cx, cy, r = 11.0, 1.9, 0.92
    ax.add_patch(Circle((cx, cy), r, fill=False, ls=(0, (4, 3)), ec=R["ref"],
                        lw=1.3, zorder=2))
    ax.text(cx - r - 0.12, cy, "prior", fontsize=7, color=R["ref"], ha="right", va="center")
    ang = 30.0
    th = np.radians(ang)
    u_sloppy = np.array([np.cos(th), np.sin(th)])
    u_stiff = np.array([-np.sin(th), np.cos(th)])
    ax.add_patch(Ellipse((cx, cy), 2 * 0.86, 2 * 0.17, angle=ang, fill=True,
                         fc=R["ggn"], ec=R["ggn"], alpha=0.18, lw=2.0, zorder=3))
    ax.add_patch(Ellipse((cx, cy), 2 * 0.86, 2 * 0.17, angle=ang, fill=False,
                         ec=R["ggn"], lw=2.0, zorder=3))
    s = u_stiff * 0.30
    ax.add_patch(FancyArrowPatch((cx, cy), (cx + s[0], cy + s[1]), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.8, color=R["truth"], zorder=5))
    l = u_sloppy * 0.84
    ax.add_patch(FancyArrowPatch((cx, cy), (cx + l[0], cy + l[1]), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.8, color=R["residual"], zorder=5))
    ax.plot(cx, cy, "o", color=R["truth"], ms=4, zorder=6)
    ax.annotate("stiff: data pins it", xy=(cx + s[0], cy + s[1]),
                xytext=(cx + 0.15, cy + 1.28), fontsize=7.5, color=R["truth"], ha="center",
                arrowprops=dict(arrowstyle="->", color=R["truth"], lw=1.0))
    ax.text(cx + 0.15, cy - 1.12,
            "sloppy: data leaves it free\n$\\rightarrow$ the counterfactual lives here",
            fontsize=7.5, color=R["residual"], ha="center", va="top")

    fig.suptitle("A good fit does not say which parameter combinations the data constrains",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FL.savefig(fig, "fig01_schematic")
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    run()
