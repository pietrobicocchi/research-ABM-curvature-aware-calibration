"""Booklet 2, Figure 6: the two-parameter stiff/sloppy worked example (concept)."""
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
OUT_NAME = "fig_06_stiff_sloppy_toy"

_STIFF_COLOR = QUAL[1]    # warm red  (#c0392b)
_SLOPPY_COLOR = "#7f8c8d"  # grey
_DARK = "#2c3e50"
_SQRT2 = np.sqrt(2.0)

# eigen-directions
V_STIFF = np.array([1.0, 1.0]) / _SQRT2   # along a + b  (large curvature)
V_SLOPPY = np.array([1.0, -1.0]) / _SQRT2  # along a - b  (near-zero curvature)


def _loss(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """L(a, b) = (a+b)^2 + 0.02*(a-b)^2 — ratio of curvatures is 50."""
    return (a + b) ** 2 + 0.02 * (a - b) ** 2


def _build_landscape(ax: plt.Axes) -> None:
    """Left panel: filled-contour loss landscape with stiff/sloppy arrows."""
    a = np.linspace(-3.0, 3.0, 300)
    b = np.linspace(-3.0, 3.0, 300)
    A, B = np.meshgrid(a, b)
    Z = _loss(A, B)

    cf = ax.contourf(A, B, Z, levels=30, cmap="viridis")
    ax.contour(A, B, Z, levels=30, colors="white", linewidths=0.4, alpha=0.5)
    plt.colorbar(cf, ax=ax, label="loss $L$", fraction=0.046, pad=0.04)

    # ── eigen-direction arrows ────────────────────────────────────────────────
    arrow_len = 2.0
    _arrow_kw = dict(xytext=(0.0, 0.0), textcoords="data")

    # stiff arrow: runs across the valley (large curvature)
    ax.annotate(
        "", xy=tuple(arrow_len * V_STIFF), **_arrow_kw,
        arrowprops=dict(arrowstyle="-|>", color=_STIFF_COLOR,
                        lw=2.2, mutation_scale=16),
    )
    ax.text(
        arrow_len * V_STIFF[0] + 0.15,
        arrow_len * V_STIFF[1] + 0.18,
        r"stiff: $a+b$",
        color=_STIFF_COLOR, fontsize=9.5, fontweight="bold",
        ha="left", va="bottom",
    )

    # sloppy arrow: runs along the flat valley (near-zero curvature)
    ax.annotate(
        "", xy=tuple(arrow_len * V_SLOPPY), **_arrow_kw,
        arrowprops=dict(arrowstyle="-|>", color=_SLOPPY_COLOR,
                        lw=2.2, mutation_scale=16),
    )
    ax.text(
        arrow_len * V_SLOPPY[0] + 0.15,
        arrow_len * V_SLOPPY[1] - 0.18,
        r"sloppy: $a-b$",
        color=_SLOPPY_COLOR, fontsize=9.5, fontweight="bold",
        ha="left", va="top",
    )

    ax.set_aspect("equal")
    ax.set_xlabel("$a$")
    ax.set_ylabel("$b$")
    ax.set_title("(a) loss landscape: a flat valley")


def _build_profiles(ax: plt.Axes) -> None:
    """Right panel: loss vs step along each direction."""
    s = np.linspace(-2.0, 2.0, 200)

    # Points along each direction starting from origin
    pts_stiff = np.outer(s, V_STIFF)   # shape (200, 2)
    pts_sloppy = np.outer(s, V_SLOPPY)

    l_stiff = _loss(pts_stiff[:, 0], pts_stiff[:, 1])
    l_sloppy = _loss(pts_sloppy[:, 0], pts_sloppy[:, 1])

    ax.plot(s, l_stiff, color=_STIFF_COLOR, lw=2.2, label="stiff direction")
    ax.plot(s, l_sloppy, color=_SLOPPY_COLOR, lw=2.2, label="sloppy direction",
            linestyle="--")

    # Callout on the flat sloppy curve: place annotation above the near-flat line
    # Find a good x-position (avoid s=0 where stiff is also 0)
    s_callout = 1.2
    idx = np.argmin(np.abs(s - s_callout))
    y_callout = l_sloppy[idx]
    callout(
        ax,
        xy=(s_callout, y_callout),
        text="data barely constrains this direction\n"
             r"$\Rightarrow$ small OPG eigenvalue",
        xytext=(0.2, 3.5),
        color=_SLOPPY_COLOR,
        fontsize=8.0,
    )

    ax.legend(loc="upper center", framealpha=0.9)
    ax.set_xlabel("step along direction $s$")
    ax.set_ylabel("loss $L$")
    ax.set_title(
        "(b) the data sees the stiff direction,\nnot the sloppy one",
    )
    ax.set_xlim(-2.0, 2.0)


def main() -> None:
    apply_booklet_style()
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.6))

    _build_landscape(ax_left)
    _build_profiles(ax_right)

    fig.suptitle(
        "Stiff vs sloppy: large eigenvalues are constrained combinations,"
        " small ones are not",
        fontweight="bold",
    )
    fig.tight_layout()

    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)


if __name__ == "__main__":
    main()
