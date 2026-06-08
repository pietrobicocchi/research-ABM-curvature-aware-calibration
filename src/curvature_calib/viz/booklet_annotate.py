"""In-figure annotation helpers for the booklets (explainer-style callouts)."""
from __future__ import annotations

from curvature_calib.viz.style import QUAL

STIFF_COLOR = QUAL[1]   # warm red
SLOPPY_COLOR = "#7f8c8d"  # grey

__all__ = ["callout", "brace", "tag_stiff_sloppy"]


def callout(ax, xy, text, xytext, *, color="#2c3e50", fontsize=8.5):
    """Labelled arrow pointing at data coordinate `xy`, text placed at `xytext`."""
    return ax.annotate(
        text, xy=xy, xytext=xytext, textcoords="data",
        fontsize=fontsize, color=color, fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=0.8, alpha=0.95),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
    )


def brace(ax, x0, x1, y, text, *, color="#2c3e50", fontsize=8.5, dy=0.04):
    """Horizontal annotation bracket from x0..x1 at height y with a label."""
    ax.annotate("", xy=(x0, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-", color=color, lw=1.0))
    ax.text((x0 + x1) / 2, y + dy, text, ha="center", va="bottom",
            color=color, fontsize=fontsize, fontweight="bold")


def tag_stiff_sloppy(ax, stiff_xy, sloppy_xy):
    """Convenience: tag a stiff and a sloppy point with the canonical colors."""
    callout(ax, stiff_xy, "stiff", (stiff_xy[0] + 0.6, stiff_xy[1]),
            color=STIFF_COLOR)
    callout(ax, sloppy_xy, "sloppy", (sloppy_xy[0] - 0.6, sloppy_xy[1]),
            color=SLOPPY_COLOR)
