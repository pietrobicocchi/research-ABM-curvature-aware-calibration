"""FIG-04 — Brock–Hommes: real-ABM validation and the horizon boundary.

In an actual ABM the GGN equals the exact Hessian at the fit and inherits the same
residual-curvature bias off-fit; in the chaotic regime the exact geometry is
dominated by intrinsic pathwise sensitivity and depends materially on the
differentiation horizon. Loads the frozen EXP-004 run record.

Run:  uv run python -m experiments.fig04_brock_hommes
"""
from __future__ import annotations

from curvature_calib.config import enable_x64, require_x64

enable_x64()
require_x64()

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from curvature_calib.viz import style as STY  # noqa: E402
from experiments import _figlib as FL  # noqa: E402

PLABEL = {"P1_complex": "β=50 (complex)", "P2_chaotic": "β=80 (chaotic)"}
HORIZ_ORDER = ["1", "5", "10", "20", "50", "None"]      # ascending; None = full trajectory
HORIZ_LABELS = ["1", "5", "10", "20", "50", "full"]


def _hseries(point, field):
    hs = point["horizon_sweep"]
    return np.array([hs[k][field] for k in HORIZ_ORDER])


def run():
    require_x64()
    STY.apply_style()
    R = STY.ROLE
    m, run = FL.load_metrics("EXP-004")
    P = m["points"]
    pcol = {"P1_complex": R["ggn"], "P2_chaotic": R["opg"]}

    fig, ax = plt.subplots(1, 3, figsize=STY.figsize("double", 0.37))

    # (a) H = G at the fit; R grows off-fit
    for name in P:
        hg = P[name]["H_vs_G"]
        deltas = sorted(hg, key=lambda k: float(k.split("_")[1]))
        d = [float(k.split("_")[1]) for k in deltas]
        rh = [max(hg[k]["R_over_H"], 1e-17) for k in deltas]
        ax[0].semilogy(d, rh, "-o", ms=5, color=pcol[name], label=PLABEL[name])
    ax[0].set_xlabel(r"off-fit displacement $\delta$")
    ax[0].set_ylabel(r"$\|R\|/\|H\|$")
    ax[0].set_title("H = G at the fit")
    ax[0].legend(fontsize=7, loc="lower right")
    STY.panel_label(ax[0], "a")

    # (b) leading eigenvalue explodes toward the full horizon (categorical axis)
    xpos = np.arange(len(HORIZ_ORDER))
    for name in P:
        ax[1].semilogy(xpos, _hseries(P[name], "lead_eigval"), "-o", ms=4,
                       color=pcol[name], label=PLABEL[name])
    ax[1].set_xticks(xpos); ax[1].set_xticklabels(HORIZ_LABELS)
    ax[1].set_xlabel("differentiation horizon"); ax[1].set_ylabel(r"leading eigenvalue $\lambda_1$")
    ax[1].set_title("curvature explodes")
    ax[1].legend(fontsize=7, loc="upper left")
    STY.panel_label(ax[1], "b")

    # (c) leading direction distorted at short horizon (chaotic)
    for name in P:
        ax[2].plot(xpos, _hseries(P[name], "lead_angle_vs_full_deg"), "-o", ms=4,
                   color=pcol[name], label=PLABEL[name])
    ax[2].set_xticks(xpos); ax[2].set_xticklabels(HORIZ_LABELS)
    ax[2].set_xlabel("differentiation horizon"); ax[2].set_ylabel("angle vs full (deg)")
    ax[2].set_title("direction distorts")
    ax[2].legend(fontsize=7, loc="upper right")
    STY.panel_label(ax[2], "c")

    fig.suptitle("Brock–Hommes: exact at the fit, an honest boundary in chaos",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FL.savefig(fig, "fig04_brock_hommes")
    plt.close(fig)
    print("source run:", run)
    print("saved:", out)


if __name__ == "__main__":
    run()
