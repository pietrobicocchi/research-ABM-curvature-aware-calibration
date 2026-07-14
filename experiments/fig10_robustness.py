"""FIG-10 — robustness under stochasticity, discreteness, surrogate gradients.

On stochastic discrete network-SIR at two operating points, the leading eigenspace
is stable across surrogate constructions and FD-validated; truncating the
differentiation horizon destroys it. Loads the frozen EXP-008 run record.

Run:  uv run python -m experiments.fig10_robustness
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

OP_LABEL = {"A_near_threshold": "OP-A (attack 29%)",
            "B_supercritical_denser": "OP-B (attack 52%)"}


def run():
    require_x64()
    STY.apply_style()
    R = STY.ROLE
    m, run = FL.load_metrics("EXP-008")
    ops = list(m["operating_points"])
    opcol = {ops[0]: R["ggn"], ops[1]: R["alt"]}

    fig, ax = plt.subplots(1, 3, figsize=STY.figsize("double", 0.36))

    # (a) cross-surrogate leading-k principal angles at both operating points
    ks = ["k1", "k2", "k3"]
    x = np.arange(len(ks)); w = 0.36
    for s, op in enumerate(ops):
        d = m["operating_points"][op]["C14_cross_surrogate"]["subspace_max_angle_deg_by_k"]
        ax[0].bar(x + (s - 0.5) * w, [d[k] for k in ks], w, color=opcol[op],
                  label=OP_LABEL[op], zorder=3)
    ax[0].axhline(15.0, ls=STY.LS["ref"], c=R["ref"], lw=1.2)
    ax[0].text(2.4, 15.0, "robust < 15°", va="bottom", ha="right", fontsize=7, color=R["ref"])
    ax[0].set_xticks(x); ax[0].set_xticklabels(["1", "2", "3"])
    ax[0].set_xlabel("leading-$k$ subspace"); ax[0].set_ylabel("Gumbel↔ST angle (deg)")
    ax[0].set_title("directions stable")
    ax[0].legend(fontsize=7); ax[0].set_ylim(0, 20)
    STY.panel_label(ax[0], "a")

    # (b) finite-difference validation: Gumbel (validated) vs straight-through
    surrs = ["gumbel", "straight_through"]
    scol = {"gumbel": R["ggn"], "straight_through": R["opg"]}
    x2 = np.arange(len(ops))
    for s, sur in enumerate(surrs):
        vals = [m["operating_points"][op]["per_surrogate"][sur]["fd_reljac_error_median"]
                for op in ops]
        ax[1].bar(x2 + (s - 0.5) * w, vals, w, color=scol[sur],
                  label=sur.replace("_", "-"), zorder=3)
    ax[1].axhline(1e-3, ls=STY.LS["ref"], c=R["ref"], lw=1.2)
    ax[1].text(1.4, 1e-3, "FD-validated", va="bottom", ha="right", fontsize=7, color=R["ref"])
    ax[1].set_yscale("log"); ax[1].set_ylim(1e-6, 5)
    ax[1].set_xticks(x2); ax[1].set_xticklabels(["OP-A", "OP-B"])
    ax[1].set_ylabel("median rel. Jacobian error")
    ax[1].set_title("FD-validated")
    ax[1].legend(fontsize=7, loc="lower left")
    STY.panel_label(ax[1], "b")

    # (c) horizon truncation destroys the geometry
    for op in ops:
        hs = m["operating_points"][op]["C15_horizon_sweep"]
        pts = sorted((v["grad_horizon"], v["max_leading_angle_vs_full_deg"]) for v in hs.values())
        gh = [p[0] for p in pts] + [40]         # 40 = full horizon → 0°
        ang = [p[1] for p in pts] + [0.0]
        ax[2].plot(gh, ang, "-o", ms=4, color=opcol[op], label=OP_LABEL[op])
    ax[2].axhline(15.0, ls=STY.LS["ref"], c=R["ref"], lw=1.2)
    ax[2].set_xlabel("differentiation horizon"); ax[2].set_ylabel("angle vs full (deg)")
    ax[2].set_title("horizon truncation")
    ax[2].legend(fontsize=7, loc="center right")
    STY.panel_label(ax[2], "c")

    fig.suptitle("The geometry survives stochasticity, discreteness, and surrogate gradients",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FL.savefig(fig, "fig10_robustness")
    plt.close(fig)
    print("source run:", run)
    print("saved:", out)


if __name__ == "__main__":
    run()
