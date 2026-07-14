"""FIG-08 — observation design.

Only the observation that measures what incidence is blind to (transmission across
the lockdown transition) recovers the weak direction and collapses the policy
range; a redundant one does not. Loads the frozen EXP-007 run record.

Run:  uv run python -m experiments.fig08_obs_design
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

NAMES = {"1_incidence_only": "incidence only",
         "2_incidence+prevalence": "+ prevalence (redundant)",
         "3_incidence+compliance": "+ compliance (targeted)"}


def run():
    require_x64()
    STY.apply_style()
    R = STY.ROLE
    m, run = FL.load_metrics("EXP-007")
    designs = m["designs"]
    keys = list(NAMES)
    cols = STY.seq_colors(len(keys))            # ordered by information richness

    fig, ax = plt.subplots(1, 2, figsize=STY.figsize("double", 0.46))

    # (a) prior-relative spectra for the three designs
    for c, k in zip(cols, keys):
        ev = np.array(designs[k]["prior_relative_eigs"])
        idx = np.arange(1, len(ev) + 1)
        ax[0].semilogy(idx, np.maximum(ev, 1e-12), "-o", ms=4, color=c, label=NAMES[k])
    ax[0].axhline(1.0, ls=STY.LS["ref"], c=R["ref"], lw=1.2)
    ax[0].text(len(idx), 1.0, r" $\lambda=1$", va="center", fontsize=7, color=R["ref"])
    ax[0].set_xticks(idx)
    ax[0].set_xlabel("GGN direction  (stiff → sloppy)")
    ax[0].set_ylabel(r"prior-relative $\lambda$")
    ax[0].set_title("spectrum by observation design")
    ax[0].legend(loc="lower left", fontsize=7.5)
    STY.panel_label(ax[0], "a")

    # (b) in-fit intervention-value range per design (cases)
    y = np.arange(len(keys))[::-1]
    for c, k, yy in zip(cols, keys, y):
        lo, hi = m["headline"][k]["intervention_range_cases"]
        ax[1].plot([lo, hi], [yy, yy], "-", lw=7, color=c, solid_capstyle="round",
                   alpha=0.85, zorder=3)
        ax[1].text(hi + 40, yy, f"{hi - lo:.0f} cases wide", va="center",
                   fontsize=7.5, color=R["truth"])
    ax[1].set_yticks(y); ax[1].set_yticklabels([NAMES[k] for k in keys], fontsize=8)
    ax[1].set_xlabel("in-fit counterfactual range (cases averted)")
    ax[1].set_title("counterfactual range collapses 57×")
    ax[1].set_xlim(-150, 3200)
    ax[1].grid(axis="y", visible=False)
    STY.panel_label(ax[1], "b")

    fig.suptitle("The geometry prescribes which observation to add",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FL.savefig(fig, "fig08_obs_design")
    plt.close(fig)
    print("source run:", run)
    print("saved:", out)


if __name__ == "__main__":
    run()
