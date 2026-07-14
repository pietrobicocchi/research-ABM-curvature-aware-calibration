"""FIG-07 — the payoff: policy underdetermination (hero).

Directly-observed epidemic functionals are pinned by the fit; the counterfactual
intervention value is not, and its freedom lies along the sloppiest GGN direction.
Loads the frozen EXP-006 run record.

Run:  uv run python -m experiments.fig07_payoff
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

LABELS = {"total_cases": "total cases (observed)",
          "peak_incidence": "peak incidence (observed)",
          "intervention_value": "intervention value (counterfactual)"}


def run():
    require_x64()
    STY.apply_style()
    R = STY.ROLE
    m, run = FL.load_metrics("EXP-006")
    budget = m["fit_budget_nats"]
    fig, ax = plt.subplots(1, 2, figsize=STY.figsize("double", 0.46),
                           gridspec_kw={"width_ratios": [1.25, 1]})

    # (a) profiled fit cost vs policy value change
    obs_col = {"total_cases": R["truth"], "peak_incidence": R["ref"]}
    for name in ("total_cases", "peak_incidence", "intervention_value"):
        rows = m["functionals"][name]["curve"]
        q = np.array([r["Q_frac_change"] for r in rows]) * 100.0
        L = np.array([r["profiled_loss"] for r in rows])
        o = np.argsort(q)
        col = R["ggn"] if name == "intervention_value" else obs_col[name]
        lw = 2.2 if name == "intervention_value" else 1.6
        ax[0].plot(q[o], L[o], "-o", ms=3, lw=lw, color=col, label=LABELS[name])
    ax[0].axhline(budget, ls=STY.LS["ref"], c=R["ref"], lw=1.2)
    ax[0].set_xlim(-260, 2600)
    ax[0].set_ylim(-0.15, 3.0)
    ax[0].text(-230, budget, f"fit budget {budget:g} nats",
               va="bottom", ha="left", fontsize=7, color=R["ref"])
    ax[0].annotate("observed:\npinned (±0.3%)", xy=(0, 0.05), xytext=(650, 0.55),
                   fontsize=7.5, color=R["truth"], ha="left",
                   arrowprops=dict(arrowstyle="->", color=R["truth"], lw=1.1))
    ax[0].set_xlabel("policy value change (%)")
    ax[0].set_ylabel("profiled fit cost (nats)")
    ax[0].set_title("fit cost vs policy value")
    ax[0].legend(loc="upper right", fontsize=7.5)
    STY.panel_label(ax[0], "a")

    # (b) the intervention freedom loads on the sloppiest GGN direction
    rows = m["functionals"]["intervention_value"]["curve"]
    good = [r for r in rows if r["profiled_loss"] <= budget]
    extreme = max(good, key=lambda r: abs(r["Q_frac_change"]))
    align = np.array(extreme["alignment_abs_cos"])
    eigs = np.array(m["prior_relative_eigs"])
    idx = np.arange(1, len(align) + 1)
    bars = ax[1].bar(idx, align, color=R["ref"], width=0.68, zorder=3)
    kstar = int(np.argmax(align))
    bars[kstar].set_color(R["ggn"])
    ax[1].set_xticks(idx)
    ax[1].set_xticklabels([rf"$\lambda_{{{k}}}$" + "\n" + f"{e:.1g}"
                           for k, e in zip(idx, eigs)], fontsize=6.5)
    ax[1].set_ylim(0, 1.05)
    ax[1].set_xlabel("GGN direction  (stiff → sloppy)")
    ax[1].set_ylabel(r"$|\cos\angle|$ with policy freedom")
    ax[1].set_title("alignment with GGN axes")
    ax[1].annotate(f"loads on the sloppiest\ndirection ($\\lambda$={eigs[kstar]:.1g})",
                   xy=(kstar + 1, align[kstar]), xytext=(kstar + 1 - 1.6, 0.6),
                   fontsize=7.5, color=R["ggn"], ha="center",
                   arrowprops=dict(arrowstyle="->", color=R["ggn"], lw=1.2))
    STY.panel_label(ax[1], "b")

    fig.suptitle("A good fit does not determine the counterfactual",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FL.savefig(fig, "fig07_payoff")
    plt.close(fig)
    print("source run:", run)
    print("saved:", out)


if __name__ == "__main__":
    run()
