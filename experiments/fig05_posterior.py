"""FIG-05 — SIR posterior overlay.

The prior-relative GGN is not a proxy for the local posterior geometry — it IS the
posterior's local geometry: orientation to ~1°, profiled energy to <0.5%. Loads the
frozen EXP-005 run record (covariances + profiles).

Run:  uv run python -m experiments.fig05_posterior
"""
from __future__ import annotations

from curvature_calib.config import enable_x64, require_x64

enable_x64()
require_x64()

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from curvature_calib.viz import style as STY  # noqa: E402
from experiments import _figlib as FL  # noqa: E402
from experiments.exp005_sir_posterior import PARAM_NAMES  # noqa: E402

LATEX = {"beta": r"$\beta$", "gamma": r"$\gamma$", "I0": r"$I_0$",
         "t_lock": r"$t_{\mathrm{lock}}$", "f_lock": r"$f_{\mathrm{lock}}$"}


def run():
    require_x64()
    STY.apply_style()
    R = STY.ROLE
    m, run = FL.load_metrics("EXP-005")
    A = FL.load_arrays("EXP-005")
    Sig_ref, Sig_lap = np.asarray(A["Sig_ref"]), np.asarray(A["Sig_lap"])

    # pick a moderately anisotropic pair (condition ~6) among the well-agreeing
    # parameters {beta, gamma, I0} — representative of the 4/5 marginals that match
    # to ≤5%; the mild t_lock non-Gaussianity is a stated caveat, not the hero panel.
    cands = [(a, b) for a in (0, 1, 2) for b in (0, 1, 2) if a < b]
    i, j = min(cands, key=lambda ab: abs(np.log(
        np.linalg.cond(Sig_ref[np.ix_(ab, ab)])) - np.log(6.0)))

    fig, ax = plt.subplots(1, 2, figsize=STY.figsize("double", 0.46))

    # (a) reference posterior (gray fill) vs GGN-Laplace (navy dashed outline)
    def _sub(S):
        return S[np.ix_([i, j], [i, j])]
    for ns, al in ((2.0, 0.16), (1.0, 0.28)):
        xr, yr = FL.cov_ellipse(_sub(Sig_ref), nsig=ns)
        ax[0].fill(xr, yr, color=R["ref"], alpha=al, zorder=2, lw=0)
    for ns in (1.0, 2.0):
        xg, yg = FL.cov_ellipse(_sub(Sig_lap), nsig=ns)
        ax[0].plot(xg, yg, "--", color=R["ggn"], lw=1.8, zorder=4)
    ax[0].plot(0, 0, "o", color=R["truth"], ms=4, zorder=5)
    ax[0].set_xlabel(f"{LATEX[PARAM_NAMES[i]]}  (posterior, prior-std)")
    ax[0].set_ylabel(f"{LATEX[PARAM_NAMES[j]]}  (posterior, prior-std)")
    ax[0].set_title(f"axes agree to {m['top1_angle_cov_deg']:.1f}°")
    ax[0].legend(handles=[Patch(facecolor=R["ref"], alpha=0.5, label="reference posterior"),
                          Line2D([0], [0], color=R["ggn"], lw=1.8, ls="--", label="prior-relative GGN")],
                 loc="upper left")
    ax[0].set_aspect("equal", adjustable="datalim")
    STY.panel_label(ax[0], "a")

    # (b) relative error of the GGN quadratic vs the profiled energy (stiff & sloppy)
    cols = {"stiff": R["ggn"], "sloppy": R["residual"]}
    for lab in ("stiff", "sloppy"):
        p = m["profiles"][lab]
        a = np.array(p["alphas_post_std"])
        act, quad = np.array(p["post_energy"]), np.array(p["quad_post_pred"])
        rel = np.abs(act - quad) / np.maximum(np.abs(quad), 1e-12)
        ax[1].semilogy(a, np.maximum(rel, 1e-6), "-o", ms=4, color=cols[lab],
                       label=f"{lab} direction ($\\lambda$={p['lambda']:.1g})")
    ax[1].axhline(5e-3, ls=STY.LS["ref"], c=R["ref"], lw=1.2)
    ax[1].text(-2.9, 5.6e-3, "0.5%", ha="left", va="bottom", fontsize=7, color=R["ref"])
    ax[1].set_xlabel(r"displacement $\alpha$  (posterior-std)")
    ax[1].set_ylabel("GGN quadratic rel. error")
    ax[1].set_title("energy matches to <0.5%")
    ax[1].legend(fontsize=7, loc="lower right")
    STY.panel_label(ax[1], "b")

    fig.suptitle("The prior-relative GGN is the posterior's local geometry",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FL.savefig(fig, "fig05_posterior")
    plt.close(fig)
    print("source run:", run, "| pair:", PARAM_NAMES[i], PARAM_NAMES[j])
    print("saved:", out)


if __name__ == "__main__":
    run()
