"""FIG-03 — where it breaks.

Two distinct failure modes bound where the GGN quadratic can be trusted:
residual-curvature bias (nonzero as the perturbation shrinks) and higher-order
nonlinear error (grows with distance). Loads the frozen EXP-002 run record.

Run:  uv run python -m experiments.fig03_where_it_breaks
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
from experiments.exp002_nonlinear_validity import ALPHAS, TAU  # noqa: E402

# path columns: [t, E_G=‖G−H‖/‖H‖, leading1_angle, R_over_H, max_eigval_err]
LOWRES = "rosenbrock_a1.0_b1.0"       # low-residual: E_G → 0 at the fit
RESID = "irreducible_lam1.0_c4.0"     # residual curvature: irreducible floor


def run():
    require_x64()
    STY.apply_style()
    R = STY.ROLE
    m, run = FL.load_metrics("EXP-002")
    A = FL.load_arrays("EXP-002")

    fig, ax = plt.subplots(1, 2, figsize=STY.figsize("double", 0.46))

    # (a) two failure modes vs distance from the fit (log-y makes the floor visible)
    pl = np.asarray(A[f"{LOWRES}__path_v0"])
    pr = np.asarray(A[f"{RESID}__path_v0"])
    ax[0].semilogy(pl[:, 0], np.maximum(np.abs(pl[:, 3]), 1e-4), "-o", ms=4,
                   color=R["residual"], label="low residual → 0 at fit")
    ax[0].semilogy(pr[:, 0], np.maximum(np.abs(pr[:, 3]), 1e-4), "-s", ms=4,
                   color=R["opg"], label="residual curvature → floor")
    r0 = float(np.abs(pr[np.argmin(np.abs(pr[:, 0])), 3]))
    ax[0].axhline(r0, ls=STY.LS["ref"], c=R["opg"], lw=1.0, alpha=0.6)
    ax[0].text(-2.0, r0 * 1.3, f"irreducible bias {r0:.1%}", fontsize=7,
               color=R["opg"], va="bottom")
    ax[0].set_xlabel(r"distance from fit  $t$")
    ax[0].set_ylabel(r"$\|R\|/\|H\| = \|G-H\|/\|H\|$")
    ax[0].set_title("two failure modes")
    ax[0].legend(fontsize=7.5, loc="lower right")
    STY.panel_label(ax[0], "a")

    # (b) validity radius along the stiff direction
    rad = m["cells"][LOWRES]["validity_radii"]["v0"]
    qa = np.array(rad["quad_error"])
    al = np.array(ALPHAS)
    ax[1].semilogy(al, np.maximum(qa, 1e-16), "-o", ms=4, color=R["ggn"],
                   label="GGN quadratic error")
    ax[1].axhline(TAU, ls=STY.LS["ref"], c=R["ref"], lw=1.2)
    ax[1].text(al.max(), TAU, f" τ={TAU:g}", va="bottom", ha="right", fontsize=7, color=R["ref"])
    ax[1].axvspan(-rad["rho_minus"], rad["rho_plus"], color=R["ggn"], alpha=0.10, zorder=0)
    ax[1].text(0, ax[1].get_ylim()[0] * 2, f"valid radius ρ={rad['rho']:.2f}",
               ha="center", va="bottom", fontsize=7.5, color=R["ggn"])
    ax[1].set_xlabel(r"displacement $\alpha$ along stiff direction")
    ax[1].set_ylabel("relative quadratic error")
    ax[1].set_title("the validity radius")
    STY.panel_label(ax[1], "b")

    fig.suptitle("The GGN quadratic is valid locally — and states its own limits",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FL.savefig(fig, "fig03_where_it_breaks")
    plt.close(fig)
    print("source run:", run)
    print("saved:", out)


if __name__ == "__main__":
    run()
