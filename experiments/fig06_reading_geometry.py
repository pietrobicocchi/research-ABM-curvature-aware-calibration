"""FIG-06 — reading the geometry: spectrum + eigenvector participation.

The prior-relative spectrum says HOW MANY combinations the data informs; the
eigenvector-participation heatmap says WHICH parameters form each stiff/sloppy
combination. Recomputes the smooth-SIR GGN at the fit (deterministic, cheap).

Run:  uv run python -m experiments.fig06_reading_geometry
"""
from __future__ import annotations

from curvature_calib.config import enable_x64, require_x64

enable_x64()
require_x64()

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from curvature_calib.calibration import diagnostic as DIAG  # noqa: E402
from curvature_calib.geometry import ggn as GGN  # noqa: E402
from curvature_calib.viz import style as STY  # noqa: E402
from experiments import _figlib as FL  # noqa: E402
from experiments.exp005_sir_posterior import T, m, SIGMA_DATA, Z0, PARAM_NAMES  # noqa: E402

LATEX = {"beta": r"$\beta$", "gamma": r"$\gamma$", "I0": r"$I_0$",
         "t_lock": r"$t_{\mathrm{lock}}$", "f_lock": r"$f_{\mathrm{lock}}$"}


def run():
    require_x64()
    STY.apply_style()
    R = STY.ROLE
    J = jax.jacfwd(m)(Z0)
    G = jnp.asarray(GGN.symmetrize((J.T @ J) / SIGMA_DATA ** 2))
    eig = DIAG.eigendecompose(G)
    lam = np.asarray(eig.eigvals)                     # descending
    V = np.asarray(eig.eigvecs)                       # column k = eigenvector k
    d_data = int((lam > 1).sum())
    plabels = [LATEX[p] for p in PARAM_NAMES]
    dirlabels = [f"$v_{{{k+1}}}$" for k in range(len(lam))]

    fig, ax = plt.subplots(1, 2, figsize=STY.figsize("double", 0.5),
                           gridspec_kw={"width_ratios": [1, 1.15]})

    # (a) prior-relative spectrum — data-dominant (λ>1) vs prior-dominant
    idx = np.arange(1, len(lam) + 1)
    cols = [R["ggn"] if l > 1 else R["ref"] for l in lam]
    ax[0].bar(idx, np.maximum(lam, 1e-7), color=cols, width=0.62, zorder=3)
    ax[0].set_yscale("log")
    ax[0].axhline(1.0, ls=STY.LS["ref"], c=R["residual"], lw=1.4)
    ax[0].text(len(lam) + 0.3, 1.0, r"$\lambda=1$", va="center", ha="right",
               fontsize=8, color=R["residual"])
    ax[0].set_xticks(idx); ax[0].set_xticklabels(dirlabels)
    ax[0].set_xlabel("GGN direction  (stiff → sloppy)")
    ax[0].set_ylabel(r"prior-relative eigenvalue $\lambda$")
    ax[0].set_title(rf"spectrum  ($d_{{\mathrm{{data}}}} = {d_data}$)")
    ax[0].annotate("data-dominant\n" + r"($\lambda>1$)", xy=(1, lam[0]),
                   xytext=(1.4, lam[0] * 0.04), fontsize=7.5, color=R["ggn"])
    ax[0].annotate("prior-dominant\n" + r"($\lambda<1$)", xy=(len(lam), lam[-1]),
                   xytext=(len(lam) - 2.3, lam[-1] * 8), fontsize=7.5, color=R["ref"])
    STY.panel_label(ax[0], "a")

    # (b) eigenvector participation heatmap — signed loadings, diverging
    vmax = float(np.abs(V).max())
    im = ax[1].imshow(V, cmap=STY.DIV, vmin=-vmax, vmax=vmax, aspect="auto")
    ax[1].set_xticks(range(len(lam))); ax[1].set_xticklabels(dirlabels)
    ax[1].set_yticks(range(len(plabels))); ax[1].set_yticklabels(plabels)
    ax[1].set_xlabel("GGN direction  (stiff → sloppy)")
    ax[1].set_title("eigenvector participation")
    for a in range(len(plabels)):
        for b in range(len(lam)):
            ax[1].text(b, a, f"{V[a, b]:+.2f}", ha="center", va="center",
                       fontsize=6.5,
                       color="white" if abs(V[a, b]) > 0.55 * vmax else R["truth"])
    cb = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03)
    cb.set_label("loading", fontsize=8)
    STY.panel_label(ax[1], "b")

    fig.suptitle("Reading the local geometry", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FL.savefig(fig, "fig06_reading_geometry")
    plt.close(fig)
    print("spectrum:", [f"{x:.2g}" for x in lam], "| d_data =", d_data)
    print("saved:", out)


if __name__ == "__main__":
    run()
