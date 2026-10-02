"""Brock-Hommes at beta = 50 (complex) and beta = 80 (chaotic).

(a) G against the exact Hessian of the MMD loss at the fit and at increasing
    off-fit displacements delta along the stiffest direction (R = H - G grows).
(b, c) Leading eigenvalue and leading direction of G as a function of the
    differentiation horizon, against the full horizon.

Run from the repository root:  python extras/brock_hommes/bh_ggn_vs_hessian.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gndiag.config import enable_x64  # noqa: E402

enable_x64()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import bh_common as bh  # noqa: E402
from gndiag import plotting  # noqa: E402
from gndiag.diagnostic.ggn import eigendecompose, principal_angles, symmetrize  # noqa: E402

NAME = "bh_ggn_vs_hessian"
HORIZONS = [1, 5, 10, 20, 50, None]          # None: full trajectory
DELTAS = (0.0, 0.3, 0.8)
LABEL = {"complex": r"$\beta=50$ (complex)", "chaotic": r"$\beta=80$ (chaotic)"}


def analyze(theta0):
    keys, features, _ = bh.setup(theta0)
    eigs = {}
    for h in HORIZONS:
        G, _ = bh.ggn(bh.embedding(theta0, features, keys, h), bh.Z0)
        eigs[str(h)] = eigendecompose(G)
    full = eigs["None"]
    horizon = [{"horizon": h,
                "lead_eigval": float(eigs[str(h)].eigvals[0]),
                "lead_angle_to_full_deg": float(np.degrees(jnp.max(principal_angles(
                    eigs[str(h)].eigvecs[:, :1], full.eigvecs[:, :1]))))} for h in HORIZONS]

    eta = bh.embedding(theta0, features, keys, bh.WORKING_HORIZON)
    G0, _ = bh.ggn(eta, bh.Z0)
    v_stiff = eigendecompose(G0).eigvecs[:, 0]
    off_fit = []
    for delta in DELTAS:
        eta_y = eta(bh.Z0 + delta * v_stiff)
        L = lambda z: 0.5 * jnp.sum((eta(z) - eta_y) ** 2)
        H = jnp.asarray(symmetrize(jax.hessian(L)(bh.Z0)))
        off_fit.append({"delta": delta, "residual_over_hessian":
                        float(jnp.linalg.norm(H - G0)) / max(float(jnp.linalg.norm(H)), 1e-300)})
    return {"theta": list(theta0), "horizon_sweep": horizon, "off_fit": off_fit}


def compute():
    bh.save(NAME, {name: analyze(th) for name, th in bh.POINTS.items()})


def plot():
    res = bh.load(NAME)
    plotting.apply_style()
    R = plotting.ROLE
    col = {"complex": R["ggn"], "chaotic": R["opg"]}
    fig, ax = plt.subplots(1, 3, figsize=plotting.figsize("double", 0.37))
    xpos = np.arange(len(HORIZONS))
    xlabels = [str(h) if h else "full" for h in HORIZONS]
    for name, p in res.items():
        ax[0].semilogy([r["delta"] for r in p["off_fit"]],
                       [max(r["residual_over_hessian"], 1e-17) for r in p["off_fit"]],
                       "-o", ms=5, color=col[name], label=LABEL[name])
        ax[1].semilogy(xpos, [r["lead_eigval"] for r in p["horizon_sweep"]], "-o", ms=4,
                       color=col[name], label=LABEL[name])
        ax[2].plot(xpos, [r["lead_angle_to_full_deg"] for r in p["horizon_sweep"]], "-o", ms=4,
                   color=col[name], label=LABEL[name])
    ax[0].set_xlabel(r"off-fit displacement $\delta$")
    ax[0].set_ylabel(r"$\|R\|/\|H\|$")
    ax[0].legend(fontsize=7, loc="lower right")
    for a in ax[1:]:
        a.set_xticks(xpos); a.set_xticklabels(xlabels)
        a.set_xlabel("differentiation horizon")
    ax[1].set_ylabel(r"leading eigenvalue $\lambda_1$")
    ax[1].legend(fontsize=7, loc="upper left")
    ax[2].set_ylabel("angle to full horizon (deg)")
    ax[2].legend(fontsize=7, loc="upper right")
    for a, letter in zip(ax, "abc"):
        plotting.panel_label(a, letter)
    fig.tight_layout()
    bh.save_figure(fig, NAME)
    plt.close(fig)


if __name__ == "__main__":
    compute()
    plot()
