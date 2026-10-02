"""Brock-Hommes at beta = 50: spectrum of G at differentiation horizons 10, 20 and 40,
and whether beta remains the sloppiest direction.

Run from the repository root:  python extras/brock_hommes/bh_horizon_stability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gndiag.config import enable_x64  # noqa: E402

enable_x64()

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import bh_common as bh  # noqa: E402
from gndiag import plotting  # noqa: E402
from gndiag.diagnostic.ggn import eigendecompose, principal_angles  # noqa: E402

NAME = "bh_horizon_stability"
HORIZONS = [10, 20, 40]


def compute():
    theta0 = bh.POINTS["complex"]
    keys, features, _ = bh.setup(theta0)
    per_horizon = {}
    for h in HORIZONS:
        e = eigendecompose(jnp.asarray(bh.ggn(bh.embedding(theta0, features, keys, h), bh.Z0)[0]))
        lam, V = np.asarray(e.eigvals), np.asarray(e.eigvecs)
        per_horizon[str(h)] = {"eigvals": lam.tolist(), "relative_spectrum": (lam / lam[0]).tolist(),
                               "eigvecs": V.tolist(), "beta_loading_sloppiest": float(abs(V[0, -1])),
                               "sloppiest_param": int(np.argmax(np.abs(V[:, -1])))}

    def angle(ha, hb, k):
        Va = np.asarray(per_horizon[str(ha)]["eigvecs"])
        Vb = np.asarray(per_horizon[str(hb)]["eigvecs"])
        return float(np.degrees(np.max(np.asarray(
            principal_angles(jnp.asarray(Va[:, :k]), jnp.asarray(Vb[:, :k]))))))

    angles = {f"{ha}_vs_{hb}_leading{k}": angle(ha, hb, k)
              for ha, hb in ((10, 20), (20, 40)) for k in (1, 2)}
    bh.save(NAME, {"per_horizon": per_horizon, "leading_angles_deg": angles})


def plot():
    res = bh.load(NAME)
    plotting.apply_style()
    fig, ax = plt.subplots(1, 2, figsize=plotting.figsize("double", 0.44))
    for h, c in zip(HORIZONS, plotting.seq_colors(len(HORIZONS))):
        ax[0].semilogy(range(1, 6), res["per_horizon"][str(h)]["relative_spectrum"], "o-", c=c,
                       label=f"horizon {h}")
    ax[0].set_xlabel(r"GGN direction (stiff $\to$ sloppy)")
    ax[0].set_ylabel(r"$\lambda_k/\lambda_1$")
    ax[0].set_xticks(range(1, 6))
    ax[0].legend(fontsize=7.5)
    ax[1].bar([str(h) for h in HORIZONS],
              [res["per_horizon"][str(h)]["beta_loading_sloppiest"] for h in HORIZONS],
              color=plotting.ROLE["ggn"], width=0.55)
    ax[1].axhline(1.0, ls=":", c="0.6", lw=1)
    ax[1].set_ylim(0, 1.08)
    ax[1].set_xlabel("differentiation horizon")
    ax[1].set_ylabel(r"$|$loading of $\beta$ on sloppiest$|$")
    for a, letter in zip(ax, "ab"):
        plotting.panel_label(a, letter)
    fig.tight_layout()
    bh.save_figure(fig, NAME)
    plt.close(fig)


if __name__ == "__main__":
    compute()
    plot()
