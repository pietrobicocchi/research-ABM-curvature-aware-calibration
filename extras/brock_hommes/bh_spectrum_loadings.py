"""Brock-Hommes at beta = 50, differentiation horizon 20: spectrum of G (relative to
the largest eigenvalue) and the loading of each parameter on each eigenvector.

Run from the repository root:  python extras/brock_hommes/bh_spectrum_loadings.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gndiag.config import enable_x64  # noqa: E402

enable_x64()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import bh_common as bh  # noqa: E402
from experiments.mfsir_spectrum_loadings import loadings_heatmap  # noqa: E402
from gndiag import plotting  # noqa: E402
from gndiag.diagnostic.ggn import eigendecompose  # noqa: E402

NAME = "bh_spectrum_loadings"


def compute():
    theta0 = bh.POINTS["complex"]
    keys, features, _ = bh.setup(theta0)
    G, _ = bh.ggn(bh.embedding(theta0, features, keys, bh.WORKING_HORIZON), bh.Z0)
    e = eigendecompose(G)
    bh.save(NAME, {"eigvals": bh.to_list(e.eigvals), "eigvecs": bh.to_list(e.eigvecs)})


def plot():
    res = bh.load(NAME)
    plotting.apply_style()
    lam, V = np.array(res["eigvals"]), np.array(res["eigvecs"])
    dirs = [f"$v_{{{k + 1}}}$" for k in range(len(lam))]
    fig, ax = plt.subplots(1, 2, figsize=plotting.figsize("double", 0.52),
                           gridspec_kw={"width_ratios": [1, 1.15]})
    rel = lam / lam[0]
    ax[0].bar(np.arange(1, len(lam) + 1), np.maximum(rel, rel.min() * 0.5),
              color=plotting.ROLE["ggn"], width=0.62, zorder=3)
    ax[0].set_yscale("log")
    ax[0].set_xticks(np.arange(1, len(lam) + 1)); ax[0].set_xticklabels(dirs)
    ax[0].set_xlabel(r"GGN direction (stiff $\to$ sloppy)")
    ax[0].set_ylabel(r"eigenvalue $\lambda_k/\lambda_1$")
    plotting.panel_label(ax[0], "a")
    im = loadings_heatmap(ax[1], V, bh.PARAM_LABELS, dirs, highlight_last=True)
    ax[1].set_xlabel(r"GGN direction (stiff $\to$ sloppy)")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03).set_label("loading", fontsize=8)
    plotting.panel_label(ax[1], "b")
    fig.tight_layout()
    bh.save_figure(fig, NAME)
    plt.close(fig)


if __name__ == "__main__":
    compute()
    plot()
