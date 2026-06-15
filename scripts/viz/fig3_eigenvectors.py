"""Figure 3 - Eigenvector structure at convergence (Brock-Hommes).

Three |V_T| heatmaps (stable, periodic, chaotic): parameters on the y-axis,
eigenvector index ordered stiff -> sloppy on the x-axis, cell = |V_{jk}|.
A slim companion strip above each heatmap encodes log10 lambda_k so stiffness
is read off alongside structure. Single-hue white -> deep ramp; no diverging
map (the quantity |V| has no meaningful midpoint).
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import _bhdata
from curvature_calib.viz import paper_style as ps

ps.setup()
data = _bhdata.load()
P = 5
names = _bhdata.PARAM_NAMES
xt = [rf"$v_{{{k + 1}}}$" for k in range(P)]

# shared log-lambda colour scale for the spectrum strips
all_log = np.concatenate([np.log10(np.clip(data[r]["eigvals_T"], 1e-30, None))
                          for r in _bhdata.REGIME_ORDER])
lnorm = Normalize(vmin=all_log.min(), vmax=all_log.max())
# single-hue sequential, dark = large lambda = stiff -> same "dark = more"
# convention as the |V| heatmap below (no contradictory meaning for dark).
spec_cmap = plt.colormaps["Greys"]

fig = plt.figure(figsize=(ps.FULL, 2.95))
gs = fig.add_gridspec(2, 4, height_ratios=[0.5, 5.0],
                      width_ratios=[1, 1, 1, 0.06],
                      hspace=0.07, wspace=0.16)

im = None
for c, regime in enumerate(_bhdata.REGIME_ORDER):
    d = data[regime]
    V = np.abs(d["V_T"])
    lam = np.clip(d["eigvals_T"], 1e-30, None)
    loglam = np.log10(lam)

    # spectrum strip
    axs = fig.add_subplot(gs[0, c])
    axs.imshow(loglam[None, :], cmap=spec_cmap, norm=lnorm, aspect="auto")
    for k in range(P):
        axs.text(k, 0, f"{loglam[k]:.0f}", ha="center", va="center", fontsize=6.5,
                 color="white" if lnorm(loglam[k]) > 0.55 else "black")
    axs.set_xticks([]); axs.set_yticks([])
    axs.text(0.04, 1.5, ps.smallcaps(regime), transform=axs.transAxes,
             fontsize=9, va="bottom", ha="left", color=ps.INK)
    if c == 0:
        axs.set_ylabel(r"$\log_{10}\lambda_k$", rotation=0, ha="right",
                       va="center", fontsize=7.5, labelpad=2)

    # |V| heatmap
    ax = fig.add_subplot(gs[1, c])
    im = ax.imshow(V, cmap=ps.SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(P)); ax.set_xticklabels(xt)
    if c == 0:
        ax.set_yticks(range(P)); ax.set_yticklabels(names)
    else:
        ax.set_yticks(range(P)); ax.set_yticklabels([])
    ax.set_xlabel("eigenvector (stiff " + r"$\rightarrow$" + " sloppy)")
    for i in range(P):
        for j in range(P):
            ax.text(j, i, f"{V[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if V[i, j] > 0.5 else ps.INK)

cax = fig.add_subplot(gs[1, 3])
cb = fig.colorbar(im, cax=cax)
cb.set_label(r"$|V_{jk}|$", fontsize=8)
cb.ax.tick_params(labelsize=7)

ps.save(fig, "fig3_eigenvectors")
print("saved fig3_eigenvectors")
