"""Figure 7b - network-SIR: rotation of the identifiable subspace.

Companion to fig7. Plots the cumulative reorientation of the top-k eigenvector
subspace away from its initial geometry,
    Theta_k(t) = largest principal angle between S_k(0) and S_k(t),
so Theta_k(0)=0 for every k. Three columns (slow/moderate/fast R0), linear axis
in degrees. Nested k=1..P-1; solid = identifiable (k<=d_eff), dashed-faint =
into the numerical noise floor (k>d_eff).
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt

import _netsirdata
from curvature_calib.calibration.diagnostic import principal_angles
from curvature_calib.viz import paper_style as ps

ps.setup()
data = _netsirdata.load()
P = _netsirdata.P
KS = list(range(1, P))
colors = ps.rank_colors(len(KS))

R0_LABEL = {"slow": "slow", "moderate": "moderate", "fast": "fast"}


def drift_series(eigvecs_traj: np.ndarray, k: int) -> np.ndarray:
    V0 = eigvecs_traj[0][:, :k]
    n = eigvecs_traj.shape[0]
    out = np.empty(n)
    for t in range(n):
        ang = np.asarray(principal_angles(V0, eigvecs_traj[t][:, :k]))
        out[t] = np.degrees(ang.max())
    return out


series = {r: {k: drift_series(data[r]["eigvecs_traj"], k) for k in KS}
          for r in _netsirdata.REGIME_ORDER}
ymax = max(s.max() for r in series for s in series[r].values())

fig = plt.figure(figsize=(ps.FULL, 2.7))
gs = fig.add_gridspec(1, 3, wspace=0.16)

for c, regime in enumerate(_netsirdata.REGIME_ORDER):
    ax = fig.add_subplot(gs[0, c])
    deff = int(data[regime]["d_eff"][-1])
    it = np.arange(data[regime]["eigvecs_traj"].shape[0])
    for i, k in enumerate(KS):
        identified = k <= deff
        ax.plot(it, series[regime][k], color=colors[i],
                lw=1.2 if identified else 1.0,
                alpha=1.0 if identified else 0.5,
                ls="-" if identified else (0, (3, 2)), zorder=3)
    # right-margin k labels, nudged apart in linear space so curves that
    # converge to nearly the same angle keep legible, separated labels
    ys = np.array([series[regime][k][-1] for k in KS], float)
    order = np.argsort(-ys)
    y_lab = ys.copy()
    min_gap = 0.05 * ymax
    for rank in range(1, len(KS)):
        cur, prev = order[rank], order[rank - 1]
        if y_lab[prev] - y_lab[cur] < min_gap:
            y_lab[cur] = y_lab[prev] - min_gap
    for i, k in enumerate(KS):
        identified = k <= deff
        ax.text(it[-1] + 0.8, y_lab[i], rf"$k={k}$",
                color=colors[i], fontsize=7, va="center", ha="left",
                alpha=1.0 if identified else 0.55, clip_on=False)
    ax.set_ylim(-2, ymax * 1.05)
    ax.set_xlim(it[0], it[-1])
    ax.set_xlabel("calibration iteration")
    if c == 0:
        ax.set_ylabel(r"$\Theta_k(t)\ [^\circ]$")
    else:
        ax.tick_params(labelleft=False)
    ax.text(0.04, 0.96,
            ps.smallcaps(R0_LABEL[regime]) + rf"  ($d_{{\mathrm{{eff}}}}={deff}$)",
            transform=ax.transAxes, fontsize=8.5, va="top", ha="left", color=ps.INK)

fig.text(0.5, -0.02,
         r"solid: identifiable subspace ($k\leq d_{\mathrm{eff}}$)"
         r"      dashed: reaches the numerical noise floor ($k>d_{\mathrm{eff}}$)",
         ha="center", va="top", fontsize=7.5, color=ps.INK)

ps.save(fig, "fig7b_netsir_subspace_rotation")
print("saved fig7b_netsir_subspace_rotation")
