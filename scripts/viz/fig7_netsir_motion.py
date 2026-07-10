"""Figure 7 - network-SIR: the diagnostic in motion.

Three columns (slow / moderate / fast R0), shared axes. Top (tall): log10
lambda_k(t) for all P=5 eigenvalues across calibration, rank-ordered single-hue
ramp (stiff dark -> sloppy pale) with a faint bootstrap band per line and the
bootstrap noise floor tau_t drawn as the accent line. Bottom: d_eff(t) step.
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt

import _netsirdata
from curvature_calib.viz import paper_style as ps

ps.setup()
data = _netsirdata.load()
P = _netsirdata.P
colors = ps.rank_colors(P)
FLOOR = 1e-13

R0_VALUE = {"slow": "1.3", "moderate": "2.5", "fast": "5"}

ymax = max(np.nanmax(data[r]["eigvals_traj"]) for r in _netsirdata.REGIME_ORDER)

fig = plt.figure(figsize=(ps.FULL, 4.3))
gs = fig.add_gridspec(2, 3, height_ratios=[3.0, 1.0], hspace=0.10, wspace=0.16)

for c, regime in enumerate(_netsirdata.REGIME_ORDER):
    d = data[regime]
    ev = np.clip(d["eigvals_traj"], FLOOR, None)
    lo = np.clip(d["boot_lo"], FLOOR, None)
    hi = np.clip(d["boot_hi"], FLOOR, None)
    tau = np.clip(d["tau"], FLOOR, None)
    deff = d["d_eff"]
    it = np.arange(ev.shape[0])

    ax = fig.add_subplot(gs[0, c])
    ax.fill_between(it, FLOOR, tau, color="0.5", alpha=0.16, lw=0, zorder=0)
    for k in range(P):
        ax.fill_between(it, lo[:, k], hi[:, k], color=colors[k], alpha=0.13, lw=0, zorder=1)
        ax.plot(it, ev[:, k], color=colors[k], lw=1.2, zorder=3)
    ax.plot(it, tau, color=ps.ACCENT, lw=1.4, zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR, ymax * 3)
    ax.set_xlim(it[0], it[-1])
    ax.tick_params(labelbottom=False)
    if c == 0:
        ax.set_ylabel(r"$\lambda_k(t)$")
    else:
        ax.tick_params(labelleft=False)
    ax.text(0.04, 0.96, ps.smallcaps(regime) + f"  ($R_0\\!\\approx\\!{R0_VALUE[regime]}$)",
            transform=ax.transAxes, fontsize=8.5, va="top", ha="left", color=ps.INK)

    xr = it[-1]
    ys = ev[-1].copy()
    order = np.argsort(-ys)
    log_ys = np.log10(ys)
    min_gap = 0.9
    for rank in range(1, P):
        i_cur, i_prev = order[rank], order[rank - 1]
        if log_ys[i_prev] - log_ys[i_cur] < min_gap:
            log_ys[i_cur] = log_ys[i_prev] - min_gap
    for k in range(P):
        ax.text(xr + 0.8, 10 ** log_ys[k], rf"$\lambda_{{{k + 1}}}$",
                color=colors[k], fontsize=7.5, va="center", ha="left", clip_on=False)
    ax.text(xr + 0.8, tau[-1], r"$\tau_t$", color=ps.ACCENT, fontsize=8,
            va="center", ha="left", clip_on=False)

    axd = fig.add_subplot(gs[1, c], sharex=ax)
    axd.step(it, deff, where="post", color=ps.INK, lw=1.2)
    axd.set_ylim(-0.4, P + 0.4)
    axd.set_yticks(range(0, P + 1, 1))
    axd.set_xlim(it[0], it[-1])
    axd.set_xlabel("calibration iteration")
    if c == 0:
        axd.set_ylabel(r"$d_{\mathrm{eff}}(t)$")
    else:
        axd.tick_params(labelleft=False)

ps.save(fig, "fig7_netsir_motion")
print("saved fig7_netsir_motion")
