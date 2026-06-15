"""Figure 1 - Method schematic (designed, not measured).

A left-to-right flow in four stations on one 2-D parameter slice, matching the
analytic two-parameter example of section 3.2 exactly: the per-seed gradient
cloud collapses onto the (1,1) direction (stiff / constrained), the null
direction is (1,-1) (sloppy / free).

    (i)  gradient cloud {g_m} as arrows from theta_t (anisotropic)
    (ii) outer-product accumulation -> 1-sigma OPG ellipse over the cloud
    (iii) eigendecomposition: principal axes v_1, v_P with length ~ sqrt(lambda)
    (iv) the reading: v_1 stiff, v_P sloppy; loss contours elongated along v_P
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

from curvature_calib.viz import paper_style as ps

ps.setup()

# ---- synthetic gradient cloud aligned with (1,1) -----------------------------
rng = np.random.default_rng(7)
M = 44
u_stiff = np.array([1.0, 1.0]) / np.sqrt(2)
u_sloppy = np.array([1.0, -1.0]) / np.sqrt(2)
a = rng.normal(0.0, 0.46, M)    # large spread along stiff axis
b = rng.normal(0.0, 0.055, M)   # tiny spread along sloppy axis
G = a[:, None] * u_stiff + b[:, None] * u_sloppy     # (M, 2)

F = (G.T @ G) / M                                    # OPG
w, V = np.linalg.eigh(F)
order = np.argsort(-w)
w, V = w[order], V[:, order]
# canonical sign so v_1 points into the first quadrant
if V[0, 0] < 0:
    V[:, 0] *= -1
if V[0, 1] > 0:
    V[:, 1] *= -1

LIM = 1.15


def _frame(ax):
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    # faint origin crosshair at theta_t
    ax.plot(0, 0, marker="+", ms=6, mew=1.0, color=ps.INK, zorder=5)


def ellipse(ax, n_std, **kw):
    ang = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(w, 0))
    ax.add_patch(mpatches.Ellipse((0, 0), width, height, angle=ang, **kw))


# build manually for tight control of inter-panel arrows
fig = plt.figure(figsize=(ps.FULL, 1.95))
axes = [fig.add_axes([0.01 + i * 0.252, 0.10, 0.205, 0.80]) for i in range(4)]

# ---- (i) gradient cloud ------------------------------------------------------
ax = axes[0]
_frame(ax)
for g in G:
    ax.add_patch(FancyArrowPatch((0, 0), (g[0], g[1]),
                 arrowstyle="-|>", mutation_scale=5,
                 lw=0.6, color=ps.MUTED, alpha=0.75, zorder=2))
ax.text(0.0, -LIM - 0.06, r"$\{g_m\}$ from $\theta_t$",
        ha="center", va="top", fontsize=8, color=ps.INK)

# ---- (ii) OPG accumulation -> ellipse ---------------------------------------
ax = axes[1]
_frame(ax)
ax.scatter(G[:, 0], G[:, 1], s=6, color=ps.MUTED, alpha=0.7, zorder=2,
           edgecolors="none")
ellipse(ax, 1.0, facecolor="#1f4e79", alpha=0.16, edgecolor="#1f4e79", lw=1.0,
        zorder=3)
ax.text(0.0, -LIM - 0.06,
        r"$\hat{F}=\frac{1}{M}\sum_m g_m g_m^{\top}$",
        ha="center", va="top", fontsize=8, color=ps.INK)

# ---- (iii) eigendecomposition ------------------------------------------------
ax = axes[2]
_frame(ax)
ellipse(ax, 1.0, facecolor="none", edgecolor="#1f4e79", lw=1.0, alpha=0.7,
        zorder=2)
L1 = np.sqrt(w[0]); LP = np.sqrt(w[1])
v1 = V[:, 0] * L1 * 2.0
vP = V[:, 1] * max(LP * 2.0, 0.30)
ax.add_patch(FancyArrowPatch((0, 0), tuple(v1), arrowstyle="-|>",
             mutation_scale=9, lw=1.6, color=ps.ACCENT, zorder=4))
ax.add_patch(FancyArrowPatch((0, 0), tuple(vP), arrowstyle="-|>",
             mutation_scale=8, lw=1.2, color=ps.INK, zorder=4))
ax.text(v1[0] * 1.02 + 0.04, v1[1] * 1.02, r"$v_1\;(\sqrt{\lambda_1})$",
        color=ps.ACCENT, fontsize=7.5, ha="left", va="center")
ax.text(-0.55, -0.78, r"$v_P\;(\sqrt{\lambda_P})$",
        color=ps.INK, fontsize=7.5, ha="center", va="center")
ax.text(0.0, -LIM - 0.06, r"$\hat{F}=V\Lambda V^{\top}$",
        ha="center", va="top", fontsize=8, color=ps.INK)

# ---- (iv) the reading --------------------------------------------------------
ax = axes[3]
_frame(ax)
# loss contours: long axis along the sloppy direction v_P
ang_sloppy = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
for r in (0.38, 0.74, 1.08):
    ax.add_patch(mpatches.Ellipse((0, 0), 2 * r, 2 * r * 0.26,
                 angle=ang_sloppy, facecolor="none", edgecolor=ps.MUTED,
                 lw=0.7, alpha=0.8, zorder=1))
ax.add_patch(FancyArrowPatch((0, 0), tuple(V[:, 0] * 0.92), arrowstyle="-|>",
             mutation_scale=9, lw=1.6, color=ps.ACCENT, zorder=4))
ax.add_patch(FancyArrowPatch((0, 0), tuple(V[:, 1] * 0.92), arrowstyle="-|>",
             mutation_scale=8, lw=1.2, color=ps.INK, zorder=4))
ax.text(V[0, 0] * 0.96 + 0.02, V[1, 0] * 0.96 + 0.08,
        "stiff\nconstrained",
        color=ps.ACCENT, fontsize=7.5, ha="center", va="bottom")
ax.text(V[0, 1] * 1.16, V[1, 1] * 1.16, "sloppy\nfree",
        color=ps.INK, fontsize=7.5, ha="center", va="center")
ax.text(0.0, -LIM - 0.06, r"loss flat along $v_P$",
        ha="center", va="top", fontsize=8, color=ps.INK)

# ---- inter-station flow arrows ----------------------------------------------
for i in range(3):
    x = 0.222 + i * 0.252
    fig.add_artist(FancyArrowPatch((x, 0.5), (x + 0.028, 0.5),
                   transform=fig.transFigure, arrowstyle="-|>",
                   mutation_scale=10, lw=1.0, color=ps.INK))

ps.save(fig, "fig1_schematic")
print("saved fig1_schematic")
