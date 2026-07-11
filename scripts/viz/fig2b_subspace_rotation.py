"""Figure 2b - Brock-Hommes: rotation of the identifiable subspace.

Companion to fig2_bh_motion. Eigenvalue trajectories say *how stiff* each
direction is; they say nothing about whether the stiff directions are *holding
still*. Individual eigenvectors are not robust across iterates: when two
eigenvalues are close, small perturbations to F-hat_t interchange the
corresponding eigenvectors, an artefact of the decomposition rather than a real
change in identifiability geometry. The robust object is the eigenvector
subspace S_k(t) = span(v_1(t), ..., v_k(t)) (Transtrum et al., 2011).

We plot the *cumulative* reorientation of S_k away from its initial geometry,

    Theta_k(t) = largest principal angle between S_k(0) and S_k(t)
               = arccos(sigma_k),  sigma = svd(V_0^(k)^T V_t^(k)),

so Theta_k(0) = 0 for every k by construction and the curve shows how far the
identifiable subspace has drifted by iterate t (not the per-step rate, which is
largest at the start when the optimiser takes its biggest steps). Bounded in
[0, 90] deg, so a *linear* axis is the natural choice.

Three columns (stable, periodic, chaotic), shared linear axis in degrees. Each
column plots Theta_k(t) for every nested subspace k = 1 .. P-1, dark (stiff) ->
pale (sloppy). Subspaces within the certified identifiable dimension
(k <= d_eff) are drawn solid; subspaces that reach into the numerical noise
floor (k > d_eff) are drawn faint and dashed. The reading: solid lines that stay
near 0 = the identifiable subspace is locked to its initial orientation
(stable); solid lines that climb = genuine reorientation as the optimiser moves
(chaotic); faint lines that shoot up = unconstrained sloppy directions drifting
freely (stable, d_eff = 2).
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt

import _bhdata
from curvature_calib.calibration.diagnostic import principal_angles
from curvature_calib.viz import paper_style as ps

ps.setup()
data = _bhdata.load()
P = 5
KS = list(range(1, P))            # k = 1 .. P-1 = 4 nested subspaces
colors = ps.rank_colors(len(KS))  # dark (stiff) -> pale (sloppy)


def drift_series(eigvecs_traj: np.ndarray, k: int) -> np.ndarray:
    """Theta_k(t) in degrees: top-k subspace vs its initial orientation S_k(0).

    Reports the largest principal angle theta_max (Asimov / projection-2-norm
    metric), bounded at 90 deg so the linear axis stays an angle. For the top-2
    identifiable plane this equals the Grassmann geodesic distance
    sqrt(sum theta_i^2) to 2 d.p., because the smaller angle theta_1 ~ 0 (the
    plane tilts along a single direction); see FIGURE_NOTES.md "METRIC NOTE".
    """
    V0 = eigvecs_traj[0][:, :k]
    n = eigvecs_traj.shape[0]
    out = np.empty(n)
    for t in range(n):
        ang = np.asarray(principal_angles(V0, eigvecs_traj[t][:, :k]))
        out[t] = np.degrees(ang.max())
    return out


series = {r: {k: drift_series(data[r]["eigvecs_traj"], k) for k in KS}
          for r in _bhdata.REGIME_ORDER}
ymax = max(s.max() for r in series for s in series[r].values())

fig = plt.figure(figsize=(ps.FULL, 2.7))
gs = fig.add_gridspec(1, 3, wspace=0.16)

for c, regime in enumerate(_bhdata.REGIME_ORDER):
    ax = fig.add_subplot(gs[0, c])
    deff = int(data[regime]["d_eff"][-1])     # certified dimension at convergence
    it = np.arange(data[regime]["eigvecs_traj"].shape[0])   # t = 0 .. n-1
    for i, k in enumerate(KS):
        identified = k <= deff
        ax.plot(it, series[regime][k], color=colors[i],
                lw=1.2 if identified else 1.0,
                alpha=1.0 if identified else 0.5,
                ls="-" if identified else (0, (3, 2)), zorder=3)
        ax.text(it[-1] + 0.8, series[regime][k][-1], rf"$k={k}$",
                color=colors[i], fontsize=7, va="center", ha="left",
                alpha=1.0 if identified else 0.55, clip_on=False)
    ax.set_ylim(-2, ymax * 1.05)
    ax.set_xlim(it[0], it[-1])
    ax.set_xlabel("calibration iteration")
    if c == 0:
        ax.set_ylabel(r"$\Theta_k(t)\ [^\circ]$")
    else:
        ax.tick_params(labelleft=False)
    # regime + certified dimension as the quiet small-caps marker
    ax.text(0.04, 0.96,
            ps.smallcaps(regime) + rf"  ($d_{{\mathrm{{eff}}}}={deff}$)",
            transform=ax.transAxes, fontsize=8.5, va="top", ha="left",
            color=ps.INK)

# one-line key for the solid/dashed encoding (no title block)
fig.text(0.5, -0.02,
         r"solid: identifiable subspace ($k\leq d_{\mathrm{eff}}$)"
         r"      dashed: reaches the numerical noise floor ($k>d_{\mathrm{eff}}$)",
         ha="center", va="top", fontsize=7.5, color=ps.INK)

ps.save(fig, "fig2b_subspace_rotation")
print("saved fig2b_subspace_rotation")
