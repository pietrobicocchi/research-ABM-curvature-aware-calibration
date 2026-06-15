"""Figure 5 - State-of-the-art contrast (Network-SIR).

Two panels, same parameter y-axis.
  Left : the per-parameter first-order Jacobian sensitivity of Quera-Bofarull
         et al. (2025, sec 5.4) - magnitude per *individual* parameter, as a
         horizontal bar chart. "All of these matter."
  Right: the F_hat eigenvector |V| heatmap for the same model. The sloppy
         (rightmost) eigenvector shows that a particular *combination* of the
         individually-influential parameters is unconstrained.

A connecting annotation links the per-parameter view to the eigenstructure.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from curvature_calib.calibration.diagnostic import eigendecompose
from curvature_calib.calibration.jacobian_sensitivity import per_param_jacobian_sensitivity
from curvature_calib.calibration.per_seed_grads import per_seed_loss_and_grads, vmap_simulate
from curvature_calib.models.network_sir import simulate as net_simulate
from curvature_calib.viz import paper_style as ps

ps.setup()

T, N_NODES, MEAN_DEG, GUMBEL_TAU = 200, 250, 6.0, 0.5
# Interventionist lockdown: R0~2.4 epidemic (peak ~day17), lockdown fires day~5
# so it bends the curve (~57%) and the sloppy direction is a real lockdown
# timing-vs-strength combination rather than an inert coordinate.
THETA_STAR = jnp.array([0.04, 0.10, 0.01, 0.025, 0.50], dtype=jnp.float64)
NAMES = [r"$\beta$", r"$\gamma$", r"$I_0$", r"$t_{\mathrm{lock}}$", r"$f_{\mathrm{lock}}$"]
P = 5


def _sim(theta, key):
    return net_simulate(theta, key, T=T, N=N_NODES, mean_degree=MEAN_DEG,
                        gumbel_tau=GUMBEL_TAU, grad_horizon=None)


# left: per-parameter Jacobian sensitivity (Quera-Bofarull style)
print("per-parameter Jacobian sensitivity ...", flush=True)
jac_keys = jax.random.split(jax.random.PRNGKey(42), 24)
sens = np.asarray(per_param_jacobian_sensitivity(_sim, THETA_STAR, jac_keys))
sens_norm = sens / sens.max()

# right: OPG eigenvector structure at a point just off theta*
print("OPG eigendecomposition ...", flush=True)
ref_keys = jax.random.split(jax.random.PRNGKey(0), 96)
Y_ref = vmap_simulate(_sim, THETA_STAR, ref_keys)
theta_eval = THETA_STAR + jnp.array([0.003, -0.003, 0.002, 0.005, -0.02], dtype=jnp.float64)
stats = per_seed_loss_and_grads(_sim, theta_eval, jax.random.split(jax.random.PRNGKey(1), 96), Y_ref)
eig = eigendecompose(stats.opg)
eigvals = np.asarray(eig.eigvals)
V = np.abs(np.asarray(eig.eigvecs))
print("eigvals", eigvals)
print("v_sloppy", V[:, -1].round(3))

# ---- plot --------------------------------------------------------------------
fig = plt.figure(figsize=(ps.FULL, 2.7))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.45, 0.06], wspace=0.10)

# rows ordered top->bottom = param 0..P-1; y position i for param i, with y=0 top
ypos = np.arange(P)

# Left: horizontal bars (individual sensitivity)
axL = fig.add_subplot(gs[0, 0])
axL.barh(ypos, sens_norm, color="#1f4e79", alpha=0.85, height=0.62,
         edgecolor="white", lw=0.5)
axL.set_yticks(ypos); axL.set_yticklabels(NAMES)
axL.set_ylim(P - 0.5, -0.5)             # param 0 at top, aligned with heatmap
axL.invert_xaxis()                       # bars grow leftward, toward labels-right
axL.set_xlabel(r"per-parameter $\|\partial x/\partial\theta_k\|$" + "\n(normalised)",
               fontsize=8)
axL.spines["left"].set_visible(False)
axL.tick_params(left=False)

# Right: |V| heatmap
axR = fig.add_subplot(gs[0, 1])
im = axR.imshow(V, cmap=ps.SEQ, vmin=0, vmax=1, aspect="auto")
axR.set_xticks(range(P)); axR.set_xticklabels([rf"$v_{{{k+1}}}$" for k in range(P)])
axR.set_yticks(range(P)); axR.set_yticklabels([])
axR.set_xlabel("eigenvector (stiff " + r"$\rightarrow$" + " sloppy)")
for i in range(P):
    for j in range(P):
        axR.text(j, i, f"{V[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                 color="white" if V[i, j] > 0.5 else ps.INK)
# OPG-only insight: beta, I0, gamma are cleanly identified one-by-one (v1-v3 are
# near axis-aligned, so the per-parameter view is fine for them). The lockdown
# parameters are the opposite: t_lock and f_lock are entangled across v4-v5, so
# only a *combination* (effective lockdown = timing x strength) is constrained -
# precisely what the per-parameter bar chart cannot express.
axR.add_patch(plt.Rectangle((-0.5, -0.5), 3.0, P, fill=False,
                            edgecolor=ps.INK, lw=1.0, ls=(0, (3, 2)), zorder=5))
axR.text(1.0, -0.72, "identified one-by-one", color=ps.INK, fontsize=7.5,
         ha="center", va="bottom")
axR.add_patch(plt.Rectangle((P - 2.5, -0.5), 2.0, P, fill=False,
                            edgecolor=ps.ACCENT, lw=1.6, zorder=5))
axR.text(P - 1.5, -0.72, r"lockdown $t,f$: combination only", color=ps.ACCENT,
         fontsize=7.5, ha="center", va="bottom")
span = np.log10(eigvals[0] / max(eigvals[-1], 1e-30))
axR.text(0.99, -0.18, rf"$\lambda_1/\lambda_P \sim 10^{{{span:.1f}}}$",
         transform=axR.transAxes, ha="right", va="bottom", fontsize=7.5,
         color=ps.INK)

cax = fig.add_subplot(gs[0, 2])
cb = fig.colorbar(im, cax=cax)
cb.set_label(r"$|V_{jk}|$", fontsize=8)
cb.ax.tick_params(labelsize=7)

ps.save(fig, "fig5_sota_contrast")
print("saved fig5_sota_contrast")
