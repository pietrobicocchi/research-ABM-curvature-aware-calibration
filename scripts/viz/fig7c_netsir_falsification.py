"""Figure 7c - Falsification (network-SIR, interventionist lockdown).

Network analog of fig6. theta* = the interventionist moderate-R0 network point
(lockdown fires early, during the epidemic). 2x3 grid: rows = stiff v_1 / sloppy
v_P, columns = moments / ACF sup-norm / tail quantiles, discrepancy vs signed
step alpha. Trimmed batch for the network model's per-sim cost.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from curvature_calib.calibration.diagnostic import eigendecompose
from curvature_calib.calibration.falsification import (
    moments_difference, acf_difference, quantile_difference,
)
from curvature_calib.calibration.per_seed_grads import per_seed_loss_and_grads, vmap_simulate
from curvature_calib.models.network_sir import simulate
from curvature_calib.viz import paper_style as ps

ps.setup()

T, N = 200, 250
# interventionist moderate-R0 network theta* (R0 = beta*6/gamma ~ 2.5)
THETA_STAR = jnp.array([0.0417, 0.10, 0.01, 0.025, 0.50], dtype=jnp.float64)
M = 48
N_BATCH = 4
ALPHAS = np.linspace(-0.06, 0.06, 9)


def _sim(theta, key):
    return simulate(theta, key, T=T, N=N, mean_degree=6.0, surrogate="gumbel")


ref_keys = jax.random.split(jax.random.PRNGKey(0), 96)
Y_ref = vmap_simulate(_sim, THETA_STAR, ref_keys)
theta_eval = THETA_STAR + jnp.array([0.004, -0.005, 0.003, 0.005, -0.02], dtype=jnp.float64)
stats = per_seed_loss_and_grads(_sim, theta_eval, jax.random.split(jax.random.PRNGKey(1), 96), Y_ref)
eig = eigendecompose(stats.opg)
v_stiff = eig.eigvecs[:, 0]
v_sloppy = eig.eigvecs[:, -1]
print("v_stiff ", np.asarray(v_stiff).round(3))
print("v_sloppy", np.asarray(v_sloppy).round(3))

CHANNELS = ["moments", "ACF", "quantiles"]


def discrepancies(X_a, X_b):
    return (
        float(np.sum(moments_difference(X_a, X_b))),
        float(acf_difference(X_a, X_b)),
        float(np.sum(quantile_difference(X_a, X_b))),
    )


DIRS = {"stiff": v_stiff, "sloppy": v_sloppy}
curves = {d: np.zeros((len(CHANNELS), N_BATCH, len(ALPHAS))) for d in DIRS}

for b in range(N_BATCH):
    keys = jax.random.split(jax.random.PRNGKey(100 + b), M)
    X_base = np.asarray(vmap_simulate(_sim, THETA_STAR, keys))
    for dname, v in DIRS.items():
        for ai, a in enumerate(ALPHAS):
            X_a = np.asarray(vmap_simulate(_sim, THETA_STAR + float(a) * v, keys))
            for ci, val in enumerate(discrepancies(X_a, X_base)):
                curves[dname][ci, b, ai] = val
    print(f"  batch {b + 1}/{N_BATCH} done", flush=True)

fig, axes = plt.subplots(2, 3, figsize=(ps.FULL, 3.6), sharex=True)
row_for = {"stiff": 0, "sloppy": 1}
row_color = {"stiff": ps.ACCENT, "sloppy": ps.INK}
ylabels = {"moments": r"$\sum_j|\Delta m_j|$",
           "ACF": r"$\|\Delta\,\mathrm{ACF}\|_\infty$",
           "quantiles": r"$\sum_q|\Delta Q_q|$"}

for ci, ch in enumerate(CHANNELS):
    ymax = max(curves[d][ci].max() for d in DIRS)
    for dname in DIRS:
        r = row_for[dname]
        ax = axes[r, ci]
        arr = curves[dname][ci]
        ax.fill_between(ALPHAS, arr.min(0), arr.max(0), color=row_color[dname], alpha=0.15, lw=0)
        ax.plot(ALPHAS, arr.mean(0), color=row_color[dname], lw=1.4)
        ps.rule(ax, x=0.0)
        ax.set_ylim(-0.03 * ymax, 1.08 * ymax)
        if ci == 0:
            tag = r"$v_1$ (stiff)" if dname == "stiff" else r"$v_P$ (sloppy)"
            ax.set_ylabel(tag + "\n" + ylabels[ch], fontsize=8)
        else:
            ax.set_ylabel(ylabels[ch], fontsize=8)
        if r == 1:
            ax.set_xlabel(r"signed step $\alpha$")
        ax.set_xlim(ALPHAS[0], ALPHAS[-1])

heads = {"moments": "moments", "ACF": "autocorrelation", "quantiles": "tail quantiles"}
for ci, ch in enumerate(CHANNELS):
    axes[0, ci].text(0.5, 1.06, ps.smallcaps(heads[ch]), transform=axes[0, ci].transAxes,
                     ha="center", va="bottom", fontsize=8.5, color=ps.INK)

fig.subplots_adjust(hspace=0.16, wspace=0.34)
ps.save(fig, "fig7c_netsir_falsification")
print("saved fig7c_netsir_falsification")
