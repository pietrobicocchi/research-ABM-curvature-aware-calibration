"""Figure 4 - Falsification (Brock-Hommes).

2x3 grid. Rows: perturbation along v_1 (stiff) and v_P (sloppy). Columns: the
three non-MMD discrepancies (moments, ACF sup-norm, tail quantiles). Each panel
plots discrepancy vs signed step alpha, a single curve with a faint seed-
variability band. Identical y-scales within a column so the rows are directly
comparable: the entire claim is the row contrast - F_hat's sloppy direction is
invisible to discrepancies that never saw the MMD kernel.
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
from curvature_calib.models.brock_hommes import simulate
from curvature_calib.viz import paper_style as ps

ps.setup()

T, SIGMA, R = 200, 0.05, 1.1
THETA_STAR = jnp.array([3.0, 1.2, 0.2, 1.2, -0.2], dtype=jnp.float64)
M = 96
N_BATCH = 6
ALPHAS = np.linspace(-0.12, 0.12, 13)


def _sim(theta, key):
    return simulate(theta, key, T=T, sigma=SIGMA, R=R, x_init=0.0)


# OPG eigenbasis at a point just off theta* (mean gradient non-vanishing there)
ref_keys = jax.random.split(jax.random.PRNGKey(0), 128)
Y_ref = vmap_simulate(_sim, THETA_STAR, ref_keys)
theta_eval = THETA_STAR + jnp.array([0.0, 0.0, 0.05, 0.0, 0.05], dtype=jnp.float64)
stats = per_seed_loss_and_grads(_sim, theta_eval, jax.random.split(jax.random.PRNGKey(1), 128), Y_ref)
eig = eigendecompose(stats.opg)
v_stiff = eig.eigvecs[:, 0]
v_sloppy = eig.eigvecs[:, -1]
print("v_stiff", np.asarray(v_stiff).round(3))
print("v_sloppy", np.asarray(v_sloppy).round(3))

CHANNELS = ["moments", "ACF", "quantiles"]


def discrepancies(X_a, X_b):
    return (
        float(np.sum(moments_difference(X_a, X_b))),
        float(acf_difference(X_a, X_b)),
        float(np.sum(quantile_difference(X_a, X_b))),
    )


# curves[direction][channel] -> (N_BATCH, n_alpha)
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

# ---- plot --------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(ps.FULL, 3.6), sharex=True)
row_for = {"stiff": 0, "sloppy": 1}
row_color = {"stiff": ps.ACCENT, "sloppy": ps.INK}
ylabels = {"moments": r"$\sum_j|\Delta m_j|$",
           "ACF": r"$\|\Delta\,\mathrm{ACF}\|_\infty$",
           "quantiles": r"$\sum_q|\Delta Q_q|$"}

for ci, ch in enumerate(CHANNELS):
    ymax = 0.0
    for dname in DIRS:
        ymax = max(ymax, curves[dname][ci].max())
    for dname in DIRS:
        r = row_for[dname]
        ax = axes[r, ci]
        arr = curves[dname][ci]
        mean = arr.mean(0)
        lo, hi = arr.min(0), arr.max(0)
        ax.fill_between(ALPHAS, lo, hi, color=row_color[dname], alpha=0.15, lw=0)
        ax.plot(ALPHAS, mean, color=row_color[dname], lw=1.4)
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

# column headers (channel identity) - content, not a figure title
heads = {"moments": "moments", "ACF": "autocorrelation",
         "quantiles": "tail quantiles"}
for ci, ch in enumerate(CHANNELS):
    axes[0, ci].text(0.5, 1.06, ps.smallcaps(heads[ch]), transform=axes[0, ci].transAxes,
                     ha="center", va="bottom", fontsize=8.5, color=ps.INK)

fig.subplots_adjust(hspace=0.16, wspace=0.32)
ps.save(fig, "fig4_falsification")
print("saved fig4_falsification")
