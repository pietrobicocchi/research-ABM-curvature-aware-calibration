"""Appendix B (ii) - Surrogate-gradient robustness (network-SIR).

Left : the OPG spectrum under the two surrogate estimators (Gumbel-Sigmoid and
       straight-through Bernoulli) overlaid.
Right: principal angles between the top-k OPG eigen-subspaces under the two
       surrogates, as a function of k. Small angles on the stiff subspace
       (robust); growing angles into the sloppy tail (surrogate-dependent) -
       the diagnostic's identifiable findings are surrogate-invariant, its
       sloppiest-direction claims inherit the estimator's bias.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from curvature_calib.calibration.diagnostic import eigendecompose
from curvature_calib.calibration.per_seed_grads import per_seed_loss_and_grads, vmap_simulate
from curvature_calib.models.network_sir import simulate as net_simulate
from curvature_calib.viz import paper_style as ps

ps.setup()

T, N_NODES, MEAN_DEG = 200, 250, 6.0
# interventionist lockdown (matches fig5): R0~2.4, lockdown fires day~5
THETA_STAR = jnp.array([0.04, 0.10, 0.01, 0.025, 0.50], dtype=jnp.float64)
THETA_EVAL = THETA_STAR + jnp.array([0.003, -0.003, 0.002, 0.005, -0.02], dtype=jnp.float64)
P = 5


def make_sim(surrogate):
    def _sim(theta, key):
        return net_simulate(theta, key, T=T, N=N_NODES, mean_degree=MEAN_DEG,
                            gumbel_tau=0.5, surrogate=surrogate, grad_horizon=None)
    return _sim


def eig_for(surrogate):
    _sim = make_sim(surrogate)
    ref_keys = jax.random.split(jax.random.PRNGKey(0), 96)
    Y_ref = vmap_simulate(_sim, THETA_STAR, ref_keys)
    keys = jax.random.split(jax.random.PRNGKey(1), 96)
    stats = per_seed_loss_and_grads(_sim, THETA_EVAL, keys, Y_ref)
    return eigendecompose(stats.opg)


print("Gumbel ...", flush=True)
eg = eig_for("gumbel")
print("straight-through ...", flush=True)
es = eig_for("straight_through")

lam_g = np.clip(np.asarray(eg.eigvals), 1e-30, None)
lam_s = np.clip(np.asarray(es.eigvals), 1e-30, None)
Vg, Vs = np.asarray(eg.eigvecs), np.asarray(es.eigvecs)
# per-eigenvector angle between the two surrogates: stiff directions should be
# robust (small angle), the sloppy tail surrogate-dependent (large angle).
per_angle = [float(np.degrees(np.arccos(
    np.clip(abs(np.dot(Vg[:, k], Vs[:, k])), 0.0, 1.0)))) for k in range(P)]

fig, axes = plt.subplots(1, 2, figsize=(ps.FULL, 2.7))

# left: spectra overlaid
ax = axes[0]
xs = np.arange(P)
ax.semilogy(xs, lam_g, "o-", color=ps.ACCENT, lw=1.3, ms=4)
ax.semilogy(xs, lam_s, "s--", color=ps.INK, lw=1.3, ms=4)
ax.text(2.4, 3e2, "straight-through", color=ps.INK, fontsize=7.5, ha="left", va="center")
ax.text(3.1, 7e-3, "Gumbel", color=ps.ACCENT, fontsize=7.5, ha="left", va="center")
ax.set_xticks(xs); ax.set_xticklabels([rf"$v_{{{k+1}}}$" for k in range(P)])
ax.set_xlabel("eigendirection (stiff " + r"$\rightarrow$" + " sloppy)")
ax.set_ylabel(r"$\lambda_k$")

# right: per-eigenvector angle between surrogates (stiff -> sloppy)
ax = axes[1]
xs2 = np.arange(P)
ax.plot(xs2, per_angle, "o-", color=ps.ACCENT, lw=1.4, ms=4)
ps.rule(ax, y=90)
ax.text(0.0, 91, "orthogonal", fontsize=7, color=ps.MUTED, va="bottom")
ax.set_ylim(0, 95)
ax.set_xticks(xs2); ax.set_xticklabels([rf"$v_{{{k+1}}}$" for k in range(P)])
ax.set_xlabel("eigendirection (stiff " + r"$\rightarrow$" + " sloppy)")
ax.set_ylabel(r"angle $\angle(v_k^{\mathrm{Gum}},v_k^{\mathrm{ST}})$ (deg)")

fig.subplots_adjust(wspace=0.3)
ps.save(fig, "figB2_surrogate")
print("saved figB2_surrogate")
