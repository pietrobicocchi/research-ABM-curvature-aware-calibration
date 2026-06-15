"""Appendix B (i) - Gradient-horizon sensitivity sweep.

log10 lambda_k of the OPG matrix as a function of the gradient horizon
H in {5, 10, 20, 40, T}, for Brock-Hommes (left) and mean-field SIR (right).
Short horizons underestimate the spectrum because truncated gradients miss
long-timescale sensitivity; the structure converges to the full-T value by
H ~ 20-40. One rank-ordered line per eigenvalue (stiff dark -> sloppy pale).
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from curvature_calib.calibration.diagnostic import eigendecompose
from curvature_calib.calibration.per_seed_grads import per_seed_loss_and_grads, vmap_simulate
from curvature_calib.models.brock_hommes import simulate as bh_sim
from curvature_calib.models.sir import simulate as sir_sim
from curvature_calib.viz import paper_style as ps

ps.setup()

T = 200
HS = [5, 10, 20, 40, 200]
P = 5
colors = ps.rank_colors(P)

MODELS = {
    "Brock-Hommes": dict(
        theta=jnp.array([3.0, 1.2, 0.2, 1.2, -0.2], dtype=jnp.float64),
        delta=jnp.array([0.0, 0.0, 0.05, 0.0, 0.05], dtype=jnp.float64),
        sim=lambda th, k, H: bh_sim(th, k, T=T, sigma=0.05, R=1.1, grad_horizon=H),
    ),
    "mean-field SIR": dict(
        theta=jnp.array([0.40, 0.10, 1e-3, 0.08, 0.50], dtype=jnp.float64),
        delta=jnp.array([0.01, -0.005, 1e-4, 0.005, -0.02], dtype=jnp.float64),
        sim=lambda th, k, H: sir_sim(th, k, T=T, N=1e5, sigma_obs=10.0, grad_horizon=H),
    ),
}


def spectrum(simH, theta_star, theta_eval):
    ref_keys = jax.random.split(jax.random.PRNGKey(0), 96)
    Y_ref = vmap_simulate(lambda t, k: simH(t, k), theta_star, ref_keys)
    keys = jax.random.split(jax.random.PRNGKey(1), 96)
    stats = per_seed_loss_and_grads(lambda t, k: simH(t, k), theta_eval, keys, Y_ref)
    return np.asarray(eigendecompose(stats.opg).eigvals)


fig, axes = plt.subplots(1, 2, figsize=(ps.FULL, 2.7))
for ax, (mname, cfg) in zip(axes, MODELS.items()):
    theta_star = cfg["theta"]
    theta_eval = cfg["theta"] + cfg["delta"]
    spec = np.zeros((len(HS), P))
    for i, H in enumerate(HS):
        simH = lambda t, k, H=H: cfg["sim"](t, k, H)
        # reference uses full gradient horizon (forward pass identical anyway)
        spec[i] = spectrum(simH, theta_star, theta_eval)
        print(f"  {mname} H={H} done", flush=True)
    spec = np.clip(spec, 1e-30, None)
    for k in range(P):
        ax.plot(HS, spec[:, k], "o-", color=colors[k], lw=1.2, ms=3.5)
        ps.direct_label(ax, HS[-1] * 1.05, spec[-1, k], rf"$\lambda_{{{k+1}}}$",
                        colors[k])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(HS); ax.set_xticklabels([str(h) for h in HS])
    ax.set_xlabel(r"gradient horizon $H$")
    ax.text(0.04, 0.04, mname, transform=ax.transAxes, fontsize=8.5,
            va="bottom", ha="left", color=ps.INK, style="italic")
    ax.minorticks_off()
axes[0].set_ylabel(r"$\lambda_k(H)$")

fig.subplots_adjust(wspace=0.22)
ps.save(fig, "figB1_horizon")
print("saved figB1_horizon")
