"""Brock-Hommes at beta = 50: G (full differentiation horizon) against a central
finite-difference Hessian of the MMD loss, at the fit, with common random numbers.

Run from the repository root:  python extras/brock_hommes/bh_ggn_vs_fd_hessian.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gndiag.config import enable_x64, load_config  # noqa: E402

enable_x64()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import bh_common as bh  # noqa: E402
from experiments.mfsir_ggn_vs_fd_hessian import select_step  # noqa: E402
from gndiag import plotting  # noqa: E402
from gndiag.diagnostic.ggn import eigendecompose, principal_angles, symmetrize  # noqa: E402

NAME = "bh_ggn_vs_fd_hessian"


def compute():
    theta0 = bh.POINTS["complex"]
    keys, features, _ = bh.setup(theta0)
    eta = bh.embedding(theta0, features, keys, None)
    mu_y = eta(bh.Z0)
    L = jax.jit(lambda z: 0.5 * jnp.sum((eta(z) - mu_y) ** 2))
    G = np.asarray(jax.jit(lambda z: bh.ggn(eta, z)[0])(bh.Z0))
    steps = load_config("mfsir_ggn_vs_fd_hessian")["fd_steps"]
    h, H, rel, evals = select_step(lambda z: float(L(z)), bh.Z0, steps)
    eG = eigendecompose(jnp.asarray(symmetrize(G)))
    eH = eigendecompose(jnp.asarray(symmetrize(H)))
    lam_G, lam_H = np.asarray(eG.eigvals), np.asarray(eH.eigvals)
    bh.save(NAME, {
        "eigvals_G": lam_G.tolist(), "eigvals_fd_hessian": lam_H.tolist(),
        "leading_angles_deg": {str(k): float(np.degrees(np.max(np.asarray(
            principal_angles(eG.eigvecs[:, :k], eH.eigvecs[:, :k]))))) for k in (1, 2, 3)},
        "rel_eigval_error": (np.abs(lam_G - lam_H) / (np.abs(lam_H) + 1e-30)).tolist(),
        "fd_step": h, "fd_step_rel_change": rel, "fd_loss_evaluations": evals})


def plot():
    res = bh.load(NAME)
    plotting.apply_style()
    lg, lh = np.abs(res["eigvals_G"]), np.abs(res["eigvals_fd_hessian"])
    lo, hi = min(lg.min(), lh.min()) * 0.5, max(lg.max(), lh.max()) * 2
    fig, ax = plt.subplots(1, 1, figsize=plotting.figsize(4.0, 0.86))
    ax.plot([lo, hi], [lo, hi], ls=":", c="0.6", lw=1)
    ax.loglog(lh, lg, "o", c=plotting.ROLE["ggn"], ms=8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("finite-difference Hessian eigenvalue")
    ax.set_ylabel("Gauss–Newton (GGN) eigenvalue")
    fig.tight_layout()
    bh.save_figure(fig, NAME)
    plt.close(fig)


if __name__ == "__main__":
    compute()
    plot()
