"""Fig. 3, App. C.2: G from P forward-mode JVPs against the Hessian of the loss formed
by central finite differences of loss evaluations alone, at the fit."""
from __future__ import annotations

from gndiag.config import enable_x64, load_config

enable_x64()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gndiag import io, plotting  # noqa: E402
from gndiag.diagnostic.ggn import eigendecompose, principal_angles, symmetrize  # noqa: E402
from gndiag.models.mfsir import MeanFieldSIR  # noqa: E402

NAME = "mfsir_ggn_vs_fd_hessian"


def fd_hessian(L, z0, h):
    """Central-difference Hessian; returns (H, number of loss evaluations = 2P^2 + 1)."""
    z0 = np.asarray(z0, np.float64)
    P = len(z0)
    H = np.zeros((P, P))
    L0 = float(L(jnp.asarray(z0)))
    evals = 1
    for i in range(P):
        ei = np.zeros(P); ei[i] = h
        H[i, i] = (float(L(jnp.asarray(z0 + ei))) - 2 * L0 + float(L(jnp.asarray(z0 - ei)))) / h ** 2
        evals += 2
        for j in range(i + 1, P):
            ej = np.zeros(P); ej[j] = h
            a = float(L(jnp.asarray(z0 + ei + ej)))
            b = float(L(jnp.asarray(z0 + ei - ej)))
            c = float(L(jnp.asarray(z0 - ei + ej)))
            d = float(L(jnp.asarray(z0 - ei - ej)))
            H[i, j] = H[j, i] = (a - b - c + d) / (4 * h ** 2)
            evals += 4
    return H, evals


def select_step(L, z0, steps):
    """Compare H(h) with H(h/2) for each candidate h; keep the finer Hessian of the
    most self-consistent pair. Returns (h/2, H(h/2), relative change, evaluations)."""
    best = None
    for h in steps:
        H_h, _ = fd_hessian(L, z0, h)
        H_half, evals = fd_hessian(L, z0, h / 2)
        rel = float(np.linalg.norm(H_h - H_half) / (np.linalg.norm(H_half) + 1e-30))
        if best is None or rel < best[2]:
            best = (h / 2, H_half, rel, evals)
    return best


def compute():
    cfg = load_config(NAME)
    sir = MeanFieldSIR(cfg["model"])
    z_hat = jnp.zeros(5)
    y = sir.incidence(z_hat)
    L = jax.jit(lambda z: sir.loss(z, y))

    J = np.asarray(jax.jit(jax.jacfwd(sir.incidence))(z_hat))
    G = J.T @ J / sir.sigma_obs ** 2
    h, H, rel, evals = select_step(lambda z: float(L(z)), z_hat, cfg["fd_steps"])

    eG = eigendecompose(jnp.asarray(symmetrize(G)))
    eH = eigendecompose(jnp.asarray(symmetrize(H)))
    lam_G, lam_H = np.asarray(eG.eigvals), np.asarray(eH.eigvals)
    angles = {str(k): float(np.degrees(np.max(np.asarray(
        principal_angles(eG.eigvecs[:, :k], eH.eigvecs[:, :k]))))) for k in cfg["leading_k"]}
    results = {
        "eigvals_G": lam_G.tolist(), "eigvals_fd_hessian": lam_H.tolist(),
        "leading_angles_deg": angles,
        "rel_eigval_error": (np.abs(lam_G - lam_H) / (np.abs(lam_H) + 1e-30)).tolist(),
        "G_jvps": int(J.shape[1]), "fd_loss_evaluations": evals,
        "fd_step": h, "fd_step_rel_change": rel,
    }
    io.save_results(NAME, results, cfg)


def plot():
    res = io.load_results(NAME)
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
    io.save_figure(fig, NAME)
    plt.close(fig)


if __name__ == "__main__":
    io.main(compute, plot)
