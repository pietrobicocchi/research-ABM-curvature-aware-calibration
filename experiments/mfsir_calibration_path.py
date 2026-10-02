"""Fig. 1: the diagnostic along the calibration path of the mean-field SIR model.

Adam descends from z_hat + 1 to the fit; at every checkpoint G = J^T W J is
recomputed and compared with its value at the final iterate (Sec. 3.3).
"""
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

NAME = "mfsir_calibration_path"


def _adam_path(loss, z0, cfg):
    vg = jax.jit(jax.value_and_grad(loss))
    b1, b2 = cfg["adam_betas"]
    eps, lr = cfg["adam_eps"], cfg["learning_rate"]
    z = jnp.asarray(z0)
    mt, vt = jnp.zeros_like(z), jnp.zeros_like(z)
    steps, zs = [0], [np.asarray(z)]
    for t in range(1, cfg["steps"] + 1):
        _, g = vg(z)
        mt = b1 * mt + (1 - b1) * g
        vt = b2 * vt + (1 - b2) * g ** 2
        z = z - lr * (mt / (1 - b1 ** t)) / (jnp.sqrt(vt / (1 - b2 ** t)) + eps)
        if t % cfg["checkpoint_every"] == 0 or t == cfg["steps"]:
            steps.append(t)
            zs.append(np.asarray(z))
    return steps, zs


def compute():
    cfg = load_config(NAME)
    sir = MeanFieldSIR(cfg["model"])
    z_hat = jnp.zeros(5)
    y = sir.incidence(z_hat)
    loss = lambda z: sir.loss(z, y)

    def ggn(z):
        J = jax.jacfwd(sir.incidence)(z)
        return symmetrize((J.T @ J) / sir.sigma_obs ** 2)

    steps, zs = _adam_path(loss, z_hat + cfg["start_offset"], cfg)
    eigs = [eigendecompose(ggn(jnp.asarray(z))) for z in zs]
    final = eigs[-1]
    checkpoints = []
    for step, z, eig in zip(steps, zs, eigs):
        lam = np.asarray(eig.eigvals)
        checkpoints.append({
            "step": int(step),
            "loss": float(loss(jnp.asarray(z))),
            "eigvals": lam.tolist(),
            "d_data": int((lam > 1).sum()),
            "angle_to_final_deg": {str(k): float(np.degrees(np.max(np.asarray(
                principal_angles(eig.eigvecs[:, :k], final.eigvecs[:, :k])))))
                for k in cfg["leading_k"]},
            "z": np.asarray(z).tolist(),
        })
    results = {"loss_final": checkpoints[-1]["loss"], "checkpoints": checkpoints}
    io.save_results(NAME, results, cfg)


def plot():
    cfg = load_config(NAME)
    res = io.load_results(NAME)
    plotting.apply_style()
    R = plotting.ROLE
    ch = res["checkpoints"]
    steps = np.array([c["step"] for c in ch])
    losses = np.array([c["loss"] for c in ch])
    eigs = np.array([c["eigvals"] for c in ch])
    d_data = np.array([c["d_data"] for c in ch])
    ks = cfg["leading_k"]

    fig, ax = plt.subplots(2, 2, figsize=plotting.figsize("double", 0.62))

    ax[0, 0].semilogy(steps, np.maximum(losses, 1e-300), "-", color=R["truth"], lw=1.7)
    ax[0, 0].set_xlabel("iteration")
    ax[0, 0].set_ylabel("loss")
    plotting.panel_label(ax[0, 0], "a")

    for j, c in enumerate(plotting.seq_colors(eigs.shape[1])):
        ax[0, 1].semilogy(steps, np.maximum(eigs[:, j], 1e-300), "-", lw=1.5, color=c,
                          label=rf"$\lambda_{{{j + 1}}}$")
    ax[0, 1].axhline(1.0, ls=plotting.LS["ref"], c=R["ref"], lw=1.2)
    ax[0, 1].text(steps[-1], 1.0, r" $\lambda=1$", va="center", ha="left", fontsize=7,
                  color=R["ref"])
    ax[0, 1].set_xlabel("iteration")
    ax[0, 1].set_ylabel(r"prior-relative $\lambda$")
    ax[0, 1].legend(fontsize=7, ncol=2)
    plotting.panel_label(ax[0, 1], "b")

    for c, k in zip(plotting.seq_colors(len(ks)), ks):
        ax[1, 0].plot(steps, [p["angle_to_final_deg"][str(k)] for p in ch], "-", lw=1.7,
                      color=c, label=f"leading {k}")
    for lvl in (10.0, 5.0):
        ax[1, 0].axhline(lvl, ls=plotting.LS["ref"], c=R["ref"], lw=1.0)
    ax[1, 0].set_xlabel("iteration")
    ax[1, 0].set_ylabel("angle to fit (deg)")
    ax[1, 0].legend()
    plotting.panel_label(ax[1, 0], "c")

    ax[1, 1].step(steps, d_data, where="post", color=R["ggn"], lw=1.7)
    ax[1, 1].set_xlabel("iteration")
    ax[1, 1].set_ylabel(r"directions with $\lambda>1$")
    ax[1, 1].set_ylim(-0.2, eigs.shape[1] + 0.2)
    plotting.panel_label(ax[1, 1], "d")

    fig.tight_layout()
    io.save_figure(fig, NAME)
    plt.close(fig)


if __name__ == "__main__":
    io.main(compute, plot)
