"""Fig. 2, App. C.1: the Laplace posterior N(z_hat, (wG + I)^-1) (Eq. 10) against the
generalized posterior sampled by self-normalized importance sampling."""
from __future__ import annotations

from gndiag.config import enable_x64, load_config

enable_x64()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from gndiag import io, plotting  # noqa: E402
from gndiag.diagnostic.ggn import (  # noqa: E402
    directional_loss_change, eigendecompose, principal_angles, rel_frobenius_error, symmetrize)
from gndiag.models.mfsir import PARAM_NAMES, MeanFieldSIR  # noqa: E402

NAME = "mfsir_laplace_vs_posterior"


def _importance_sampling(log_target, P_lap, key, n, inflate):
    """Self-normalized IS with proposal N(0, inflate (G + I)^-1)."""
    Sig = jnp.linalg.inv(P_lap)
    Lc = jnp.linalg.cholesky(Sig) * jnp.sqrt(inflate)
    z = (Lc @ jax.random.normal(key, (5, n))).T
    logpost = jax.vmap(log_target)(z)
    logprop = jax.vmap(lambda zz: -0.5 * (zz @ (P_lap @ zz)) / inflate)(z)
    lw = logpost - logprop
    w = jnp.exp(lw - jnp.max(lw))
    w = w / jnp.sum(w)
    mean = w @ z
    dz = z - mean
    cov = (dz * w[:, None]).T @ dz
    ess_frac = float(1.0 / jnp.sum(w ** 2) / n)
    return np.asarray(cov), ess_frac, np.asarray(z), np.asarray(w)


def _weighted_shape(x, w):
    mu = np.sum(w * x)
    u = (x - mu) / (np.sqrt(np.sum(w * (x - mu) ** 2)) + 1e-30)
    return float(np.sum(w * u ** 3)), float(np.sum(w * u ** 4) - 3.0)


def compute():
    cfg = load_config(NAME)
    sir = MeanFieldSIR(cfg["model"])
    z_hat = jnp.zeros(5)
    y = sir.incidence(z_hat)
    loss = lambda z: sir.loss(z, y)

    J = jax.jacfwd(sir.incidence)(z_hat)
    G = jnp.asarray(symmetrize((J.T @ J) / sir.sigma_obs ** 2))
    H = jnp.asarray(symmetrize(jax.hessian(loss)(z_hat)))
    eig = eigendecompose(G)
    lam = np.asarray(eig.eigvals)

    P_lap = sir.w * G + jnp.eye(5)
    Sig_lap = np.asarray(jnp.linalg.inv(P_lap))
    Sig_ref, ess_frac, z_is, w_is = _importance_sampling(
        lambda z: -sir.w * loss(z) - 0.5 * jnp.sum(z ** 2), P_lap,
        jax.random.PRNGKey(cfg["is_key"]), cfg["is_draws"], cfg["is_inflation"])

    eig_l = eigendecompose(jnp.asarray(Sig_lap))
    eig_r = eigendecompose(jnp.asarray(Sig_ref))
    axis_angles = [float(np.degrees(np.max(np.asarray(
        principal_angles(eig_l.eigvecs[:, :k], eig_r.eigvecs[:, :k]))))) for k in (1, 2)]
    marginals = {}
    for i, p in enumerate(PARAM_NAMES):
        sd_ref, sd_lap = np.sqrt(Sig_ref[i, i]), np.sqrt(Sig_lap[i, i])
        skew, kurt = _weighted_shape(z_is[:, i], w_is)
        marginals[p] = {"sd_reference": float(sd_ref), "sd_laplace": float(sd_lap),
                        "sd_rel_error": float(abs(sd_ref - sd_lap) / sd_ref),
                        "skew": skew, "excess_kurtosis": kurt}

    # profiled posterior energy U(alpha) = w L(z_hat + alpha v) + alpha^2 / 2 against
    # its quadratic prediction; alpha in units of the posterior s.d. along v
    profiles = {}
    alphas = np.asarray(cfg["alphas"])
    for label, k in (("stiffest", 0), ("sloppiest", 4)):
        v, lam_k = eig.eigvecs[:, k], float(eig.eigvals[k])
        a = alphas * (1.0 / np.sqrt(sir.w * lam_k + 1.0))
        dL = np.asarray(directional_loss_change(loss, z_hat, v, jnp.asarray(a)))
        profiles[label] = {"lambda": lam_k, "alphas_posterior_sd": alphas.tolist(),
                           "energy": (sir.w * dL + 0.5 * a ** 2).tolist(),
                           "prediction": (0.5 * a ** 2 * (sir.w * lam_k + 1.0)).tolist()}

    results = {
        "eigvals": lam.tolist(),
        "d_data": int((lam > 1).sum()),
        "residual_over_hessian": float(jnp.linalg.norm(H - G)) / float(jnp.linalg.norm(H)),
        "is_ess": ess_frac * cfg["is_draws"],
        "axis_angles_deg": axis_angles,
        "cov_rel_frobenius_error": rel_frobenius_error(jnp.asarray(Sig_lap), jnp.asarray(Sig_ref)),
        "marginals": marginals,
        "profiles": profiles,
    }
    io.save_results(NAME, results, cfg,
                    arrays={"G": np.asarray(G), "cov_laplace": Sig_lap, "cov_reference": Sig_ref})


def _ellipse(cov2, nsig, n=200):
    w, U = np.linalg.eigh(np.asarray(cov2))
    t = np.linspace(0, 2 * np.pi, n)
    pts = (U @ (np.sqrt(np.maximum(w, 0))[:, None] * np.stack([np.cos(t), np.sin(t)]))) * nsig
    return pts[0], pts[1]


def plot():
    res = io.load_results(NAME)
    arr = io.load_arrays(NAME)
    plotting.apply_style()
    R = plotting.ROLE
    Sig_ref, Sig_lap = arr["cov_reference"], arr["cov_laplace"]

    # the pair of dynamic parameters whose reference covariance is closest to
    # condition number 6, so that both ellipse axes are visible
    pairs = [(a, b) for a in (0, 1, 2) for b in (0, 1, 2) if a < b]
    i, j = min(pairs, key=lambda ab: abs(np.log(
        np.linalg.cond(Sig_ref[np.ix_(ab, ab)])) - np.log(6.0)))
    sub = lambda S: S[np.ix_([i, j], [i, j])]

    fig, ax = plt.subplots(1, 2, figsize=plotting.figsize("double", 0.46))
    for ns, al in ((2.0, 0.16), (1.0, 0.28)):
        ax[0].fill(*_ellipse(sub(Sig_ref), ns), color=R["ref"], alpha=al, zorder=2, lw=0)
    for ns in (1.0, 2.0):
        ax[0].plot(*_ellipse(sub(Sig_lap), ns), "--", color=R["ggn"], lw=1.8, zorder=4)
    ax[0].plot(0, 0, "o", color=R["truth"], ms=4, zorder=5)
    lab = plotting.PARAM_LABELS
    ax[0].set_xlabel(f"{lab[PARAM_NAMES[i]]}  (prior s.d.)")
    ax[0].set_ylabel(f"{lab[PARAM_NAMES[j]]}  (prior s.d.)")
    ax[0].legend(handles=[
        Patch(facecolor=R["ref"], alpha=0.5, label="sampled posterior"),
        Line2D([0], [0], color=R["ggn"], lw=1.8, ls="--", label="Laplace from $G$")],
        loc="upper left")
    ax[0].set_aspect("equal", adjustable="datalim")
    plotting.panel_label(ax[0], "a")

    cols = {"stiffest": R["ggn"], "sloppiest": R["residual"]}
    for label, p in res["profiles"].items():
        a = np.array(p["alphas_posterior_sd"])
        energy, pred = np.array(p["energy"]), np.array(p["prediction"])
        rel = np.abs(energy - pred) / np.maximum(np.abs(pred), 1e-12)
        ax[1].semilogy(a, np.maximum(rel, 1e-6), "-o", ms=4, color=cols[label],
                       label=rf"{label} direction ($\lambda$={p['lambda']:.1g})")
    ax[1].axhline(5e-3, ls=plotting.LS["ref"], c=R["ref"], lw=1.2)
    ax[1].text(-2.9, 5.6e-3, "0.5%", ha="left", va="bottom", fontsize=7, color=R["ref"])
    ax[1].set_xlabel(r"displacement $\alpha$  (posterior s.d.)")
    ax[1].set_ylabel("relative error of the prediction")
    ax[1].legend(fontsize=7, loc="lower right")
    plotting.panel_label(ax[1], "b")

    fig.tight_layout()
    io.save_figure(fig, NAME)
    plt.close(fig)


if __name__ == "__main__":
    io.main(compute, plot)
