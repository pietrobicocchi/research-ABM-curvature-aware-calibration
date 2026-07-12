"""EXP-005 — Smooth SIR posterior validation.

Does the prior-relative GGN agree with the true local generalized posterior?
Smooth (deterministic) mean-field SIR, Gaussian-noise loss, prior-scaled z.
Compares the GGN-Laplace posterior (G + I) against a preconditioned-MCMC
reference posterior. Supports C10, C11. See docs/experiments/EXP-005.md.

Run:  uv run python -m experiments.exp005_sir_posterior
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from curvature_calib.config import enable_x64, require_x64

enable_x64()
require_x64()

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from curvature_calib import metrics as MET  # noqa: E402
from curvature_calib import provenance as prov  # noqa: E402
from curvature_calib.calibration import diagnostic as DIAG  # noqa: E402
from curvature_calib.geometry import ggn as GGN  # noqa: E402
from curvature_calib.geometry import validity as V  # noqa: E402
from curvature_calib.models.sir import simulate  # noqa: E402

_COMMAND = "uv run python -m experiments.exp005_sir_posterior"

T_SIR, N_POP, SIGMA_DATA = 200, 1e5, 150.0
Z0 = jnp.zeros(5)
PARAM_NAMES = ["beta", "gamma", "I0", "t_lock", "f_lock"]
# reference posterior via importance sampling (Laplace proposal, variance-inflated)
N_IS, IS_INFLATE = 400000, 1.3
ALPHAS = np.array([-3.0, -2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0, 3.0])  # in prior-std units


def _logit(p):
    return jnp.log(p / (1 - p))


def T(z):
    """θ = T(z), prior z~N(0,I). θ* = T(0) = (0.4, 0.1, 1e-3, 0.4, 0.5)."""
    return jnp.array([
        jnp.exp(jnp.log(0.4) + 0.3 * z[0]),
        jnp.exp(jnp.log(0.1) + 0.3 * z[1]),
        jax.nn.sigmoid(_logit(1e-3) + 0.5 * z[2]),
        jax.nn.sigmoid(_logit(0.4) + 0.5 * z[3]),
        jax.nn.sigmoid(_logit(0.5) + 0.5 * z[4]),
    ])


_KEY0 = jax.random.PRNGKey(0)


def m(z):
    """Deterministic expected incidence trajectory (sigma_obs=0)."""
    return simulate(T(z), _KEY0, T=T_SIR, N=N_POP, sigma_obs=0.0)


def _importance_sampling(loss, P_lap, key, n, inflate):
    """Self-normalized IS reference for π(z) ∝ exp(-loss(z)) N(z;0,I).

    Proposal = N(0, inflate·Σ_lap) (heavier-tailed than the posterior ⇒ bounded
    weights). Returns weighted mean, weighted covariance, and the effective
    sample size fraction (coverage/reliability diagnostic).
    """
    Sig = jnp.linalg.inv(P_lap)
    Lc = jnp.linalg.cholesky(Sig) * jnp.sqrt(inflate)     # proposal cov = inflate·Σ
    z = (Lc @ jax.random.normal(key, (5, n))).T           # (n, 5), mean 0
    logpost = jax.vmap(lambda zz: -loss(zz) - 0.5 * jnp.sum(zz ** 2))(z)
    logprop = jax.vmap(lambda zz: -0.5 * (zz @ (P_lap @ zz)) / inflate)(z)  # up to const
    lw = logpost - logprop
    lw = lw - jnp.max(lw)
    w = jnp.exp(lw); w = w / jnp.sum(w)
    mean = w @ z
    dz = z - mean
    cov = (dz * w[:, None]).T @ dz
    ess_frac = float(1.0 / jnp.sum(w ** 2) / n)
    return np.asarray(mean), np.asarray(cov), ess_frac, np.asarray(z), np.asarray(w)


def run(out_root="outputs/EXP-005") -> str:
    require_x64()
    y = m(Z0)
    loss = lambda z: 0.5 * jnp.sum((m(z) - y) ** 2) / SIGMA_DATA ** 2

    # geometry at the mode (z=0, residual 0 -> H = G)
    J = jax.jacfwd(m)(Z0)
    G = jnp.asarray(GGN.symmetrize((J.T @ J) / SIGMA_DATA ** 2))
    H = jnp.asarray(GGN.symmetrize(jax.hessian(loss)(Z0)))
    R_over_H = float(jnp.linalg.norm(H - G)) / max(float(jnp.linalg.norm(H)), 1e-300)
    eigG = DIAG.eigendecompose(G)
    prior_rel_eigs = np.asarray(eigG.eigvals)          # w=1, prior precision I
    d_data = int((prior_rel_eigs > 1).sum())

    # Laplace posterior precision/covariance (w=1): P = G + I
    P_lap = G + jnp.eye(5)
    Sig_lap = jnp.linalg.inv(P_lap)
    P_lapH = H + jnp.eye(5)

    # reference posterior via importance sampling (Laplace proposal)
    mean_ref, Sig_ref, ess_frac, z_is, w_is = _importance_sampling(
        loss, P_lap, jax.random.PRNGKey(1), N_IS, IS_INFLATE)

    # comparison metrics
    Sl, Sr = np.asarray(Sig_lap), Sig_ref
    eig_l = DIAG.eigendecompose(jnp.asarray(Sl))
    eig_r = DIAG.eigendecompose(jnp.asarray(Sr))
    # whitened posterior covariance: P_lap^{1/2} Σ_ref P_lap^{1/2} ~ I if Laplace exact
    Phalf = np.asarray(jnp.real(jax.scipy.linalg.sqrtm(P_lap)))
    whit_cov = Phalf @ Sr @ Phalf.T

    def _wmoments(x):
        mu = np.sum(w_is * x)
        sd = np.sqrt(np.sum(w_is * (x - mu) ** 2)) + 1e-30
        u = (x - mu) / sd
        return sd, float(np.sum(w_is * u ** 3)), float(np.sum(w_is * u ** 4) - 3.0)

    marg = {}
    for i in range(5):
        _, sk, ku = _wmoments(z_is[:, i])
        marg[PARAM_NAMES[i]] = {
            "post_std_ref": float(np.sqrt(Sr[i, i])), "post_std_lap": float(np.sqrt(Sl[i, i])),
            "std_rel_err": abs(np.sqrt(Sr[i, i]) - np.sqrt(Sl[i, i])) / max(np.sqrt(Sr[i, i]), 1e-300),
            "skew": sk, "excess_kurt": ku,
        }

    comparison = {
        "is_ess_fraction": ess_frac,
        "R_over_H_at_mode": R_over_H,
        "prior_relative_eigs_G": prior_rel_eigs.tolist(),
        "d_data_lambda_gt_1": d_data,
        "cov_rel_fro_err_lap_vs_ref": MET.rel_frobenius_error(jnp.asarray(Sl), jnp.asarray(Sr)),
        "top1_angle_cov_deg": float(np.degrees(MET.max_principal_angle(eig_l.eigvecs[:, :1], eig_r.eigvecs[:, :1]))),
        "top2_angle_cov_deg": float(np.degrees(MET.max_principal_angle(eig_l.eigvecs[:, :2], eig_r.eigvecs[:, :2]))),
        "posterior_mean_shift_norm": float(np.linalg.norm(mean_ref)),
        "whitened_cov_rel_err_vs_I": MET.rel_frobenius_error(jnp.asarray(whit_cov), jnp.eye(5)),
        "whitened_marginal_std": np.sqrt(np.diag(whit_cov)).tolist(),
        "marginals": marg,
    }

    # posterior-energy profile for stiffest and sloppiest prior-relative direction.
    # α is scaled by each direction's POSTERIOR std σ=1/sqrt(λ+1) so both are probed
    # over ±3 of their own posterior width (prior-std units are meaningless for the
    # ultra-tight stiff direction).
    profiles = {}
    for label, k in [("stiff", 0), ("sloppy", 4)]:
        v, lam = eigG.eigvecs[:, k], float(eigG.eigvals[k])
        sig_dir = 1.0 / np.sqrt(lam + 1.0)
        a_dir = ALPHAS * sig_dir
        dL = np.asarray(V.directional_loss_change(loss, Z0, v, jnp.asarray(a_dir)))
        Uprof = dL + 0.5 * a_dir ** 2               # U(a) = L(ẑ+a v) + ½‖z‖²
        Qpost = 0.5 * a_dir ** 2 * (lam + 1.0)      # quad posterior energy
        profiles[label] = {"lambda": lam, "posterior_std": float(sig_dir),
                           "alphas_post_std": ALPHAS.tolist(), "alphas_z": a_dir.tolist(),
                           "loss_change": dL.tolist(), "post_energy": Uprof.tolist(),
                           "quad_post_pred": Qpost.tolist()}

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(out_root) / f"{ts}_{prov.short_commit()}"
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    config = {"T": T_SIR, "N": N_POP, "sigma_data": SIGMA_DATA, "theta_star": [0.4, 0.1, 1e-3, 0.4, 0.5],
              "transform": "beta,gamma log-normal(0.3); I0,t_lock,f_lock logit-normal(0.5)",
              "w": 1.0, "reference": "importance sampling, Laplace proposal",
              "n_is": N_IS, "is_inflate": IS_INFLATE, "alphas_prior_std": ALPHAS.tolist()}
    prov.write_json(run_dir / "config.json", config)
    prov.write_json(run_dir / "provenance.json",
                    prov.run_metadata("EXP-005", config, {"data": 0, "mcmc": 1}, command=_COMMAND))
    prov.write_json(run_dir / "metrics.json", comparison | {"profiles": profiles})
    np.savez_compressed(run_dir / "arrays.npz", G=np.asarray(G), Sig_lap=Sl, Sig_ref=Sr,
                        prior_rel_eigs=prior_rel_eigs)
    _figure(comparison, profiles, run_dir / "figures" / "fig05_sir_posterior")
    return str(run_dir)


def _figure(comp, profiles, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    ev = np.array(comp["prior_relative_eigs_G"])
    ax[0].semilogy(range(1, 6), ev, "o-"); ax[0].axhline(1.0, ls=":", c="k")
    ax[0].set_xlabel("direction"); ax[0].set_ylabel("prior-relative eigenvalue λ")
    ax[0].set_title(f"data-dominant dims = {comp['d_data_lambda_gt_1']}", fontsize=10)
    for lab in ("stiff", "sloppy"):
        p = profiles[lab]; a = np.array(p["alphas_post_std"])
        ax[1].plot(a, p["post_energy"], "o-", label=f"{lab} actual U")
        ax[1].plot(a, p["quad_post_pred"], "--", label=f"{lab} quad")
    ax[1].set_xlabel("α (posterior-std)"); ax[1].set_ylabel("posterior energy"); ax[1].legend(fontsize=7)
    ax[1].set_title("profiled posterior energy", fontsize=10)
    fig.suptitle("EXP-005 — smooth SIR posterior validation", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("run:", run())
