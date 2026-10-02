"""Fig. 6: is the sloppy direction a property of the data or of the loss?

The fit is perturbed along the stiffest and the sloppiest eigenvectors of G, and the
perturbed and fitted models are compared through three summaries of noisy incidence
that play no part in calibration. The same noise is used at every alpha.
"""
from __future__ import annotations

from gndiag.config import enable_x64, load_config

enable_x64()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gndiag import io, plotting  # noqa: E402
from gndiag.diagnostic.ggn import eigendecompose, symmetrize  # noqa: E402
from gndiag.models.mfsir import MeanFieldSIR  # noqa: E402

NAME = "mfsir_sloppy_direction_check"
SUMMARIES = ("moments", "autocorrelation", "tail_quantiles")
SUMMARY_LABEL = {"moments": "moments", "autocorrelation": "autocorrelation",
                 "tail_quantiles": "tail quantiles"}
DIRECTION_LABEL = {"stiffest": "stiffest direction", "sloppiest": "sloppiest direction"}


def _peak_and_total(ens):
    return ens.max(axis=1), ens.sum(axis=1)


def _moments(x):
    mu, sd = x.mean(), x.std() + 1e-12
    z = (x - mu) / sd
    return np.array([mu, x.var(), (z ** 3).mean(), (z ** 4).mean()])


def _acf(x, max_lag):
    x = x - x.mean()
    v = np.dot(x, x) + 1e-12
    return np.array([np.dot(x[:-l], x[l:]) / v for l in range(1, max_lag + 1)])


def discrepancies(ens_p, ens_f, cfg):
    """Relative differences in the moments and in the tail quantiles of the per-replicate
    peak and total, and the sup-norm difference of the mean trajectory's autocorrelation."""
    mom = quant = 0.0
    for xp, xf in zip(_peak_and_total(ens_p), _peak_and_total(ens_f)):
        mp, mf = _moments(xp), _moments(xf)
        mom += float(np.sum(np.abs(mp - mf) / (np.abs(mf) + 1e-9)))
        qp, qf = np.percentile(xp, cfg["quantiles"]), np.percentile(xf, cfg["quantiles"])
        quant += float(np.sum(np.abs(qp - qf) / (np.abs(qf) + 1e-9)))
    lag = cfg["acf_max_lag"]
    acf = float(np.max(np.abs(_acf(ens_p.mean(0), lag) - _acf(ens_f.mean(0), lag))))
    return {"moments": mom, "autocorrelation": acf, "tail_quantiles": quant}


def compute():
    cfg = load_config(NAME)
    sir = MeanFieldSIR(cfg["model"])
    z_hat = jnp.zeros(5)
    J = jax.jacfwd(sir.incidence)(z_hat)
    eig = eigendecompose(jnp.asarray(symmetrize((J.T @ J) / sir.sigma_obs ** 2)))
    V = np.asarray(eig.eigvecs)
    alphas = np.linspace(*cfg["alphas"])
    directions = {"stiffest": V[:, 0], "sloppiest": V[:, -1]}

    curves = {}
    for d, v in directions.items():
        out = {k: np.zeros((cfg["noise_draws"], len(alphas))) for k in SUMMARIES}
        mean_fit = np.asarray(sir.incidence(z_hat + 0.0 * jnp.asarray(v)))
        means = [np.asarray(sir.incidence(z_hat + a * jnp.asarray(v))) for a in alphas]
        for s in range(cfg["noise_draws"]):
            noise = np.asarray(jax.random.normal(
                jax.random.PRNGKey(cfg["noise_key_offset"] + s), (cfg["replicates"], sir.T)))
            ens_f = mean_fit[None, :] + sir.sigma_obs * noise
            for j, mt in enumerate(means):
                for k, val in discrepancies(mt[None, :] + sir.sigma_obs * noise, ens_f, cfg).items():
                    out[k][s, j] = val
        curves[d] = out

    edge = len(alphas) - 1
    results = {
        "alphas": alphas.tolist(),
        "eigvals": np.asarray(eig.eigvals).tolist(),
        "directions": {d: v.tolist() for d, v in directions.items()},
        "ratio_stiff_over_sloppy_at_max_alpha": {
            k: float(curves["stiffest"][k][:, edge].mean()
                     / (curves["sloppiest"][k][:, edge].mean() + 1e-12)) for k in SUMMARIES},
        "mean": {d: {k: c[k].mean(0).tolist() for k in SUMMARIES} for d, c in curves.items()},
        "min": {d: {k: c[k].min(0).tolist() for k in SUMMARIES} for d, c in curves.items()},
        "max": {d: {k: c[k].max(0).tolist() for k in SUMMARIES} for d, c in curves.items()},
    }
    io.save_results(NAME, results, cfg)


def plot():
    res = io.load_results(NAME)
    plotting.apply_style()
    R = plotting.ROLE
    alphas = np.array(res["alphas"])
    col = {"stiffest": R["ggn"], "sloppiest": R["ref"]}
    dirs = list(res["mean"])
    fig, ax = plt.subplots(len(dirs), len(SUMMARIES), sharex=True, squeeze=False,
                           figsize=plotting.figsize("double", 0.30 * len(dirs)))
    for i, d in enumerate(dirs):
        for j, k in enumerate(SUMMARIES):
            a = ax[i][j]
            a.fill_between(alphas, res["min"][d][k], res["max"][d][k], color=col[d], alpha=0.25)
            a.plot(alphas, res["mean"][d][k], "-", c=col[d], lw=1.4)
            a.axvline(0, ls=":", c="0.6", lw=0.8)
            if i == 0:
                a.set_title(SUMMARY_LABEL[k], fontsize=9.5)
            if j == 0:
                a.set_ylabel(f"{DIRECTION_LABEL[d]}\ndiscrepancy", fontsize=8)
            if i == len(dirs) - 1:
                a.set_xlabel(r"$\alpha$ (prior s.d. along $v$)")
    fig.tight_layout()
    io.save_figure(fig, NAME)
    plt.close(fig)


if __name__ == "__main__":
    io.main(compute, plot)
