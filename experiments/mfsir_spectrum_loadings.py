"""Fig. 4: prior-relative spectrum of G at the fit and the loading of each parameter on
each eigenvector, mean-field SIR."""
from __future__ import annotations

from gndiag.config import enable_x64, load_config

enable_x64()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gndiag import io, plotting  # noqa: E402
from gndiag.diagnostic.ggn import eigendecompose, symmetrize  # noqa: E402
from gndiag.models.mfsir import PARAM_NAMES, MeanFieldSIR  # noqa: E402

NAME = "mfsir_spectrum_loadings"


def compute():
    cfg = load_config(NAME)
    sir = MeanFieldSIR(cfg["model"])
    J = jax.jacfwd(sir.incidence)(jnp.zeros(5))
    eig = eigendecompose(jnp.asarray(symmetrize((J.T @ J) / sir.sigma_obs ** 2)))
    io.save_results(NAME, {"param_names": list(PARAM_NAMES),
                           "eigvals": np.asarray(eig.eigvals).tolist(),
                           "eigvecs": np.asarray(eig.eigvecs).tolist()}, cfg)


def loadings_heatmap(ax, V, row_labels, col_labels, highlight_last=False):
    """Signed loadings of each parameter (rows) on each eigenvector (columns)."""
    R = plotting.ROLE
    vmax = float(np.abs(V).max())
    im = ax.imshow(V, cmap=plotting.DIV, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(V.shape[1])); ax.set_xticklabels(col_labels)
    ax.set_yticks(range(V.shape[0])); ax.set_yticklabels(row_labels)
    for a in range(V.shape[0]):
        for b in range(V.shape[1]):
            ax.text(b, a, f"{V[a, b]:+.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(V[a, b]) > 0.55 * vmax else R["truth"])
    if highlight_last:
        n = V.shape[1]
        ax.add_patch(plt.Rectangle((n - 1.5, -0.5), 1, V.shape[0], fill=False,
                                   ec=R["residual"], lw=2))
    return im


def plot():
    res = io.load_results(NAME)
    plotting.apply_style()
    R = plotting.ROLE
    lam, V = np.array(res["eigvals"]), np.array(res["eigvecs"])
    dirs = [f"$v_{{{k + 1}}}$" for k in range(len(lam))]

    fig, ax = plt.subplots(1, 2, figsize=plotting.figsize("double", 0.5),
                           gridspec_kw={"width_ratios": [1, 1.15]})
    idx = np.arange(1, len(lam) + 1)
    ax[0].bar(idx, np.maximum(lam, 1e-7), color=[R["ggn"] if v > 1 else R["ref"] for v in lam],
              width=0.62, zorder=3)
    ax[0].set_yscale("log")
    ax[0].axhline(1.0, ls=plotting.LS["ref"], c=R["residual"], lw=1.4)
    ax[0].text(len(lam) + 0.3, 1.0, r"$\lambda=1$", va="center", ha="right", fontsize=8,
               color=R["residual"])
    ax[0].annotate("data-dominant\n" + r"($\lambda>1$)", xy=(1, lam[0]),
                   xytext=(1.4, lam[0] * 0.04), fontsize=7.5, color=R["ggn"])
    ax[0].annotate("prior-dominant\n" + r"($\lambda<1$)", xy=(len(lam), lam[-1]),
                   xytext=(len(lam) - 2.3, lam[-1] * 8), fontsize=7.5, color=R["ref"])
    ax[0].set_xticks(idx); ax[0].set_xticklabels(dirs)
    ax[0].set_xlabel("GGN direction  (stiff → sloppy)")
    ax[0].set_ylabel(r"prior-relative eigenvalue $\lambda$")
    plotting.panel_label(ax[0], "a")

    im = loadings_heatmap(ax[1], V, [plotting.PARAM_LABELS[p] for p in res["param_names"]], dirs)
    ax[1].set_xlabel("GGN direction  (stiff → sloppy)")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03).set_label("loading", fontsize=8)
    plotting.panel_label(ax[1], "b")

    fig.tight_layout()
    io.save_figure(fig, NAME)
    plt.close(fig)


if __name__ == "__main__":
    io.main(compute, plot)
