"""Fig. 7, Sec. 6: geometry of the network SIR model as the observation window lengthens.

At point A with the lockdown off, the incidence over the first T steps is embedded
with D random Fourier features (bandwidth by the median heuristic at each T) and the
plug-in G_V (Eq. 14) is formed from M per-seed feature Jacobians at theta* for
P = 3 parameters (beta, gamma, I0).
"""
from __future__ import annotations

from gndiag.config import enable_x64, load_config

enable_x64()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gndiag import io, plotting  # noqa: E402
from gndiag.diagnostic.estimators import feature_jacobians, ggn_plugin  # noqa: E402
from gndiag.diagnostic.ggn import eigendecompose  # noqa: E402
from gndiag.losses import rff  # noqa: E402
from gndiag.losses.mmd import median_heuristic  # noqa: E402
from gndiag.models.netsir import NetworkSIR  # noqa: E402
from experiments.mfsir_spectrum_loadings import loadings_heatmap  # noqa: E402

NAME = "netsir_observation_window"
PARAMS = ("beta", "gamma", "I0")


def window_jacobians(cfg, T):
    """Attack rate and per-seed feature Jacobians (M, D, 3) for a window of T steps."""
    sir = NetworkSIR(cfg["model"], cfg["operating_point"], horizon=T)
    z = jnp.zeros(5)
    keys = sir.seed_keys()
    inc = lambda zz, key: sir.incidence(sir.theta(zz).at[4].set(cfg["f_lock"]), key)
    ref = jax.vmap(lambda k: inc(z, k))(keys)
    attack = float(jnp.mean(jnp.sum(ref, 1)) / sir.N)
    features = rff.frozen_rff(sir.rff_key(), sir.n_features, T, float(median_heuristic(ref)))
    feat = lambda zz, key: rff.feature_map(features, inc(zz, key))
    return attack, feature_jacobians(feat, z, keys)[:, :, :len(PARAMS)]


def compute():
    cfg = load_config(NAME)
    windows = []
    for T in cfg["windows"]:
        attack, A = window_jacobians(cfg, T)
        eig = eigendecompose(ggn_plugin(A))
        windows.append({"T": T, "attack_rate": attack,
                        "eigvals": np.asarray(eig.eigvals).tolist(),
                        "eigvecs": np.asarray(eig.eigvecs).tolist()})
    final = windows[-1]["attack_rate"]
    for w in windows:
        w["fraction_of_final_size"] = w["attack_rate"] / final
        w["I0_loading_sloppiest"] = abs(w["eigvecs"][PARAMS.index("I0")][-1])
    io.save_results(NAME, {"param_names": list(PARAMS), "windows": windows}, cfg)


def plot():
    cfg = load_config(NAME)
    res = io.load_results(NAME)
    plotting.apply_style()
    R = plotting.ROLE
    ws = res["windows"]
    i0 = PARAMS.index("I0")
    frac = np.array([w["fraction_of_final_size"] for w in ws])
    load = np.array([w["I0_loading_sloppiest"] for w in ws])
    i0_is_sloppiest = [int(np.argmax(np.abs(np.array(w["eigvecs"])[:, -1]))) == i0 for w in ws]

    fig = plt.figure(figsize=plotting.figsize("double", 0.52))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.5)
    axa = fig.add_subplot(gs[0, 0])
    axa.plot(frac, load, "-", c=R["truth"], lw=1.3, zorder=2)
    axa.scatter(frac, load, c=[R["ggn"] if s else R["ref"] for s in i0_is_sloppiest], s=34,
                zorder=3, edgecolor="k", linewidth=0.4)
    axa.set_xlabel("fraction of final size observed")
    axa.set_ylabel(r"$|$loading of $I_0$ on sloppiest$|$")
    axa.set_ylim(0.22, 0.80)
    plotting.panel_label(axa, "a")

    labels = [plotting.PARAM_LABELS[p] for p in PARAMS]
    dirs = [f"$v_{{{k + 1}}}$" for k in range(len(PARAMS))]
    by_T = {w["T"]: w for w in ws}
    for n, (letter, T) in enumerate(zip("bc", cfg["panel_windows"])):
        ax = fig.add_subplot(gs[0, 1 + n])
        w = by_T[T]
        im = loadings_heatmap(ax, np.array(w["eigvecs"]), labels, dirs, highlight_last=True)
        ax.set_xlabel(r"direction (stiff $\to$ sloppy)")
        ax.set_title(f"attack rate {w['attack_rate']:.0%}", fontsize=9)
        plotting.panel_label(ax, letter)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("loading", fontsize=8)
    io.save_figure(fig, NAME)
    plt.close(fig)


if __name__ == "__main__":
    io.main(compute, plot)
