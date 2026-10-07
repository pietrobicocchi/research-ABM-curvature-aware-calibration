"""Fig. 8, App. D: robustness of the network-SIR geometry at points A and B.

(a) Gumbel-sigmoid against straight-through gradients: angles between the leading
eigenspaces of G_V. (b) Each construction's seed-mean Jacobian against central finite
differences of its own forward model (common random numbers). (c) Leading eigenspace
with a truncated differentiation horizon against the full horizon.
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
from gndiag.diagnostic.ggn import eigendecompose, principal_angles  # noqa: E402
from gndiag.losses import rff  # noqa: E402
from gndiag.losses.mmd import median_heuristic  # noqa: E402
from gndiag.models.netsir import NetworkSIR  # noqa: E402

NAME = "netsir_gradient_robustness"
SURROGATE_LABEL = {"gumbel": "Gumbel-sigmoid", "straight_through": "straight-through"}


def _max_angle_deg(V1, V2, k):
    return float(np.max(np.degrees(np.asarray(principal_angles(V1[:, :k], V2[:, :k])))))


def analyze(sir, cfg):
    z_star = jnp.zeros(5)
    keys = sir.seed_keys()
    incidence = lambda z, key, surrogate, horizon: sir.incidence(sir.theta(z), key, surrogate, horizon)

    ref = jax.vmap(lambda k: incidence(z_star, k, "gumbel", None))(keys)
    features = rff.frozen_rff(sir.rff_key(), sir.n_features, sir.T, float(median_heuristic(ref)))

    def feature_fn(surrogate, horizon):
        return lambda z, key: rff.feature_map(features, incidence(z, key, surrogate, horizon))

    def jacobians(z, ky, surrogate, horizon=None):
        return feature_jacobians(feature_fn(surrogate, horizon), z, ky)

    def mean_embedding(z, surrogate):
        f = feature_fn(surrogate, None)
        return jnp.mean(jax.vmap(lambda k: f(z, k))(keys), axis=0)

    def fd_jacobian(surrogate, step):
        cols = []
        for i in range(5):
            e = jnp.zeros(5).at[i].set(step)
            cols.append((mean_embedding(z_star + e, surrogate)
                         - mean_embedding(z_star - e, surrogate)) / (2.0 * step))
        return jnp.stack(cols, axis=1)

    jac_full = jax.jit(lambda z, ky, s: jacobians(z, ky, s), static_argnums=(2,))

    per_surrogate, eig = {}, {}
    for s in cfg["surrogates"]:
        A = jac_full(z_star, keys, s)
        eig[s] = eigendecompose(ggn_plugin(A))
        J_ad, J_fd = jnp.mean(A, axis=0), fd_jacobian(s, cfg["fd_step"])
        col_err = [float(jnp.linalg.norm(J_ad[:, i] - J_fd[:, i]) /
                         (jnp.linalg.norm(J_fd[:, i]) + 1e-30)) for i in range(5)]
        per_surrogate[s] = {"eigvals": np.asarray(eig[s].eigvals).tolist(),
                            "fd_rel_jacobian_error": col_err,
                            "fd_rel_jacobian_error_median": float(np.median(col_err))}

    Vg, Vs = eig["gumbel"].eigvecs, eig["straight_through"].eigvecs
    surrogate_angles = {str(k): _max_angle_deg(Vg, Vs, k) for k in (1, 2, 3)}

    k_lead = cfg["leading_k"]
    horizon = []
    for frac in sorted(cfg["horizon_fractions"], reverse=True):
        h = int(round(frac * sir.T))
        A = jac_full(z_star, keys, "gumbel") if h >= sir.T else jacobians(z_star, keys, "gumbel", h)
        e = eigendecompose(ggn_plugin(A))
        horizon.append({"grad_horizon": h,
                        "angle_to_full_deg": _max_angle_deg(Vg, e.eigvecs, k_lead)})

    return {"attack_rate": float(jnp.mean(jnp.sum(ref, axis=1)) / sir.N),
            "per_surrogate": per_surrogate,
            "surrogate_angle_deg_by_k": surrogate_angles,
            "horizon_truncation": horizon}


def compute():
    cfg = load_config(NAME)
    points = {p: analyze(NetworkSIR(cfg["model"], p), cfg) for p in cfg["operating_points"]}
    io.save_results(NAME, {"operating_points": points}, cfg)


def plot():
    res = io.load_results(NAME)["operating_points"]
    plotting.apply_style()
    R = plotting.ROLE
    pts = list(res)
    pcol = dict(zip(pts, (R["ggn"], R["alt"])))
    plabel = {p: f"point {p}" for p in pts}
    w = 0.36
    fig, ax = plt.subplots(1, 3, figsize=plotting.figsize("double", 0.36))

    ks = ["1", "2", "3"]
    x = np.arange(len(ks))
    for s, p in enumerate(pts):
        ax[0].bar(x + (s - 0.5) * w, [res[p]["surrogate_angle_deg_by_k"][k] for k in ks], w,
                  color=pcol[p], label=plabel[p], zorder=3)
    ax[0].axhline(15.0, ls=plotting.LS["ref"], c=R["ref"], lw=1.2)
    ax[0].set_xticks(x); ax[0].set_xticklabels(ks)
    ax[0].set_xlabel("leading-$k$ eigenspace")
    ax[0].set_ylabel("angle (deg)")
    ax[0].legend(fontsize=7)
    ax[0].set_ylim(0, 20)
    plotting.panel_label(ax[0], "a")

    scol = {"gumbel": R["ggn"], "straight_through": R["contrast"]}
    x2 = np.arange(len(pts))
    for s, sur in enumerate(scol):
        ax[1].bar(x2 + (s - 0.5) * w,
                  [res[p]["per_surrogate"][sur]["fd_rel_jacobian_error_median"] for p in pts], w,
                  color=scol[sur], label=SURROGATE_LABEL[sur], zorder=3)
    ax[1].axhline(1e-3, ls=plotting.LS["ref"], c=R["ref"], lw=1.2)
    ax[1].set_yscale("log")
    ax[1].set_ylim(1e-6, 5)
    ax[1].set_xticks(x2); ax[1].set_xticklabels([plabel[p] for p in pts])
    ax[1].set_ylabel("median rel. Jacobian error")
    ax[1].legend(fontsize=7, loc="lower left")
    plotting.panel_label(ax[1], "b")

    for p in pts:
        rows = sorted((r["grad_horizon"], r["angle_to_full_deg"]) for r in res[p]["horizon_truncation"])
        ax[2].plot([r[0] for r in rows], [r[1] for r in rows], "-o", ms=4, color=pcol[p],
                   label=plabel[p])
    ax[2].axhline(15.0, ls=plotting.LS["ref"], c=R["ref"], lw=1.2)
    ax[2].set_xlabel("differentiation horizon")
    ax[2].set_ylabel("angle to full horizon (deg)")
    ax[2].legend(fontsize=7, loc="center right")
    plotting.panel_label(ax[2], "c")

    fig.tight_layout()
    io.save_figure(fig, NAME)
    plt.close(fig)


if __name__ == "__main__":
    io.main(compute, plot)
