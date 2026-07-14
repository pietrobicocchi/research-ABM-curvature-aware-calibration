"""FIG-02 — "the ellipse is the valley".

The soundness figure a reader verifies by eye. Left: a 2D slice of the smooth-SIR
calibration loss around the fit (the (gamma, I0) plane, the one cleanly readable
plane — every other pair is dominated by the single ultra-stiff GGN direction). The
true loss contours are overlaid with the GGN quadratic level sets AT THE SAME LOSS
LEVELS and the two GGN eigenvectors: the ellipse hugs the contours near the fit and
honestly separates at the edge (a preview of the residual/validity limit). Right:
the affine benchmark — AD reproduces the analytic GGN A^T W A to float64, while the
scalar-gradient outer product collapses at the exact fit.

Run:  uv run python -m experiments.fig02_ellipse
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

from curvature_calib import provenance as prov  # noqa: E402
from curvature_calib.geometry import ggn as GGN  # noqa: E402
from curvature_calib.benchmarks import linear_gaussian as lg  # noqa: E402
from curvature_calib.viz import style as STY  # noqa: E402
from experiments.exp005_sir_posterior import T, m, SIGMA_DATA, Z0, PARAM_NAMES  # noqa: E402

_COMMAND = "uv run python -m experiments.fig02_ellipse"

PLANE = (1, 2)                     # (gamma, I0) — the cleanly-readable anisotropic plane
LATEX = {"beta": r"\beta", "gamma": r"\gamma", "I0": r"I_0",
         "t_lock": r"t_{\mathrm{lock}}", "f_lock": r"f_{\mathrm{lock}}"}
LEVELS = [0.5, 2.0, 8.0]          # loss levels (nats) for both contours and ellipses
GRID_N = 121


def _sir_slice():
    """Loss on the (i,j) plane through the fit; the 2x2 GGN block and its ellipses."""
    i, j = PLANE
    y = m(Z0)
    loss = lambda z: 0.5 * jnp.sum((m(z) - y) ** 2) / SIGMA_DATA ** 2

    J = jax.jacfwd(m)(Z0)
    G = np.asarray(GGN.symmetrize((J.T @ J) / SIGMA_DATA ** 2))
    S = G[np.ix_([i, j], [i, j])]                       # 2x2 GGN block in the plane
    w, U = np.linalg.eigh(S)                            # ascending
    Sinv = U @ np.diag(1.0 / w) @ U.T

    # window: fit the largest ellipse (level max(LEVELS)) with margin
    cmax = max(LEVELS)
    wa = 1.35 * np.sqrt(2 * cmax * Sinv[0, 0])
    wb = 1.35 * np.sqrt(2 * cmax * Sinv[1, 1])
    a = np.linspace(-wa, wa, GRID_N)
    b = np.linspace(-wb, wb, GRID_N)
    A, B = np.meshgrid(a, b)

    ei = np.zeros(5); ei[i] = 1.0
    ej = np.zeros(5); ej[j] = 1.0
    pts = Z0[None, :] + A.ravel()[:, None] * ei[None, :] + B.ravel()[:, None] * ej[None, :]
    Lg = np.asarray(jax.vmap(loss)(jnp.asarray(pts))).reshape(A.shape)

    # GGN ellipse points at each level: delta(t) = sqrt(2c) * S^{-1/2} [cos t, sin t]
    Sinv_half = U @ np.diag(1.0 / np.sqrt(w)) @ U.T
    t = np.linspace(0, 2 * np.pi, 200)
    circ = np.stack([np.cos(t), np.sin(t)])
    ellipses = {c: (np.sqrt(2 * c) * (Sinv_half @ circ)) for c in LEVELS}
    # eigenvector arrows (major = smaller eigenvalue = sloppy), length at level=2
    axes = {"stiff": U[:, 1] * np.sqrt(2 * 2.0 / w[1]),
            "sloppy": U[:, 0] * np.sqrt(2 * 2.0 / w[0])}
    return {"a": a, "b": b, "L": Lg, "ellipses": ellipses, "axes": axes,
            "block_eigs": w[::-1].tolist(), "window": (wa, wb)}


def _affine_benchmark():
    """AD GGN vs analytic A^T W A, and the scalar-gradient outer product at the fit."""
    key = jax.random.PRNGKey(0)
    kA, kw = jax.random.split(key)
    P, K = 6, 12
    A = lg.random_A(kA, K, P, cond=1e2, dtype=jnp.float64)
    B = jax.random.normal(kw, (K, K), dtype=jnp.float64)
    W = (B @ B.T) / K + jnp.eye(K)
    z_true = jax.random.normal(jax.random.PRNGKey(1), (P,), dtype=jnp.float64)
    model = lg.LinearGaussian(A=A, y=A @ z_true, W=W)          # exact fit at z_true
    rep, loss = lg.make_representation(model), lg.make_loss(model)

    G_an = np.asarray(lg.analytic_ggn(model))
    G_ad = np.asarray(GGN.ggn_dense(rep, z_true, W))
    g = np.asarray(jax.grad(loss)(z_true))                     # gradient at exact fit
    opg = np.outer(g, g)
    nrm = lambda X: float(np.linalg.norm(X))
    eig_an = np.sort(np.linalg.eigvalsh(G_an))[::-1]
    eig_ad = np.sort(np.linalg.eigvalsh(G_ad))[::-1]
    return {
        "ggn_rel_err": nrm(G_ad - G_an) / nrm(G_an),           # ≤ float64 eps (here 0)
        "opg_over_ggn": nrm(opg) / nrm(G_an),                  # ~0 at the fit
        "grad_norm_at_fit": nrm(g),
        "eig_analytic": eig_an.tolist(),
        "eig_ad": eig_ad.tolist(),
    }


def run(out_root="outputs/FIG-02") -> str:
    require_x64()
    sir = _sir_slice()
    aff = _affine_benchmark()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(out_root) / f"{ts}_{prov.short_commit()}"
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    config = {"plane": [PARAM_NAMES[PLANE[0]], PARAM_NAMES[PLANE[1]]], "levels": LEVELS,
              "grid_n": GRID_N, "reference": "EXP-005 SIR setup; EXP-001 affine"}
    prov.write_json(run_dir / "config.json", config)
    prov.write_json(run_dir / "provenance.json",
                    prov.run_metadata("FIG-02", config, {"data": 0}, command=_COMMAND))
    metrics = {"sir_block_eigs": sir["block_eigs"], "sir_window": list(sir["window"]),
               "affine": aff}
    prov.write_json(run_dir / "metrics.json", metrics)
    _figure(sir, aff, run_dir / "figures" / "fig02_ellipse")
    return str(run_dir)


def _figure(sir, aff, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    STY.apply_style()

    from matplotlib.lines import Line2D
    R = STY.ROLE
    fig, ax = plt.subplots(1, 2, figsize=STY.figsize("double", 0.47),
                           gridspec_kw={"width_ratios": [1.3, 1]})
    i, j = PLANE
    A, Bb = np.meshgrid(sir["a"], sir["b"])

    # --- (a) SIR loss slice + GGN ellipses at the same loss levels ---
    ax[0].contour(A, Bb, sir["L"], levels=LEVELS, colors=R["truth"],
                  linewidths=1.5, zorder=2)
    ax[0].text(0.03, 0.97, "contours & ellipses at\n$L = 0.5,\\ 2,\\ 8$ nats",
               transform=ax[0].transAxes, fontsize=7.5, color=R["truth"],
               ha="left", va="top")
    for c, E in sir["ellipses"].items():
        ax[0].plot(E[0], E[1], ls=STY.LS["ggn_quad"], lw=1.7, color=R["ggn"], zorder=3)
    # fixed-length eigenvector arrows (direction, not magnitude), labels at the tips
    wa, wb = sir["window"]
    L_arrow = 0.5 * min(wa, wb)
    for lab, v in sir["axes"].items():
        d = v / (np.linalg.norm(v) + 1e-30)
        if d[1] < 0:
            d = -d                                              # point into the upper half
        tip = d * L_arrow
        ax[0].annotate("", xy=(tip[0], tip[1]), xytext=(0, 0),
                       arrowprops=dict(arrowstyle="-|>", color=R["ggn"], lw=1.8), zorder=5)
        ax[0].text(tip[0] * 1.18, tip[1] * 1.18, lab, color=R["ggn"], fontsize=9,
                   fontweight="bold", ha="center", va="center", zorder=6)
    ax[0].plot(0, 0, "o", color=R["truth"], ms=4, zorder=6)
    proxies = [Line2D([0], [0], color=R["truth"], lw=1.5, label="true loss"),
               Line2D([0], [0], color=R["ggn"], lw=1.7, ls="--", label="GGN quadratic")]
    ax[0].legend(handles=proxies, loc="lower right")
    ax[0].set_xlabel(rf"$z_{{{LATEX[PARAM_NAMES[i]]}}}$  (prior-std)")
    ax[0].set_ylabel(rf"$z_{{{LATEX[PARAM_NAMES[j]]}}}$  (prior-std)")
    ax[0].set_title(r"GGN ellipse vs true loss  ($\gamma,\ I_0$ plane)")
    ax[0].set_aspect("equal", adjustable="box")
    STY.panel_label(ax[0], "a")

    # --- (b) affine exactness — AD tracks analytic; OPG collapses at the fit ---
    ea, ed = np.array(aff["eig_analytic"]), np.array(aff["eig_ad"])
    idx = np.arange(1, len(ea) + 1)
    ax[1].plot(idx, ea, "o", ms=9, mfc="none", mec=R["truth"], mew=1.8,
               label=r"analytic  $A^\top W A$", zorder=3)
    ax[1].plot(idx, ed, "x", ms=7, color=R["ggn"], mew=2.0,
               label=r"AD (matches to $\leq\varepsilon$)", zorder=4)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("eigenvalue index"); ax[1].set_ylabel("eigenvalue (log)")
    ax[1].set_xticks(idx)
    ax[1].set_title("affine benchmark: exact recovery")
    ax[1].legend(loc="upper right")
    ax[1].annotate("scalar-gradient OPG $\\equiv 0$ at the fit\n(the GGN is not)",
                   xy=(0.5, 0.04), xycoords="axes fraction", ha="center", va="bottom",
                   fontsize=8, color=R["opg"],
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=R["opg"], lw=1.0))
    STY.panel_label(ax[1], "b")

    fig.suptitle("The GGN is the local loss geometry", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    print("run:", run())
