"""EXP-002 — Nonlinear local-validity benchmark.

Validates ∇²L = G + R (Math-Spec §7) and measures the neighborhood over which
the GGN predicts the local loss geometry, on two controlled nonlinear cases:
Case A (Rosenbrock exact-fit curved valley, R=0 at optimum) and Case B
(irreducible overdetermined residual, R≠0 at the minimizer). Supports C03, C09.

Run:  uv run python -m experiments.exp002_nonlinear_validity
Writes: outputs/EXP-002/<UTC-timestamp>_<short-commit>/{config,provenance,metrics}.json,
        arrays.npz, figures/fig02_nonlinear_validity.{png,pdf}
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
from curvature_calib.benchmarks import nonlinear_residual as nr  # noqa: E402
from curvature_calib.calibration.diagnostic import eigendecompose  # noqa: E402
from curvature_calib.geometry import ggn as G  # noqa: E402
from curvature_calib.geometry import validity as V  # noqa: E402

_COMMAND = "uv run python -m experiments.exp002_nonlinear_validity"

# Prespecified grids/tolerances (fixed BEFORE running; recorded in config.json).
ALPHAS = np.array([-2.0, -1.5, -1.0, -0.75, -0.5, -0.3, -0.2, -0.1, -0.05, -0.02, -0.01,
                   0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0])
TAU = 0.10
T_POS = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0])
T_SIGNED = np.concatenate([-T_POS[::-1][:-1], T_POS])


def _case_fns(kind, params):
    if kind == "rosenbrock":
        model = nr.RosenbrockResidual(a=params["a"], b=params["b"])
        return (model, nr.rosen_residual_fn(model), nr.rosen_loss_fn(model),
                nr.rosen_ggn, nr.rosen_hessian, nr.rosen_residual_curvature,
                nr.rosen_optimum(model))
    model = nr.IrreducibleResidual(lam=params["lam"], c=params["c"])
    z_hat = nr.newton_minimize(nr.irr_loss_fn(model), jnp.array([1.0, 1.0]))
    return (model, nr.irr_residual_fn(model), nr.irr_loss_fn(model),
            nr.irr_ggn, nr.irr_hessian, nr.irr_residual_curvature, z_hat)


def _matrix_agreement(Gm, Hm):
    """Four distinct axes: matrix, leading-subspace, eigenvalue, (radius elsewhere)."""
    eg, eh = eigendecompose(Gm), eigendecompose(Hm)
    return {
        "rel_fro_G_vs_H": MET.rel_frobenius_error(Gm, Hm),
        "rel_fro_R_over_H": float(jnp.linalg.norm(Hm - Gm)) / max(float(jnp.linalg.norm(Hm)), 1e-300),
        "leading1_principal_angle": MET.max_principal_angle(eg.eigvecs[:, :1], eh.eigvecs[:, :1]),
        "full_principal_angle": MET.max_principal_angle(eg.eigvecs, eh.eigvecs),
        "eigval_rel_error": [float(x) for x in MET.eigenvalue_rel_error(eg.eigvals, eh.eigvals)],
    }


def _cell(kind, params):
    model, r, loss, ggn_an, hess_an, rcurv_an, z_hat = _case_fns(kind, params)
    grad_norm = float(jnp.linalg.norm(jax.grad(loss)(z_hat)))

    G_ad = G.ggn_dense(r, z_hat)
    H_ad = G.exact_hessian(loss, z_hat)
    R_ad = H_ad - G_ad

    # AD-vs-analytic consistency (float64 references)
    consistency = {
        "rel_fro_G_ad_vs_analytic": MET.rel_frobenius_error(G_ad, ggn_an(model, z_hat)),
        "rel_fro_H_ad_vs_analytic": MET.rel_frobenius_error(H_ad, hess_an(model, z_hat)),
        "rel_fro_R_ad_vs_analytic": MET.rel_frobenius_error(R_ad, rcurv_an(model, z_hat)),
    }

    at_opt = _matrix_agreement(G_ad, H_ad)

    # validity radii per GGN eigenvector
    eigG = eigendecompose(G_ad)
    radii = {}
    for k in range(z_hat.shape[0]):
        v, lam = eigG.eigvecs[:, k], float(eigG.eigvals[k])
        rad = V.validity_radius(loss, z_hat, v, lam, ALPHAS, TAU)
        radii[f"v{k}"] = {"lambda": lam, **rad,
                          "quad_error": [float(x) for x in
                                         V.quadratic_model_error(loss, z_hat, v, lam, ALPHAS)]}

    # paths: along the two GGN eigenvectors + a fixed random unit direction
    tgrid = T_SIGNED if kind == "rosenbrock" else T_POS
    rng = np.array(jax.random.normal(jax.random.PRNGKey(7), (2,)))
    rand_dir = rng / np.linalg.norm(rng)
    directions = {"v0": np.array(eigG.eigvecs[:, 0]),
                  "v1": np.array(eigG.eigvecs[:, 1]),
                  "rand": rand_dir}
    paths = {}
    arrays = {}
    for name, d in directions.items():
        rows = []
        for t in tgrid:
            z = z_hat + jnp.asarray(t * d, dtype=z_hat.dtype)
            Gt, Ht = G.ggn_dense(r, z), G.exact_hessian(loss, z)
            a = _matrix_agreement(Gt, Ht)
            rows.append([float(t), a["rel_fro_G_vs_H"], a["leading1_principal_angle"],
                         a["rel_fro_R_over_H"], max(a["eigval_rel_error"])])
        arr = np.array(rows)  # (nt, 5): t, E_G, angle1, R/H, max eigval err
        paths[name] = {"columns": ["t", "E_G", "leading1_angle", "R_over_H", "max_eigval_err"],
                       "rows": arr.tolist()}
        arrays[f"path_{name}"] = arr

    arrays["G_ad"] = np.array(G_ad)
    arrays["H_ad"] = np.array(H_ad)
    arrays["R_ad"] = np.array(R_ad)

    cell = {
        "kind": kind, "params": params, "z_hat": [float(x) for x in z_hat],
        "grad_norm_at_z_hat": grad_norm,
        "consistency": consistency, "at_optimum": at_opt,
        "validity_radii": radii, "paths": paths,
    }
    return cell, arrays


def _new_run_dir(out_root: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = Path(out_root) / f"{ts}_{prov.short_commit()}"
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


def _figure(arrays, cell, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    p = np.array(arrays["path_v0"])  # along stiff eigenvector
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(p[:, 0], p[:, 1], "o-", label=r"$E_G=\|G-H\|/\|H\|$")
    ax[0].plot(p[:, 0], p[:, 3], "s--", label=r"$\|R\|/\|H\|$")
    ax[0].set_xlabel("t along $v_0$"); ax[0].set_ylabel("relative error"); ax[0].legend(fontsize=8)
    ax[0].set_title("matrix disagreement vs distance", fontsize=10)
    rad = cell["validity_radii"]["v0"]
    qa = np.array(rad["quad_error"])
    ax[1].semilogy(np.array(ALPHAS), np.maximum(qa, 1e-16), "o-")
    ax[1].axhline(TAU, ls=":", c="k"); ax[1].axvline(rad["rho_plus"], ls="--", c="g")
    ax[1].axvline(-rad["rho_minus"], ls="--", c="g")
    ax[1].set_xlabel(r"$\alpha$ along $v_0$"); ax[1].set_ylabel("quad. rel. error")
    ax[1].set_title(rf"validity radius $\rho$={rad['rho']:.2f}", fontsize=10)
    fig.suptitle(f"EXP-002 FIG-02 — {cell['kind']} {cell['params']}", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)


def run(out_root="outputs/EXP-002") -> str:
    require_x64()
    run_dir = _new_run_dir(out_root)

    grid = [
        ("rosenbrock", {"a": 1.0, "b": 1.0}),
        ("rosenbrock", {"a": 5.0, "b": 1.0}),
        ("rosenbrock", {"a": 10.0, "b": 1.0}),
        ("irreducible", {"lam": 0.3, "c": 0.0}),
        ("irreducible", {"lam": 1.0, "c": 4.0}),
    ]
    cells, npz = {}, {}
    fig_ref = None
    for kind, params in grid:
        cell, arr = _cell(kind, params)
        key = f"{kind}_" + "_".join(f"{k}{v}" for k, v in params.items())
        cells[key] = cell
        for k, v in arr.items():
            npz[f"{key}__{k}"] = v
        if kind == "rosenbrock" and params["a"] == 5.0:
            fig_ref = (arr, cell)

    config = {"cases": grid, "alphas": ALPHAS.tolist(), "tau": TAU,
              "t_grid_pos": T_POS.tolist(), "seed": 7}
    prov.write_json(run_dir / "config.json", config)
    prov.write_json(run_dir / "provenance.json",
                    prov.run_metadata("EXP-002", config, {"path_dir": 7}, command=_COMMAND))
    prov.write_json(run_dir / "metrics.json", {"cells": cells})
    np.savez_compressed(run_dir / "arrays.npz", **npz)
    if fig_ref is not None:
        _figure(fig_ref[0], fig_ref[1], run_dir / "figures" / "fig02_nonlinear_validity")
    return str(run_dir)


if __name__ == "__main__":
    print("run:", run())
