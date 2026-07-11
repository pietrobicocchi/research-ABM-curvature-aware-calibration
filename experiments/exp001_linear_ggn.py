"""EXP-001 — Analytic linear GGN recovery.

Validates that AD reproduces the analytic GGN G = A^T W A for the linear model
m(z) = A z, L(z) = 1/2 ||A z - y||_W^2 (registry EXP-001; supports C01).

Ordinary validation gate: cond(A^T A) in {1, 1e2, 1e6}, float64.
Separate non-gating stress case: cond ~ 1e13 (SIR-scale), via run_stress().

Run:  uv run python -m experiments.exp001_linear_ggn
Writes: outputs/EXP-001/<UTC-timestamp>_<short-commit>/{config,provenance,metrics}.json,
        arrays.npz, figures/fig01_linear_benchmark.{png,pdf}
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

# Precision MUST be enabled before any significant array is created / traced.
from curvature_calib.config import enable_x64, require_x64

enable_x64()
require_x64()

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from curvature_calib import metrics as M  # noqa: E402
from curvature_calib import provenance as prov  # noqa: E402
from curvature_calib.benchmarks import linear_gaussian as lg  # noqa: E402
from curvature_calib.calibration.diagnostic import eigendecompose  # noqa: E402
from curvature_calib.geometry import ggn as G  # noqa: E402

_DTYPE = {"float64": jnp.float64, "float32": jnp.float32}
_FD_STEPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
_COMMAND = "uv run python -m experiments.exp001_linear_ggn"


def _weight(kind: str, K: int, dtype, key):
    if kind == "none":
        return None
    if kind == "diag":
        return jnp.linspace(0.5, 2.0, K, dtype=dtype)
    if kind == "dense":
        B = jax.random.normal(key, (K, K), dtype=dtype)
        return (B @ B.T) / K + jnp.eye(K, dtype=dtype)
    raise ValueError(kind)


def _test_vectors(P: int, dtype, key):
    axes = [jnp.eye(P, dtype=dtype)[i] for i in (0, P // 2, P - 1)]
    ones = jnp.ones((P,), dtype=dtype)
    rand = jax.random.normal(key, (P,), dtype=dtype)
    return axes + [ones, rand]


def _cell(P, cond, rank_kind, dtype_name, weight_kind, seed):
    dtype = _DTYPE[dtype_name]
    K = 2 * P
    rank = P if rank_kind == "full" else max(1, P // 2)
    key = jax.random.PRNGKey(seed)
    kA, kz, kw, kv, kd = jax.random.split(key, 5)

    A = lg.random_A(kA, K, P, cond=cond, rank=(None if rank_kind == "full" else rank),
                    dtype=dtype)
    W = _weight(weight_kind, K, dtype, kw)
    z_true = jax.random.normal(kz, (P,), dtype=dtype)
    y = A @ z_true                      # exact fit exists at z_true (r=0)
    model = lg.LinearGaussian(A=A, y=y, W=W)

    rep = lg.make_representation(model)
    loss = lg.make_loss(model)
    z_hat = z_true                      # r = 0 here

    G_analytic = lg.analytic_ggn(model)
    G_ad = G.ggn_dense(rep, z_hat, W)
    H = G.exact_hessian(loss, z_hat)

    t0 = time.perf_counter()
    _ = G.ggn_dense(rep, z_hat, W).block_until_ready()
    runtime_s = time.perf_counter() - t0

    # matrix-free vs dense over deterministic vectors
    matvec_errs = []
    for v in _test_vectors(P, dtype, kv):
        gv_free = G.ggn_matvec(rep, z_hat, v, W)
        gv_dense = G_ad @ v
        matvec_errs.append(M.rel_l2_error(gv_free, gv_dense))

    eig_ad = eigendecompose(G_ad)
    eig_an = eigendecompose(G_analytic)

    # OPG comparison: at the fit (r=0) it is ~0; at a perturbation it is rank 1.
    opg_fit = G.scalar_gradient_outer_product(loss, z_hat)
    z_pert = z_hat + jnp.asarray(0.1, dtype) * jax.random.normal(kd, (P,), dtype=dtype)
    opg_pert = G.scalar_gradient_outer_product(loss, z_pert)

    g_fro = float(jnp.linalg.norm(G_ad))
    out = {
        "P": P, "K": K, "cond_target": cond, "rank_kind": rank_kind,
        "rank_expected": rank, "dtype": dtype_name, "weight": weight_kind,
        "cond_AtA_actual": float(eig_an.eigvals[0] / jnp.clip(eig_an.eigvals[-1], min=1e-300))
        if weight_kind == "none" else None,
        "rel_fro_ggn_vs_analytic": M.rel_frobenius_error(G_ad, G_analytic),
        "rel_fro_hessian_vs_ggn": M.rel_frobenius_error(H, G_ad),  # R = H - G (=0 for affine)
        "matvec_max_rel_err": max(matvec_errs),
        "numerical_rank_ggn": M.numerical_rank(eig_ad.eigvals),
        "eigvec_max_principal_angle": M.max_principal_angle(eig_ad.eigvecs, eig_an.eigvecs),
        "opg_fit_fro_over_ggn_fro": float(jnp.linalg.norm(opg_fit)) / max(g_fro, 1e-300),
        "opg_pert_numerical_rank": M.numerical_rank(jnp.linalg.eigvalsh(G.symmetrize(opg_pert))),
        "ggn_runtime_s": runtime_s,
    }

    # null-space recovery for rank-deficient cells
    if rank_kind != "full":
        V = lg.right_singular_vectors(kA, K, P, dtype)
        null_true = V[:, rank:]
        null_rec = eig_ad.eigvecs[:, rank:]
        out["nullspace_max_principal_angle"] = M.max_principal_angle(null_rec, null_true)

    arrays = {
        "ggn_ad": np.asarray(G_ad),
        "ggn_analytic": np.asarray(G_analytic),
        "eigvals_ggn": np.asarray(eig_ad.eigvals),
    }
    return out, arrays


def _fd_step_study(P=5, cond=1.0, dtype=jnp.float64, seed=0):
    """FD-Hessian relative error vs step size (separate, looser regime)."""
    K = 2 * P
    key = jax.random.PRNGKey(seed)
    kA, kz = jax.random.split(key)
    A = lg.random_A(kA, K, P, cond=cond, dtype=dtype)
    z_true = jax.random.normal(kz, (P,), dtype=dtype)
    model = lg.LinearGaussian(A=A, y=A @ z_true, W=None)
    loss = lg.make_loss(model)
    G_analytic = lg.analytic_ggn(model)
    curve = {}
    for h in _FD_STEPS:
        H_fd = G.finite_difference_hessian(loss, z_true, h)
        curve[f"{h:.0e}"] = M.rel_frobenius_error(H_fd, G_analytic)
    best = min(curve.values())
    return {"cond": cond, "P": P, "curve": curve, "best_rel_error": best}


def _new_run_dir(out_root: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{ts}_{prov.short_commit()}"
    d = Path(out_root) / run_id
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


def _figure(cell_arrays, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    G_an = cell_arrays["ggn_analytic"]
    G_ad = cell_arrays["ggn_ad"]
    R = G_an - G_ad
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, mat, title in zip(
        axes, [G_an, G_ad, R], ["analytic $A^T W A$", "AD GGN", "residual (analytic-AD)"]
    ):
        im = ax.imshow(mat, cmap="RdBu_r")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("EXP-001 FIG-01 — linear GGN recovery (P=5, cond=1)", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)


def run(P_list=(5, 20, 100), conds=(1.0, 1e2, 1e6), ranks=("full", "deficient"),
        dtypes=("float64", "float32"), seed=0, out_root="outputs/EXP-001") -> str:
    """Ordinary validation gate. Returns the run directory path."""
    require_x64()
    run_dir = _new_run_dir(out_root)

    weights_grid = ("none",)          # full grid uses W=None (cond target exact)
    weights_extra = ("none", "diag", "dense")  # weight-variant coverage at one cell

    cells, npz = {}, {}
    fig_arrays = None
    for P in P_list:
        for cond in conds:
            for rank_kind in ranks:
                for dt in dtypes:
                    for wk in weights_grid:
                        m, arr = _cell(P, cond, rank_kind, dt, wk, seed)
                        ckey = f"P{P}_cond{cond:.0e}_{rank_kind}_{dt}_{wk}"
                        cells[ckey] = m
                        for k, v in arr.items():
                            npz[f"{ckey}__{k}"] = v
                        if P == 5 and cond == 1.0 and rank_kind == "full" and dt == "float64":
                            fig_arrays = arr

    # weight-variant coverage (P=5, cond=1e2, full, float64)
    for wk in weights_extra:
        m, _ = _cell(5, 1e2, "full", "float64", wk, seed)
        cells[f"weightcov_{wk}"] = m

    fd_study = _fd_step_study(P=5, cond=1.0, dtype=jnp.float64, seed=seed)

    config = {
        "kind": "main", "P_list": list(P_list), "conds": list(conds),
        "ranks": list(ranks), "dtypes": list(dtypes), "seed": seed,
        "fd_steps": list(_FD_STEPS),
    }
    prov.write_json(run_dir / "config.json", config)
    prov.write_json(run_dir / "provenance.json",
                    prov.run_metadata("EXP-001", config, {"master": seed}, command=_COMMAND))
    prov.write_json(run_dir / "metrics.json", {"cells": cells, "fd_step_study": fd_study})
    np.savez_compressed(run_dir / "arrays.npz", **npz)
    if fig_arrays is not None:
        _figure(fig_arrays, run_dir / "figures" / "fig01_linear_benchmark")
    return str(run_dir)


def run_stress(P=20, cond=1e13, seed=0, out_root="outputs/EXP-001") -> str:
    """Non-gating ill-conditioning stress case (~1e13). Characterizes degradation."""
    require_x64()
    run_dir = _new_run_dir(out_root)
    cells, npz = {}, {}
    for dt in ("float64", "float32"):
        m, arr = _cell(P, cond, "full", dt, "none", seed)
        ckey = f"stress_P{P}_cond{cond:.0e}_{dt}"
        cells[ckey] = m
        for k, v in arr.items():
            npz[f"{ckey}__{k}"] = v
    fd_study = {dt: _fd_step_study(P=P, cond=cond, dtype=_DTYPE[dt], seed=seed)
                for dt in ("float64", "float32")}
    config = {"kind": "stress", "P": P, "cond": cond, "seed": seed}
    prov.write_json(run_dir / "config.json", config)
    prov.write_json(run_dir / "provenance.json",
                    prov.run_metadata("EXP-001-stress", config, {"master": seed}, command=_COMMAND))
    prov.write_json(run_dir / "metrics.json", {"cells": cells, "fd_step_study": fd_study})
    np.savez_compressed(run_dir / "arrays.npz", **npz)
    return str(run_dir)


if __name__ == "__main__":
    main_dir = run()
    print(f"main run:   {main_dir}")
    stress_dir = run_stress()
    print(f"stress run: {stress_dir}")
