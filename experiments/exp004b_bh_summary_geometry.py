"""EXP-004b — BH geometry with a finite SUMMARY-STATISTIC representation.

Fork-resolver for EXP-004: is the chaotic horizon-explosion of the GGN intrinsic,
or an artifact of the trajectory-RFF representation (which maximally amplifies
chaotic sensitivity)? Here the calibrated representation is a fixed vector of
robust summary statistics S(X) (moments, ACF lags, tail quantiles) — the
finite-summary route DEC-005 recommends as primary. If summaries tame the
explosion, EXP-004's limitation was representation-driven, not fundamental.

Run:  uv run python -m experiments.exp004b_bh_summary_geometry
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
from curvature_calib.calibration.per_seed_grads import vmap_simulate  # noqa: E402
from curvature_calib.models.brock_hommes import simulate  # noqa: E402

_COMMAND = "uv run python -m experiments.exp004b_bh_summary_geometry"

T, R_GR, SIGMA = 100, 1.01, 0.04
S_PRIOR = jnp.array([0.30, 0.10, 0.05, 0.10, 0.05])
POINTS = {"P1_complex": (50.0, 0.9, 0.2, 0.9, -0.2),
          "P2_chaotic": (80.0, 0.9, 0.2, 0.9, -0.2)}
HORIZONS = [1, 5, 10, 20, 50, None]
WORKING_HORIZON = 20
M = 128
_A = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0])
ALPHAS = np.concatenate([-_A[::-1], _A])
TAU = 0.10
Z0 = jnp.zeros(5)
_RTOL = 1e-8
_ACF_LAGS = (1, 2, 3, 5, 10)
_QUANTILES = jnp.array([0.05, 0.25, 0.5, 0.75, 0.95])


def summary_features(X):
    """Differentiable robust summaries of a trajectory X:(T,) -> (14,)."""
    mu = jnp.mean(X)
    sd = jnp.std(X) + 1e-8
    Xc = X - mu
    skew = jnp.mean(Xc ** 3) / sd ** 3
    kurt = jnp.mean(Xc ** 4) / sd ** 4
    acfs = jnp.array([jnp.mean(Xc[:-k] * Xc[k:]) / sd ** 2 for k in _ACF_LAGS])
    qs = jnp.quantile(X, _QUANTILES)
    return jnp.concatenate([jnp.array([mu, sd, skew, kurt]), acfs, qs])


def _sim(theta0, gh):
    return lambda z, key: simulate(jnp.asarray(theta0) + S_PRIOR * z, key,
                                   T=T, R=R_GR, sigma=SIGMA, grad_horizon=gh)


def _feats(theta0, gh, keys, z):
    sim = _sim(theta0, gh)
    return jax.vmap(lambda k: summary_features(sim(z, k)))(keys)   # (M, K)


def _eta_hat(theta0, gh, keys, w):
    """Whitened mean-embedding η̂(z) = w ⊙ mean_m S(X_z(ε_m))."""
    return lambda z: w * jnp.mean(_feats(theta0, gh, keys, z), axis=0)


def _spectrum(Gm):
    ev = np.sort(np.asarray(jnp.linalg.eigvalsh(Gm)))[::-1]
    return {"eigvals": ev.tolist(),
            "effective_rank": int((ev > _RTOL * ev[0]).sum()),
            "cond": float(ev[0] / max(ev[-1], 1e-300))}


def _point(theta0):
    keys = jax.random.split(jax.random.PRNGKey(0), M)
    # whitening weights: inverse across-seed std of each raw summary at z0
    F0 = np.asarray(_feats(theta0, WORKING_HORIZON, keys, Z0))
    w = jnp.asarray(1.0 / (F0.std(axis=0) + 1e-8))

    eta_ref = _eta_hat(theta0, WORKING_HORIZON, keys, w)
    eta_star = eta_ref(Z0)
    loss = lambda z: 0.5 * jnp.sum((eta_ref(z) - eta_star) ** 2)

    Gs, eigs, Jn = {}, {}, {}
    for h in HORIZONS:
        J = jax.jacfwd(_eta_hat(theta0, h, keys, w))(Z0)      # (K, P)
        Gs[str(h)] = jnp.asarray(GGN.symmetrize(J.T @ J)); eigs[str(h)] = DIAG.eigendecompose(Gs[str(h)])
        Jn[str(h)] = float(jnp.linalg.norm(J))
    G_full, eig_full = Gs["None"], eigs["None"]
    lam1_full, v1_full = float(eig_full.eigvals[0]), eig_full.eigvecs[:, :1]

    rows = {}
    for h in HORIZONS:
        eig = eigs[str(h)]
        v1, lam1 = eig.eigvecs[:, 0], float(eig.eigvals[0])
        rad = V.validity_radius(loss, Z0, v1, lam1, jnp.asarray(ALPHAS), TAU)
        rows[str(h)] = {
            "Jeta_norm": Jn[str(h)], "spectrum": _spectrum(Gs[str(h)]), "lead_eigval": lam1,
            "lead_eigval_ratio_vs_full": lam1 / max(lam1_full, 1e-300),
            "lead_angle_vs_full_deg": float(np.degrees(
                jnp.max(DIAG.principal_angles(eig.eigvecs[:, :1], v1_full)))),
            "rel_fro_vs_full": MET.rel_frobenius_error(Gs[str(h)], G_full),
            "validity_radius": rad,
        }
    Xp = vmap_simulate(_sim(theta0, WORKING_HORIZON), Z0, keys)
    return {"theta_physical": list(theta0), "traj_std": float(np.std(np.asarray(Xp))),
            "horizon_sweep": rows,
            "explosion_ratio_full_over_h1": Jn["None"] / max(Jn["1"], 1e-300)}


def run(out_root="outputs/EXP-004b") -> str:
    require_x64()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(out_root) / f"{ts}_{prov.short_commit()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    points = {name: _point(th) for name, th in POINTS.items()}
    config = {"T": T, "R": R_GR, "sigma": SIGMA, "points": {k: list(v) for k, v in POINTS.items()},
              "horizons": [str(h) for h in HORIZONS], "M": M,
              "summaries": ["mean", "std", "skew", "kurt", *[f"acf{k}" for k in _ACF_LAGS],
                            *[f"q{int(100*q)}" for q in np.asarray(_QUANTILES)]],
              "whitening": "inverse across-seed std at z0", "alphas": ALPHAS.tolist(), "tau": TAU}
    prov.write_json(run_dir / "config.json", config)
    prov.write_json(run_dir / "provenance.json",
                    prov.run_metadata("EXP-004b", config, {"sim": 0}, command=_COMMAND))
    prov.write_json(run_dir / "metrics.json", {"points": points})
    return str(run_dir)


if __name__ == "__main__":
    print("run:", run())
