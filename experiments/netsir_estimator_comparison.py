"""Sec. 6: plug-in G_V (Eq. 14) against cross-seed G_U (Eq. 16) on the same per-seed
feature Jacobians, at the windows of Fig. 7b-c. No figure."""
from __future__ import annotations

from gndiag.config import enable_x64, load_config

enable_x64()

import numpy as np  # noqa: E402

from gndiag import io  # noqa: E402
from gndiag.diagnostic.estimators import ggn_cross_seed, ggn_plugin  # noqa: E402
from gndiag.diagnostic.ggn import eigendecompose  # noqa: E402
from experiments.netsir_observation_window import window_jacobians  # noqa: E402

NAME = "netsir_estimator_comparison"


def _angle_deg(u, v):
    c = abs(float(np.dot(u, v))) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(min(c, 1.0))))


def compute():
    cfg = load_config(NAME)
    cfg_window = load_config("netsir_observation_window")
    windows = []
    for T in cfg["windows"]:
        attack, A = window_jacobians(cfg_window, T)
        ev = eigendecompose(ggn_plugin(A))
        eu = eigendecompose(ggn_cross_seed(A))
        lam_v, lam_u = np.asarray(ev.eigvals), np.asarray(eu.eigvals)
        windows.append({
            "T": T, "attack_rate": attack,
            "eigvals_plugin": lam_v.tolist(), "eigvals_cross_seed": lam_u.tolist(),
            "sloppiest_angle_deg": _angle_deg(np.asarray(ev.eigvecs)[:, -1],
                                              np.asarray(eu.eigvecs)[:, -1]),
            # share of the smallest plug-in eigenvalue due to the O(1/M) bias (Eq. 15)
            "plugin_bias_fraction_of_lambda_min": float((lam_v[-1] - lam_u[-1]) / lam_v[-1]),
        })
    io.save_results(NAME, {"windows": windows}, cfg)


def plot():
    pass


if __name__ == "__main__":
    io.main(compute, plot)
