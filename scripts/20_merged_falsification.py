"""Script 20: merged falsification panel (the paper's Figure 2).

Runs the §5.4 non-MMD falsification protocol on three models:
    * Brock-Hommes  (financial heterogeneous-agent, smooth)
    * mean-field SIR (epidemic, smooth)
    * network-SIR    (epidemic, discrete-state, Gumbel-Sigmoid surrogate)

For each model: locate stiff v_1 and sloppy v_P via the OPG eigendecomposition
at a theta_eval near theta*; perturb theta* by +/- alpha along each direction;
simulate at all five points using the same M=128 keys; measure three
non-MMD discrepancies (four moments, ACF up to lag 20, four tail quantiles).
Aggregate to scalar |Delta| per discrepancy.

Single figure, 3 rows (models) x 4 columns (moments, ACF, tail quantiles,
aggregate stiff/sloppy ratio).

This consolidates parts of scripts 08, 16, 18 into the canonical Figure 2
of the AI4ABM 2026 paper. Required by docs/memory/paper_story_arc.md.

Run: uv run python scripts/20_merged_falsification.py
"""

from __future__ import annotations

from pathlib import Path

import jax
# Enable float64. The SIR eigenstructure has condition number ~1e13;
# without float64 the bottom of the spectrum is float32 noise (negative
# eigenvalues) and v_P points in a numerically meaningless direction,
# under-stating the stiff/sloppy ratio by 2-3 orders of magnitude.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

from curvature_calib.calibration.opg import eigendecompose
from curvature_calib.calibration.per_seed_grads import (
    per_seed_loss_and_grads,
    vmap_simulate,
)
from curvature_calib.models.brock_hommes import simulate as bh_simulate
from curvature_calib.models.network_sir import simulate as net_simulate
from curvature_calib.models.sir import simulate as mf_simulate
from curvature_calib.viz.style import QUAL, apply_style, save


# ---------------------------------------------------------------------------
# Model configurations (kept identical to scripts 08, 16, 18 so headline
# numbers match what's already reported in memory).

T = 200
ALPHA = 1e-2
N_FALSIFY = 128
N_EIG = 96


def bh_simulate_fn(theta, key):
    return bh_simulate(theta, key, T=T, sigma=0.05, R=1.1, x_init=0.0,
                       grad_horizon=None)


def mf_simulate_fn(theta, key):
    return mf_simulate(theta, key, T=T, N=1e5, sigma_obs=10.0,
                       grad_horizon=None)


def net_simulate_fn(theta, key):
    return net_simulate(theta, key, T=T, N=250, mean_degree=6.0,
                        gumbel_tau=0.5, grad_horizon=None)


def _bh_theta_T_from_log():
    """Mirror script 08 convention: use the OPG saved at the last iterate of
    the saved BH calibration log (i.e. theta_T, not theta*). The two are very
    close — within ~1% — but the existing headline numbers report falsification
    at theta_T."""
    log_path = Path("outputs/brock_hommes/calibration_log.npz")
    if not log_path.exists():
        raise FileNotFoundError(
            f"Need {log_path}. Run scripts/06_calibration_dashboard.py first.")
    log = np.load(log_path)
    return jnp.asarray(log["thetas"][-1], dtype=jnp.float64)


MODEL_CONFIGS = [
    {
        "name": "Brock-Hommes",
        "simulate_fn": bh_simulate_fn,
        # OPG eigenbasis at theta_T (matches script 08).
        "theta_star": jnp.array([3.0, 1.2, 0.2, 1.2, -0.2], dtype=jnp.float64),
        "theta_eval": _bh_theta_T_from_log(),
        "theta_ref": "theta_eval",  # falsify reference = theta_T as in script 08
    },
    {
        "name": "mean-field SIR",
        "simulate_fn": mf_simulate_fn,
        "theta_star": jnp.array([0.40, 0.10, 1e-3, 0.40, 0.50], dtype=jnp.float64),
        "theta_eval": jnp.array([0.40, 0.10, 1e-3, 0.40, 0.50], dtype=jnp.float64)
                       + jnp.array([0.01, -0.005, 1e-4, 0.02, -0.02], dtype=jnp.float64),
        "theta_ref": "theta_star",
    },
    {
        "name": "network-SIR (Gumbel)",
        "simulate_fn": net_simulate_fn,
        "theta_star": jnp.array([0.30, 0.10, 0.05, 0.40, 0.50], dtype=jnp.float64),
        "theta_eval": jnp.array([0.30, 0.10, 0.05, 0.40, 0.50], dtype=jnp.float64)
                       + jnp.array([0.005, -0.003, 0.005, 0.02, -0.02], dtype=jnp.float64),
        "theta_ref": "theta_star",
    },
]


# ---------------------------------------------------------------------------
# Non-MMD discrepancy library.

def four_moments(X):
    x = np.asarray(X).reshape(-1)
    return np.array([x.mean(), x.std(),
                     float(sps.skew(x)), float(sps.kurtosis(x))])


def autocorr_mean(X, max_lag=20):
    X = np.asarray(X)
    out = np.zeros(max_lag + 1)
    for m in range(X.shape[0]):
        x = X[m] - X[m].mean()
        var = x.var() + 1e-12
        out += np.array([np.mean(x[: x.size - k] * x[k:]) / var
                         for k in range(max_lag + 1)])
    return out / X.shape[0]


def tail_quantiles(X, qs=(0.01, 0.05, 0.95, 0.99)):
    x = np.asarray(X).reshape(-1)
    return np.array([np.quantile(x, q) for q in qs])


def discrepancies(X_T, X_a):
    return {
        "moments": float(np.sum(np.abs(four_moments(X_a) - four_moments(X_T)))),
        "ACF":     float(np.sum(np.abs(autocorr_mean(X_a) - autocorr_mean(X_T)))),
        "quant":   float(np.sum(np.abs(tail_quantiles(X_a) - tail_quantiles(X_T)))),
    }


# ---------------------------------------------------------------------------
# Per-model falsification.

def run_one_model(cfg):
    """Return dict with eigvals + raw discrepancies for stiff +/- and sloppy +/- ."""
    print(f"\n=== {cfg['name']} ===")
    sim = cfg["simulate_fn"]
    theta_star = cfg["theta_star"]
    theta_eval = cfg["theta_eval"]

    # Reference Y at theta* (sets the MMD inner product).
    M_ref = 96
    ref_keys = jax.random.split(jax.random.PRNGKey(0), M_ref)
    Y_ref = vmap_simulate(sim, theta_star, ref_keys)

    # F_hat at theta_eval (BH: theta_T from log; SIR/netSIR: slight offset
    # from theta* so the per-seed-grad mean is not vanishingly small).
    eig_keys = jax.random.split(jax.random.PRNGKey(1), N_EIG)
    stats = per_seed_loss_and_grads(sim, theta_eval, eig_keys, Y_ref)
    eig = eigendecompose(stats.opg)
    eigvals = np.asarray(eig.eigvals)
    V = np.asarray(eig.eigvecs)
    v_stiff = jnp.asarray(V[:, 0], dtype=jnp.float64)
    v_sloppy = jnp.asarray(V[:, -1], dtype=jnp.float64)

    print(f"  eigvals: {eigvals}")
    print(f"  condition: {eigvals[0] / max(eigvals[-1], 1e-30):.2e}")

    # Falsification reference: same convention as the corresponding original
    # script. BH: theta_T; SIR/netSIR: theta_star.
    theta_ref = theta_eval if cfg["theta_ref"] == "theta_eval" else theta_star

    falsify_keys = jax.random.split(jax.random.PRNGKey(404), N_FALSIFY)

    def at(theta):
        return np.asarray(vmap_simulate(sim, theta, falsify_keys))

    X_T = at(theta_ref)
    perturbations = {
        r"$+\alpha v_1$ (stiff)":  at(theta_ref + ALPHA * v_stiff),
        r"$-\alpha v_1$ (stiff)":  at(theta_ref - ALPHA * v_stiff),
        r"$+\alpha v_P$ (sloppy)": at(theta_ref + ALPHA * v_sloppy),
        r"$-\alpha v_P$ (sloppy)": at(theta_ref - ALPHA * v_sloppy),
    }
    results = {name: discrepancies(X_T, X_a) for name, X_a in perturbations.items()}

    for name, d in results.items():
        print(f"  {name:30s} moments={d['moments']:.3e}  "
              f"ACF={d['ACF']:.3e}  quant={d['quant']:.3e}")

    return {
        "name": cfg["name"],
        "eigvals": eigvals,
        "results": results,
    }


def stiff_sloppy_ratios(model_out):
    """Aggregate +/- stiff and +/- sloppy then compute stiff/sloppy ratio per channel."""
    r = model_out["results"]
    out = {}
    for chan in ("moments", "ACF", "quant"):
        stiff = 0.5 * (r[r"$+\alpha v_1$ (stiff)"][chan]
                       + r[r"$-\alpha v_1$ (stiff)"][chan])
        sloppy = 0.5 * (r[r"$+\alpha v_P$ (sloppy)"][chan]
                        + r[r"$-\alpha v_P$ (sloppy)"][chan])
        out[chan] = {"stiff": stiff, "sloppy": sloppy,
                     "ratio": stiff / max(sloppy, 1e-30)}
    return out


def main() -> None:
    apply_style()
    out_dir = Path("outputs/paper")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running §5.4 falsification on three models...")
    all_out = [run_one_model(cfg) for cfg in MODEL_CONFIGS]

    # ============================================================ FIGURE
    fig, axes = plt.subplots(3, 4, figsize=(15, 10),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})

    channels = ["moments", "ACF", "quant"]
    channel_titles = {"moments": r"$\sum|\Delta$ moments$|$",
                      "ACF":     r"$\sum|\Delta$ ACF$|$",
                      "quant":   r"$\sum|\Delta$ tail quantiles$|$"}

    for i, mo in enumerate(all_out):
        ratios = stiff_sloppy_ratios(mo)

        for j, chan in enumerate(channels):
            ax = axes[i, j]
            bars = [mo["results"][n][chan] for n in mo["results"]]
            colors = [QUAL[1], QUAL[1], QUAL[2], QUAL[2]]  # stiff (orange), sloppy (green)
            alphas = [0.95, 0.55, 0.95, 0.55]
            xs = np.arange(4)
            for k in range(4):
                ax.bar(xs[k], bars[k], color=colors[k], alpha=alphas[k],
                       edgecolor="white", linewidth=0.8)
            ax.set_yscale("log")
            ax.set_xticks(xs)
            ax.set_xticklabels([r"$+v_1$", r"$-v_1$", r"$+v_P$", r"$-v_P$"],
                               fontsize=8)
            if i == 0:
                ax.set_title(channel_titles[chan], fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(mo["name"], fontsize=11, fontweight="bold")
            ratio = ratios[chan]["ratio"]
            ax.text(0.97, 0.93, f"ratio: {ratio:.2e}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, bbox=dict(facecolor="white",
                                          edgecolor="grey", alpha=0.85,
                                          boxstyle="round,pad=0.3"))

        # 4th column: per-model summary bar
        ax = axes[i, 3]
        chs = channels
        rats = [stiff_sloppy_ratios(mo)[c]["ratio"] for c in chs]
        ax.barh(range(3), rats, color=QUAL[0], alpha=0.85, edgecolor="white")
        ax.set_yticks(range(3))
        ax.set_yticklabels([channel_titles[c] for c in chs], fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("stiff / sloppy ratio")
        for k, r in enumerate(rats):
            ax.text(r * 1.1, k, f"{r:.1e}", va="center", fontsize=8)
        if i == 0:
            ax.set_title("aggregate ratios", fontsize=11, fontweight="bold")

    # Single legend at the bottom.
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=QUAL[1], alpha=0.95, label=r"stiff direction $v_1$"),
        Patch(facecolor=QUAL[2], alpha=0.95, label=r"sloppy direction $v_P$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=11)

    fig.suptitle(
        "§5.4 falsification across three models — non-MMD discrepancies, "
        r"$\alpha = $" + f"{ALPHA}",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = save(fig, "20_merged_falsification.png", out_dir=out_dir)
    print(f"\nSaved figure: {p}")

    # Save underlying ratios.
    summary = {}
    for mo in all_out:
        rs = stiff_sloppy_ratios(mo)
        summary[mo["name"]] = {
            "eigvals": mo["eigvals"],
            **{f"{c}_stiff": rs[c]["stiff"] for c in channels},
            **{f"{c}_sloppy": rs[c]["sloppy"] for c in channels},
            **{f"{c}_ratio": rs[c]["ratio"] for c in channels},
        }

    np.savez_compressed(
        out_dir / "20_merged_falsification.npz",
        **{f"{name}__{k}": np.asarray(v)
           for name, d in summary.items() for k, v in d.items()},
    )

    # Console table.
    print("\nFinal ratios table:")
    print(f"{'model':28s}  {'moments':>12s}  {'ACF':>12s}  {'quant':>12s}")
    for mo in all_out:
        rs = stiff_sloppy_ratios(mo)
        print(f"{mo['name']:28s}  "
              f"{rs['moments']['ratio']:12.3e}  "
              f"{rs['ACF']['ratio']:12.3e}  "
              f"{rs['quant']['ratio']:12.3e}")


if __name__ == "__main__":
    main()
