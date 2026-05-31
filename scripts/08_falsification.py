"""The §5.4 falsification protocol.

After calibration, identify
    v_P:  the SLOPPIEST direction (smallest eigenvalue)
    v_1:  the STIFFEST direction  (largest eigenvalue)

Perturb theta_T by +/- alpha * v in each direction (alpha = 10 * bootstrap
std of the projection), then compare simulator outputs at the perturbed
parameters against outputs at theta_T under THREE non-MMD discrepancies:
    - moments (mean, std, skew, kurt)
    - autocorrelation function
    - tail quantiles (1, 5, 95, 99 %)

If the OPG eigenstructure captures genuine simulator non-identifiability,
perturbation along v_P should be near-invisible under all three; perturbation
along v_1 should produce clearly distinguishable outputs.

Run: uv run python scripts/08_falsification.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

from curvature_calib.calibration.opg import eigendecompose
from curvature_calib.calibration.per_seed_grads import vmap_simulate
from curvature_calib.models.brock_hommes import simulate
from curvature_calib.viz.style import QUAL, apply_style, save


T = 200
SIGMA = 0.05
R = 1.1

PARAM_NAMES = [r"$\beta$", r"$g_1$", r"$b_1$", r"$g_2$", r"$b_2$"]


def _sim(theta, key):
    return simulate(theta, key, T=T, sigma=SIGMA, R=R, x_init=0.0)


def autocorr_mean(X, max_lag=20):
    out = np.zeros(max_lag + 1)
    for m in range(X.shape[0]):
        x = X[m] - X[m].mean()
        var = x.var() + 1e-12
        out += np.array([np.mean(x[: x.size - k] * x[k:]) / var
                          for k in range(max_lag + 1)])
    return out / X.shape[0]


def four_moments(X):
    x = X.reshape(-1)
    return np.array([x.mean(), x.std(),
                     float(sps.skew(x)), float(sps.kurtosis(x))])


def tail_quantiles(X, qs=(0.01, 0.05, 0.95, 0.99)):
    x = X.reshape(-1)
    return np.array([np.quantile(x, q) for q in qs])


def main() -> None:
    apply_style()

    log_path = Path("outputs/calibration_log.npz")
    if not log_path.exists():
        raise FileNotFoundError(
            "outputs/calibration_log.npz not found. Run "
            "scripts/06_calibration_dashboard.py first."
        )
    log = np.load(log_path)

    theta_T = jnp.asarray(log["thetas"][-1])
    F_final = jnp.asarray(log["opgs"][-1])
    eig = eigendecompose(F_final)
    eigvals = np.asarray(eig.eigvals)
    V = np.asarray(eig.eigvecs)
    v_stiff = jnp.asarray(V[:, 0])
    v_sloppy = jnp.asarray(V[:, -1])

    # Effective std along each direction: 1/sqrt(M*lambda_k) is the variability
    # of the mean gradient projection at a fixed sample size, but for a clean
    # falsification we want a perturbation that is well beyond statistical
    # noise. Use alpha = 10 * standard error along the direction.
    # SE_along_v = sqrt( v^T cov(g) v / M ).  cov(g) ~ F here.
    P = F_final.shape[0]
    M_eff = 64
    se_stiff = float(np.sqrt(max(eigvals[0], 1e-30) / M_eff))
    se_sloppy = float(np.sqrt(max(eigvals[-1], 1e-30) / M_eff))

    # Match the absolute perturbation magnitude across directions so the
    # comparison is honest: equal step size in theta-space, not statistically
    # equalised.  This is what "10 x sloppy SE in sloppy direction, same
    # magnitude in stiff direction" comes down to.
    alpha = max(10.0 * se_sloppy, 1e-2)
    print(f"alpha = {alpha:.4e}")
    print(f"eigvals: {eigvals}")

    theta_p_stiff  = theta_T + alpha * v_stiff
    theta_m_stiff  = theta_T - alpha * v_stiff
    theta_p_sloppy = theta_T + alpha * v_sloppy
    theta_m_sloppy = theta_T - alpha * v_sloppy

    M_show = 128
    keys = jax.random.split(jax.random.PRNGKey(404), M_show)
    X_T          = np.asarray(vmap_simulate(_sim, theta_T,          keys))
    X_p_stiff    = np.asarray(vmap_simulate(_sim, theta_p_stiff,    keys))
    X_m_stiff    = np.asarray(vmap_simulate(_sim, theta_m_stiff,    keys))
    X_p_sloppy   = np.asarray(vmap_simulate(_sim, theta_p_sloppy,   keys))
    X_m_sloppy   = np.asarray(vmap_simulate(_sim, theta_m_sloppy,   keys))

    # Discrepancies relative to theta_T.
    def disc_moments(X_a):
        return np.abs(four_moments(X_a) - four_moments(X_T))

    def disc_acf(X_a):
        return np.abs(autocorr_mean(X_a) - autocorr_mean(X_T))

    def disc_quant(X_a):
        return np.abs(tail_quantiles(X_a) - tail_quantiles(X_T))

    # Aggregate scalar disagreement: sum over the relevant components.
    rows = []
    for name, X_a in [
        (r"$+\alpha v_1$ (stiff)",  X_p_stiff),
        (r"$-\alpha v_1$ (stiff)",  X_m_stiff),
        (r"$+\alpha v_P$ (sloppy)", X_p_sloppy),
        (r"$-\alpha v_P$ (sloppy)", X_m_sloppy),
    ]:
        rows.append({
            "name": name,
            "moments": disc_moments(X_a),
            "acf": disc_acf(X_a),
            "quant": disc_quant(X_a),
        })

    # --------------------------- plot
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.40,
                          height_ratios=[1.0, 1.0, 1.0])

    # Top: sample trajectories.
    for col, (label, X_a, color) in enumerate([
        (r"$\theta_T$",            X_T,        QUAL[0]),
        (r"$+\alpha v_1$ (stiff)", X_p_stiff,  QUAL[1]),
        (r"$+\alpha v_P$ (sloppy)", X_p_sloppy, QUAL[2]),
    ]):
        ax = fig.add_subplot(gs[0, col])
        for m in range(min(8, X_a.shape[0])):
            ax.plot(X_a[m], color=color, lw=0.9, alpha=0.6)
        ax.set_title(f"trajectories: {label}")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$x_t$")

    # 0,3: eigenvalues + which is stiff/sloppy
    ax = fig.add_subplot(gs[0, 3])
    xs = np.arange(P)
    ax.semilogy(xs, eigvals, "o-", color=QUAL[0], markersize=10, lw=1.5,
                markerfacecolor=QUAL[0], markeredgecolor="white",
                markeredgewidth=1.4)
    ax.scatter([0], [eigvals[0]], s=320, marker="o",
               edgecolor=QUAL[1], facecolor="none", linewidth=2.2,
               label="stiff $v_1$")
    ax.scatter([P - 1], [eigvals[-1]], s=320, marker="o",
               edgecolor=QUAL[2], facecolor="none", linewidth=2.2,
               label="sloppy $v_P$")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$v_{k+1}$" for k in xs])
    ax.set_ylabel(r"$\lambda_k$")
    ax.set_title("Eigenvalues at $\\theta_T$ — direction selection")
    ax.legend(fontsize=8)

    # Middle / bottom: discrepancy bars per channel.
    moment_names = ["mean", "std", "skew", "kurt"]
    ax = fig.add_subplot(gs[1, 0:2])
    width = 0.18
    xs_m = np.arange(len(moment_names))
    for j, row in enumerate(rows):
        color = QUAL[1] if "stiff" in row["name"] else QUAL[2]
        ax.bar(xs_m + (j - 1.5) * width, row["moments"], width,
               color=color, alpha=0.5 if j % 2 else 0.95,
               edgecolor="white", label=row["name"])
    ax.set_xticks(xs_m)
    ax.set_xticklabels(moment_names)
    ax.set_ylabel(r"$|$stat at perturbed $-$ stat at $\theta_T|$")
    ax.set_title("Channel 1: first four moments of $x_t$")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.set_yscale("log")

    ax = fig.add_subplot(gs[1, 2:])
    max_lag = rows[0]["acf"].size - 1
    lags = np.arange(max_lag + 1)
    for j, row in enumerate(rows):
        color = QUAL[1] if "stiff" in row["name"] else QUAL[2]
        ax.plot(lags, row["acf"], "-",
                color=color, alpha=0.5 if j % 2 else 0.95,
                lw=1.6, label=row["name"])
    ax.set_xlabel("lag k")
    ax.set_ylabel(r"$|$ACF at perturbed $-$ ACF at $\theta_T|$")
    ax.set_title("Channel 2: autocorrelation discrepancy")
    ax.legend(fontsize=8, ncol=2)
    ax.set_yscale("log")

    qs = ["1%", "5%", "95%", "99%"]
    ax = fig.add_subplot(gs[2, 0:2])
    xs_q = np.arange(len(qs))
    for j, row in enumerate(rows):
        color = QUAL[1] if "stiff" in row["name"] else QUAL[2]
        ax.bar(xs_q + (j - 1.5) * width, row["quant"], width,
               color=color, alpha=0.5 if j % 2 else 0.95,
               edgecolor="white", label=row["name"])
    ax.set_xticks(xs_q)
    ax.set_xticklabels(qs)
    ax.set_xlabel("quantile")
    ax.set_ylabel(r"$|$quantile at perturbed $-$ quantile at $\theta_T|$")
    ax.set_title("Channel 3: tail quantile discrepancy")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.set_yscale("log")

    # Overall summary heatmap.
    summary = np.zeros((4, 3))
    for j, row in enumerate(rows):
        summary[j, 0] = row["moments"].sum()
        summary[j, 1] = row["acf"].sum()
        summary[j, 2] = row["quant"].sum()
    ax = fig.add_subplot(gs[2, 2:])
    im = ax.imshow(np.log10(summary + 1e-12), cmap="magma", aspect="auto")
    ax.set_yticks(range(4))
    ax.set_yticklabels([r["name"] for r in rows])
    ax.set_xticks(range(3))
    ax.set_xticklabels(["moments", "ACF", "quantiles"])
    ax.set_title("Aggregate discrepancy: $\\log_{10}\\sum$ |channel diff|")
    for i in range(4):
        for j in range(3):
            ax.text(j, i, f"{summary[i, j]:.2e}", ha="center", va="center",
                    color="white" if np.log10(summary[i, j] + 1e-12) < np.log10(summary).mean() else "black",
                    fontsize=9)
    plt.colorbar(im, ax=ax, label=r"$\log_{10}$ aggregate discrepancy")

    fig.suptitle(
        "§5.4 Falsification: sloppy vs stiff perturbations under non-MMD discrepancies",
        fontsize=14, fontweight="bold", y=0.995,
    )
    p = save(fig, "08_falsification.png")
    print(f"saved {p}")

    # Console summary.
    print("\nAggregate discrepancy (summed |channel diff|):")
    print(f"  {'perturbation':<28s}  {'moments':>10s}  {'ACF':>10s}  {'quant':>10s}")
    for j, row in enumerate(rows):
        m = row["moments"].sum()
        a = row["acf"].sum()
        q = row["quant"].sum()
        print(f"  {row['name']:<28s}  {m:10.3e}  {a:10.3e}  {q:10.3e}")


if __name__ == "__main__":
    main()
