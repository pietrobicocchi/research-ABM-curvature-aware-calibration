"""Booklet 2, Figure 13: Same ε, opposite identifiability consequence.

Two models × two perturbation directions:
  Row 1 — Brock-Hommes (stochastic financial ABM)
    (a) stiff direction v₁ (b₁+b₂ combination): price distributions separate
    (b) sloppy direction v_P (β intensity):       distributions completely overlap
  Row 2 — Mean-field SIR (deterministic epidemic model)
    (c) stiff direction v₁ (I₀ initial seeding): epidemic peak shifts ~7 days/decade
    (d) sloppy direction v_P (f_lock strength):   curves identical — epidemic over
        before lockdown fires (day 80)

For BH: OPG eigenvectors are recomputed at θ* (M=100 seeds); ε=0.05 applied
along v₁ and v_P.  The same ε in parameter space causes dramatically different
observable effects, which is the point of the figure.

For SIR: physically-motivated parameter sweeps that match the known eigenvector
structure (v₁ ≈ I₀, v_P ≈ f_lock) — no OPG recompute needed.
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curvature_calib.calibration.opg import eigendecompose  # noqa: E402
from curvature_calib.calibration.per_seed_grads import (  # noqa: E402
    per_seed_loss_and_grads,
    vmap_simulate,
)
from curvature_calib.models.brock_hommes import simulate as bh_simulate  # noqa: E402
from curvature_calib.models.sir import simulate as sir_simulate  # noqa: E402
from curvature_calib.viz.booklet_annotate import SLOPPY_COLOR, STIFF_COLOR  # noqa: E402
from curvature_calib.viz.booklet_style import QUAL, apply_booklet_style, save_vector  # noqa: E402
from scripts.booklets._figbase import out_dir  # noqa: E402

OUT_AREA = "methodology"
OUT_NAME = "fig_13_indistinguishable"

BH_T     = 200
BH_SIGMA = 0.05
BH_R     = 1.1
BH_EPS   = 0.05    # parameter-space perturbation magnitude (same for stiff & sloppy)
BH_M_OPG = 100     # seeds for OPG computation
BH_M_VIZ = 25      # seeds for trajectory visualisation

SIR_T = 200
SIR_N = 100_000.0
SIR_LOCK_DAY = 80  # = 0.40 × T


def _bh_sim(theta, key):
    return bh_simulate(theta, key, T=BH_T, sigma=BH_SIGMA, R=BH_R, x_init=0.0)


def _sir_sim(theta):
    return sir_simulate(theta, jax.random.PRNGKey(0), T=SIR_T, N=SIR_N)


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    apply_booklet_style()

    # ── BH: compute OPG eigenvectors at θ* ──────────────────────────────────
    theta_bh = jnp.array([3.0, 1.2, 0.2, 1.2, -0.2], dtype=jnp.float64)

    ref_keys  = jax.random.split(jax.random.PRNGKey(0), BH_M_OPG)
    Y_ref     = vmap_simulate(_bh_sim, theta_bh, ref_keys)
    eval_keys = jax.random.split(jax.random.PRNGKey(7), BH_M_OPG)
    stats     = per_seed_loss_and_grads(_bh_sim, theta_bh, eval_keys, Y_ref)
    eig       = eigendecompose(jnp.asarray(stats.opg))

    V  = np.asarray(eig.eigvecs)
    v1 = V[:, 0]    # stiff: mainly b₁, b₂
    vP = V[:, -1]   # sloppy: mainly β

    print("BH eigenvalues:", np.asarray(eig.eigvals))
    print("BH v₁:", v1)
    print("BH vP:", vP)

    theta_bh_stiff  = theta_bh + BH_EPS * jnp.array(v1)
    theta_bh_sloppy = theta_bh + BH_EPS * jnp.array(vP)

    print("Δθ stiff  :", np.array(theta_bh_stiff  - theta_bh))
    print("Δθ sloppy :", np.array(theta_bh_sloppy - theta_bh))

    # ── BH: simulate M_VIZ seed bundles ─────────────────────────────────────
    viz_keys = jax.random.split(jax.random.PRNGKey(42), BH_M_VIZ)

    sim_base   = partial(_bh_sim, theta_bh)
    sim_stiff  = partial(_bh_sim, theta_bh_stiff)
    sim_sloppy = partial(_bh_sim, theta_bh_sloppy)

    trajs_base   = np.asarray(jax.vmap(sim_base)(viz_keys))    # (M, T)
    trajs_stiff  = np.asarray(jax.vmap(sim_stiff)(viz_keys))
    trajs_sloppy = np.asarray(jax.vmap(sim_sloppy)(viz_keys))

    # ── SIR: parameter sweeps ────────────────────────────────────────────────
    sir_star = jnp.array([0.40, 0.10, 1e-3, 0.40, 0.50], dtype=jnp.float64)

    # Stiff: I₀ × 0.1, ×1, ×10  (v₁ ≈ I₀ direction)
    sir_I_lo = jnp.array([0.40, 0.10, 1e-4, 0.40, 0.50], dtype=jnp.float64)
    sir_I_hi = jnp.array([0.40, 0.10, 1e-2, 0.40, 0.50], dtype=jnp.float64)

    # Sloppy: f_lock ∈ {0, 0.5, 1}  (v_P ≈ f_lock direction)
    sir_f_lo = jnp.array([0.40, 0.10, 1e-3, 0.40, 0.00], dtype=jnp.float64)
    sir_f_hi = jnp.array([0.40, 0.10, 1e-3, 0.40, 1.00], dtype=jnp.float64)

    pct = 100.0 / SIR_N
    sir_base   = np.asarray(_sir_sim(sir_star))  * pct
    sir_I_lo_c = np.asarray(_sir_sim(sir_I_lo))  * pct
    sir_I_hi_c = np.asarray(_sir_sim(sir_I_hi))  * pct
    sir_f_lo_c = np.asarray(_sir_sim(sir_f_lo))  * pct
    sir_f_hi_c = np.asarray(_sir_sim(sir_f_hi))  * pct

    peak_pct = sir_base.max()
    peak_day = int(sir_base.argmax())
    print(f"SIR peak: day {peak_day}, {peak_pct:.2f}%")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5),
                             gridspec_kw={"hspace": 0.48, "wspace": 0.33})

    t_bh  = np.arange(BH_T)
    t_sir = np.arange(SIR_T)

    _BASE  = "#555555"
    _ALPHA = 0.20    # individual trace transparency

    # ── (a) BH stiff ─────────────────────────────────────────────────────────
    ax = axes[0, 0]
    for i in range(BH_M_VIZ):
        ax.plot(t_bh, trajs_base[i],  color=_BASE,       lw=0.45, alpha=_ALPHA)
        ax.plot(t_bh, trajs_stiff[i], color=STIFF_COLOR, lw=0.45, alpha=_ALPHA)
    ax.plot(t_bh, trajs_base.mean(0),  color=_BASE,       lw=2.2,
            label=r"baseline $\theta^*$")
    ax.plot(t_bh, trajs_stiff.mean(0), color=STIFF_COLOR, lw=2.2,
            label=rf"$\theta^*+\varepsilon\, v_1$  ($\varepsilon={BH_EPS}$)")
    ax.legend(fontsize=8, loc="upper right", handlelength=1.6, frameon=True)
    ax.set_ylabel(r"price deviation $x_t$", fontsize=9)
    ax.set_xlabel("time step", fontsize=9)
    ax.set_title(r"(a)  BH: stiff direction $v_1$  ($b_1{+}b_2$ combination)",
                 fontweight="bold", fontsize=9)
    ax.text(0.97, 0.04, "distributions separate",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color=STIFF_COLOR, fontstyle="italic")

    # ── (b) BH sloppy ────────────────────────────────────────────────────────
    ax = axes[0, 1]
    for i in range(BH_M_VIZ):
        ax.plot(t_bh, trajs_base[i],   color=_BASE,        lw=0.45, alpha=_ALPHA)
        ax.plot(t_bh, trajs_sloppy[i], color=SLOPPY_COLOR, lw=0.45, alpha=_ALPHA)
    ax.plot(t_bh, trajs_base.mean(0),   color=_BASE,        lw=2.2,
            label=r"baseline $\theta^*$")
    ax.plot(t_bh, trajs_sloppy.mean(0), color=SLOPPY_COLOR, lw=2.2,
            label=rf"$\theta^*+\varepsilon\, v_P$  ($\varepsilon={BH_EPS}$)")
    ax.legend(fontsize=8, loc="upper right", handlelength=1.6, frameon=True)
    ax.set_xlabel("time step", fontsize=9)
    ax.set_title(r"(b)  BH: sloppy direction $v_P$  ($\beta$ intensity)",
                 fontweight="bold", fontsize=9)
    ax.text(0.97, 0.04, "distributions overlap",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color=SLOPPY_COLOR, fontstyle="italic")

    # ── (c) SIR stiff (I₀ sweep) ─────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(t_sir, sir_I_lo_c, color=STIFF_COLOR, lw=2.0, alpha=0.75, ls="--",
            label=r"$I_0=10^{-4}$  (0.1× $\theta^*$)")
    ax.plot(t_sir, sir_base,   color=_BASE,       lw=2.6,
            label=r"$I_0=10^{-3}$ = $\theta^*$")
    ax.plot(t_sir, sir_I_hi_c, color=STIFF_COLOR, lw=2.0, alpha=0.75, ls=":",
            label=r"$I_0=10^{-2}$  (10× $\theta^*$)")
    ax.legend(fontsize=7.5, loc="upper right", handlelength=2.2, frameon=True)
    ax.set_ylabel(r"infected $I(t)/N$  (%)", fontsize=9)
    ax.set_xlabel("day", fontsize=9)
    ax.set_title(r"(c)  SIR: stiff direction $v_1$  ($I_0$ initial seeding)",
                 fontweight="bold", fontsize=9)
    # Annotate peak shift arrow
    ax.annotate(
        r"peak shifts $\approx7$ days" "\nper decade of $I_0$",
        xy=(peak_day, peak_pct),
        xytext=(peak_day + 18, peak_pct * 0.78),
        fontsize=7.5, color=STIFF_COLOR,
        arrowprops=dict(arrowstyle="->", color=STIFF_COLOR, lw=1.0),
        ha="left",
    )

    # ── (d) SIR sloppy (f_lock sweep) ────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(t_sir, sir_f_lo_c, color=QUAL[2],     lw=3.2, alpha=0.45, ls="--",
            label=r"$f_{\rm lock}=0$  (no lockdown)")
    ax.plot(t_sir, sir_base,   color=_BASE,        lw=2.6,
            label=r"$f_{\rm lock}=0.5$ = $\theta^*$")
    ax.plot(t_sir, sir_f_hi_c, color=SLOPPY_COLOR, lw=2.0, ls=":",
            label=r"$f_{\rm lock}=1$  (full lockdown)")
    ax.axvline(SIR_LOCK_DAY, color="#c0c0c0", ls=":", lw=1.2, zorder=0)
    ax.text(SIR_LOCK_DAY + 2, peak_pct * 0.60,
            f"lockdown fires\n(day {SIR_LOCK_DAY})\n—epidemic\nalready over",
            fontsize=7.0, color="#999999", va="top")
    ax.legend(fontsize=7.5, loc="upper right", handlelength=2.2, frameon=True)
    ax.set_xlabel("day", fontsize=9)
    ax.set_title(r"(d)  SIR: sloppy direction $v_P$  ($f_{\rm lock}$ strength)",
                 fontweight="bold", fontsize=9)
    ax.text(0.97, 0.04, "curves identical",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color=SLOPPY_COLOR, fontstyle="italic")

    fig.suptitle(
        rf"Same perturbation magnitude $|\varepsilon|={BH_EPS}$ in parameter space:"
        "\nstiff direction changes outputs — sloppy direction does not",
        fontweight="bold", fontsize=11,
    )

    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)
    print(f"Saved {OUT_NAME}")


if __name__ == "__main__":
    main()
