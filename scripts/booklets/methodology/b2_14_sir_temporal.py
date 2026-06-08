"""Booklet 2, Figure 14: Time-varying identifiability in the mean-field SIR model.

Two stacked panels:
  (a) SIR epidemic curve I(t) at θ* — provides epidemic context.  Key events
      are annotated: the epidemic peak (~day 24) and the lockdown firing (day 80).
  (b) Per-parameter OPG diagonal F̂_{ii}(t) vs observation horizon t (log y).
      Each line shows how strongly the loss constrains parameter i if we observe
      only the first t days of the epidemic.

Key story:
  - I₀ (initial seeding): identifiable from the very first days (it governs
    the epidemic onset timing, so even 5 days of data carry its signal).
  - β (transmission): identifiable during the exponential growth phase (~days 5–30).
  - γ (recovery): identifiable once the epidemic turns over (~days 25–60).
  - t_lock_norm, f_lock: near-zero identifiability throughout — the lockdown
    fires at day 80 after the epidemic is already over, so observing the full
    T=200 trajectory provides essentially no signal about lockdown parameters.

Computation: for each horizon t, we clip the simulated output to the first t
days and compute the MMD loss against clipped reference trajectories.  The OPG
diagonal diag(F̂) gives the average squared partial derivative of the loss with
respect to each parameter — a direct per-parameter identifiability measure.

θ* = (β=0.40, γ=0.10, I₀=1e-3, t_lock_norm=0.40, f_lock=0.50)
     Lockdown fires at day 0.40 × 200 = 80.  Epidemic peak ≈ day 24.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curvature_calib.calibration.per_seed_grads import (  # noqa: E402
    per_seed_loss_and_grads,
    vmap_simulate,
)
from curvature_calib.models.sir import simulate as sir_simulate  # noqa: E402
from curvature_calib.viz.booklet_annotate import SLOPPY_COLOR, STIFF_COLOR  # noqa: E402
from curvature_calib.viz.booklet_style import QUAL, apply_booklet_style, save_vector  # noqa: E402
from scripts.booklets._figbase import out_dir  # noqa: E402

OUT_AREA = "methodology"
OUT_NAME = "fig_14_sir_temporal"

SIR_T   = 200
SIR_N   = 100_000.0
M_REF   = 64    # reference seeds per horizon
M_EVAL  = 64    # evaluation seeds per horizon

# Horizons at which we compute the OPG diagonal (days)
HORIZONS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 120, 150, 200]

# Parameter labels and colors (order matches θ* vector: β, γ, I₀, t_lock, f_lock)
_PARAM_LABELS  = [r"$\beta$", r"$\gamma$", r"$I_0$",
                  r"$t_{\rm lock}$", r"$f_{\rm lock}$"]
_PARAM_COLORS  = [QUAL[0], QUAL[1], STIFF_COLOR, "#aaaaaa", SLOPPY_COLOR]


def _full_sim(theta, key):
    """Always run T=SIR_T so that t_lock_norm=0.40 → day 80 regardless of horizon."""
    return sir_simulate(theta, key, T=SIR_T, N=SIR_N)


def _trunc_sim(t_trunc: int):
    """Return a sim function whose output is the first t_trunc days of the epidemic."""
    def _sim(theta, key):
        return _full_sim(theta, key)[:t_trunc]
    return _sim


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    apply_booklet_style()

    theta_star = jnp.array([0.40, 0.10, 1e-3, 0.40, 0.50], dtype=jnp.float64)
    P = len(theta_star)  # 5 parameters

    # ── Full epidemic curve (for panel a) ────────────────────────────────────
    full_curve = np.asarray(_full_sim(theta_star, jax.random.PRNGKey(0))) / SIR_N * 100
    peak_day   = int(full_curve.argmax())
    lock_day   = int(0.40 * SIR_T)
    print(f"Epidemic peak: day {peak_day}, {full_curve.max():.2f}%   Lock day: {lock_day}")

    # ── OPG diagonal vs horizon ──────────────────────────────────────────────
    opg_diag = np.zeros((len(HORIZONS), P), dtype=np.float64)

    for h_idx, t in enumerate(HORIZONS):
        sim_t = _trunc_sim(t)
        ref_keys  = jax.random.split(jax.random.PRNGKey(0), M_REF)
        Y_ref     = vmap_simulate(sim_t, theta_star, ref_keys)
        eval_keys = jax.random.split(jax.random.PRNGKey(1), M_EVAL)
        stats     = per_seed_loss_and_grads(sim_t, theta_star, eval_keys, Y_ref)
        opg_diag[h_idx] = np.diag(np.asarray(stats.opg))
        print(f"  horizon t={t:3d}: Fii = {opg_diag[h_idx]}")

    horizons = np.array(HORIZONS)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_epi, ax_id) = plt.subplots(
        2, 1, figsize=(10, 9),
        gridspec_kw={"hspace": 0.42, "height_ratios": [1.0, 2.2]},
    )

    # ── Panel (a): epidemic curve ─────────────────────────────────────────────
    t_full = np.arange(SIR_T)
    ax_epi.plot(t_full, full_curve, color=QUAL[1], lw=2.2)
    ax_epi.fill_between(t_full, 0, full_curve, color=QUAL[1], alpha=0.12)

    ax_epi.axvline(peak_day, color=QUAL[1], ls="--", lw=1.0, alpha=0.8)
    ax_epi.axvline(lock_day, color="#c0c0c0", ls="--", lw=1.0)

    ax_epi.text(peak_day + 2, full_curve.max() * 0.85,
                f"epidemic\npeak (day {peak_day})",
                fontsize=7.5, color=QUAL[1], va="top")
    ax_epi.text(lock_day + 2, full_curve.max() * 0.55,
                f"lockdown fires\n(day {lock_day})\n→ epidemic over",
                fontsize=7.0, color="#999999", va="top")

    ax_epi.set_xlim(0, SIR_T)
    ax_epi.set_ylabel(r"infected $I(t)/N$  (%)", fontsize=9)
    ax_epi.set_title(
        r"(a)  Mean-field SIR at $\theta^*$: epidemic is over before the lockdown fires",
        fontweight="bold", fontsize=9,
    )
    ax_epi.set_xlabel("day", fontsize=9)

    # ── Panel (b): per-parameter OPG diagonal ────────────────────────────────
    eps_floor = 1e-30
    for i, (label, color) in enumerate(zip(_PARAM_LABELS, _PARAM_COLORS)):
        vals = np.clip(opg_diag[:, i], eps_floor, None)
        lw   = 2.5 if i in (0, 1, 2) else 1.5   # thicker for identifiable params
        ls   = "-"  if i in (0, 1, 2) else "--"
        ax_id.plot(horizons, vals, color=color, lw=lw, ls=ls,
                   marker="o", ms=4.5, label=label)

    ax_id.set_yscale("log")

    # Choose y position for bottom annotations = min visible Fii across all horizons
    y_bottom = float(np.clip(opg_diag, eps_floor, None).min()) * 0.5

    ax_id.axvline(peak_day, color=QUAL[1],   ls="--", lw=0.9, alpha=0.6)
    ax_id.axvline(lock_day, color="#c0c0c0", ls="--", lw=0.9)
    ax_id.text(peak_day + 2, y_bottom,
               f"peak\n(day {peak_day})", fontsize=7.0, color=QUAL[1], va="bottom")
    ax_id.text(lock_day + 2, y_bottom,
               f"lockdown\n(day {lock_day})", fontsize=7.0, color="#999999", va="bottom")
    ax_id.set_xlim(0, SIR_T)
    ax_id.set_xlabel("observation horizon $t$ (days)", fontsize=9)
    ax_id.set_ylabel(r"per-parameter OPG diagonal $\hat{F}_{ii}(t)$", fontsize=9)
    ax_id.set_title(
        r"(b)  Identifiability emerges at different epidemic phases; "
        r"$t_{\rm lock}$, $f_{\rm lock}$ remain non-identifiable",
        fontweight="bold", fontsize=9,
    )
    ax_id.legend(title="parameter", ncol=5, loc="upper left",
                 fontsize=8.5, title_fontsize=8, frameon=True,
                 handlelength=1.8, columnspacing=0.8)

    # Annotate which parameter leads identifiability at key epochs
    _annotate_lead(ax_id, horizons, opg_diag, eps_floor)

    fig.suptitle(
        "Time-varying identifiability: when can each SIR parameter be inferred?",
        fontweight="bold", fontsize=11,
    )

    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)
    print(f"Saved {OUT_NAME}")


def _annotate_lead(ax, horizons, opg_diag, eps_floor):
    """Annotate key identifiability phases above the I₀ line."""
    # I₀ (index 2) is always the top line — annotate it at three representative points
    phase_t   = [10,  40,  100]
    phase_txt = [r"$I_0$ identifiable",
                 r"$\beta$, $\gamma$ peak",
                 r"$t_{\rm lock}$, $f_{\rm lock}$" "\nnever identifiable"]
    phase_col = [STIFF_COLOR, QUAL[0], "#999999"]
    phase_ha  = ["left", "center", "right"]

    for t_check, txt, color, ha in zip(phase_t, phase_txt, phase_col, phase_ha):
        if t_check not in horizons.tolist():
            # find nearest horizon
            idx = int(np.argmin(np.abs(horizons - t_check)))
            t_check = int(horizons[idx])
        else:
            idx = horizons.tolist().index(t_check)

        # Place text above the dominant (I₀) value at this horizon
        y_top = float(np.clip(opg_diag[idx, 2], eps_floor, None))  # I₀ = index 2
        ax.text(t_check, y_top * 3.5, txt,
                fontsize=7.0, color=color, ha=ha, va="bottom", fontstyle="italic")


if __name__ == "__main__":
    main()
