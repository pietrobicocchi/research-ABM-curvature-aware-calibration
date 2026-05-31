"""Bifurcation sanity check for the Brock-Hommes simulator.

Sweep beta (intensity of choice) at zero noise; for each beta, simulate one
long trajectory and record the steady-state amplitude (max |x| over the last
window) and the unique-value count (a coarse periodicity proxy).

Expected qualitative behaviour, matching Brock & Hommes 1998 Sec. 3 with the
trend-follower + opposing-bias setup (g_h ~ 0.9, b_h = +/- 0.2):
    - small beta:                  amplitude -> 0       (stable fundamental)
    - intermediate beta:           amplitude > 0, low period count
    - large beta:                  amplitude > 0, high period count (chaos)

Run:
    uv run python scripts/01_brock_hommes_bifurcation.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from curvature_calib.models.brock_hommes import simulate


def amplitude_and_periodicity(xs: jax.Array, tail: int = 200) -> tuple[float, int]:
    tail_xs = np.asarray(xs[-tail:])
    amp = float(np.max(np.abs(tail_xs)))
    rounded = np.round(tail_xs, 3)
    return amp, int(np.unique(rounded).size)


def main() -> None:
    # Brock & Hommes 1998 Example 3.1 canonical values: g_2 = g_3 = 1.2,
    # b_2 = -b_3 = 0.2, R = 1.1. Trend coefficient g/R > 1 makes trend strategies
    # locally unstable; beta governs the selection pressure between them and
    # the fundamentalist.
    g, b, R = 1.2, 0.2, 1.1
    betas = np.linspace(0.0, 120.0, 80)
    T = 1500
    key = jax.random.PRNGKey(0)

    # Two diagnostics:
    #   (a) noiseless with strong initial perturbation -> deterministic regime.
    #   (b) noisy from rest -> trajectory variance, the calibration-relevant signal.
    amps_det, periods_det, vars_noisy = [], [], []
    for beta in betas:
        theta = jnp.array([beta, g, b, g, -b])
        xs_det = simulate(theta, key, T=T, sigma=0.0, x_init=1.0, R=R)
        a, p = amplitude_and_periodicity(xs_det)
        amps_det.append(a)
        periods_det.append(p)

        xs_noisy = simulate(theta, key, T=T, sigma=0.05, x_init=0.0, R=R)
        vars_noisy.append(float(jnp.var(xs_noisy[-500:])))

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    ax1.plot(betas, amps_det, "o-")
    ax1.set_xlabel(r"$\beta$")
    ax1.set_ylabel("max |x| over last 200 steps")
    ax1.set_title("Noiseless: amplitude (x_init=1.0)")
    ax1.grid(alpha=0.3)

    ax2.plot(betas, periods_det, "o-", color="C1")
    ax2.set_xlabel(r"$\beta$")
    ax2.set_ylabel("unique-value count")
    ax2.set_title("Noiseless: periodicity proxy")
    ax2.grid(alpha=0.3)

    ax3.plot(betas, vars_noisy, "o-", color="C2")
    ax3.set_xlabel(r"$\beta$")
    ax3.set_ylabel(r"Var(x) over last 500 steps")
    ax3.set_title(r"Noisy ($\sigma=0.05$): stationary variance")
    ax3.set_yscale("log")
    ax3.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_dir / "bh_bifurcation.png", dpi=120)
    print(f"saved {out_dir / 'bh_bifurcation.png'}")

    print(f"beta={betas[0]:.2f}  amp={amps_det[0]:.4f}  periods={periods_det[0]}  var_noisy={vars_noisy[0]:.4e}")
    mid = len(betas) // 2
    print(f"beta={betas[mid]:.2f}  amp={amps_det[mid]:.4f}  periods={periods_det[mid]}  var_noisy={vars_noisy[mid]:.4e}")
    print(f"beta={betas[-1]:.2f}  amp={amps_det[-1]:.4f}  periods={periods_det[-1]}  var_noisy={vars_noisy[-1]:.4e}")


if __name__ == "__main__":
    main()
