"""Booklet 2, Figure 8: BH OPG eigenvector content heatmap |V| (result)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curvature_calib.calibration.opg import eigendecompose  # noqa: E402
from curvature_calib.calibration.per_seed_grads import (  # noqa: E402
    per_seed_loss_and_grads,
    vmap_simulate,
)
from curvature_calib.models.brock_hommes import simulate  # noqa: E402
from curvature_calib.viz.booklet_annotate import (  # noqa: E402
    SLOPPY_COLOR,
    STIFF_COLOR,
    callout,
)
from curvature_calib.viz.booklet_style import apply_booklet_style, save_vector  # noqa: E402
from scripts.booklets._figbase import out_dir  # noqa: E402

OUT_AREA = "methodology"
OUT_NAME = "fig_08_eigenvector_heatmap"

# ── constants (verbatim from script 21) ──────────────────────────────────────
T = 200
SIGMA = 0.05
R = 1.1
THETA_STAR = jnp.array([3.0, 1.2, 0.2, 1.2, -0.2], dtype=jnp.float64)
THETA_EVAL = THETA_STAR + jnp.array([0.0, 0.05, 0.03, 0.05, -0.03],
                                     dtype=jnp.float64)
PARAM_NAMES = [r"$\beta$", r"$g_1$", r"$b_1$", r"$g_2$", r"$b_2$"]


def _sim(theta, key):
    return simulate(theta, key, T=T, sigma=SIGMA, R=R, x_init=0.0)


def main() -> None:
    apply_booklet_style()

    # ── computation (verbatim from script 21 / b2_07) ────────────────────────
    M_ref = 128
    M_eval = 200
    ref_keys = jax.random.split(jax.random.PRNGKey(0), M_ref)
    Y_ref = vmap_simulate(_sim, THETA_STAR, ref_keys)
    keys = jax.random.split(jax.random.PRNGKey(11), M_eval)
    stats = per_seed_loss_and_grads(_sim, THETA_EVAL, keys, Y_ref)

    F = np.asarray(stats.opg)
    eig = eigendecompose(jnp.asarray(F))
    eigvals = np.asarray(eig.eigvals)
    V = np.asarray(eig.eigvecs)

    # ── sanity print ─────────────────────────────────────────────────────────
    print("eigenvalues:", eigvals)
    print("\n|V| matrix (rows = parameters, cols = eigenvectors v_1..v_5):")
    print("       v_1    v_2    v_3    v_4    v_5")
    for i, name in enumerate(PARAM_NAMES):
        row = "  ".join(f"{np.abs(V[i, j]):.3f}" for j in range(5))
        print(f"  {name:>8s}  {row}")

    P = len(eigvals)
    xs = np.arange(P)

    # ── identify dominant parameters per column for annotations ──────────────
    abs_V = np.abs(V)
    # v_1 (stiff, col 0): top-2 rows
    stiff_col = abs_V[:, 0]
    stiff_top2 = np.argsort(-stiff_col)[:2]
    stiff_labels = [PARAM_NAMES[i] for i in stiff_top2]
    stiff_vals = [stiff_col[i] for i in stiff_top2]
    stiff_pairs = list(zip(stiff_labels, [f"{v:.3f}" for v in stiff_vals], strict=True))
    print(f"\nv_1 (stiff): top entries → {stiff_pairs}")

    # v_5 (sloppy, col 4): dominant row
    sloppy_col = abs_V[:, 4]
    sloppy_top = int(np.argmax(sloppy_col))
    sloppy_label = PARAM_NAMES[sloppy_top]
    sloppy_val = sloppy_col[sloppy_top]
    print(f"v_5 (sloppy): dominant entry → {sloppy_label} = {sloppy_val:.3f}")

    # ── figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    im = ax.imshow(abs_V, cmap="magma", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"$v_{k+1}$" for k in xs])
    ax.set_yticks(np.arange(P))
    ax.set_yticklabels(PARAM_NAMES)

    ax.set_title(
        r"OPG eigenvector content $|V|$ — which combinations the data constrains",
        fontweight="bold",
    )

    plt.colorbar(im, ax=ax, label=r"$|v_{k,j}|$")

    # Per-cell text annotations
    for i in range(P):
        for j in range(P):
            val = abs_V[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val < 0.55 else "black", fontsize=8)

    # ── callout annotations ───────────────────────────────────────────────────
    # v_1 (stiff): b_2 (0.716) + b_1 (0.696) — symmetric-bias combination
    stiff_text = r"$v_1$ (stiff): $b_1$+$b_2$" + "\nsymmetric bias\n(data constrains)"
    # v_5 (sloppy): beta alone (0.999) — almost entirely beta
    sloppy_text = r"$v_5$ (sloppy): $\beta$ alone" + "\n(data barely constrains)"

    # Callout pointing at the dominant cell in v_1 (b_2 row = row 4, col 0)
    callout(
        ax,
        xy=(0, stiff_top2[0]),
        text=stiff_text,
        xytext=(-0.7, stiff_top2[0] - 1.7),
        color=STIFF_COLOR,
        fontsize=8.0,
    )
    # Callout pointing at the dominant cell in v_5 (beta row = row 0, col 4)
    callout(
        ax,
        xy=(4, sloppy_top),
        text=sloppy_text,
        xytext=(4.0, sloppy_top - 1.7),
        color=SLOPPY_COLOR,
        fontsize=8.0,
    )

    fig.tight_layout()

    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)


if __name__ == "__main__":
    main()
