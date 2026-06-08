"""Booklet 2, Figure 11: Falsification protocol — stiff vs sloppy perturbation test.

Two panels:
  (a) Grouped bars: discrepancy under stiff (red) vs sloppy (grey) perturbation
      across 3 models × 3 non-MMD metrics (log y-axis)
  (b) Log-ratio heatmap: log10(stiff/sloppy) per model × metric, showing
      the diagnostic separates the two directions by 2–6 orders of magnitude

Data: outputs/paper/20_merged_falsification.npz (float64, canonical paper numbers)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curvature_calib.viz.booklet_annotate import SLOPPY_COLOR, STIFF_COLOR  # noqa: E402
from curvature_calib.viz.booklet_style import QUAL, apply_booklet_style, save_vector  # noqa: E402
from scripts.booklets._figbase import out_dir  # noqa: E402

OUT_AREA = "methodology"
OUT_NAME = "fig_11_falsification"

_DARK = "#2c3e50"
_GREY = "#7f8c8d"

_MODELS = ["Brock-Hommes", "mean-field SIR", "network-SIR (Gumbel)"]
_MODEL_LABELS = ["Brock–Hommes", "Mean-field SIR", "Network-SIR\n(Gumbel)"]
_METRICS = ["moments", "ACF", "quant"]
_METRIC_LABELS = ["Moments", "ACF", "Tail quantiles"]


def main() -> None:
    apply_booklet_style()

    root = Path(__file__).resolve().parents[3]
    d = np.load(root / "outputs" / "paper" / "20_merged_falsification.npz")

    # Build 3×3 arrays: rows=models, cols=metrics
    stiff = np.array([
        [d[f"{m}__{mt}_stiff"] for mt in _METRICS]
        for m in _MODELS
    ], dtype=float)
    sloppy = np.array([
        [d[f"{m}__{mt}_sloppy"] for mt in _METRICS]
        for m in _MODELS
    ], dtype=float)
    ratio = stiff / np.maximum(sloppy, 1e-30)

    print("Stiff discrepancies:")
    for i, m in enumerate(_MODELS):
        print(f"  {m}: {stiff[i]}")
    print("Sloppy discrepancies:")
    for i, m in enumerate(_MODELS):
        print(f"  {m}: {sloppy[i]}")
    print("Ratios (log10):")
    for i, m in enumerate(_MODELS):
        print(f"  {m}: {np.log10(ratio[i])}")

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.9, 1.0], wspace=0.38)
    ax_bars = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])

    # ── Panel (a): grouped bars ───────────────────────────────────────────────
    n_models = len(_MODELS)
    n_metrics = len(_METRICS)
    group_gap = 1.2
    bar_w = 0.28
    inner_gap = 0.08   # gap between stiff/sloppy pair within a metric
    metric_gap = 0.18  # gap between metric groups within a model

    x_ticks = []
    x_labels = []
    metric_tick_positions: list[list[float]] = [[] for _ in range(n_metrics)]

    x = 0.0
    for i, m_label in enumerate(_MODEL_LABELS):
        group_start = x
        for j in range(n_metrics):
            pos_stiff = x
            pos_sloppy = x + bar_w + inner_gap
            ax_bars.bar(pos_stiff, stiff[i, j], width=bar_w,
                        color=STIFF_COLOR, alpha=0.85)
            ax_bars.bar(pos_sloppy, sloppy[i, j], width=bar_w,
                        color=SLOPPY_COLOR, alpha=0.85)
            metric_tick_positions[j].append((pos_stiff + pos_sloppy) / 2)
            x += 2 * bar_w + inner_gap + metric_gap

        # Model label centred under its group
        group_center = (group_start + x - metric_gap) / 2
        x_ticks.append(group_center)
        x_labels.append(m_label)
        x += group_gap

    ax_bars.set_yscale("log")
    ax_bars.set_xticks(x_ticks)
    ax_bars.set_xticklabels(x_labels, fontsize=8.5)
    ax_bars.set_ylabel("non-MMD discrepancy", fontsize=9)
    ax_bars.set_title("(a)  Stiff vs sloppy perturbation — same magnitude, opposite effect",
                      fontweight="bold", fontsize=9.5)

    # Metric sub-labels beneath each pair (axes-fraction y so log scale doesn't matter)
    for j, (ml, positions) in enumerate(zip(_METRIC_LABELS, metric_tick_positions, strict=True)):
        for pos in positions:
            ax_bars.annotate(
                ml,
                xy=(pos, 0), xycoords=("data", "axes fraction"),
                xytext=(0, -14), textcoords="offset points",
                ha="center", va="top", fontsize=7, color=_GREY,
            )

    # Legend
    ax_bars.legend(
        handles=[
            mpatches.Patch(facecolor=STIFF_COLOR, alpha=0.85, label="stiff  $v_1$"),
            mpatches.Patch(facecolor=SLOPPY_COLOR, alpha=0.85, label="sloppy  $v_P$"),
        ],
        fontsize=8.5, loc="upper right", frameon=True,
    )

    # ── Panel (b): log-ratio heatmap ──────────────────────────────────────────
    log_ratio = np.log10(ratio)

    im = ax_heat.imshow(log_ratio, cmap="RdYlGn", aspect="auto",
                        vmin=2.0, vmax=6.5)

    for i in range(n_models):
        for j in range(n_metrics):
            val = ratio[i, j]
            txt = f"{val:.0f}×" if val < 1e6 else f"{val/1e6:.2f}M×"
            ax_heat.text(j, i, txt, ha="center", va="center",
                         fontsize=8, color="white" if log_ratio[i, j] > 4.5 else _DARK,
                         fontweight="bold")

    ax_heat.set_xticks(range(n_metrics))
    ax_heat.set_xticklabels(_METRIC_LABELS, fontsize=8.5)
    ax_heat.set_yticks(range(n_models))
    ax_heat.set_yticklabels(_MODEL_LABELS, fontsize=8)
    ax_heat.set_title("(b)  Ratio stiff / sloppy\n(log₁₀ colour scale)",
                      fontweight="bold", fontsize=9.5)
    cb = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cb.set_label(r"$\log_{10}$(ratio)", fontsize=8)

    fig.suptitle(
        "OPG falsification: stiff perturbations always detectable, "
        "sloppy ones always missed",
        fontweight="bold", fontsize=11,
    )

    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)


if __name__ == "__main__":
    main()
