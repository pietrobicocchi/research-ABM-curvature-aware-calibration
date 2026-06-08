"""Booklet 1, Figure 8: Erdős–Rényi contact network (concept)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curvature_calib.viz.booklet_annotate import callout  # noqa: E402
from curvature_calib.viz.booklet_style import apply_booklet_style, save_vector  # noqa: E402
from curvature_calib.viz.style import REGIME  # noqa: E402
from scripts.booklets._figbase import out_dir  # noqa: E402

OUT_AREA = "models"
OUT_NAME = "fig_08_sir_network"

# SIR compartment colors
_S_COLOR = REGIME["fundamental"]   # blue
_I_COLOR = REGIME["chaotic"]       # red
_R_COLOR = REGIME["periodic"]      # orange


def main() -> None:
    apply_booklet_style()

    rng = np.random.default_rng(3)
    N = 45
    pos = rng.uniform(0, 1, size=(N, 2))

    # Erdős–Rényi edges (p = 0.07)
    p = 0.07
    edges = [
        (i, j)
        for i in range(N)
        for j in range(i + 1, N)
        if rng.random() < p
    ]

    # Assign epidemic states: 6 Infected, 8 Recovered, rest Susceptible
    all_idx = np.arange(N)
    infected_idx = rng.choice(all_idx, size=6, replace=False)
    remaining = np.setdiff1d(all_idx, infected_idx)
    recovered_idx = rng.choice(remaining, size=8, replace=False)
    susceptible_idx = np.setdiff1d(remaining, recovered_idx)

    # Build per-node color array
    colors = np.empty(N, dtype=object)
    colors[susceptible_idx] = _S_COLOR
    colors[infected_idx] = _I_COLOR
    colors[recovered_idx] = _R_COLOR

    # Build adjacency for each infected node so we can pick one with neighbours
    adj: dict[int, list[int]] = {i: [] for i in range(N)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Pick an infected node that has at least one susceptible neighbour
    callout_node = int(infected_idx[0])  # fallback
    for cand in infected_idx:
        if any(nb in susceptible_idx for nb in adj[int(cand)]):
            callout_node = int(cand)
            break

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Edges
    for u, v in edges:
        xu, yu = pos[u]
        xv, yv = pos[v]
        ax.plot([xu, xv], [yu, yv], color="#cfd6dd", lw=0.7, zorder=1)

    # Nodes
    ax.scatter(
        pos[:, 0], pos[:, 1],
        c=list(colors),
        s=140,
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    ax.scatter([], [], c=_S_COLOR, s=80, label="Susceptible")
    ax.scatter([], [], c=_I_COLOR, s=80, label="Infected")
    ax.scatter([], [], c=_R_COLOR, s=80, label="Recovered")
    ax.legend(loc="upper left", frameon=True, fontsize=8, edgecolor="#cccccc")

    # ── Callout on the selected infected node ─────────────────────────────────
    cx, cy = pos[callout_node]
    # Place text label offset diagonally from the node
    offset_x = cx + 0.20 if cx < 0.75 else cx - 0.20
    offset_y = cy + 0.18 if cy < 0.80 else cy - 0.18
    callout(
        ax,
        xy=(cx, cy),
        text="infection spreads\nalong edges",
        xytext=(offset_x, offset_y),
        color=_I_COLOR,
        fontsize=8,
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    try:
        ax.set_title(
            "Network-SIR: transmission on an Erdős–Rényi contact graph",
            fontweight="bold",
        )
    except Exception:  # noqa: BLE001  # fallback for missing glyph
        ax.set_title(
            "Network-SIR: transmission on an Erdos-Renyi contact graph",
            fontweight="bold",
        )

    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)


if __name__ == "__main__":
    main()
