"""EXP-010 — Calibration-path stability of the local geometry (FIG-09).

Is the prior-relative local GGN geometry available DURING calibration, or only as
an offline snapshot at the optimum? Descend from a displaced start to the smooth
SIR fit, checkpointing z along the way, and at each checkpoint recompute the
prior-relative GGN. Track whether the leading eigenspace LOCKS IN (stops rotating)
and the data-dominant dimension stabilizes at a loss still clearly above
convergence — the property finite-difference / surrogate-network Hessians (offline,
single snapshot at the fit) do not have.

Supporting figure for C20 / the Background offline-snapshot contrast; NOT a
registered headline claim. See docs/experiments/EXP-010.md.

Run:  uv run python -m experiments.exp010_calibration_path
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

from curvature_calib import provenance as prov  # noqa: E402
from curvature_calib.calibration import diagnostic as DIAG  # noqa: E402
from curvature_calib.geometry import ggn as GGN  # noqa: E402
from experiments.exp005_sir_posterior import T, m, SIGMA_DATA, Z0, PARAM_NAMES  # noqa: E402

_COMMAND = "uv run python -m experiments.exp010_calibration_path"

# Displaced start: +1 prior-std in every coordinate (recorded initial loss checked
# against a floor). Adam handles the ~1e9 GGN condition number that plain GD cannot.
Z_INIT = Z0 + 1.0
LR = 0.05
N_STEPS = 800
CHECKPOINT_EVERY = 8
KS = (1, 2, 3)                     # leading-k subspaces to track (d_data = 3 here)
INIT_LOSS_FLOOR = 1e2             # sanity: initial profiled loss must exceed this


def _ggn(z):
    """Prior-relative GGN G(z) = J_mᵀ J_m / σ² (z prior-whitened ⇒ compare λ to 1)."""
    J = jax.jacfwd(m)(z)
    return GGN.symmetrize((J.T @ J) / SIGMA_DATA ** 2)


def _adam_path(loss, z0):
    """Adam descent; return (steps, zs) with a checkpoint every CHECKPOINT_EVERY
    steps plus the final iterate."""
    vg = jax.jit(jax.value_and_grad(loss))
    b1, b2, eps = 0.9, 0.999, 1e-8
    z = jnp.asarray(z0)
    mt = jnp.zeros_like(z)
    vt = jnp.zeros_like(z)
    steps, zs = [0], [np.asarray(z)]
    for t in range(1, N_STEPS + 1):
        _, g = vg(z)
        mt = b1 * mt + (1 - b1) * g
        vt = b2 * vt + (1 - b2) * g ** 2
        mhat = mt / (1 - b1 ** t)
        vhat = vt / (1 - b2 ** t)
        z = z - LR * mhat / (jnp.sqrt(vhat) + eps)
        if t % CHECKPOINT_EVERY == 0 or t == N_STEPS:
            steps.append(t)
            zs.append(np.asarray(z))
    return steps, zs


def run(out_root="outputs/EXP-010") -> str:
    require_x64()
    y = m(Z0)
    loss = lambda z: 0.5 * jnp.sum((m(z) - y) ** 2) / SIGMA_DATA ** 2

    L_init = float(loss(Z_INIT))
    if L_init < INIT_LOSS_FLOOR:
        raise RuntimeError(f"initial loss {L_init:.3g} below floor {INIT_LOSS_FLOOR}; "
                           "increase the Z_INIT displacement.")

    steps, zs = _adam_path(loss, Z_INIT)

    # geometry at each checkpoint; reference = the final (converged) iterate.
    eigs = [DIAG.eigendecompose(_ggn(jnp.asarray(z))) for z in zs]
    eig_ref = eigs[-1]
    L_final = float(loss(jnp.asarray(zs[-1])))

    checkpoints = []
    for step, z, eig in zip(steps, zs, eigs):
        lam = np.asarray(eig.eigvals)                       # descending
        angles = {}
        for k in KS:
            a = DIAG.principal_angles(eig.eigvecs[:, :k], eig_ref.eigvecs[:, :k])
            angles[k] = float(np.degrees(np.max(np.asarray(a))))
        checkpoints.append({
            "step": int(step),
            "loss": float(loss(jnp.asarray(z))),
            "prior_relative_eigs": lam.tolist(),
            "d_data": int((lam > 1).sum()),
            "angle_to_final_deg": {str(k): angles[k] for k in KS},
            "z": np.asarray(z).tolist(),
        })

    # --- lock-in reads (leading-2 subspace) --------------------------------
    total_drop = max(L_init - L_final, 1e-300)

    def _first_below(thresh_deg):
        for c in checkpoints:
            if c["angle_to_final_deg"]["2"] < thresh_deg:
                frac_remaining = (c["loss"] - L_final) / total_drop  # descent still to go
                return {"step": c["step"], "loss": c["loss"],
                        "loss_over_final": c["loss"] / max(L_final, 1e-300),
                        "frac_of_total_drop_remaining": float(frac_remaining)}
        return None

    d_data_final = checkpoints[-1]["d_data"]
    stabilize_step = None
    for i, c in enumerate(checkpoints):
        if all(cc["d_data"] == d_data_final for cc in checkpoints[i:]):
            stabilize_step = c["step"]
            break

    lockin = {
        "leading2_below_10deg": _first_below(10.0),
        "leading2_below_5deg": _first_below(5.0),
        "d_data_final": d_data_final,
        "d_data_stabilizes_at_step": stabilize_step,
        "d_data_stabilizes_at_loss_over_final": (
            next(c["loss"] for c in checkpoints if c["step"] == stabilize_step)
            / max(L_final, 1e-300) if stabilize_step is not None else None),
    }

    summary = {
        "loss_init": L_init,
        "loss_final": L_final,
        "n_checkpoints": len(checkpoints),
        "prior_relative_eigs_final": checkpoints[-1]["prior_relative_eigs"],
        "z_final": zs[-1].tolist(),
        "z_init": np.asarray(Z_INIT).tolist(),
        "lockin": lockin,
        "checkpoints": checkpoints,
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(out_root) / f"{ts}_{prov.short_commit()}"
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    config = {"reference": "EXP-005 SIR setup", "sigma_data": SIGMA_DATA,
              "z_init": np.asarray(Z_INIT).tolist(), "optimizer": "adam", "lr": LR,
              "n_steps": N_STEPS, "checkpoint_every": CHECKPOINT_EVERY,
              "ks_tracked": list(KS), "param_names": PARAM_NAMES}
    prov.write_json(run_dir / "config.json", config)
    prov.write_json(run_dir / "provenance.json",
                    prov.run_metadata("EXP-010", config, {"data": 0}, command=_COMMAND))
    prov.write_json(run_dir / "metrics.json", summary)
    steps_arr = np.array([c["step"] for c in checkpoints])
    np.savez_compressed(
        run_dir / "arrays.npz",
        steps=steps_arr,
        losses=np.array([c["loss"] for c in checkpoints]),
        eigs=np.array([c["prior_relative_eigs"] for c in checkpoints]),
        d_data=np.array([c["d_data"] for c in checkpoints]),
        angles=np.array([[c["angle_to_final_deg"][str(k)] for k in KS] for c in checkpoints]),
    )
    _figure(summary, run_dir / "figures" / "fig09_calibration_path")
    return str(run_dir)


def _figure(summary, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from curvature_calib.viz import style as STY
    except Exception:
        return
    STY.apply_style()
    R = STY.ROLE
    ch = summary["checkpoints"]
    steps = np.array([c["step"] for c in ch])
    losses = np.array([c["loss"] for c in ch])
    eigs = np.array([c["prior_relative_eigs"] for c in ch])       # (T, 5)
    d_data = np.array([c["d_data"] for c in ch])
    angles = {k: np.array([c["angle_to_final_deg"][str(k)] for c in ch]) for k in KS}

    fig, ax = plt.subplots(2, 2, figsize=STY.figsize("double", 0.62))

    # (a) descent — single series (neutral truth ink)
    ax[0, 0].semilogy(steps, np.maximum(losses, 1e-300), "-", color=R["truth"], lw=1.7)
    ax[0, 0].set_xlabel("iteration"); ax[0, 0].set_ylabel("profiled fit loss (nats)")
    ax[0, 0].set_title("descent")
    STY.panel_label(ax[0, 0], "a")

    # (b) spectrum — ordered stiff→sloppy family, sequential ramp
    ecol = STY.seq_colors(eigs.shape[1])
    for j in range(eigs.shape[1]):
        ax[0, 1].semilogy(steps, np.maximum(eigs[:, j], 1e-300), "-", lw=1.5,
                          color=ecol[j], label=rf"$\lambda_{{{j+1}}}$")
    ax[0, 1].axhline(1.0, ls=STY.LS["ref"], c=R["ref"], lw=1.2)
    ax[0, 1].text(steps[-1], 1.0, r" $\lambda=1$", va="center", ha="left",
                  fontsize=7, color=R["ref"])
    ax[0, 1].set_xlabel("iteration"); ax[0, 1].set_ylabel(r"prior-relative $\lambda$")
    ax[0, 1].set_title("spectrum settling"); ax[0, 1].legend(fontsize=7, ncol=2)
    STY.panel_label(ax[0, 1], "b")

    # (c) eigenspace stability — ordered leading-k family, sequential ramp
    kcol = STY.seq_colors(len(KS))
    for c, k in zip(kcol, KS):
        ax[1, 0].plot(steps, angles[k], "-", lw=1.7, color=c, label=f"leading-{k}")
    for lvl in (10.0, 5.0):
        ax[1, 0].axhline(lvl, ls=STY.LS["ref"], c=R["ref"], lw=1.0)
    ax[1, 0].set_xlabel("iteration"); ax[1, 0].set_ylabel("angle to final (deg)")
    ax[1, 0].set_title("eigenspace stability along the path")
    ax[1, 0].legend()
    STY.panel_label(ax[1, 0], "c")

    # (d) d_data — single series
    ax[1, 1].step(steps, d_data, where="post", color=R["ggn"], lw=1.7)
    ax[1, 1].set_xlabel("iteration"); ax[1, 1].set_ylabel(r"data-dominant dim ($\lambda>1$)")
    ax[1, 1].set_ylim(-0.2, eigs.shape[1] + 0.2)
    ax[1, 1].set_title("data-dominant dimension")
    STY.panel_label(ax[1, 1], "d")

    fig.suptitle("The local geometry is stable along the whole calibration path",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    print("run:", run())
