"""Phase 2 convergence comparison: OPG-preconditioned vs Adam vs SGD.

Project plan §6 Phase 2 protocol (down-scaled for one workstation):
    3 difficulty levels x 5 random (theta_0, theta*) pairs x 3 optimisers
    = 45 runs, 80 iterations each, M = 64 seeds.

Original plan was 3 x 20 x 4 = 240 runs incl. L-BFGS. We skip L-BFGS for
this first pass (would need jaxopt wiring) and reduce to 5 pairs to fit one
workstation budget. Easy to scale up once everything works.

Difficulty levels:
    easy  : ||theta_0 - theta*||_2 = 0.05  (basin-local)
    medium: ||theta_0 - theta*||_2 = 0.15
    hard  : ||theta_0 - theta*||_2 = 0.40

Why these distances (not the project plan's 0.1 / 0.5 / 1.5)?
    The plan's distances were specified before the explosion threshold was
    characterised. With theta* = (3, 1.2, 0.2, 1.2, -0.2) and R = 1.1, a
    perturbation that pushes g_h above ~1.5 makes g/R > 1.36 and trajectories
    diverge in float32. The smaller distances stay safely below this
    threshold for every random direction we sample.

Reported metrics per run:
    - parameter recovery error ||theta_T - theta*||_2 at the final iterate
    - iteration at which ||theta_t - theta*||_2 first drops below 0.5 * d
      ("iterations to halve")
    - final MMD^2 loss (a sanity check; convergence is judged in parameter
      space, since MMD bottoms out at its noise floor)

Run: uv run python scripts/10_phase2_convergence.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from curvature_calib.calibration.baselines import adam, sgd
from curvature_calib.calibration.calibrate import calibrate
from curvature_calib.calibration.per_seed_grads import vmap_simulate
from curvature_calib.losses.mmd import mmd_sq_with_median_bandwidth
from curvature_calib.models.brock_hommes import simulate
from curvature_calib.viz.style import QUAL, apply_style


T = 200
SIGMA = 0.05
R = 1.1
THETA_STAR = jnp.array([3.0, 1.2, 0.2, 1.2, -0.2])
PARAM_NAMES = [r"$\beta$", r"$g_1$", r"$b_1$", r"$g_2$", r"$b_2$"]

DIFFICULTIES = [
    ("easy",   0.05),
    ("medium", 0.15),
    ("hard",   0.40),
]
N_PAIRS = 5
N_ITER = 80
M = 64


def _sim(theta, key):
    return simulate(theta, key, T=T, sigma=SIGMA, R=R, x_init=0.0,
                    grad_horizon=None)  # full horizon -- killswitch verdict


def is_safe(theta_np: np.ndarray) -> bool:
    """Reject candidates that would push the simulator past the explosion
    threshold g_h / R > ~1.36, or produce numerical issues."""
    beta, g1, b1, g2, b2 = theta_np
    if beta <= 0 or beta > 50:
        return False
    if g1 > 1.45 or g2 > 1.45:
        return False
    if g1 < 0 or g2 < 0:
        return False
    if abs(b1) > 0.6 or abs(b2) > 0.6:
        return False
    return True


def produces_finite_loss(theta: jax.Array, ref: jax.Array,
                         key: jax.Array, M_check: int = 16) -> bool:
    keys = jax.random.split(key, M_check)
    X = vmap_simulate(_sim, theta, keys)
    if not bool(jnp.all(jnp.isfinite(X))):
        return False
    v = float(mmd_sq_with_median_bandwidth(X, ref))
    return np.isfinite(v)


def sample_safe_theta(theta_star: jax.Array, distance: float,
                      key: jax.Array, ref: jax.Array,
                      max_attempts: int = 60) -> jax.Array:
    star_np = np.asarray(theta_star)
    for attempt in range(max_attempts):
        k = jax.random.fold_in(key, attempt)
        u = jax.random.normal(k, (5,))
        u = u / jnp.linalg.norm(u)
        candidate = theta_star + distance * u
        cand_np = np.asarray(candidate)
        if is_safe(cand_np) and produces_finite_loss(candidate, ref, k):
            return candidate
    raise RuntimeError(
        f"No safe theta_0 at distance {distance} after {max_attempts} attempts."
    )


def run_one(opt_name: str, theta_0: jax.Array, Y_ref: jax.Array) -> dict:
    """Run one optimiser to N_ITER. Returns log + summary stats."""
    if opt_name == "opg":
        log = calibrate(_sim, theta_0, Y_ref, M=M, n_iter=N_ITER,
                        verbose=False)
        arrs = log.as_arrays()
        thetas = arrs["thetas"]   # (n_iter+1, P)
        losses = arrs["losses"]   # (n_iter+1,)
    elif opt_name == "adam":
        log = adam(_sim, theta_0, Y_ref, M=M, n_iter=N_ITER, lr=1e-2)
        arrs = log.as_arrays()
        thetas = arrs["thetas"]
        losses = arrs["losses"]
    elif opt_name == "sgd":
        log = sgd(_sim, theta_0, Y_ref, M=M, n_iter=N_ITER, lr=1e-3)
        arrs = log.as_arrays()
        thetas = arrs["thetas"]
        losses = arrs["losses"]
    else:
        raise ValueError(opt_name)

    err = np.linalg.norm(thetas - np.asarray(THETA_STAR), axis=1)  # (n_iter+1,)
    return {
        "thetas": thetas,
        "losses": losses,
        "val_losses": arrs.get("val_losses", losses),
        "err": err,
    }


def iters_to_halve(err: np.ndarray) -> int:
    """First iteration at which err drops below 0.5 * err[0].
    Returns len(err) if never crossed."""
    threshold = 0.5 * err[0]
    below = np.where(err <= threshold)[0]
    return int(below[0]) if below.size else len(err)


def main() -> None:
    apply_style()
    out = Path("outputs")
    out.mkdir(exist_ok=True)

    # Reference distribution at theta*.
    print(f"Building reference at theta* (M_ref={128})...")
    ref_keys = jax.random.split(jax.random.PRNGKey(0), 128)
    Y_ref = vmap_simulate(_sim, THETA_STAR, ref_keys)

    # Run all combinations.
    print("Running Phase 2 convergence race...")
    results = {}  # (difficulty, pair, opt_name) -> dict
    for diff_name, d in DIFFICULTIES:
        for pair_idx in range(N_PAIRS):
            theta_0 = sample_safe_theta(
                THETA_STAR, distance=d,
                key=jax.random.PRNGKey(1000 + 100 * DIFFICULTIES.index((diff_name, d)) + pair_idx),
                ref=Y_ref,
            )
            err0 = float(jnp.linalg.norm(theta_0 - THETA_STAR))
            for opt_name in ["opg", "adam", "sgd"]:
                r = run_one(opt_name, theta_0, Y_ref)
                results[(diff_name, pair_idx, opt_name)] = r
                err_end = r["err"][-1]
                it_halve = iters_to_halve(r["err"])
                print(f"  {diff_name:>6s}/pair{pair_idx} {opt_name:>4s}: "
                      f"d0={err0:.3f}  err_end={err_end:.3f}  "
                      f"iter_halve={it_halve}  "
                      f"loss_end={r['losses'][-1]:+.2e}")

    # ===================================================== figure
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    # Rows: difficulty. Columns: 0 = loss curves, 1 = err curves, 2 = histogram.
    opt_colors = {"opg": QUAL[0], "adam": QUAL[1], "sgd": QUAL[2]}
    opt_labels = {"opg": "OPG (LM)", "adam": "Adam", "sgd": "SGD"}

    for row, (diff_name, d) in enumerate(DIFFICULTIES):
        # Aggregate per optimizer across pairs.
        for opt in ["opg", "adam", "sgd"]:
            losses_stack = np.stack([
                np.clip(results[(diff_name, p, opt)]["losses"], 1e-6, None)
                for p in range(N_PAIRS)
            ])  # (N_PAIRS, n_iter+1)
            errs_stack = np.stack([
                results[(diff_name, p, opt)]["err"]
                for p in range(N_PAIRS)
            ])
            median_loss = np.median(losses_stack, axis=0)
            q25_loss = np.percentile(losses_stack, 25, axis=0)
            q75_loss = np.percentile(losses_stack, 75, axis=0)
            median_err = np.median(errs_stack, axis=0)
            q25_err = np.percentile(errs_stack, 25, axis=0)
            q75_err = np.percentile(errs_stack, 75, axis=0)

            ax = fig.add_subplot(gs[row, 0]) if 'ax' not in locals() else \
                 fig.axes[gs[row, 0].get_geometry()[2] - 1] if False else \
                 fig.axes[-1]
        # Re-create cleanly:
    fig.clear()
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    for row, (diff_name, d) in enumerate(DIFFICULTIES):
        ax_loss = fig.add_subplot(gs[row, 0])
        ax_err = fig.add_subplot(gs[row, 1])
        ax_box = fig.add_subplot(gs[row, 2])

        final_errs = {opt: [] for opt in ["opg", "adam", "sgd"]}
        iter_halves = {opt: [] for opt in ["opg", "adam", "sgd"]}

        for opt in ["opg", "adam", "sgd"]:
            # Use val_losses if available (clean), else fall back to losses.
            r0 = results[(diff_name, 0, opt)]
            loss_key = "val_losses" if "val_losses" in r0 else "losses"
            losses_stack = np.stack([
                np.clip(results[(diff_name, p, opt)][loss_key], 1e-6, None)
                for p in range(N_PAIRS)
            ])
            errs_stack = np.stack([
                results[(diff_name, p, opt)]["err"]
                for p in range(N_PAIRS)
            ])
            its = np.arange(losses_stack.shape[1])

            med_loss = np.median(losses_stack, axis=0)
            q25_l = np.percentile(losses_stack, 25, axis=0)
            q75_l = np.percentile(losses_stack, 75, axis=0)
            ax_loss.semilogy(its, med_loss, color=opt_colors[opt],
                             lw=2, label=opt_labels[opt])
            ax_loss.fill_between(its, q25_l, q75_l,
                                 color=opt_colors[opt], alpha=0.18)

            med_err = np.median(errs_stack, axis=0)
            q25_e = np.percentile(errs_stack, 25, axis=0)
            q75_e = np.percentile(errs_stack, 75, axis=0)
            its_err = np.arange(errs_stack.shape[1])
            ax_err.semilogy(its_err, med_err, color=opt_colors[opt],
                            lw=2, label=opt_labels[opt])
            ax_err.fill_between(its_err, q25_e, q75_e,
                                color=opt_colors[opt], alpha=0.18)

            for p in range(N_PAIRS):
                final_errs[opt].append(results[(diff_name, p, opt)]["err"][-1])
                iter_halves[opt].append(
                    iters_to_halve(results[(diff_name, p, opt)]["err"]))

        ax_loss.set_xlabel("iteration")
        ax_loss.set_ylabel(r"MMD$^2$ (median + IQR)")
        ax_loss.set_title(rf"{diff_name.capitalize()} (d={d}): loss")
        if row == 0:
            ax_loss.legend(fontsize=8)

        ax_err.set_xlabel("iteration")
        ax_err.set_ylabel(r"$\|\theta_t - \theta^*\|_2$ (median + IQR)")
        ax_err.set_title(rf"{diff_name.capitalize()}: parameter recovery")
        if row == 0:
            ax_err.legend(fontsize=8)

        # Box plot of final errors per optimizer.
        data = [final_errs[opt] for opt in ["opg", "adam", "sgd"]]
        bp = ax_box.boxplot(data, labels=[opt_labels[opt] for opt in ["opg", "adam", "sgd"]],
                            patch_artist=True, widths=0.55)
        for patch, opt in zip(bp["boxes"], ["opg", "adam", "sgd"]):
            patch.set_facecolor(opt_colors[opt])
            patch.set_alpha(0.7)
        ax_box.set_ylabel(r"final $\|\theta_T - \theta^*\|_2$")
        ax_box.set_title(rf"{diff_name.capitalize()}: final error ({N_PAIRS} pairs)")
        ax_box.set_yscale("log")

    fig.suptitle(
        rf"Phase 2 convergence comparison — "
        rf"{len(DIFFICULTIES)} difficulties $\times$ {N_PAIRS} pairs $\times$ 3 opt's "
        rf"(M={M}, T={T}, {N_ITER} iter, full horizon)",
        fontsize=14, fontweight="bold", y=0.995,
    )

    p = out / "10_phase2_convergence.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {p}")

    # ============================================== console summary
    print("\n" + "=" * 78)
    print("PHASE 2 SUMMARY")
    print("=" * 78)
    for diff_name, d in DIFFICULTIES:
        print(f"\n{diff_name.upper()} (d={d}):")
        print(f"  {'optimizer':<10s} {'med final err':>14s} {'med iter→half':>15s}")
        for opt in ["opg", "adam", "sgd"]:
            fe = np.median([results[(diff_name, p, opt)]["err"][-1]
                            for p in range(N_PAIRS)])
            ih = np.median([iters_to_halve(results[(diff_name, p, opt)]["err"])
                            for p in range(N_PAIRS)])
            print(f"  {opt_labels[opt]:<10s} {fe:14.4f} {ih:15.1f}")

    # Speedup table: median iter-to-halve for Adam vs OPG.
    print("\nSpeedup: median iter-to-halve of Adam / OPG (ratio > 1 means OPG faster):")
    for diff_name, d in DIFFICULTIES:
        ih_opg = np.median([iters_to_halve(results[(diff_name, p, "opg")]["err"])
                            for p in range(N_PAIRS)])
        ih_adam = np.median([iters_to_halve(results[(diff_name, p, "adam")]["err"])
                             for p in range(N_PAIRS)])
        speedup = ih_adam / max(ih_opg, 1e-3)
        print(f"  {diff_name:<6s}: {speedup:.2f}x  (OPG={ih_opg:.0f},  Adam={ih_adam:.0f})")

    # Save raw results for downstream inspection.
    flat = {}
    for (d, p, o), r in results.items():
        key = f"{d}_pair{p}_{o}"
        flat[key + "_thetas"] = r["thetas"]
        flat[key + "_losses"] = r["losses"]
        flat[key + "_err"] = r["err"]
    np.savez(out / "10_phase2_convergence.npz", **flat)
    print(f"saved {out / '10_phase2_convergence.npz'}")


if __name__ == "__main__":
    main()
