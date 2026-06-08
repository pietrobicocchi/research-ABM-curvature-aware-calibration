# curvature-aware-calibration

Research code for the paper *"Curvature-Aware Calibration of Differentiable Agent-Based Models"*.

The central contribution is a **diagnostic**, not an optimiser. Differentiable ABM calibration hands you per-seed gradients for free. Their outer product — the OPG matrix F̂ — has an eigendecomposition that names which parameter combinations the loss cannot constrain, *before* you waste compute trying to recover them. We show that first-order optimisers (Adam in particular) silently fail precisely along the directions F̂ flags as sloppy, verified non-circularly on two structurally different models (Brock-Hommes financial, mean-field and network SIR epidemic) under three discrepancies the calibrator never saw.

## Setup

Requires Python 3.12 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync --extra dev
```

## Tests

```bash
uv run pytest          # 75 tests, ~2 min
```

## Package layout

```
src/curvature_calib/
  models/
    brock_hommes.py       # Brock-Hommes heterogeneous-agent financial model
    sir.py                # Mean-field SIR with differentiable lockdown
    network_sir.py        # Network-SIR with Erdős-Rényi contact graph
    surrogates.py         # Differentiable relaxations: gumbel_sigmoid, straight_through_bernoulli
  losses/
    mmd.py                # Unbiased MMD² with median-heuristic bandwidth
  calibration/
    per_seed_grads.py     # VJP-based per-seed gradients → CalibStats(loss, mean_grad, per_seed_grads, opg)
    diagnostic.py         # EigDecomp, eigendecompose, principal_angles, effective_dimension
    bootstrap.py          # bootstrap_eigvals, eigenvalue_cis, bootstrap_subspace_cis
    falsification.py      # moments_difference, acf_difference, quantile_difference, run_falsification
    calibrate.py          # LM-adaptive calibration loop
    baselines.py          # SGD and Adam baselines
    opg.py                # Backwards-compat re-export shim (prefer importing from diagnostic/bootstrap directly)
  viz/
    style.py              # Shared palette and rcParams
```

## Running the experiments

Outputs land in `outputs/` (gitignored). Scripts are numbered in dependency order.

### Brock-Hommes (start here)

```bash
# 1. Calibrate: runs BH with OPG-LM, SGD, Adam; saves calibration_log.npz
uv run python scripts/06_calibration_dashboard.py

# 2. Core result: stiff vs sloppy perturbation under non-MMD discrepancies
uv run python scripts/08_falsification.py

# 3. Visualise the gradient cloud and OPG spectrum at a fixed theta
uv run python scripts/05_gradient_cloud.py
```

Deeper analysis (all require `calibration_log.npz` from script 06):

| Script | What it shows |
|---|---|
| `07_optimizer_comparison.py` | OPG vs Adam vs SGD: loss, distance-to-truth, eigenbasis trajectories |
| `09_horizon_bias.py` | Eigenvalue hierarchy vs gradient-truncation horizon H ∈ {5,10,20,40,80,200} |
| `11_stiff_sloppy_decomposition.py` | Recovery error decomposed along F̂(θ\*) eigenbasis — Adam wanders 280 000× further along sloppy directions |
| `13_jacobian_comparison.py` | Jacobian per-parameter sensitivity (500× range) vs OPG eigenvalues (5×10⁶ range) |
| `14_multiseed_far_from_eq.py` | N=15 random inits: OPG and SGD beat Adam **15/15 seeds** |
| `25_eigenvalue_trajectory.py` | OPG spectra across stable / periodic / chaotic BH regimes |

### Mean-field SIR

```bash
uv run python scripts/16_sir_diagnostic.py    # OPG spectrum + §5.4 falsification
uv run python scripts/17_sir_calibration_race.py  # Calibration race: OPG vs Adam vs SGD
uv run python scripts/26_horizon_sensitivity_sir.py  # Gradient-horizon sensitivity
uv run python scripts/27_jacobian_comparison_sir.py  # Jacobian vs OPG
```

### Network-SIR (Gumbel-Sigmoid surrogate)

```bash
uv run python scripts/18_network_sir_diagnostic.py   # Diagnostic under surrogate bias
uv run python scripts/24_surrogate_comparison.py     # Gumbel vs straight-through gradients
```

### Paper figures

```bash
uv run python scripts/21_fig1_spectrum.py          # Fig 1: BH OPG spectrum + gradient cloud
uv run python scripts/20_merged_falsification.py   # Fig 2: three models × three discrepancies
uv run python scripts/22_fig3_predicts_adam.py     # Fig 3: Adam failure predicted by F̂
uv run python scripts/23_fig4_network_sir.py       # Fig 4: network-SIR with surrogate
```

## Key empirical results

- **BH OPG spectrum spans 8.4 orders of magnitude.** Sloppiest direction v₅ is dominated by β (trend strength); stiffest v₁ is the symmetric-bias combination (b₁+b₂)/√2 — a direction no per-parameter sensitivity analysis can surface.
- **Falsification ratios ≥ 10² on every channel and every model.** Stiff-direction perturbations produce discrepancies orders of magnitude larger than sloppy-direction perturbations under moments, ACF, and tail quantiles.
- **Adam diverges along sloppy directions, 25/25 seeds (BH + SIR).** The exact failure the OPG eigendecomposition predicted. Cross-optimiser ratio Adam-v_P / OPG-v₁ = 9.4×10⁴.
- **Diagnostic generalises to network-SIR under Gumbel-Sigmoid surrogate bias.** Spectrum spans 20.7 OOM; falsification ratios grow, not shrink.

## Environment notes

- `jax==jaxlib==0.4.30` pinned for Intel-Mac x86_64 wheel compatibility. Unpin on Apple Silicon.
- `jax_enable_x64=True` is set at the top of every script — required for accurate OPG spectra (SIR condition number ~10¹³).
- `docs/memory/` is gitignored (Claude session notes, symlinked from `~/.claude/`).
