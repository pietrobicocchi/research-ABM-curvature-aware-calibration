# curvature-aware-calibration

Research code for the paper *"Curvature-Aware Calibration of Differentiable Agent-Based Models"*.

The central contribution is a **diagnostic** to investigate identifiability in differentiable ABMs. 

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


## Environment notes

- `jax==jaxlib==0.4.30` pinned for Intel-Mac x86_64 wheel compatibility. Unpin on Apple Silicon.
- `jax_enable_x64=True` is set at the top of every script — required for accurate OPG spectra (SIR condition number ~10¹³).
- `docs/memory/` is gitignored (Claude session notes, symlinked from `~/.claude/`).
