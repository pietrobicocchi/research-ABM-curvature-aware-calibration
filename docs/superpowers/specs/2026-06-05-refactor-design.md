# Design: `curvature_calib` Refactor

**Date:** 2026-06-05
**Author:** Pietro Bicocchi
**Status:** Approved

## Goal

Refactor the `curvature_calib` library from an organically-grown research codebase into a clean, extensible research library. The paper is being substantially rewritten; more analysis and visualizations are expected. The library must support that ongoing work without friction.

This is not a full rewrite. Package name, tests, and working experiments are preserved. The structural work is: extract three new library modules, add the straight-through Bernoulli surrogate, strip dead weight, and add four missing experiment scripts.

---

## 1. New Library Module Layout

```
src/curvature_calib/
├── models/
│   ├── brock_hommes.py         unchanged
│   ├── sir.py                  unchanged (mean-field SIR, P=5, fully smooth)
│   ├── network_sir.py          extract _gumbel_sigmoid → surrogates.py; import back
│   └── surrogates.py           NEW — see §2
├── losses/
│   └── mmd.py                  unchanged
├── calibration/
│   ├── per_seed_grads.py       unchanged — VJP pipeline, M-scaling convention locked
│   ├── diagnostic.py           NEW — see §3
│   ├── bootstrap.py            NEW — see §4
│   ├── falsification.py        NEW — see §5
│   ├── calibrate.py            keep; deduplicate _loss_only; keep val_losses + NaN guard
│   ├── baselines.py            keep; import _loss_only from calibrate.py
│   ├── preconditioner.py       unchanged
│   ├── jacobian_sensitivity.py unchanged
│   └── opg.py                  slimmed — re-exports from diagnostic.py + bootstrap.py
│                               for backwards compatibility; no logic of its own
└── viz/
    └── style.py                unchanged
```

---

## 2. `models/surrogates.py` (NEW)

Extract `_gumbel_sigmoid` from `network_sir.py` and add `straight_through_bernoulli`.

```python
def gumbel_sigmoid(
    p: jax.Array,
    key: jax.Array,
    tau: float = 0.5,
    eps: float = 1e-6,
) -> jax.Array:
    """Differentiable Bernoulli via Gumbel-Sigmoid at temperature tau.
    Extracted from network_sir._gumbel_sigmoid; identical behaviour."""

@jax.custom_jvp
def straight_through_bernoulli(p: jax.Array, key: jax.Array) -> jax.Array:
    """Hard Bernoulli sample; gradient passes through as if the output were p.
    Forward: jr.bernoulli(key, p).  JVP: tangent of p passes straight through."""
```

`network_sir.py` imports `gumbel_sigmoid` from `surrogates.py`; its internal `_gumbel_sigmoid` is deleted.

A `tau` parameter is added to `network_sir.simulate()` with default `tau=0.5` (matching current behaviour) so the surrogate comparison experiment can vary it without changing existing scripts.

---

## 3. `calibration/diagnostic.py` (NEW)

Consolidates the pure mathematical layer currently split between `opg.py` and ad-hoc script code.

Moves from `opg.py`:
- `EigDecomp` (NamedTuple)
- `eigendecompose(F)`
- `principal_angles(V1, V2)`
- `opg_from_grads(G)`

New additions:
- `effective_dimension(eigvals, noise_floor) -> int` — count eigenvalues above `noise_floor`
- `d_eff_from_bootstrap(eigvals, cis) -> int` — count where CI lower bound exceeds zero

`opg.py` is reduced to re-exports of these for backwards compat.

---

## 4. `calibration/bootstrap.py` (NEW)

Moves from `opg.py`:
- `bootstrap_eigvals(per_seed_grads, n_boot, key) -> (n_boot, P)` — bootstrap distribution of eigenvalues

New additions:
- `bootstrap_subspace_cis(per_seed_grads, k, n_boot, key) -> float` — largest principal angle CI for top-k subspace
- `noise_threshold(eigval_cis) -> float` — upper CI of smallest eigenvalue
- `eigenvalue_cis(boot_eigvals, confidence) -> (P, 2)` — percentile CIs from bootstrap distribution

---

## 5. `calibration/falsification.py` (NEW)

Consolidates falsification logic currently scattered across scripts 08 and 20.

```python
def perturbed_parameters(
    theta: Array["P"],
    direction: Array["P"],
    alpha_grid: Array["n_alpha"],
) -> Array["n_alpha 2 P"]:
    """theta ± alpha * direction for each alpha. Last axis indexes +/-."""

def moments_difference(X: Array["n T"], Y: Array["m T"]) -> Array["4"]:
    """Differences in mean, std, skewness, kurtosis (scipy)."""

def acf_difference(X: Array["n T"], Y: Array["m T"], max_lag: int = 20) -> float:
    """Sup-norm of empirical ACF difference."""

def quantile_difference(
    X: Array["n T"],
    Y: Array["m T"],
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.95, 0.99),
) -> Array["n_q"]:
    """Differences in empirical quantiles."""

@dataclass
class FalsificationResult:
    alpha_grid: Array["n_alpha"]
    stiff_moments: Array["n_alpha 4"]
    stiff_acf: Array["n_alpha"]
    stiff_quantiles: Array["n_alpha n_q"]
    sloppy_moments: Array["n_alpha 4"]
    sloppy_acf: Array["n_alpha"]
    sloppy_quantiles: Array["n_alpha n_q"]

def run_falsification(
    simulate_fn: Callable,
    theta_T: Array["P"],
    eig: EigDecomp,
    alpha_grid: Array["n_alpha"],
    M: int,
    key: jax.Array,
) -> FalsificationResult:
    """For v_1 (stiff) and v_P (sloppy): perturb, simulate M seeds,
    compute three discrepancies. Returns FalsificationResult."""
```

---

## 6. What is kept untouched

- `outputs/` directory and all its contents (produced figures — not regenerated by default)
- All 26 existing tests
- `scripts/05_*.py` through `scripts/23_*.py` (simplified but not deleted)
- `docs/memory/` files listed in §11 as "keep"

---

## 7. Dead weight to delete

```
paper/                              all LaTeX — DELETE
appendix/                           supp.tex — DELETE
notebooks/                          all notebooks — DELETE
scripts/build_brock_hommes_paper.py
scripts/build_brock_hommes_walkthrough.py
scripts/build_sir_paper.py
scripts/01_brock_hommes_bifurcation.py
scripts/02_simulation_gallery.py
scripts/03_phase_portraits.py
scripts/04_mmd_landscape.py
docs/memory/paper_style_guide.md    (if present — already superseded)
```

---

## 8. Scripts 05–23: simplification target

Each remaining script currently embeds simulation setup, discrepancy functions, and boilerplate inline. After the new modules exist, scripts become thin callers:
- Import `run_falsification` from `falsification.py` (scripts 08, 20)
- Import `effective_dimension` / `d_eff_from_bootstrap` from `diagnostic.py` (scripts 05, 06, 16, 18, 21, 23)
- Import `bootstrap_eigvals` / `eigenvalue_cis` from `bootstrap.py` (scripts 19, 21)
- Import `gumbel_sigmoid` / `straight_through_bernoulli` from `surrogates.py` (script 18, new script 24)

Scripts 12 and 15 (Adam lr sweep, hyperparam robustness) are kept — they support the "divergence is structural" narrative.

---

## 9. New experiment scripts

| Script | Experiment | Key library calls |
|--------|-----------|------------------|
| `24_surrogate_comparison.py` | Network-SIR: compare Gumbel-Softmax vs straight-through eigenstructure (spectrum overlay + principal angles between top-k subspaces). §4.2 "critical additional result". | `surrogates.straight_through_bernoulli`, `diagnostic.eigendecompose`, `diagnostic.principal_angles` |
| `25_eigenvalue_trajectory.py` | Polished paper figure: log λ_k(t) vs iteration for BH calibration across 3 regimes (stable / periodic / chaotic). §4.1 explicit requirement. | `calibrate.calibrate`, `diagnostic.eigendecompose`, `diagnostic.effective_dimension` |
| `26_horizon_sensitivity_sir.py` | Gradient-horizon sensitivity for mean-field SIR (mirrors script 09 for BH). Appendix B. | `sir.simulate` with `grad_horizon`, `diagnostic.principal_angles` |
| `27_jacobian_comparison_sir.py` | Jacobian vs OPG for SIR (mirrors script 13 for BH). §4.2. | `jacobian_sensitivity.per_param_jacobian_sensitivity`, `diagnostic.eigendecompose` |

---

## 10. Implementation tricks locked in (must survive refactor)

| Trick | Location | Reason |
|-------|----------|--------|
| `g_m = M * (∂L/∂x_m)(∂x_m/∂θ)` scaling | `per_seed_grads.py` | Guarantees `mean(g_m) = ∇L` exactly |
| `stop_gradient` on median bandwidth | `mmd.py` | Prevents spurious second-order terms from non-differentiable median |
| `val_losses` on fixed held-out seed | `calibrate.py`, `baselines.py` | Training MMD² is too noisy to monitor convergence otherwise |
| NaN guard + force damping ×10 | `calibrate.py` | BH explodes at `g/R > ~1.5`; bad proposals must be hard-rejected |
| `init_damping=100` default | `calibrate.py` | Stiff-init BH: default damping=1 overshoots into explosion regime |
| `jax.config.update("jax_enable_x64", True)` | paper figure scripts | float32 gives 16 OOM spectrum; float64 gives 20.7 OOM on Network-SIR |

---

## 11. Tests

Existing 26 tests continue to pass. New tests needed:
- `tests/test_surrogates.py`: straight-through forward returns hard sample; `jax.grad` of `p → E[sample]` returns ≈ 1
- `tests/test_diagnostic.py`: `effective_dimension` on hand-crafted eigenvalues; `d_eff_from_bootstrap` counts correctly
- `tests/test_bootstrap.py`: `bootstrap_subspace_cis` on identical subspaces returns near-zero
- `tests/test_falsification.py`: toy sum-only simulator — sloppy-direction perturbations leave output unchanged

Existing tests import from `curvature_calib.calibration.opg` — `opg.py` re-exports preserve this without changes to test files.

---

## 12. Memory cleanup

Memory files to delete (superseded / paper-writing-specific):
- `docs/memory/paper_style_guide.md`
- Any paper-v2 review checklist if present

Memory files to keep (contain empirical findings and implementation decisions):
- `state.md` — update after refactor completes
- `phase1_horizon_bias_result.md`
- `phase2_convergence_result.md`
- `sir_generalization.md`
- `reviewer_concerns_audit.md`
- `honest_appraisal.md`
- `falsification_protocol.md`
- `framing_kunstner_opg_not_fisher.md`
- `paper_story_arc.md` — still relevant for which numbers are locked
- `literature_positioning.md`
- `project_overview.md`
- `project_plan_phases.md`
- `feedback_token_efficiency.md`
