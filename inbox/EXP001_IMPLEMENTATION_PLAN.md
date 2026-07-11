---
title: EXP-001 Implementation Plan + Minimum Reusable GGN Infrastructure
status: approved-with-revisions
date: 2026-07-11
revised: 2026-07-11
base_commit: e1b3a6f
authority: subordinate to 02_MATHEMATICAL_SPECIFICATION.md, 04_EXPERIMENT_REGISTRY.md, 07_DECISION_LOG.md
scope: EXP-001 only + shared infra. NO Brock–Hommes / SIR / net-SIR changes.
---

> **Revision note (2026-07-11).** Plan approved subject to seven decisions,
> folded in below: (1) condition-number gate `κ∈{1,1e2,1e6}`, `~1e13` demoted
> to a labelled stress test; (2) run records under `outputs/EXP-001/<run-id>/`;
> (3) scale-aware relative tolerances, separate looser FD tolerance + step-size
> study; (4) minimal 5-function API, `ggn_linear_operator` as a thin wrapper,
> `R=H−G`; (5) explicit weight semantics (None/diagonal/dense SPD) with
> validation; (6) x64 enabled+verified at entry point before array/JIT; (7)
> production OPG untouched + tracked blocker recorded.

## TRACKED BLOCKER (must clear before EXP-000 or any BH/SIR rerun)

> Before EXP-000 or any Brock–Hommes/SIR rerun, **rename and redocument the
> historical per-seed OPG** (`calibration/per_seed_grads.py` `opg`/`CalibStats.opg`,
> `diagnostic.opg_from_grads`) so it cannot be mistaken for GGN curvature
> (DEC-001, Math-Spec §14). EXP-001 leaves it untouched; this blocker is the
> gate for the next phase.

# EXP-001 Implementation Plan

Turns the accepted `CODEBASE_AUDIT.md` findings into a concrete build for
**EXP-001 — Analytic linear GGN recovery** (`04_EXPERIMENT_REGISTRY.md`) and
the **minimum reusable infrastructure** that all later experiments will share.

This is a plan only. No code is written in this step. No canonical vault file,
no production model file (`models/*.py`, `surrogates.py`), and no existing
calibration/loss module is edited here.

## 0. What EXP-001 must deliver (from the registry)

- Model: `m(z)=Az`, `L(z)=½‖Az−y‖²_W`.
- Objects compared: analytic GGN `AᵀWA`; explicit-Jacobian AD GGN; matrix-free
  `Gv`; exact Hessian; raw scalar-gradient OPG; finite-difference Hessian.
- Variables: `P∈{5,20,100}`; full-rank and rank-deficient `A`; controlled
  condition numbers; float32 vs float64.
- Metrics: relative Frobenius error, eigenvalue error, principal-angle error,
  recovered rank, runtime/memory.
- **Pass criterion:** AD GGN agrees with the analytic matrix to numerical
  precision in float64. Supports **C01**.

Because `m` is affine, `R(z)=H−G≡0` (Math-Spec §9), so EXP-001 also pins down
the exact-Hessian path and demonstrates, at a fully analytic point, the
OPG≠GGN distinction (DEC-001): at `r=0`, `g=0`⇒`F_OPG=0` while `G=AᵀWA≠0`.

---

## 1. Coordinate conventions (fixed for all infra)

- The geometry is always computed in **unconstrained, prior-scaled coordinates
  `z∈R^P`** (Math-Spec §1, DEC-003). Utilities differentiate the map `m: R^P→R^K`
  supplied by the caller; the caller is responsible for expressing `m` in `z`.
- For **EXP-001 specifically**, `T=identity`, prior `z~N(0,I)`, so `z`-space *is*
  the physical space; `A` acts directly on `z`. This is the one case where the
  distinction is vacuous — deliberately, to isolate GGN correctness from the
  transform machinery (which lands with the SIR/BH work, not here).
- `W⪰0` is the **data metric** of the loss, passed explicitly. Default `W=I`
  represented as `None` (meaning identity) to avoid materializing `I_K`.
- Prior precision `P_π` is **not** applied inside the GGN; prior-relative
  whitening (`wGv=λP_πv`) is a later concern and is out of scope for EXP-001.

---

## 2. Precision settings

- New module `curvature_calib/config.py` centralizes precision.
  - `enable_x64()` → calls `jax.config.update("jax_enable_x64", True)`; **must be
    called before any `jax.numpy` array is created or any function is traced/
    JIT-compiled** (Decision 6). Idempotent.
  - `require_x64()` → raises `RuntimeError` if x64 is not active; called at the
    top of the experiment entry point and precision-sensitive utilities as a
    guard (bypassable by tests that deliberately exercise float32).
- The library is **not** forced to x64 at package import (would surprise
  importers). Instead the experiment entry point calls `enable_x64()` then
  `require_x64()` **before constructing any significant array or tracing**.
- **No module-level creation of significant JAX arrays** anywhere in the new
  code — all array construction happens inside functions called after x64 is
  enabled (Decision 6). Test modules enable x64 in a fixture before building arrays.
- EXP-001 runs the full sweep in **both** float32 and float64 to document the
  degradation; the pass criterion is asserted only in float64 (Decision 8).

---

## 3. Files to create

```
src/curvature_calib/
  config.py                      # NEW  precision + x64 guard
  provenance.py                  # NEW  run metadata capture -> dict/JSON
  metrics.py                     # NEW  matrix/subspace comparison metrics
  geometry/
    __init__.py                  # NEW
    ggn.py                       # NEW  GGN construction (dense + matrix-free), Hessian, OPG, FD-Hessian
  benchmarks/
    __init__.py                  # NEW
    linear_gaussian.py           # NEW  analytic m(z)=Az benchmark + A generators
experiments/
  __init__.py                    # NEW
  exp001_linear_ggn.py           # NEW  EXP-001 entry point (run + run_stress)
tests/
  test_config.py                 # NEW
  test_provenance.py             # NEW
  test_metrics.py                # NEW
  test_ggn.py                    # NEW  core correctness (the heart of EXP-001)
  test_linear_gaussian.py        # NEW
outputs/EXP-001/<run-id>/        # NEW (gitignored) config/provenance/metrics.json + arrays.npz + figures/
```

**Files modified:** none in `src/` production code. Everything else is new.
Machine run records go **only** to `outputs/EXP-001/<run-id>/` (Decision 2) — no
files are written under `docs/` or the canonical vault.

---

## 4. Public function interfaces

### 4.1 `curvature_calib/config.py`

```python
def enable_x64() -> None: ...
    # Idempotent. Sets jax_enable_x64=True.

def x64_enabled() -> bool: ...

def require_x64() -> None: ...
    # Raise RuntimeError if not x64_enabled().
```

### 4.2 `curvature_calib/geometry/ggn.py`

The **core reusable API is exactly five functions** (Decision 4). All take a
caller-supplied representation map `representation_fn: (z:(P,)) -> (K,)` or loss
`loss_fn: (z:(P,)) -> scalar`, and an evaluation point `z:(P,)`. `weight` is the
data metric `W`: `None` (identity), a `(K,)` diagonal, or a `(K,K)` dense SPD
matrix (Decision 5).

```python
def ggn_dense(representation_fn, z, weight=None) -> Array:        # (P, P)
    # J = _representation_jacobian(representation_fn, z)  # (K,P), jacfwd if K>=P else jacrev
    # return symmetrize(J.T @ _apply_weight(weight, J))   # J.T W J ; None -> J.T J
    # PSD by construction; symmetrized 0.5*(G+G.T) on return.

def ggn_matvec(representation_fn, z, vector, weight=None) -> Array:  # v:(P,) -> (P,)
    # Matrix-free: Jv = jax.jvp(representation_fn, (z,), (vector,))[1]  # (K,)
    # WJv = _apply_weight(weight, Jv)                                   # (K,)
    # return vjp_of_representation(z)(WJv)                              # (P,)
    # No (K,P) matrix materialized.

def exact_hessian(loss_fn, z) -> Array:                          # (P, P)
    # symmetrize(jax.hessian(loss_fn)(z)).

def finite_difference_hessian(loss_fn, z, step_size) -> Array:   # (P, P)
    # Central differences of jax.grad(loss_fn) with explicit step_size (required,
    # no default); symmetrized. Separate, looser tolerance regime (Decision 3).

def scalar_gradient_outer_product(loss_fn, z) -> Array:          # (P, P)
    # g = jax.grad(loss_fn)(z); return outer(g, g).  Rank-1 COMPARISON object
    # (DEC-001) — NOT a curvature estimate. Zero at an exact fit (g=0).
```

Thin wrapper (not an independent implementation, Decision 4):

```python
def ggn_linear_operator(representation_fn, z, weight=None) -> Callable[[Array], Array]:
    # return lambda vector: ggn_matvec(representation_fn, z, vector, weight)
```

Private helpers (module-internal, not public API):

```python
def _representation_jacobian(representation_fn, z) -> Array:      # (K, P)
def _apply_weight(weight, u) -> Array:   # None: u ; (K,) diag: w*u ; (K,K): W@u
def _validate_weight(weight, K, dtype) -> None:
    # shape ((K,) or (K,K)); symmetry (dense); dtype match; finiteness;
    # PSD (min eigenvalue >= -tol for dense; nonneg entries for diagonal).
def symmetrize(A) -> Array:   # 0.5*(A + A.T)
```

**Residual curvature `R=H−G`** is *not* a standalone library function
(Decision 4). EXP-001 computes it inline at the experiment/test layer as
`exact_hessian(loss_fn, z) - ggn_dense(representation_fn, z, weight)`; it is `0`
for affine `m`.

Design notes:
- `ggn_dense` and `ggn_matvec` must agree: `ggn_dense(...) @ v ≈ ggn_matvec(..., v)`
  over several deterministic test vectors, under the scale-aware relative
  tolerance (Decision 3; a test).
- `weight=None` avoids building `I_K`; `(K,)` is applied as elementwise scaling;
  `(K,K)` is applied as `W @ (·)`. `_validate_weight` runs once per call.
- No coordinate transform is applied inside these functions; the caller owns `T`.

### 4.3 `curvature_calib/benchmarks/linear_gaussian.py`

```python
class LinearGaussian(NamedTuple):
    A: Array      # (K, P)
    y: Array      # (K,)
    W: Array | None  # (K, K) or None

def make_representation(model) -> Callable:      # z:(P,) -> m:(K,)  ==  A @ z
def make_loss(model) -> Callable:                # z:(P,) -> scalar  ==  0.5 * r^T W r
def analytic_ggn(model) -> Array:                # (P, P)  ==  A^T W A
def analytic_gradient(model, z) -> Array:        # (P,)    ==  A^T W (A z - y)

def random_A(key, K, P, cond=None, rank=None, dtype=...) -> Array:  # (K, P)
    # SVD-constructed A with prescribed condition number and (optional) rank
    # deficiency: A = U diag(s) V^T, s log-spaced over [1, cond]; zero the
    # trailing (P-rank) singular values for rank-deficient cases.
```

Shapes throughout: `P` = parameter dim, `K` = representation dim (`K>=P` for
full-rank identifiable cases; `K<P` or reduced rank for rank-deficient cases).

### 4.4 `curvature_calib/metrics.py`

```python
def rel_frobenius_error(A_est, A_ref) -> float:          # ‖A_est-A_ref‖_F / ‖A_ref‖_F
def eigenvalue_rel_error(w_est, w_ref) -> Array:          # (P,) per-eigenvalue rel error, sorted desc
def subspace_principal_angles(V1, V2) -> Array:           # reuse diagnostic.principal_angles
def max_principal_angle(V1, V2) -> float:
def numerical_rank(w, rtol=1e-10, atol=0.0) -> int:       # count eigvals > rtol*max(w)+atol
```

`eigenvalue_rel_error` and eigen-inputs use `diagnostic.eigendecompose` (reused,
not reimplemented). `subspace_principal_angles` delegates to the existing
`calibration.diagnostic.principal_angles` (Björck–Golub) — no duplication.

### 4.5 `curvature_calib/provenance.py`

```python
def run_metadata(experiment_id, config: dict, seeds: dict) -> dict: ...
def write_metadata(path, metadata: dict) -> None: ...  # JSON, sorted keys
```

Captured keys (schema): `experiment_id`, `timestamp_utc`, `git_commit`,
`git_dirty` (bool), `jax_version`, `jaxlib_version`, `numpy_version`,
`python_version`, `platform`, `hostname`, `x64_enabled`, `default_dtype`,
`seeds` (dict), `config` (dict), `plan` = `"EXP001_IMPLEMENTATION_PLAN.md"`.
Git info via `subprocess` on `git rev-parse HEAD` / `git status --porcelain`;
degrade gracefully (record `"unknown"`) if not in a repo.

### 4.6 `experiments/exp001_linear_ggn.py`

```python
def run(P_list=(5,20,100), conds=(1.0, 1e2, 1e6), ranks=("full","deficient"),
        dtypes=("float64","float32"), seed=0, out_root="outputs/EXP-001") -> str:
    # Ordinary validation gate (Decision 1). For each (P, cond, rank, dtype):
    # build LinearGaussian, compute the objects, evaluate metrics at z=z_hat
    # (least-squares solution, r≈0) AND at a perturbed z (r≠0) to exhibit OPG
    # collapse. Persist a run record; return the run directory path.

def run_stress(P=20, cond=1e13, seed=0, out_root="outputs/EXP-001") -> str:
    # Separately labelled ill-conditioning stress case (Decision 1). Characterizes
    # float64 vs float32 degradation at SIR-scale conditioning. Does NOT gate.
```

Entry: `if __name__ == "__main__": enable_x64(); require_x64(); run(); run_stress()`.
Each call writes a run record under `outputs/EXP-001/<run-id>/` where
`<run-id> = "<UTC-timestamp>_<short-git-commit>"` (Decision 2), containing:

```text
outputs/EXP-001/<run-id>/
  config.json        # the run configuration (P grid, conds, ranks, dtypes, seed, kind=main|stress)
  provenance.json    # provenance.run_metadata schema (§4.5)
  metrics.json       # per-cell metrics, JSON-serialisable summary
  arrays.npz         # raw matrices / eigvals / timings keyed by (P,cond,rank,dtype)
  figures/           # FIG-01: analytic vs AD GGN vs Hessian vs OPG (png+pdf)
```

Figures reuse `viz/style.py`. **Nothing is written to `inbox/`, `docs/`, or the
canonical vault; the Results Ledger is not updated automatically** (Decision 2).

---

## 5. Tensor shapes (summary contract)

| Symbol | Shape | Meaning |
|---|---|---|
| `z` | `(P,)` | parameters in prior-scaled coords (≡ physical for EXP-001) |
| `A` | `(K,P)` | linear representation map |
| `m(z)=Az` | `(K,)` | calibrated representation |
| `y` | `(K,)` | reference representation |
| `r=Az−y` | `(K,)` | residual |
| `W` | `(K,K)` or `None` | data metric |
| `J=representation_jacobian` | `(K,P)` | `Dm(z)` (`=A` here) |
| `G`, `H`, `R`, `F_OPG`, `H_fd` | `(P,P)` | GGN, Hessian, residual-curv., scalar OPG, FD-Hessian |
| `g=∇L` | `(P,)` | loss gradient (`=AᵀWr`) |
| eigvals / eigvecs | `(P,)` / `(P,P)` | via `diagnostic.eigendecompose` |
| `Gv` | `(P,)` | matrix-free product |

---

## 6. Tests

Scale-aware relative tolerance (Decision 3), used for all AD-vs-analytic checks:

```
rel_fro(Ĝ, G) = ‖Ĝ − G‖_F / max(‖G‖_F, ε)   with ε = 1e-300 (guard)   ≤ 1e-10   (float64)
```

`tests/test_ggn.py` (core — this *is* the EXP-001 acceptance harness):

1. **AD GGN == analytic** — `rel_fro(ggn_dense, AᵀWA) <= 1e-10` (float64),
   across `P∈{5,20,100}`, `cond(AᵀWA)∈{1, 1e2, 1e6}`, and `weight∈{None,
   diagonal, dense SPD}`.
2. **Matrix-free == dense** — for **several deterministic test vectors** (unit
   axes `e_0,e_{P//2},e_{P-1}`, all-ones, and a fixed pseudo-random vector),
   `‖ggn_matvec(v) − ggn_dense @ v‖_2 / max(‖ggn_dense @ v‖_2, ε) <= 1e-10`
   (float64).
3. **Affine ⇒ R=0** — `rel_fro(exact_hessian, ggn_dense) <= 1e-10`; i.e.
   inline `R = H − G` has `‖R‖_F / max(‖G‖_F,ε) <= 1e-10`.
4. **OPG ≠ GGN, and OPG collapses at the fit** — at `z=z_hat` (r≈0):
   `‖scalar_gradient_outer_product‖_F <= 1e-8·‖G‖_F` while `‖G‖_F` bounded
   below; at `r≠0`: `numerical_rank(scalar_gradient_outer_product)==1` whereas
   `G` has full rank. Concrete DEC-001 demonstration.
5. **FD-Hessian — separate, looser tolerance + step-size study** (Decision 3).
   `rel_fro(finite_difference_hessian(loss, z, h), AᵀWA)` is computed over a
   grid `h∈{1e-2,1e-3,1e-4,1e-5,1e-6}`; the test asserts the **best** `h`
   achieves `<= 1e-5` for a well-conditioned case, and records the U-shaped
   error(h) curve (truncation vs round-off). The AD tolerance is **never**
   applied to FD.
6. **Rank & null-space recovery** — for rank-deficient `A` (rank `q`):
   `numerical_rank(eig(G)) == q`, and the recovered null-space (eigenvectors of
   the `P−q` smallest eigenvalues) matches `null(A)` via
   `max_principal_angle <= 1e-6` (float64).
7. **Eigenvector agreement** — `max_principal_angle(eigvecs(G_AD),
   eigvecs(AᵀWA)) <= 1e-6` on non-degenerate spectra.
8. **float32 recorded, not gated** (Decision 8) — same cells in float32
   recorded; only a loose sanity bound (e.g. `<=1e-3`) asserted; float32 never
   defines acceptance.

`tests/test_metrics.py`: identity/zero-error cases; `numerical_rank` on
prescribed-rank matrices; principal-angle 0 for identical subspaces, π/2 for
orthogonal (mirrors existing `test_per_seed_grads` angle tests).

`tests/test_linear_gaussian.py`: `random_A` yields prescribed cond/rank
(`svd` check); `analytic_ggn == AᵀWA`; `analytic_gradient == jax.grad(loss)`.

`tests/test_config.py`: `enable_x64` makes `x64_enabled()` true and default
float dtype float64; `require_x64` raises when off.

`tests/test_provenance.py`: `run_metadata` contains all schema keys; JSON
round-trips; git fields present (string) in-repo.

All x64 tests call `enable_x64()` in a module fixture. Because x64 is a global
JAX flag, x64 and float32 assertions are split across separate test modules /
processes (pytest-xdist is already a dev dep) to avoid flag bleed; the float32
sanity check constructs arrays with explicit `dtype=jnp.float32` and does not
rely on the global default.

---

## 7. Acceptance criteria

EXP-001 is **accepted** when, in float64, over the ordinary validation grid
`cond(AᵀWA)∈{1,1e2,1e6}` (Decision 1):

- A1. AD GGN matches analytic `AᵀWA` with scale-aware relative Frobenius error
  `≤1e-10` across all `(P, cond, rank, weight)` cells (registry pass, C01).
- A2. `ggn_matvec` matches `ggn_dense @ v` to `≤1e-10` (relative) over the
  deterministic test vectors.
- A3. For affine `m`, `R=H−G` has `‖R‖_F/max(‖G‖_F,ε) ≤1e-10`.
- A4. FD-Hessian matches analytic under its **own** looser regime: best-`h`
  relative error `≤1e-5` (well-conditioned), with the step-size study recorded.
  The AD `1e-10` bound is **not** applied to FD.
- A5. Recovered numerical rank equals `rank(A)`, and null-space principal angle
  `≤1e-6`, for every rank-deficient cell.
- A6. The OPG collapse demonstration (test 4) holds.
- A7. A run record exists at `outputs/EXP-001/<run-id>/` with
  `config.json`, `provenance.json` (real git commit), `metrics.json`,
  `arrays.npz`, `figures/`.
- A8. `uv run pytest` green; new tests included.

The `~1e13` **stress case does not gate** acceptance (Decision 1); it is run
separately via `run_stress` and its degradation is characterized in the review.

float32 results are **recorded** (metrics + figure) but do **not** gate
acceptance (Decision 8); they document the CLAUDE.md precision invariant.

Runtime is **captured** (per-cell `time.perf_counter`; peak memory
deferred/optional) and logged to `arrays.npz`, but no performance threshold
gates EXP-001 — scaling is EXP-009's concern.

---

## 8. Provenance metadata

Every `experiments/*.run()` writes a run directory `outputs/EXP-001/<run-id>/`
(`<run-id> = "<UTC-timestamp>_<short-git-commit>"`) containing `config.json`,
`provenance.json` (schema §4.5), `metrics.json`, `arrays.npz` (metric arrays and
timings keyed by `(P,cond,rank,dtype)`), and `figures/`. This operationalizes
the empty `05_RESULTS_LEDGER.md` template, but the plan **does not** write to the
canonical Results Ledger and **does not** write under `inbox/` or `docs/`
(Decision 2); a `RES-001` entry is drafted later by a human citing the run dir.

---

## 9. What deliberately remains untouched

- **All production model code:** `models/brock_hommes.py`, `models/sir.py`,
  `models/network_sir.py`, `models/surrogates.py` — no edits (explicit user
  constraint; BH/SIR GGN comes in a later plan).
- **`losses/mmd.py`**, `calibration/per_seed_grads.py` — unchanged. The audit's
  recommended F_OPG rename/redoc is **deferred**; EXP-001 does not depend on it.
  (`geometry/ggn.scalar_grad_opg` is a *new, correctly-labeled* comparison
  helper and does not touch the existing OPG code.)
- **`calibration/calibrate.py`, `preconditioner.py`, `baselines.py`,
  `falsification.py`, `bootstrap.py`** — unchanged. `diagnostic.eigendecompose`
  and `principal_angles` are **reused (imported), not modified**.
- **Canonical vault** (`docs/01`–`08`, `manifest.json`) — read-only.
- **Coordinate transforms and priors** — not introduced here beyond the
  `T=identity` degenerate case; the real transform layer lands with SIR.
- **Prior-relative whitening** (`wGv=λP_πv`) — out of scope; EXP-001 tests bare
  GGN correctness only.
- **Existing `outputs/viz/fig7*`** and deleted `scripts/` — left as-is.

---

## 10. Minimum reusable infrastructure (what survives EXP-001)

These are built once here and reused unchanged by EXP-002…EXP-009:

- `config.enable_x64/require_x64` — precision discipline everywhere.
- `geometry/ggn.py` — `representation_jacobian`, `ggn_dense`, `ggn_matvec`,
  `ggn_linear_operator`, `exact_hessian`, `residual_curvature`,
  `scalar_grad_opg`, `finite_difference_hessian`. Representation-agnostic:
  any later `m_fn` (BH summaries, SIR incidence, MMD finite features) plugs in.
- `metrics.py` — the standard comparison metrics used by every validation
  experiment.
- `provenance.py` — uniform run metadata for the Results Ledger.
- `benchmarks/linear_gaussian.py` — the analytic oracle reused by EXP-002's
  nonlinear benchmark scaffolding and as a regression fixture.
- `experiments/` package + `outputs/<exp>/` convention — replaces the deleted
  ad-hoc `scripts/` layer with a metadata-emitting entry-point pattern.

The one open dependency for later (not EXP-001): the **representation `ψ` /
`W` / transform `T` / prior** for BH and SIR (Audit §9). EXP-001 is designed so
that answering those questions later requires only *new* `m_fn`/model files, not
changes to the geometry, metrics, precision, or provenance layers built here.

---

## 11. Required EXP-001 checks (must all be verified)

The completed experiment must verify (mapped to the tests in §6):

1. analytic GGN equals `AᵀWA` — `test_linear_gaussian`;
2. explicit-Jacobian AD GGN equals the analytic GGN — test 1;
3. matrix-free `Gv` equals dense `Gv` — test 2;
4. exact Hessian equals the GGN for affine `m(z)=Az` (`R=H−G=0`) — test 3;
5. rank and null-space recovery for rank-deficient `A` — test 6;
6. at an exact fit, scalar-gradient OPG is zero while the GGN may remain
   nonzero — test 4;
7. finite-difference accuracy varies with step size and conditioning — test 5
   (step-size study) + stress case;
8. float32 behavior is recorded but does not define acceptance — test 8 +
   `run` dtype sweep.

## 12. Resolved review questions (from prior draft)

1. Placement: **`geometry/` subpackage** (hosts later `whitening.py`,
   `eigensolve.py`).
2. Run-record location: **`outputs/EXP-001/<run-id>/`** (Decision 2).
3. `scalar_gradient_outer_product` lives in **`geometry/ggn.py`** with an
   explicit "comparison object, not curvature" docstring (Decision 4 keeps the
   API in one module).
4. Condition grid: **`{1,1e2,1e6}` gates; `~1e13` is a non-gating stress case**
   (Decision 1). FD uses a **step-size grid**, not a single `eps` (Decision 3).

Implementation proceeds exactly as revised above.
