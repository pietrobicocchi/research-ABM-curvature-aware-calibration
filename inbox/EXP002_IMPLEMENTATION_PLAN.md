---
title: EXP-002 Implementation Plan — Nonlinear local-validity benchmark
status: proposed-for-review
date: 2026-07-11
base_commit: 7c612df
authority: subordinate to 02_MATHEMATICAL_SPECIFICATION.md, 04_EXPERIMENT_REGISTRY.md, 07_DECISION_LOG.md
scope: EXP-002 design ONLY. Not implemented. No BH/SIR/MMD/OPG/vault/calibration changes.
supports (intended): C03, C09
---

# EXP-002 Implementation Plan

Validate the nonlinear Hessian decomposition (Math-Spec §7)

    ∇²L(z) = G(z) + R(z),   G = J_m^T W J_m,   R = Σ_k [W r]_k ∇²m_k,

and determine the neighborhood over which the GGN predicts the local loss
geometry (Math-Spec §16). Two controlled nonlinear cases: an exact-fit curved
valley (R=0 at the optimum, growing disagreement away from it) and an
irreducible-residual system (nonzero R at the minimizer).

Reuses the EXP-001 infrastructure unchanged: `geometry.ggn` (`ggn_dense`,
`ggn_matvec`, `exact_hessian`, `finite_difference_hessian`), `metrics`,
`config` (float64), `provenance`, and `diagnostic.eigendecompose`. Adds a
nonlinear-benchmark module and a reusable local-validity module.

**This plan is not to be implemented in this step.**

---

## 1. Exact residual maps and parameter values

Both cases use the squared-residual loss `L(z)=½‖r(z)‖²` (i.e. representation
`m=r`, reference `m_y=0`, `W=I`), so `G=J_r^T J_r` and the residual curvature is
analytic. Coordinates are `z∈R²` directly (`T=Id`); float64 throughout.

### Case A — Exact-fit curved valley (Rosenbrock-type)

    r1(z) = a (z2 − z1²)
    r2(z) = b (1 − z1)
    L(z)  = ½ (r1² + r2²)

- Optimum `ẑ_A = (1, 1)`, `r(ẑ_A) = 0` ⇒ `R(ẑ_A)=0`, `H=G` at the optimum.
- Analytic Jacobian / curvature:
  `J_r = [[−2a·z1, a], [−b, 0]]`,
  `∇²r1 = [[−2a, 0], [0, 0]]`, `∇²r2 = 0`,
  ⇒ `R(z) = r1(z)·[[−2a, 0], [0, 0]] = [[−2a·r1, 0], [0, 0]]`.
  Away from `ẑ_A`, `r1≠0` so `‖R‖` grows — the intended increasing GGN–Hessian
  disagreement.
- Parameter grid (nonlinearity / conditioning knob): `a∈{1, 5, 10}`, `b=1`.
  Larger `a` ⇒ sharper valley, faster R growth, higher cond(G).

### Case B — Irreducible nonlinear residual (overdetermined, nonzero-residual minimizer)

    r1(z) = z1 − 1
    r2(z) = z2 − 1
    r3(z) = λ (z1² + z2² − c)
    L(z)  = ½ (r1² + r2² + r3²)

Three residual components in two parameters; they cannot all vanish (r1=r2=0
forces z=(1,1), where r3=λ(2−c)≠0 for c≠2). Hence the minimizer has nonzero
residual and nonzero `R`.

- Analytic: `J_r = [[1,0],[0,1],[2λz1, 2λz2]]`,
  `∇²r3 = 2λI`, `∇²r1=∇²r2=0` ⇒ `R(z) = r3(z)·2λI` (closed form).
  `H(z) = G(z) + 2λ·r3(z)·I`.
- The minimizer `ẑ_B` is found numerically (Newton on ∇L, small 2-D system;
  verified `‖∇L(ẑ_B)‖ ≤ 1e-10`). `R(ẑ_B) = 2λ·r3(ẑ_B)·I ≠ 0`.
- Parameter grid (residual-magnitude / R knob): `λ∈{0.3, 1.0}`, `c∈{0.0, 4.0}`
  (both ≠2 ⇒ genuine tension; `c` sign controls residual sign/magnitude).

---

## 2. Analytic / AD reference formulas

Every object has a closed form (above); AD is validated against it:
- `G_analytic = J_r^T J_r` with `J_r` as given.
- `H_analytic = G_analytic + R_analytic`, `R_analytic` as given per case.
- `∇L_analytic = J_r^T r`.
- AD references: `G_ad = geometry.ggn_dense(r, z)`,
  `H_ad = geometry.exact_hessian(loss, z)`,
  `H_fd = geometry.finite_difference_hessian(loss, z, h)` (independent cross-check).
- Consistency gates: `rel_fro(G_ad, G_analytic) ≤ 1e-10`,
  `rel_fro(H_ad, H_analytic) ≤ 1e-10`,
  `rel_fro(H_ad − G_ad, R_analytic) ≤ 1e-8` (R via AD equals analytic R).

---

## 3. Points and paths where H, G, R are evaluated

For each parameter cell:
1. **At the optimum/minimizer** `ẑ` (Case A: (1,1); Case B: numeric `ẑ_B`):
   report `H, G, R`, `‖R‖/‖H‖`, eigensystems.
2. **Radial paths** `z(t) = ẑ + t·d`, `t` on a prespecified grid
   `t∈{0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0}` (Case A also negative
   `t` — the valley is asymmetric), for directions `d ∈`:
   - the two GGN eigenvectors `v_1, v_2` at `ẑ`;
   - the valley-tangent and valley-normal directions (Case A: numerically, the
     min/max curvature directions of `H`);
   - a fixed pseudo-random unit direction (control).
   Along each path record `H(z(t)), G(z(t)), R(z(t))` and the four metric
   families (§4–§7) as functions of `t`.

All points, paths, `t`-grid, directions, seed, and tolerances are fixed in the
run `config.json` **before** execution (see §9).

---

## 4. Relative matrix-error metric

`metrics.rel_frobenius_error` (scale-aware, EXP-001 §3):

    E_G(z)  = ‖G(z) − H(z)‖_F / max(‖H(z)‖_F, ε)
    E_R(z)  = ‖R(z)‖_F        / max(‖H(z)‖_F, ε)

Reported at `ẑ` and along every path. `E_G` and `E_R` coincide here
(`H−G=R`); both are logged for clarity. This is the **exact matrix-agreement**
axis (distinct from §5–§7).

---

## 5. Principal-angle (leading-subspace) comparison

`metrics.subspace_principal_angles` / `max_principal_angle` between the leading-k
eigenspaces of `G(z)` and `H(z)` (eigsystems via `diagnostic.eigendecompose`):

    θ_k(z) = max principal angle between top-k eigvecs of G and of H,   k∈{1, 2}.

This is the **leading-subspace-agreement** axis: subspaces can remain aligned
even where `E_G` has grown (matrix magnitude disagreement without rotation), so
it is reported separately vs distance.

Sign/degeneracy handling: near-degenerate spectra (Case B is nearly isotropic
when λ small) are flagged; angles on degenerate pairs are reported at
subspace (not vector) level.

---

## 6. Eigenvalue comparison

`metrics.eigenvalue_rel_error` between sorted eigenvalues of `G` and `H`:

    δ_k(z) = |λ_k^G − λ_k^H| / max(|λ_k^H|, ε),   k = 1..P.

Reported at `ẑ` and along paths. This is the **eigenvalue-agreement** axis:
`R` can shift eigenvalues while barely rotating eigenvectors, so eigenvalue
error and principal angle are logged independently.

---

## 7. Local quadratic-prediction test

For each GGN eigenvector `v_k` (unit) with GGN eigenvalue `λ_k` at `ẑ`, over the
**prespecified** signed grid `α ∈ A` (see §8):

    ΔL_k(α) = L(ẑ + α v_k) − L(ẑ)          (actual)
    Q_k(α)  = ½ α² λ_k                       (GGN quadratic prediction)

Also the full-model check along arbitrary `δ`:
`½ δ^T G(ẑ) δ` vs `L(ẑ+δ) − L(ẑ)`, and the Hessian quadratic
`½ δ^T H(ẑ) δ` as an upper reference.

Reported quantity: relative prediction error
`e_k(α) = |ΔL_k(α) − Q_k(α)| / max(|ΔL_k(α)|, ε)`.

This is the **predictive-accuracy** axis — the operational test, distinct from
matrix/subspace/eigenvalue agreement. At the exact fit (Case A) `Q_k` should
track `ΔL_k` in a nontrivial ball; in Case B the GGN prediction is biased by the
missing `R` term and the ball is expected to be smaller.

---

## 8. Empirical validity radius (prespecified)

Fixed **before** running (no post-hoc tuning):
- signed α grid `A = ±{0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0}`;
- relative-error tolerance `τ = 0.10` (10%).

Definition. For eigenvector `v_k`, the validity radius is the largest contiguous
(from `α=0`) extent on which the quadratic prediction holds:

    ρ_k⁺ = max{ α>0 in A : e_k(α') ≤ τ for all grid 0<α'≤α }
    ρ_k⁻ = max{ |α|, α<0 in A : e_k(α') ≤ τ for all grid α≤α'<0 }
    ρ_k  = min(ρ_k⁺, ρ_k⁻)            (symmetric radius)

Report `ρ_k⁺, ρ_k⁻` separately (Case A is asymmetric) and the overall
`ρ = min_k ρ_k`. `τ` and `A` are recorded verbatim in `config.json`.

---

## 9. Tolerances and conditioning controls

- **float64 mandatory** (`config.enable_x64()` + `require_x64()` at entry, before
  any array/trace).
- Reference-consistency tolerances (§2): AD-vs-analytic `1e-10`; AD-`R`-vs-analytic
  `1e-8`. FD Hessian on its **own** looser regime (step-size grid
  `{1e-3,1e-4,1e-5}`; best-step cross-check `≤1e-5`), never held to the AD bound.
- Validity-radius tolerance `τ=0.10` and α-grid `A` prespecified (§8).
- Conditioning controls: Case A via `a` (valley sharpness / cond(G)); Case B via
  `λ, c` (residual magnitude ⇒ `‖R‖`). `cond(G(ẑ))` and `‖R(ẑ)‖/‖H(ẑ)‖` are
  recorded per cell so results are read relative to conditioning.
- Determinism: single seed (default 0) for the random control direction; the
  benchmarks are otherwise deterministic.

---

## 10. Output and provenance structure

Identical convention to EXP-001 (Decision 2), under a new experiment id:

```
outputs/EXP-002/<UTC-timestamp>_<short-commit>/
  config.json        # cases, param grids, t-grid, directions, alpha-grid A, tau, seed
  provenance.json    # provenance.run_metadata (commit, dirty, versions, x64, command, config)
  metrics.json       # per-cell: at-optimum + per-path metric families (§4–§8)
  arrays.npz         # H, G, R matrices and eigensystems along paths
  figures/           # FIG-02: E_G / angle / eigval-error / quad-error vs distance;
                     #         ΔL_k vs Q_k with validity radius marked
```

Written only under `outputs/` (gitignored). Nothing to `docs/` or the vault;
the Results Ledger is updated by a human later. Entry point
`experiments/exp002_nonlinear_validity.py::run()` calls `enable_x64()`;
provenance records `executed_command = "uv run python -m experiments.exp002_nonlinear_validity"`.

---

## 11. Files to create / unit tests

**New library / experiment files (design only):**
- `src/curvature_calib/benchmarks/nonlinear_residual.py` — Case A & B residual
  maps, analytic `J_r`, `G`, `R`, `H`, `∇L`, and a Newton solver for `ẑ_B`.
- `src/curvature_calib/geometry/validity.py` — reusable local-validity utilities:
  - `directional_loss_change(loss_fn, z_hat, direction, alphas) -> (len(A),)`;
  - `quadratic_prediction(lambda_k, alphas) -> (len(A),)`;
  - `quadratic_model_error(loss_fn, z_hat, v_k, lambda_k, alphas) -> (len(A),)`;
  - `validity_radius(loss_fn, z_hat, v_k, lambda_k, alphas, tol) -> {ρ+, ρ-, ρ}`.
  (Reused by EXP-004/005 later.)
- `experiments/exp002_nonlinear_validity.py` — `run()`.

**New tests:**
- `tests/test_nonlinear_residual.py`:
  - Case A: `G_ad == J_r^T J_r` analytic (`≤1e-10`); at `ẑ_A`, `‖R‖/‖H‖ ≤ 1e-10`
    and `H_ad == G_ad`; along a ray `‖R(z(t))‖` strictly increases in `t`.
  - Case B: Newton `ẑ_B` has `‖∇L‖ ≤ 1e-10`; `R_ad == 2λ r3 I` analytic (`≤1e-8`);
    `‖R(ẑ_B)‖/‖H(ẑ_B)‖` exceeds a floor (nonzero residual curvature).
  - AD Hessian vs analytic Hessian (`≤1e-10`); AD vs FD (own looser bound).
- `tests/test_validity.py`:
  - On a purely quadratic loss (reuse `benchmarks.linear_gaussian`) the quadratic
    prediction is exact ⇒ `validity_radius == max(A)` and `quadratic_model_error
    ≈ 0` for all α (sanity that the machinery is unbiased).
  - Case A at `ẑ_A`: `e_k(α) → 0` as `α → 0`; radius is a positive value inside
    `A`; asymmetry `ρ⁺ ≠ ρ⁻` detected along the curved direction.
  - Case B: GGN prediction biased (nonzero `e_k` even at small α scale set by
    `R`), radius smaller than the Case-A well-conditioned radius.
  - Prespecification guard: `τ` and `A` are read from arguments/config, not
    computed from the data (tested by passing them in and checking they are not
    mutated).
- `tests/test_provenance.py` extension (or reuse): EXP-002 config/provenance keys.

All EXP-002 tests require float64 and are added to the `tests/conftest.py`
x64-scoped module set, so the historical float32 suite stays unaffected.

---

## 12. Claims EXP-002 can and cannot support

**Can support (conditionally):**
- **C03** — "The GGN approximates the exact Hessian near a good fit." Case A
  shows `H=G` at the exact fit and quantifies the growth of `E_G`, subspace
  angle, and eigenvalue error with distance; Case B shows the `R`-induced bias
  at a nonzero-residual minimizer. Support is *conditional and local*, on these
  controlled maps only.
- **C09** — "The local GGN predicts actual loss changes over a nontrivial
  radius." The validity radius `ρ_k` (prespecified `τ`, `A`) is the direct
  evidence; a nontrivial `ρ>0` in Case A supports C09 at benchmark scope.

**Cannot support (out of scope for EXP-002):**
- Anything about Brock–Hommes, SIR, or network-SIR geometry (needs EXP-004/005).
- Any MMD-estimator claim (C04–C06; EXP-003).
- Prior-relative / generalized-posterior interpretation, `λ≷1` data-dominance,
  or covariance/contour agreement (C10–C11; EXP-005) — no prior/whitening here.
- Global or structural identifiability (C17 rejected); the radius is local and
  map-specific and does **not** generalize to ABMs by itself.
- Any claim that a single `τ` or `ρ` is universal — both are conventions fixed
  for this benchmark.

---

## 13. Reuse and scope

**Reused unchanged:** `geometry.ggn`, `metrics`, `config`, `provenance`,
`diagnostic.eigendecompose`, `benchmarks.linear_gaussian` (as a quadratic
sanity oracle), the `outputs/<EXP>/<run-id>/` + provenance convention.

**Deliberately untouched** (scope restriction): Brock–Hommes, SIR, network-SIR,
`surrogates.py`, `losses/mmd.py`, the historical per-seed OPG
(`per_seed_grads.py`, `diagnostic.opg_from_grads`), `calibrate.py`,
`preconditioner.py`, `baselines.py`, `falsification.py`, `bootstrap.py`, and all
canonical vault files. The tracked OPG-rename blocker
(`EXP001_IMPLEMENTATION_PLAN.md`) remains open and is **not** cleared by EXP-002.

Awaiting review before implementation. EXP-000 and EXP-003 are not begun.
