---
title: Results Ledger
status: active
last_verified: 2026-07-11
---

# Results Ledger

This document records reviewed results. Preliminary observations should not be treated as paper claims until entered here.

## Result status

- preliminary;
- replicated;
- reviewed;
- superseded;
- rejected.

---

## RES-000 — Historical OPG results

### Status

Superseded as curvature evidence.

### Observation

Previous Brock--Hommes and SIR experiments produced structured eigenspectra and perturbation directions using raw per-seed scalar-loss gradient outer products.

### Interpretation

These results may reflect useful empirical sensitivity or stochastic-gradient geometry.

### What this supports

- The old OPG may contain model-specific directional information worth comparing with the true GGN.

### What this does not establish

- that OPG equals the GGN;
- that its eigenvalues estimate Hessian eigenvalues;
- that its rank is an identifiable dimension;
- that its weak directions are structurally or practically non-identifiable.

### Follow-up

Run EXP-000.

---

## RES-001 — EXP-001 analytic linear GGN recovery

- **Status:** reviewed
- **Experiment ID:** EXP-001
- **Authoritative commit:** `7c612df` (clean tree, `git_dirty: false`)
- **Environment:** Python 3.12.13, JAX/jaxlib 0.4.30, float64 (x64 enabled)
- **Run IDs:** `outputs/EXP-001/20260711T172324Z_7c612df/` (float64 validation),
  `outputs/EXP-001/20260711T172342Z_7c612df/` (~1e13 stress). Earlier dirty-tree
  runs (`…_e1b3a6f`) retained as historical artifacts.
- **Tests:** `146 passed` (`uv run pytest`).

### Observation (float64, worst over 18 cells: P∈{5,20,100}, cond∈{1,1e2,1e6}, full & rank-deficient, weight∈{None,diag,dense SPD})

- **Dense GGN:** `‖G_AD − AᵀWA‖_F / ‖AᵀWA‖_F = 0.00e+00`.
- **Matrix-free:** `‖Gv − (AᵀWA)v‖ / ‖(AᵀWA)v‖ ≤ 7.65e-16`.
- **Hessian equality (affine ⇒ R=0):** `‖H − G‖_F / ‖G‖_F = 0.00e+00`.
- **Rank / null-space:** numerical rank = rank(A) in all cells; null-space
  principal angle `≤ 3.33e-08`.
- **OPG counterexample (DEC-001):** at the exact fit the scalar-gradient OPG is
  `0.00e+00` while the GGN is nonzero; off-fit the OPG is rank 1.
- **Finite-difference (own looser regime):** best step-size relative error
  `8.37e-15`; error is round-off-monotone (no truncation term for a quadratic).
- **Non-gating stress (cond ~1e13):** `rel_fro(G_AD, AᵀWA)=0.00e+00`; the
  numerical rank **under the stated tolerance** (`rtol=1e-10`) is 15/20 (the
  small σ² fall below threshold — a tolerance effect, not a change in the
  mathematical rank of `AᵀWA`).

### Supports

- C01 — supported (affine finite-dimensional, float64).

### Does not establish

- correctness for nonlinear representations (EXP-002);
- MMD GGN, surrogate/stochastic gradients (EXP-003, EXP-008);
- Brock–Hommes or SIR geometry (EXP-004/005);
- any prior-relative/posterior or global-identifiability claim.

### Follow-up

Run EXP-002.

---

## RES-002 — EXP-002 nonlinear local-validity benchmark

- **Status:** reviewed
- **Experiment ID:** EXP-002
- **Authoritative commits:** code `4118f4e`, review `0d1eb4b` (clean tree)
- **Environment:** Python 3.12.13, JAX/jaxlib 0.4.30, float64
- **Run ID:** `outputs/EXP-002/20260711T175059Z_4118f4e/`
- **Tests:** `166 passed` (`uv run pytest`).

### Benchmarks

- **Case A — Rosenbrock exact-fit valley:** `r=[a(z2−z1²), b(1−z1)]`,
  optimum `(1,1)`, `R=0` there; `a∈{1,5,10}`, `b=1`.
- **Case B — irreducible residual:** `r=[z1−1, z2−1, λ(z1²+z2²−c)]`,
  analytic `R=2λ·r3·I`; `(λ,c)∈{(0.3,0.0),(1.0,4.0)}`.

### Observation

- **Minimizer checks:** Case A `‖∇L‖=0`; Case B Newton `‖∇L‖≤1.4e-15`.
- **R_AD vs analytic R:** `rel_fro ≤ 3.2e-15` (all cells); `G_AD`/`H_AD` vs
  analytic `≤1.6e-16`. The decomposition `H=G+R` is numerically exact.
- **Case A validity radii (τ=10%, prespecified signed α-grid):**
  `a=1 → {v0:0.1, v1:0.3}`, `a=5 → {0.1,0.1}`, `a=10 → {0.1,0.05}` — nontrivial,
  local, direction-dependent; shrinks as the valley sharpens.
- **Case B residual-curvature magnitude at the minimizer:**
  `‖R‖/‖H‖ = 0.158` (λ=0.3) and `0.025` (λ=1,c=4); eigenvalue rel error up to
  `0.39` while the leading-eigenvector angle stays `≤2.1e-08`.
- **Zero GGN validity radius (Case B):** under the prespecified 10% tolerance the
  radius is `0` for the biased directions (`λ=0.3 → {0,0}`, `λ=1 → {0.1, 0}`),
  because the residual-curvature bias `≈ vᵀRv/vᵀHv` is roughly constant in α and
  exceeds τ — an irreducible bias that persists as `α→0`.

### Two failure modes (main scientific result)

1. **residual-curvature bias** — nonzero as `α→0`
   (`e_curv(v)=|vᵀ(H−G)v|/max(|vᵀHv|,ε)`);
2. **higher-order nonlinear error** — grows with perturbation radius (validity
   radius).

### Supports

- C03 — supported conditionally (numerically exact H=G+R; GGN predicts near
  exact/low-residual fits; irreducible bias when R non-negligible).
- C09 — supported conditionally (nontrivial radius near good fits; collapses
  under residual-curvature bias).

### Does not establish

- correctness for stochastic simulators or the MMD GGN (EXP-003);
- Brock–Hommes or SIR geometry (EXP-004/005);
- any unconditional/global validity claim.

### Follow-up

Run EXP-003.

---

## RES-003 — EXP-003 stochastic MMD GGN estimation

- **Status:** reviewed
- **Experiment ID:** EXP-003
- **Authoritative commit:** `ce45c43` (clean tree, `git_dirty: false`)
- **Environment:** Python 3.12.13, JAX/jaxlib 0.4.30, float64
- **Run ID:** `outputs/EXP-003/20260711T215027Z_ce45c43/`
- **Tests:** `181 passed`.

### Benchmark

Gaussian location-scale `X_z=a(z)+L(z)ε` (prior-scaled `z`, P=5), frozen RFF
(γ=2, D=512). Main regime `G_ref` eigenvalues `[0.116,0.112,0.0074,0.0036,0.0018]`
(gap λ2/λ3≈15); near-degenerate `[0.082,0.041,0.029,0.028,0.020]` (λ2/λ3≈1.4).

### Observation

- **Analytic reference validated:** closed-form feature mean `η` vs MC
  `rel_fro=2.4e-3` (MC floor); `J_η` vs central finite differences `3.9e-11`;
  finite-D `G_ref → G_RBF` (exact RBF kernel) `0.084→0.028` over D∈{64,256,1024}.
- **Plug-in bias (C05):** relative bias `0.187→0.011` for M=8→256; predicted
  `Ĉ/M` lies inside the observed 95% CI for M∈{8..128}; at M=256 the true bias is
  below the B=200 sampling floor (a resolution limit, not a model failure).
- **Cross-seed (C06):** bias compatible with zero at all M; negative-eigenvalue
  frequency `0.94→0.28→0` (M=8→32+) in both regimes.
- **Top-two subspace recovery (C04):** angle ≤5° for M≥16 (main and
  near-degenerate); A3 passes at M∈{16,32,64,128,256}.
- **OPG residual sweep:** `F_OPG_pop=0` at exact match (δ=0) while `G_ref≠0`;
  for δ>0 the OPG is residual-dependent and shape-distinct (trace-normalized
  `rel_fro≈0.94`) — confirms OPG ≠ GGN (DEC-001).

### Supports

- C04, C05, C06 — supported conditionally.

### Does not establish

- Brock–Hommes, SIR; production MMD code; arbitrary kernels/representations;
  surrogate gradients; discrete stochastic ABMs; posterior geometry.

### Follow-up

Run EXP-000 (Brock–Hommes historical-OPG audit).

---

# Result entry template

## RES-XXX — Experiment and result title

- **Status:** preliminary / replicated / reviewed / superseded / rejected
- **Date:**
- **Experiment ID:**
- **Git commit:**
- **Environment:**
- **Configuration:**
- **Random seeds:**
- **Raw output directory:**
- **Processed output directory:**
- **Figure script:**

### Question

### Observation

State only what was measured.

### Quantitative result

Include confidence intervals, variation across seeds, and relevant sample sizes.

### Interpretation

State the narrowest defensible interpretation.

### Supports

- Claim IDs:

### Does not establish

- 
- 
- 

### Unexpected observations

### Robustness checks

### Limitations

### Follow-up

### Review decision
