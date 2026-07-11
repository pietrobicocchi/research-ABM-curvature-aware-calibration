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
