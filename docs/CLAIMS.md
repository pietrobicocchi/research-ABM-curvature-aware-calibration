---
title: Claims Ledger
status: active
last_verified: 2026-07-11
---

# Claims Ledger

Only claims marked **supported** or **supported conditionally** may enter the abstract or conclusion.

| ID | Proposed claim | Status | Required support | Current evidence | Intended location |
|---|---|---|---|---|---|
| C01 | The implemented AD construction reproduces the analytic generalized Gauss–Newton matrix on affine finite-dimensional benchmarks in float64. | supported | EXP-001 | RES-001 (commit 7c612df) | Methods / Validation |
| C02 | The GGN is positive semidefinite and measures first-order change in the calibrated representation. | supported | algebraic derivation | Mathematical Specification | Theory |
| C03 | On controlled nonlinear residual problems the implemented decomposition H=G+R is numerically exact, and the GGN predicts local loss geometry near exact or low-residual fits; non-negligible residual curvature causes an irreducible local prediction bias even arbitrarily close to a stationary point. | supported conditionally | EXP-002 (+EXP-005) | RES-002 (commit 4118f4e) | Theory / Results |
| C04 | On the controlled Gaussian location–scale simulator with frozen finite RFF (float64), the MMD GGN is estimated consistently from simulator feature Jacobians. | supported conditionally | EXP-003 | RES-003 (commit ce45c43) | Methods |
| C05 | The PSD plug-in MMD estimator exhibits the predicted PSD O(1/M) finite-sample bias. | supported conditionally | EXP-003 | RES-003 (commit ce45c43) | Results |
| C06 | The cross-seed MMD estimator is unbiased within Monte-Carlo resolution but may be indefinite at small sample sizes. | supported conditionally | EXP-003 | RES-003 (commit ce45c43) | Results |
| C07 | Raw per-seed scalar-loss OPG estimates the GGN. | rejected | contradicted algebraically | scalar and residual counterexamples | nowhere |
| C08 | The old OPG eigenspaces align with the true GGN on Brock--Hommes. | not supported | EXP-000 | EXP-000: F_OPG ≈ gradient covariance C_g, nearly orthogonal to the GGN (top-2 angle 86° at β=80) | Comparison / Appendix |
| C09 | The local GGN predicts actual loss changes over a nontrivial radius near exact or low-residual fits; the validity radius shrinks with curvature and collapses under non-negligible residual-curvature bias. | supported conditionally | EXP-002 (+EXP-004, EXP-005) | RES-002 (commit 4118f4e) | Results |
| C10 | Prior-relative GGN directions agree with local posterior contours in smooth SIR. | proposed | EXP-005 | none | Results |
| C11 | Weak prior-relative SIR directions agree with profiled generalized-posterior energy. | proposed | EXP-005 | none | Results |
| C12 | A locally weak SIR direction materially changes a policy quantity. | proposed | EXP-006 | old OPG evidence is not sufficient | Results |
| C13 | Additional observations increase information in the weak direction. | proposed | EXP-007 | none | Results |
| C14 | Important local eigenspaces are robust to the chosen surrogate gradient. | proposed | EXP-008 | old results require recomputation | Results |
| C15 | Truncated differentiation can destroy local information geometry in transient models. | proposed | EXP-008 | old OPG evidence only | Results |
| C16 | AD-based GGN estimation is computationally preferable to finite-difference Hessians at demonstrated scales. | proposed | EXP-009 | none | Results |
| C17 | Small local eigenvalues prove structural non-identifiability. | rejected | false in general | conceptual counterexamples | nowhere |
| C18 | Posterior curvature can be interpreted as data information without separating the prior. | rejected | false by decomposition | Mathematical Specification | nowhere |
| C19 | MMD is the only common calibration loss admitting a GGN. | rejected | GGN applies more broadly | standard composite-loss theory | nowhere |
| C20 | The proposed method is a local diagnostic relevant to practical non-identifiability, not a complete global test. | supported conditionally | conceptual analysis | Mathematical Specification | Introduction / Discussion |

## Scope note on C01

C01 is supported **only** for affine finite-dimensional representations in
float64 (EXP-001). It does **not** yet establish correctness of the AD GGN
construction for: nonlinear representations, stochastic simulators, the MMD
GGN, surrogate/relaxed gradients, or the Brock–Hommes and SIR models. Those
require EXP-002 (nonlinear), EXP-003 (MMD), EXP-004/005 (BH/SIR), and EXP-008
(surrogate/stochastic) respectively.

## Scope note on C04, C05, and C06

On the controlled Gaussian location–scale simulator with frozen finite random
Fourier features in float64 (EXP-003), the MMD generalized Gauss–Newton matrix
can be estimated consistently from simulator feature Jacobians. The PSD plug-in
estimator exhibits the predicted positive-semidefinite `O(1/M)` bias, while the
cross-seed estimator is unbiased within Monte-Carlo resolution but may be
indefinite at small sample sizes. This is **not** yet established for:
Brock–Hommes; SIR; the production MMD code; arbitrary kernels or representations;
surrogate gradients; discrete stochastic ABMs; or posterior geometry.

## Scope note on C03 and C09

C03 and C09 are supported conditionally **only** on the deterministic controlled
nonlinear residual benchmarks of EXP-002 (float64). They have **not** been
established for stochastic simulators, the MMD GGN, or the Brock–Hommes and SIR
models; those require EXP-003 (MMD) and EXP-004/005 (BH/SIR). EXP-002 also fixes
the vocabulary for two failure modes: (i) **residual-curvature bias**, nonzero as
`α→0`, quantified by `e_curv(v)=|vᵀ(H−G)v| / max(|vᵀHv|, ε)`; and (ii)
**higher-order nonlinear error**, which grows with the perturbation radius and is
captured by the validity radius.

## Claim review template

### Claim ID

### Current wording

### Status

### Evidence required

### Evidence available

### Conditions and scope

### Strongest defensible wording

### What the evidence does not establish

### Decision
