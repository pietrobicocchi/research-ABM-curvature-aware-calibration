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
| C03 | On controlled nonlinear residual problems the implemented decomposition H=G+R is numerically exact, and the GGN predicts local loss geometry near exact or low-residual fits; non-negligible residual curvature causes an irreducible local prediction bias even arbitrarily close to a stationary point. | supported conditionally | EXP-002 (+EXP-004 BH, EXP-005) | EXP-002; EXP-004: H=G at the BH fit to 1e-16, R grows off-fit | Theory / Results |
| C04 | On the controlled Gaussian location–scale simulator with frozen finite RFF (float64), the MMD GGN is estimated consistently from simulator feature Jacobians. | supported conditionally | EXP-003 | RES-003 (commit ce45c43) | Methods |
| C05 | The PSD plug-in MMD estimator exhibits the predicted PSD O(1/M) finite-sample bias. | supported conditionally | EXP-003 | RES-003 (commit ce45c43) | Results |
| C06 | The cross-seed MMD estimator is unbiased within Monte-Carlo resolution but may be indefinite at small sample sizes. | supported conditionally | EXP-003 | RES-003 (commit ce45c43) | Results |
| C07 | Raw per-seed scalar-loss OPG estimates the GGN. | rejected | contradicted algebraically | scalar and residual counterexamples | nowhere |
| C08 | The old OPG eigenspaces align with the true GGN on Brock--Hommes. | not supported | EXP-000 | EXP-000: F_OPG ≈ gradient covariance C_g, nearly orthogonal to the GGN (top-2 angle 86° at β=80) | Comparison / Appendix |
| C09 | The local GGN predicts actual loss changes over a nontrivial radius near exact or low-residual fits; the validity radius shrinks with curvature and collapses under non-negligible residual-curvature bias. | supported conditionally | EXP-002 (+EXP-004 BH, EXP-005) | EXP-002; EXP-004: in chaotic BH the radius is very small (≈0.003 at β=50, →0 at β=80) | Results |
| C10 | Prior-relative GGN directions agree with local posterior contours in smooth SIR. | supported conditionally | EXP-005 | EXP-005: leading posterior axes agree to ~1°, cov error 25%, d_data=3 (mild non-Gaussianity in t_lock) | Results |
| C11 | Weak prior-relative SIR directions agree with profiled generalized-posterior energy. | supported conditionally | EXP-005 | EXP-005: profiled posterior energy matches the GGN quadratic to <0.5% (stiff & sloppy) | Results |
| C12 | A locally weak SIR direction materially changes a policy quantity. | supported conditionally | EXP-006 | EXP-006: the counterfactual intervention value spans ~0→2180 cases within a good fit, along the sloppiest GGN direction (f_lock); observed cases stay ±0.3% | Results |
| C13 | Additional observations increase information in the weak direction. | supported conditionally | EXP-007 | EXP-007: compliance data lifts the weak f_lock direction λ 8.9e-6→10 (d_data 3→5), collapsing policy uncertainty 57×; redundant prevalence does not | Results |
| C14 | Important local eigenspaces are robust to the chosen surrogate gradient. | supported conditionally | EXP-008 | EXP-008: on discrete network-SIR the leading GGN eigenspace agrees Gumbel↔straight-through to ≤8.9° (k≤3), Gumbel FD-validated (3.1e-5); eigenvalue scale is surrogate-dependent (0.22–0.43×) | Results |
| C15 | Truncated differentiation can destroy local information geometry in transient models. | supported conditionally | EXP-008 | EXP-008: halving the differentiation horizon rotates the leading eigenspace ~80° and collapses its top eigenvalue to 15% of full (→1% at gh=T/8) | Results |
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
