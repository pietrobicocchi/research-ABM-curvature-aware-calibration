---
title: Claims Ledger
status: active
last_verified: 2026-07-11
---

# Claims Ledger

Only claims marked **supported** or **supported conditionally** may enter the abstract or conclusion.

| ID | Proposed claim | Status | Required support | Current evidence | Intended location |
|---|---|---|---|---|---|
| C01 | AD reproduces analytic GGN matrices on controlled models. | proposed | EXP-001 | none | Methods / Validation |
| C02 | The GGN is positive semidefinite and measures first-order change in the calibrated representation. | supported | algebraic derivation | Mathematical Specification | Theory |
| C03 | The GGN approximates the exact Hessian near a good fit. | proposed | EXP-002, EXP-005 | theoretical condition only | Theory / Results |
| C04 | The MMD GGN can be estimated consistently from simulator feature Jacobians. | proposed | EXP-003 | derivation only | Methods |
| C05 | The PSD plug-in MMD estimator has acceptable finite-sample bias. | proposed | EXP-003 | none | Results |
| C06 | The cross-seed MMD estimator has lower bias but may be indefinite at finite sample size. | proposed | EXP-003 | derivation only | Results |
| C07 | Raw per-seed scalar-loss OPG estimates the GGN. | rejected | contradicted algebraically | scalar and residual counterexamples | nowhere |
| C08 | The old OPG eigenspaces align with the true GGN on Brock--Hommes. | proposed | EXP-000 | none | Comparison / Appendix |
| C09 | The local GGN predicts actual loss changes over a nontrivial radius. | proposed | EXP-002, EXP-004, EXP-005 | none | Results |
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
