---
title: Evidence to Paper Map
status: active
last_updated: 2026-07-14
---

# Evidence to Paper Map

The supported-claims projection of Layer 1. This file **selects and rewords**
Layer-1 facts for the manuscript; it never invents a fact. Every row cites a claim
ID from `docs/CLAIMS.md` and its evidence as **experiment + authoritative commit**
(no separate results-ledger ids — `docs/CLAIMS.md` is the single ledger).

> **Figure numbering (2026-07-14):** `FIGURE_LEDGER.md` was re-ordered to the
> reader's belief journey and is the **authority** for figure IDs. The figure
> tags in the table below still use the pre-belief-order numbering and are being
> migrated; when they disagree with the ledger, the ledger wins. Mapping: old
> FIG-01→inset of new FIG-02; old FIG-02→FIG-03; old FIG-05→FIG-04;
> old FIG-06→FIG-05/FIG-06; old FIG-07/08→FIG-07/08; old FIG-09→FIG-10;
> old FIG-03 (MMD) and old FIG-04 (OPG) → appendix. New: FIG-01 (schematic),
> FIG-06 (participation heatmap), FIG-09 (calibration-path).

No substantive scientific statement enters the paper unless it is connected to:

1. a supported or conditionally supported claim in `docs/CLAIMS.md`;
2. a reviewed experiment in `docs/experiments/EXP-*.md`;
3. an approved figure or table (`FIGURE_LEDGER.md`) when empirical.

## Current map

| Planned paper statement | Claim ID | Evidence (experiment · commit) | Figure/table | Planned section | Status |
|---|---|---|---|---|---|
| AD reproduces the analytic GGN on affine finite-dimensional benchmarks. | C01 | EXP-001 · `7c612df` | FIG-01 | Controlled validation | supported |
| The gradient second moment (empirical Fisher / OPG) can vanish at exact fit while the GGN remains nonzero. | rejected C07 clarification | EXP-001 · `7c612df`; EXP-003 · `ce45c43` | FIG-01, FIG-03 | Appendix clarification | supported as clarification |
| The exact Hessian decomposes as \(H=G+R\). | C02 / C03 | `docs/MATH.md`; EXP-002 · `4118f4e`; EXP-004 · `6ff8031` (real ABM: \(H=G\) at the BH fit to \(10^{-16}\)) | FIG-02, FIG-05 | Theory / validation | supported |
| Residual curvature can create irreducible local GGN bias. | C03, C09 | EXP-002 · `4118f4e`; EXP-004 · `6ff8031` (BH: \(R\) grows off-fit; radius ≈0.003 at β=50) | FIG-02, FIG-05 | Controlled validation | supported conditionally |
| The PSD MMD plug-in estimator has positive \(O(1/M)\) bias. | C05 | EXP-003 · `ce45c43` | FIG-03 | MMD estimation | supported conditionally |
| The cross-seed estimator is unbiased in expectation but may be indefinite at finite \(M\). | C06 | EXP-003 · `ce45c43` | FIG-03 | MMD estimation | supported conditionally |
| The leading MMD GGN subspace is recoverable at feasible \(M\) in the controlled simulator. | C04 | EXP-003 · `ce45c43` | FIG-03 | MMD estimation | supported conditionally |
| In a real ABM (Brock–Hommes), the gradient second moment (empirical Fisher / OPG) tracks the gradient covariance \(C_g\), not the GGN, and is nearly orthogonal to it in the chaotic regime. | C08 (not supported) → C07 clarification | EXP-000 · `02d6bd6` | FIG-04 | Appendix clarification | supported as clarification |
| In smooth BH the GGN equals the exact Hessian at the fit; away from it residual curvature reintroduces \(R\), and the local quadratic is predictive only over a small radius. | C03, C09 | EXP-004 · `6ff8031` | FIG-05 | BH results | supported conditionally |
| In chaotic differentiable ABMs the exact full-horizon GGN is dominated by intrinsic \(\partial X/\partial\theta\) sensitivity, so the geometry depends materially on the differentiation horizon (representation-independent). | limitation (no claim) | EXP-004 · `6ff8031`; EXP-004b | FIG-05 | BH results / Discussion | supported as limitation |
| Prior-relative GGN agrees with local SIR posterior geometry (axes ~1°, \(d_{\mathrm{data}}=3\)). | C10 | EXP-005 · `7ba9447` | FIG-06 | SIR inference | supported conditionally |
| Weak SIR directions agree with profiled posterior energy (<0.5%). | C11 | EXP-005 · `7ba9447` | FIG-06 | SIR inference | supported conditionally |
| A locally weak direction leaves a policy quantity undetermined: the counterfactual intervention value spans ~0→2180 cases within a good fit, along the sloppiest direction, while observed functionals stay ±0.3%. | C12 | EXP-006 · `91fe277` | FIG-07 | Scientific consequence | supported conditionally |
| A targeted added observation (compliance) recovers the weak direction (λ 8.9e-6→10, d_data 3→5) and collapses the policy range 57×; a redundant one (prevalence) does not. | C13 | EXP-007 · `4adb5c6` | FIG-08 | Scientific consequence | supported conditionally |
| The diagnostic loop survives in a stochastic/discrete ABM under an estimated MMD geometry; surrogate gradients preserve (or predictably distort) the important eigenspaces. | C14, C15 | EXP-008 | FIG-09 | Completion / fidelity | pending (keystone) |
| Computational cost of AD-based GGN estimation. | (no claim — parked) | EXP-009 | — | Discussion paragraph | parked |

**Note on C08 (demoted).** The claim as originally proposed ("old OPG eigenspaces
align with the true GGN on Brock–Hommes") is **not supported** (EXP-000). In the
paper it is a **demoted appendix clarification** — a real-ABM instance of the
empirical-Fisher ≠ GGN distinction — **not** a motivating pillar. The positive
problem (a good fit does not determine which combinations, or which counterfactuals,
the data constrains) carries the motivation; the OPG is a technical footnote,
anchored to the empirical-Fisher critique (verify citation) rather than to the
project's own history.

## Section-level evidence plan

### 1. Introduction

Allowed now: differentiable ABMs expose derivatives of calibrated representations and
are used for counterfactual / policy questions; a good fit does not determine which
parameter combinations — or which counterfactuals — the data constrains; the paper
studies the GGN of the calibrated representation as the correct local object; the
interpretation is local and prior-relative; the prior-relative GGN matches a real
local posterior on smooth SIR (EXP-005). (The gradient second moment / OPG
distinction is a demoted appendix clarification, not an introduction beat.)

Not yet earned: policy consequences; observation-design value; computational
scalability. (Actual-ABM *inference* is now earned on smooth SIR, EXP-005; actual-ABM
*scientific consequence* is not, pending EXP-006.)

### 2. Differentiable generalized calibration

Definitions \(z, T, X, \xi, \psi, m, \mathcal L, \pi_w\). Evidence: `docs/MATH.md`
and verified literature (`docs/papers/`).

### 3. Local Gauss–Newton geometry

Claims: \(G=Dm^\ast W Dm\) is PSD; \(v^\top Gv\) measures squared first-order
representation change; \(H=G+R\); GGN validity depends on residual curvature.
Evidence: `docs/MATH.md`; EXP-001 · `7c612df`; EXP-002 · `4118f4e`; FIG-01, FIG-02.

### 4. MMD geometry and estimation

Claims: \(G_{\mathrm{MMD}}=J_\eta^\top J_\eta\); plug-in and cross-seed estimators
have distinct finite-sample properties; scalar-gradient OPG is residual-dependent
and distinct. Evidence: `docs/MATH.md`; EXP-003 · `ce45c43`; FIG-03.

### 5. Prior-relative local information

Available: prior whitening is required for meaningful comparisons; local posterior
precision is approximated by \(P_\pi+wG\), conditional on GGN validity. Pending:
agreement with posterior contours and profiles (EXP-005).

### 6. Controlled validation

Evidence: EXP-001 · `7c612df`; EXP-002 · `4118f4e`; EXP-003 · `ce45c43`; FIG-01,
FIG-02, FIG-03. Narrative: implementation correctness → nonlinear curvature
conditions → stochastic MMD estimator behavior.

### 7. Brock–Hommes

Available now (EXP-000 · `02d6bd6`, appendix): the gradient second moment (empirical
Fisher / OPG) is the gradient covariance, not the GGN; the true GGN is
multi-directional (effective rank ~4).
Available now (EXP-004 · `6ff8031`, EXP-004b): in the actual BH model \(H=G\) at the
fit to \(10^{-16}\) and \(R\) grows off-fit (C03/C09 in a real ABM); the local
quadratic is predictive only over a small radius (≈0.003 at β=50, →0 at β=80). The
**differentiation-horizon dependence** is characterized and shown intrinsic to
\(\partial X/\partial\theta\) (representation-independent) — recorded as a scope
**limitation** of the exact geometry in chaotic ABMs, not a claim.
Pending: none for this chapter.

### 8. SIR and scientific consequence

Available now (EXP-005 · `7ba9447`): on smooth SIR the prior-relative GGN matches
the local generalized posterior — leading axes ~1°, \(d_{\mathrm{data}}=3\) (C10);
profiled posterior energy matches the GGN quadratic to <0.5% along stiff and sloppy
directions (C11).

**Scientific consequence, available now.** EXP-006 · `91fe277` (C12): observed
functionals are tightly identified, but the counterfactual intervention value spans
~0→2180 cases within a good fit, along the sloppiest GGN direction — the diagnostic
flags the policy answer the data cannot determine. EXP-007 · `4adb5c6` (C13): a
compliance observation recovers that weak direction (λ 8.9e-6→10, d_data 3→5) and
collapses the policy range 57×, while redundant prevalence does not — the diagnostic
is actionable.

**Caveat carried into the paper.** EXP-005/006/007 are all smooth *deterministic*
SIR under a Gaussian-summary loss. Pending EXP-008 (keystone): the same loop on a
stochastic/discrete ABM under an estimated MMD geometry, which also makes the
EXP-003 estimators load-bearing and tests surrogate fidelity (C14/C15).

### 9. Discussion

Evidence-backed limitations available now: locality; dependence on representation
and loss; residual-curvature bias (now in a real ABM, EXP-004); finite-sample
estimator trade-offs; OPG/GGN distinction (real ABM, EXP-000); **in chaotic
differentiable ABMs the exact geometry is intrinsically sensitivity-dominated and
horizon-dependent** (EXP-004/004b); mild posterior non-Gaussianity (EXP-005,
t_lock). Pending: surrogate-gradient distortion; computational scale.

## Claim-to-prose rule

Before adding a substantive sentence to the paper, record:

```text
Sentence:
Claim ID:          (must exist in docs/CLAIMS.md)
Evidence:          (EXP-xxx · commit, and figure if empirical)
Scope:
Figure/table:
```

If no claim ID or evidence exists, keep the sentence provisional.

## New evidence template

**Proposed paper statement:**
**Claim ID (docs/CLAIMS.md):**
**Evidence (EXP-xxx · commit):**
**Figure/table:**
**Scope:**
**Strongest defensible wording:**
**Wording to avoid:**
**Planned section:**
