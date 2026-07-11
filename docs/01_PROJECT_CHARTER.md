---
title: Project Charter
status: active
last_verified: 2026-07-11
---

# Project Charter

## 1. Provisional title

**Local Information Geometry of Differentiable Agent-Based Models**

Possible subtitle:

**Gauss--Newton diagnostics for generalized Bayesian calibration**

## 2. Research objective

Determine whether automatic differentiation can provide a reliable estimate of the local, prior-relative generalized Gauss--Newton geometry of calibration objectives for stochastic differentiable agent-based models.

The central scientific question is:

> Near a calibrated parameter value, which combinations of ABM parameters produce first-order changes in the calibrated representation large enough to dominate the local prior information?

## 3. Primary mathematical object

Let \(z \in \mathbb{R}^P\) denote unconstrained, prior-scaled coordinates and let

\[
X(z,\xi) = f(T(z),\xi)
\]

be the stochastic simulator.

Let

\[
m(z) = \mathbb{E}_{\xi}[\psi(X(z,\xi))]
\]

be the calibrated representation, and define

\[
\mathcal{L}(z)
=
\frac12 \|m(z)-m_y\|_W^2.
\]

The primary geometry is

\[
G(z)
=
D m(z)^\ast W D m(z).
\]

For MMD,

\[
G_{\mathrm{MMD}}(z)
=
J_\mu(z)^\ast J_\mu(z),
\]

where \(J_\mu(z)=D_z\mu_z\) is the derivative of the kernel mean embedding.

## 4. Generalized Bayesian interpretation

The generalized posterior is

\[
\pi_w(z\mid y)
\propto
\exp\{-w\mathcal{L}(z)\}\pi(z).
\]

With a standard-normal prior on \(z\), local posterior precision is approximated by

\[
I + wG(\hat z),
\]

provided that the generalized Gauss--Newton approximation is accurate near \(\hat z\).

The eigensystem of \(wG(\hat z)\) is interpreted relative to unit prior precision.

## 5. Intended contributions

1. A correct formulation of local generalized Gauss--Newton geometry for stochastic differentiable ABMs.
2. AD-based estimators for this geometry under moment matching and MMD.
3. A prior-relative interpretation separating data geometry from prior precision.
4. Validation against exact Hessians, local posterior contours, profiled objectives, and high-fidelity derivative references.
5. A study of the effect of simulator noise, surrogate gradients, and truncated differentiation.

## 6. Explicit non-claims

The project does **not** assume or claim that:

- raw per-seed scalar-loss gradient outer products estimate the GGN;
- local weak curvature proves global non-identifiability;
- a zero eigenvalue at one point proves structural non-identifiability;
- posterior concentration necessarily comes from the data;
- surrogate derivatives are exact derivatives of the original discrete simulator;
- the geometry is free once a scalar gradient is available;
- one numerical threshold defines a universal identifiable dimension;
- the GGN necessarily approximates the Hessian far from a good fit;
- MMD is the only loss admitting a GGN.

## 7. Terminology

Preferred:

- local data-informed direction;
- locally weakly informed parameter combination;
- prior-relative local geometry;
- local Gauss--Newton sensitivity;
- local proxy for practical non-identifiability.

Avoid unless separately established:

- identified parameter;
- structurally non-identifiable;
- globally unidentified;
- exact Fisher information of the ABM;
- curvature emitted for free.

## 8. Main research questions

### RQ1 — Estimator correctness

Can AD accurately estimate

\[
G(z)=D m(z)^\ast W D m(z)
\]

for stochastic ABMs?

### RQ2 — Curvature validity

Over what neighborhood of a fitted parameter does \(G\) adequately approximate the exact Hessian and actual loss variation?

### RQ3 — Inferential validity

Do prior-relative GGN directions agree with local generalized-posterior contours and directional profiled losses?

### RQ4 — Differentiation fidelity

How do simulator randomness, surrogate derivatives, and differentiation-horizon truncation affect the eigenspaces?

### RQ5 — Scientific utility

Can the method identify a locally weakly informed combination that materially affects a policy prediction, and can additional observations constrain it?

## 9. Decision gates

### Gate 1 — Mathematical implementation

AD reproduces analytic GGN matrices on controlled examples.

### Gate 2 — MMD estimator viability

The MMD GGN estimator converges at acceptable cost and recovers the reference eigenspace.

### Gate 3 — Local curvature validity

GGN and exact Hessian leading subspaces agree near the fitted point, with a measurable local validity radius.

### Gate 4 — Inferential relevance

Prior-relative GGN directions agree with local posterior contours and profiled objectives.

### Gate 5 — Stochastic derivative fidelity

Important eigenspaces survive stochastic and surrogate-gradient estimation with quantified uncertainty.

### Gate 6 — Scientific consequence

At least one weak direction matters for prediction, policy, or observation design.

## 10. Scope control

Minimum viable paper:

1. analytic linear benchmark;
2. nonlinear curvature benchmark;
3. MMD estimator validation;
4. smooth Brock--Hommes;
5. smooth and discrete SIR;
6. posterior/profile validation;
7. one policy or observation-design result.

Deferred unless justified:

- optimization preconditioning;
- macroeconomic-scale ABM;
- heterogeneity scaling;
- large-\(P\) applications;
- extensive calibration-trajectory analysis.
