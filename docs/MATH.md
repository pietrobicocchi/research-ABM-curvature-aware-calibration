---
title: Mathematical Specification
status: active
last_verified: 2026-07-11
---

# Mathematical Specification

This document contains the current accepted mathematical formulation. Rejected derivations should not remain here.

## 1. Coordinates and prior

Let

\[
z\in\mathbb{R}^P
\]

be unconstrained coordinates and let

\[
\theta=T(z)
\]

map them to physical model parameters.

Default prior:

\[
z\sim\mathcal{N}(0,I).
\]

This makes the Euclidean geometry in \(z\)-space equal to the prior-whitened geometry.

For a general Gaussian prior with precision \(P_\pi\), use the generalized eigenproblem

\[
wGv_k=\lambda_k P_\pi v_k.
\]

## 2. Stochastic simulator

Let

\[
X(z,\xi)=f(T(z),\xi)
\]

denote the simulator output, where \(\xi\sim P_\xi\) contains all simulator randomness.

## 3. Calibrated representation

Let

\[
\psi:\mathcal{X}\rightarrow\mathcal{H}
\]

be a feature or observable map.

Define

\[
m(z)
=
\mathbb{E}_{\xi}[\psi(X(z,\xi))]
\in\mathcal{H}.
\]

The observed or reference representation is \(m_y\).

Examples:

### Finite summaries

\[
\psi(X)=S(X)\in\mathbb{R}^K.
\]

### Finite kernel features

\[
\psi(X)\in\mathbb{R}^D.
\]

### Population kernel mean embedding

\[
m(z)=\mu_z
=
\mathbb{E}_\xi[k(X(z,\xi),\cdot)]
\in\mathcal{H}.
\]

## 4. Calibration loss

The baseline loss is

\[
\mathcal{L}(z)
=
\frac12
\|m(z)-m_y\|_W^2,
\]

where \(W\succeq 0\).

Define the residual

\[
r(z)=m(z)-m_y.
\]

Then

\[
\mathcal{L}(z)
=
\frac12\langle r(z),Wr(z)\rangle.
\]

## 5. Derivative of the calibrated representation

Define

\[
J_m(z)=D_z m(z).
\]

Assuming differentiation and expectation can be interchanged,

\[
J_m(z)
=
\mathbb{E}_{\xi}
\left[
D_z\psi(X(z,\xi))
\right].
\]

> **Status: assumption**
>
> Interchanging differentiation and expectation requires regularity and domination conditions. These must be stated for each model or treated as an empirical approximation when surrogate derivatives are used.

## 6. Gradient

\[
\nabla_z\mathcal{L}(z)
=
J_m(z)^\ast Wr(z).
\]

## 7. Exact Hessian decomposition

The exact Hessian is

\[
\nabla_z^2\mathcal{L}(z)
=
G(z)+R(z),
\]

where

\[
\boxed{
G(z)
=
J_m(z)^\ast WJ_m(z)
}
\]

and

\[
R(z)
=
\left\langle
Wr(z),
D_z^2m(z)
\right\rangle.
\]

For finite-dimensional \(m=(m_1,\ldots,m_K)\),

\[
R(z)
=
\sum_{k=1}^K
[Wr(z)]_k
\nabla_z^2m_k(z).
\]

## 8. Generalized Gauss--Newton matrix

\[
\boxed{
G(z)=J_m(z)^\ast WJ_m(z)
}
\]

is positive semidefinite.

For any direction \(v\),

\[
v^\top G(z)v
=
\|J_m(z)v\|_W^2.
\]

Interpretation:

> \(v^\top Gv\) is the squared first-order change in the calibrated representation induced by a prior-scaled movement along \(v\).

## 9. Relationship to the Hessian

The GGN equals the exact Hessian when \(R(z)=0\), including when:

- \(m(z)\) is locally affine;
- the residual is zero;
- the residual-weighted second derivative term otherwise vanishes.

The GGN is an approximation to the Hessian when \(R(z)\) is small.

> **Status: conditional**
>
> The usefulness of the approximation must be established empirically through Hessian comparison and local quadratic-prediction tests.

## 10. Generalized posterior

Define

\[
\pi_w(z\mid y)
\propto
\exp\{-w\mathcal{L}(z)\}\pi(z).
\]

The generalized-posterior energy is

\[
U(z)
=
w\mathcal{L}(z)-\log\pi(z).
\]

With \(z\sim\mathcal N(0,I)\),

\[
\nabla_z^2U(z)
=
w\nabla_z^2\mathcal L(z)+I.
\]

The GGN approximation is

\[
G_{\mathrm{post}}(z)
=
wG(z)+I.
\]

## 11. Prior-relative eigensystem

With a standard-normal prior,

\[
wG(\hat z)v_k=\lambda_kv_k.
\]

Interpretation:

- \(\lambda_k\gg1\): local data curvature dominates prior precision;
- \(\lambda_k\ll1\): local posterior precision is predominantly prior-supplied;
- \(\lambda_k\approx1\): data and prior contribute on similar scales.

Possible descriptive statistic:

\[
d_{\mathrm{data}}
=
\#\{k:\lambda_k>1\}.
\]

This is a convention called the **data-dominant local dimension**, not a universal identifiability rank.

## 12. Population MMD specialization

Let

\[
\mu_z
=
\mathbb{E}_\xi[\phi(X(z,\xi))]
\in\mathcal H.
\]

Define

\[
\mathcal L_{\mathrm{MMD}}(z)
=
\frac12\|\mu_z-\mu_y\|_{\mathcal H}^2.
\]

Then

\[
G_{\mathrm{MMD}}(z)
=
J_\mu(z)^\ast J_\mu(z),
\]

where

\[
J_\mu(z)=D_z\mu_z.
\]

## 13. Finite-feature MMD estimator

Let

\[
A_m(z)
=
D_z\psi(X(z,\xi_m))
\in\mathbb R^{D\times P}.
\]

Then

\[
J_\mu(z)
=
\mathbb E[A_m(z)].
\]

### PSD plug-in estimator

\[
\widehat J_\mu
=
\frac1M\sum_{m=1}^M A_m,
\]

\[
\boxed{
\widehat G_V
=
\widehat J_\mu^\top\widehat J_\mu.
}
\]

Properties:

- positive semidefinite;
- consistent;
- generally biased at finite \(M\).

### Cross-seed estimator

\[
\boxed{
\widehat G_U
=
\frac{1}{M(M-1)}
\sum_{m\ne n}A_m^\top A_n.
}
\]

Properties:

- unbiased under independent samples;
- not necessarily positive semidefinite at finite \(M\);
- requires careful numerical treatment.

## 14. Distinction from the old OPG

Let \(g_m\) be a per-seed scalar-loss gradient contribution.

The raw matrix

\[
F_{\mathrm{OPG}}
=
\frac1M\sum_mg_mg_m^\top
\]

is not generally equal to \(G\).

For a residual loss,

\[
g=J^\top r,
\]

so

\[
gg^\top
=
J^\top rr^\top J,
\]

not

\[
J^\top J.
\]

At an exact fit, \(g=0\) may hold while \(G\neq0\).

> **Status: rejected**
>
> The raw per-seed scalar-loss OPG is not a general GGN estimator.

## 15. Surrogate derivatives

If a surrogate tangent program yields \(\widetilde J_m\), define

\[
\widetilde G
=
\mathbb E[\widetilde J_m^\top W\widetilde J_m].
\]

This is the GGN induced by the surrogate derivative system.

It is not automatically the GGN of the original discrete stochastic simulator.

The relationship between \(\widetilde G\) and a higher-fidelity reference is an empirical question.

## 16. Local quadratic prediction

At a stationary point \(\hat z\),

\[
\mathcal L(\hat z+\delta)-\mathcal L(\hat z)
\approx
\frac12\delta^\top G(\hat z)\delta.
\]

Along eigenvector \(v_k\),

\[
\mathcal L(\hat z+\alpha v_k)-\mathcal L(\hat z)
\approx
\frac12\alpha^2\lambda_k.
\]

The empirical validity radius is the largest \(|\alpha|\) for which this approximation satisfies a prespecified error tolerance.

## 17. Directional profiled objective

For direction \(v\),

\[
\mathcal P_v(a)
=
\min_{z:\,v^\top(z-\hat z)=a}
\mathcal L(z).
\]

For posterior geometry,

\[
\mathcal P_v^U(a)
=
\min_{z:\,v^\top(z-\hat z)=a}
U(z).
\]

These profiles test whether nuisance directions can compensate for movement along \(v\).

## 18. Known limitations

- The geometry is local.
- It depends on the calibrated representation and loss.
- It depends on the prior metric.
- It may fail to approximate the Hessian when the residual-curvature term is large.
- A fixed local eigenspace may not describe a curved or multimodal posterior.
- Surrogate derivatives may distort eigenvalues and eigenspaces.
- A single observed trajectory may not identify a full output distribution.
- Local weak information does not by itself establish structural or global non-identifiability.
