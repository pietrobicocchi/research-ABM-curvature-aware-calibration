---
title: Paper Architecture
status: draft
last_verified: 2026-07-14
---

# Paper Architecture

This is a claim–evidence outline, not manuscript prose. Section headers here ARE
the intended manuscript section titles — keep them free of internal project
jargon (no "keystone", "payoff", "gap-closing", no EXP-xxx/claim-ID literals).
Everything else in this file (weight tags, claim IDs, evidence pointers) is
internal scaffolding for us and must not appear in the manuscript itself; see
`WRITING_BRIEF.md` → "No internal scaffolding in prose."

## Provisional title

**Local Information Geometry of Differentiable Agent-Based Models**

Subtitle option:

**Gauss–Newton diagnostics for generalized Bayesian calibration**

## Core narrative

1. Differentiable ABMs make first derivatives of the calibrated representation
   available, and are used to answer counterfactual / policy questions.
2. A good fit does not determine a counterfactual: many parameter combinations fit
   equally well, and neither a point estimate nor a scalar gradient reveals which
   combinations the data locally constrains.
3. The correct local object is the GGN of the calibrated representation, formed
   directly from the automatic-differentiation Jacobian and read relative to the
   prior — no detour through an empirical-Fisher identification is needed.
4. Prior whitening is required for a data-information interpretation.
5. The geometry must be validated against exact Hessians, posterior references, and
   profiled objectives, in increasingly realistic settings.
6. A locally weak direction can leave a policy counterfactual undetermined, and the
   same geometry prescribes the observation that constrains it.
7. The result is a local diagnostic, not a global identifiability theorem.

*(A tempting alternative — the per-seed gradient second moment / empirical Fisher —
is a different, residual-weighted object for discrepancy losses, and collapses at
an exact fit. This is a general methodological clarification carried in an
appendix, grounded in the empirical-Fisher literature, not a motivating pillar or
an account of our own history. See Appendix A.)*

## Asymmetric weighting

Not every chapter carries equal narrative weight. This is deliberate:

| § | Title | Weight | Role |
|---|---|---|---|
| 1 | Introduction | High (narrative) | Motivation and claim, not evidence |
| 2 | Background | High | Positions the paper in real literature |
| 3 | Framework | Medium | Definitions only |
| 4 | Local Gauss–Newton geometry | High | The theory chapter |
| 5 | MMD geometry and estimation | Medium-High | The estimator chapter |
| 6 | Prior-relative local information | Medium (glue) | Connects theory to the ABM chapters |
| 7 | Brock–Hommes: exact validation and the boundary of local geometry | Medium, deliberately small | Validation + honest scope limit, not co-equal with §8–9 |
| 8 | SIR: inferential validation | High | The positive inferential result |
| 9 | Policy counterfactuals under locally weak identifiability | Highest | The scientific consequence |
| 10 | Robustness under stochasticity, discreteness, and surrogate gradients | Medium-High | Earns "stochastic agent-based" |
| 11 | Discussion | Medium | Stated limits, one closing paragraph |
| 12 | Conclusion | Low | Supported claims only, short |

Brock–Hommes (§7) is validation and an honest scope boundary — it is kept
deliberately smaller than SIR (§8–9), which carries the paper's actual argument.

---

## 1. Introduction

### Content

Differentiable ABMs expose derivatives of a calibrated representation and are used
for counterfactual / policy analysis. A good fit does not determine which parameter
combinations — and therefore which counterfactuals — the data locally constrains.
The correct local object is the generalized Gauss–Newton (GGN) matrix of the
calibrated representation. One paragraph previews the validation strategy and the
policy/observation-design consequence, without pre-empting §9's detail.

### Claims allowed

- C02, C20; forward reference to C10, C12/C13 once earned in §8–9.

### Claims forbidden

- C07, C17, C18, C19.

---

## 2. Background

### 2.1 Differentiable ABMs and distributional calibration losses

Simulator, no tractable likelihood, calibration by distributional discrepancy
(MMD). A differentiable ABM replaces non-differentiable operations with a
surrogate-gradient scheme in the backward pass; the forward pass is unchanged.
Rewritten from the earlier draft's Background §1, redirected toward direct AD
access to the Jacobian rather than an empirical-Fisher argument.

Evidence / literature: Quera-Bofarull 2023/2025, Chopra 2023, Platt 2020,
surrogate-gradient literature (Andelfinger, Jang/Maddison, Bengio, Arya).

### 2.2 Practical identifiability as curvature

Structural vs. practical identifiability; curvature as the local instrument;
eigenvectors, not coordinates, are the identifiable object; sloppy models
(hyper-ribbon geometry, gapless log-spaced spectra). Reused near-verbatim from the
earlier draft — this material was already strong and does not depend on the OPG
argument.

Evidence / literature: Transtrum & Sethna, Gutenkunst et al., Machta et al.,
Naumann 2024, Raue et al., Eisenberg & Hayashi.

### 2.3 Curvature for ABMs: state of the art

Naumann 2024 (finite-difference Hessians, O(P²) resimulations, offline snapshot);
Scarrold 2024 (surrogate-network gradients, removes stochastic differencing noise
but interposes a second, learned model). Sharpened contrast: both reconstruct the
geometry offline and require either O(P²) resimulation or a separate learned
surrogate; the present method reads the GGN from the AD Jacobian the calibration
already computes, with neither cost.

Evidence / literature: Naumann 2024, Scarrold 2024.

### 2.4 The generalized Gauss–Newton matrix and generalized Bayesian calibration

The paper's theoretical anchor, elevated from a buried subsection in the earlier
draft to a full pillar of Background. For a residual loss, the exact Hessian
decomposes as `H = G + R`, `G` positive semidefinite; AD gives the residual
Jacobian directly, so `G` is formed without any empirical-Fisher identification.
Generalized Bayesian calibration situates the MMD loss inside a proper posterior
(Bissiri et al., Knoblauch et al., Chérief-Araouassian et al.), giving the local
geometry an inferential reading (§6, §8). One clarifying remark, not a subsection:
a natural-looking alternative construction (the per-seed gradient outer product)
is a different, residual-weighted object — detailed in Appendix A, not here.

Evidence / literature: Botev, Ollivier & Martens (GGN); Schraudolph; Bissiri et al.,
Knoblauch et al. (generalized Bayes); Gretton et al. (MMD); Kunstner et al.
(empirical Fisher, cited for the appendix pointer only).

---

## 3. Framework: differentiable generalized calibration

### Definitions

`z, T, X, ξ, ψ, m, L, π_w`; stochastic simulator; calibrated representation;
generalized posterior; exact vs. surrogate derivatives; prior-scaled coordinates.

### Evidence

`docs/MATH.md` §1–4, §10.

---

## 4. Local Gauss–Newton geometry

### Main derivation

`H = G + R`, `G = Dm* W Dm`. Positive semidefiniteness; directional interpretation
(`v^T G v` = squared first-order change in the calibrated representation);
limitations (locality, representation/loss dependence, residual curvature,
parameterization) stated immediately, not deferred to a later Discussion.

### Validation, integrated here rather than deferred

- Exact recovery on an affine finite-dimensional benchmark (float64).
- Nonlinear residual benchmarks: `H=G+R` numerically exact; two distinct failure
  modes — residual-curvature bias (nonzero as the perturbation shrinks) and
  higher-order nonlinear error (grows with distance) — with a directional bias
  measure and an empirical validity radius.

### Claims

C01, C02, C03 (conditional), C09 (conditional).

### Evidence

`docs/MATH.md` §5–9, §16; controlled affine and nonlinear benchmarks. FIG-02
(affine inset + the ellipse-is-the-valley overlay), FIG-03 (where it breaks).

---

## 5. MMD geometry and estimation

### Definition

`L_MMD(z) = ½‖μ_z − μ_y‖²_H`; `G_MMD = Jη* Jη`.

### Estimators

PSD plug-in (consistent, `O(1/M)` positive bias) and cross-seed (unbiased within
Monte Carlo resolution, may be indefinite at small `M`) — distinct finite-sample
behavior, demonstrated on a controlled Gaussian location-scale simulator with
frozen random Fourier features.

### Claims

C04, C05, C06 (all conditional, controlled Gaussian/RFF scope).

### Evidence

`docs/MATH.md` §12–13; controlled RFF estimator study. Estimator-convergence detail
in Appendix B (compressed from main text — technical, not a reader's belief gate).

---

## 6. Prior-relative local information

### Main object

Generalized eigenproblem `wGv_k = λ_k P_π v_k`; standard-normal prior reduces this
to a standard eigenproblem. Interpretation: data-dominant, prior-dominant,
comparable local precision; the data-dominant local dimension `d_data` as a
descriptive statistic, not a universal identifiability rank.

### Evidence

`docs/MATH.md` §11.

---

## 7. Brock–Hommes: exact validation and the boundary of local geometry

**Deliberately compact.** Validates the construction on a real, nonlinear,
chaotic-capable ABM, and states — as a scope boundary rather than a limitation to
apologize for — where pathwise-differentiated local geometry stops being
meaningful.

### Content

In the correct large-β regime, the GGN equals the exact Hessian at the fit to
numerical precision; away from the fit, residual curvature reintroduces `R` and
the local quadratic is predictive only over a small radius that shrinks with β. In
the chaotic regime, the exact full-horizon geometry is dominated by intrinsic
pathwise (Lyapunov) sensitivity and depends materially on the differentiation
horizon — shown to be a property of any pathwise-differentiated observable, not an
artifact of a particular summary choice.

### Claims

C03, C09 (extended to a real ABM). The horizon/chaos finding is stated as a scope
limitation of exact pathwise geometry in chaotic regimes, not a registered claim.

### Evidence

Smooth and chaotic Brock–Hommes runs; horizon sweep; representation control. FIG-04.

---

## 8. SIR: inferential validation

### Content

On smooth, deterministic mean-field SIR, the prior-relative GGN is compared
directly against a reference local posterior and against the profiled generalized-
posterior energy. This is the paper's positive inferential result: the local
geometry is not merely self-consistent, it agrees with an independently computed
posterior reference.

### Claims

C10, C11 (conditional; smooth deterministic SIR, near-Gaussian posterior scope).

### Evidence

Reference-posterior comparison (importance sampling); directional profiles. FIG-05
(posterior overlay), FIG-06 (spectrum + eigenvector-participation heatmap).

---

## 9. Policy counterfactuals under locally weak identifiability

**Highest weight — the scientific consequence.** Directly-observed epidemic
functionals (peak incidence, total cases) are tightly pinned by the fit. The
counterfactual value of an intervention — cases averted by a lockdown — is not:
it spans a policy-decisive range at an essentially unchanged fit, and that freedom
loads onto the single sloppiest direction of the prior-relative GGN. A targeted
additional observation (measuring transmission across the lockdown transition)
lifts that direction by several orders of magnitude and collapses the policy range
by more than an order of magnitude; a plausible but redundant observation does not.
The diagnostic identifies not only that a policy conclusion is undetermined, but
which observation would determine it.

### Claims

C12, C13 (conditional; this θ*, this observation design).

### Evidence

Profiled fit-cost vs. policy-value curves; prior-relative spectra under three
observation designs; in-fit intervention-value ranges. FIG-07, FIG-08 (candidates
for merging into one two-panel figure).

---

## 10. Robustness under stochasticity, discreteness, and surrogate gradients

### Content

The same diagnose-then-design loop is run on a genuinely stochastic, discrete,
network-structured SIR calibrated under an estimated MMD geometry with a
surrogate gradient (the discrete transmission/recovery events are not
differentiable as written). Two independent operating points, differing in regime
(near-threshold vs. supercritical), graph density, and graph realization, are
used to separate a property of the geometry from a property of one configuration.
The leading eigenspace is shown stable across two different surrogate-gradient
constructions and finite-difference-validated; truncating the differentiation
horizon rotates the leading eigenspace substantially and collapses its curvature,
directly instantiating the §7 horizon boundary in a genuinely stochastic setting.

### Claims

C14, C15 (conditional; two operating points, one model family, RFF/MMD
representation).

### Evidence

Cross-surrogate principal angles and finite-difference validation at two operating
points; horizon sweep. FIG-10 (robustness); FIG-09 (calibration-path stability,
the offline-snapshot contrast) sits alongside this chapter and Background.

---

## 11. Discussion

### Required topics

1. Local versus global conclusions.
2. Sensitivity versus practical identifiability.
3. Data geometry versus prior geometry.
4. Dependence on observations, summaries, kernel, and loss.
5. Surrogate-gradient and truncation bias.
6. Single-trajectory limitations.
7. Model misspecification.
8. Policy implications stated at the correct level.

### Prohibited discussion style

- No broad claim that weak directions make all counterfactuals invalid.
- No claim that the method proves structural non-identifiability.
- No repeated self-summary.
- No speculative future-work catalogue.
- No narration of how the project's own understanding evolved.

### Ending

One restrained paragraph identifying the main unresolved limitation (a
meaningful chaotic-regime geometry needs a fixed horizon or a derivative-free/
ensemble construction) and the next scientific step.

---

## 12. Conclusion

Short. Only claims marked supported or supported conditionally in the Claims
Ledger.

---

## Appendices

### Appendix A — A tempting alternative construction, and why it fails

Framed as a general methodological warning for anyone differentiating an ABM
under a discrepancy loss, not as an account of our own history: the per-seed
gradient outer product is the object closest at hand once gradients exist, and it
is provably a different, residual-weighted object from the GGN (`gg^T = J^T rr^T
J`, not `J^T J`), collapsing at an exact fit while the GGN does not. Demonstrated
on Brock–Hommes: the gradient second moment equals the gradient covariance, nearly
orthogonal to the true GGN in the chaotic regime.

Evidence: real-ABM comparison of the two matrices across regimes. Appendix A figure
(Brock–Hommes gradient-outer-product vs GGN; the former FIG-04, now appendix-only).

### Appendix B — Estimation and computation

Explicit Jacobians; JVP/VJP products; matrix-free `Gv`; eigensolvers; Monte Carlo
uncertainty; subspace bootstrap.

### Appendix C — Computational cost

One paragraph. Cost of AD-based GGN estimation at the scales demonstrated;
explicitly not a claim about scaling to large ABMs.

---

# Figure plan

Numbering follows `FIGURE_LEDGER.md`, which is ordered by the reader's belief
journey (not experiment chronology). Main text:

- **FIG-01** concept schematic (§1)
- **FIG-02** the ellipse is the valley — GGN vs true loss contours, affine inset (§4)
- **FIG-03** where it breaks — H=G+R, residual bias, validity radius (§4)
- **FIG-04** real-ABM validation and the horizon boundary, Brock–Hommes (§7)
- **FIG-05** posterior overlay, SIR (§8)
- **FIG-06** reading the geometry — spectrum + eigenvector-participation heatmap (§6/§8)
- **FIG-07** the payoff — policy underdetermination (§9)
- **FIG-08** observation design (§9)
- **FIG-09** calibration-path stability — the offline-snapshot contrast (§10 / Background)
- **FIG-10** robustness under stochasticity, discreteness, surrogate gradients (§10)

Appendix: gradient-outer-product / empirical-Fisher clarification on Brock–Hommes
(App A); MMD estimator convergence detail (App B); affine analytic-recovery detail
(App B).

*The two hero figures are FIG-02 and FIG-07. FIG-07/FIG-08 may merge into one
"scientific consequence" figure and FIG-05/FIG-06 into one "SIR local geometry"
figure, bringing the main text to ~8. Only FIG-02 (2D loss grid) and FIG-09
(checkpointed descent) need new computation. Computational-scaling is an Appendix C
paragraph, not a figure. See `FIGURE_LEDGER.md` → "Lean main-text set."*

---

# Abstract gate

The abstract may be drafted only when:

- C01 is supported;
- at least one MMD estimator claim is supported;
- C10 or an equivalent inferential-validation claim is supported;
- C12 or C13 provides scientific consequence;
- the main limitations have been empirically characterized.

**Status: met.** All conditions above are satisfied (C01, C04–C06, C10/C11,
C12/C13, C14/C15). The only remaining gate before drafting is figure approval
(`FIGURE_LEDGER.md`).

# Conclusion gate

The conclusion may contain only claims marked supported or supported
conditionally in the Claims Ledger.
