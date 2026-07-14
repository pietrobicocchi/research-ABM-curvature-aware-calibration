---
title: Figure Ledger
status: active
last_updated: 2026-07-14
---

# Figure Ledger

Every figure must answer one scientific question and support a registered claim.
Figures are ordered by the **reader's belief journey**, not by experiment
chronology: each load-bearing figure owns exactly one gate a skeptical reader must
cross.

> The reader's five gates:
> 1. the matrix is the right object and is computed correctly;
> 2. it is the local loss/posterior geometry — and we are honest about when it isn't;
> 3. it reveals structure we couldn't otherwise see, on a real model;
> 4. that structure changes an answer we care about;
> 5. it survives real ABM messiness.

## Status values

proposed · candidate · approved · superseded · rejected.

## Overview (belief order)

| ID | Figure | Gate / job | Experiment(s) | Data status | Status |
|---|---|---|---|---|---|
| FIG-01 | Concept schematic | Orient the whole idea | — | design only | proposed |
| FIG-02 | The ellipse is the valley | 1 · correct & visible | smooth-SIR slice + affine | rendered | candidate |
| FIG-03 | Where it breaks (H=G+R, validity radius) | 2 · honest scope | EXP-002 (+EXP-004) | exists | candidate |
| FIG-04 | Real-ABM validation & horizon boundary (Brock–Hommes) | 2 · sound in a real ABM + scope limit | EXP-004/004b | exists | candidate |
| FIG-05 | Posterior overlay (SIR) | 2→3 bridge · it is the real posterior | EXP-005 | exists | candidate |
| FIG-06 | Reading the geometry (spectrum + participation) | 3 · how to use it | EXP-005/006 | exists; heatmap new render | candidate |
| FIG-07 | The payoff — policy underdetermination | 4 · **the** figure | EXP-006 | exists | candidate |
| FIG-08 | Observation design | 4 · actionable | EXP-007 | exists | candidate |
| FIG-09 | Calibration-path stability | offline-snapshot contrast | EXP-010 | exists | candidate |
| FIG-10 | Robustness (stochastic/discrete/surrogate) | 5 · survives messiness | EXP-008 | exists | candidate |

**The two hero figures everything else serves: FIG-02 (the ellipse is the valley)
and FIG-07 (the payoff).** If those two are not crisp, nothing else matters.

**Only two figures need new computation** — FIG-02's 2D loss grid (trivial, smooth
SIR) and FIG-09's checkpointed-descent run (a small new experiment, not yet
specced). The other seven are re-renders of data already in hand: the science is
done, the remaining figure work is design and rendering.

### Old→new mapping (nothing lost)

| Old ID | New home |
|---|---|
| FIG-01 (affine analytic recovery) | folded into **FIG-02** as an inset + Appendix B detail |
| FIG-02 (residual curvature / validity) | **FIG-03** |
| FIG-03 (MMD estimator convergence) | **Appendix B** (technical; compressed from main text) |
| FIG-04 (OPG / empirical-Fisher, BH) | **Appendix A** (unchanged placement) |
| FIG-05 (BH GGN + horizon) | **FIG-04** |
| FIG-06 (SIR posterior) | **FIG-05** |
| FIG-07 (policy) | **FIG-07** |
| FIG-08 (observation design) | **FIG-08** |
| FIG-09 (stochastic/discrete robustness) | **FIG-10** |
| — | **FIG-01** (schematic) and **FIG-06** (participation heatmap) and **FIG-09** (path) are new |

---

## FIG-01 — Concept schematic

### Question

What is the whole idea, in one picture, before any math?

### Panels

Single schematic: parameter space → differentiable simulator → calibrated
representation `m(z)` → Jacobian `Dm` → the GGN ellipse `½δᵀGδ = c` with a stiff
axis and a sloppy axis drawn to scale, and the sloppy axis annotated "the
counterfactual lives here." Prior circle shown for the prior-relative reading.

### Main message

The GGN of the calibrated representation is a local ellipse in parameter space
whose short (sloppy) axes are the combinations the data leaves free — and a policy
counterfactual can lie along one of them.

### Claims

Sets up C02, C20 visually; no empirical claim.

### Data source

None — design/illustration only. Must be schematic, not a plot of real numbers, to
avoid implying a specific result.

### Status

Proposed.

## FIG-02 — The ellipse is the valley

### Question

Where the truth is directly visible, does the AD-computed GGN reproduce the actual
local loss geometry?

### Panels

1. A 2D parameter slice around a fit (smooth SIR, or a controlled nonlinear
   benchmark) where the true loss is brute-forced on a grid: true loss contours,
   with the GGN level set `½δᵀGδ = c` and its two eigenvectors overplotted. The
   ellipse hugs the contours.
2. Inset / small panel: affine benchmark — AD GGN equals `AᵀWA` to float64
   precision (the old FIG-01 point, demoted to an inset).

### Main message

Where a reader can see the truth directly, the GGN *is* the local loss valley —
orientation and anisotropy both. The construction is correct, shown by eye, not by
a norm.

### Design note

This figure lives or dies on being a genuine 2D overlay the eye verifies. Do not
replace it with an error-metric bar chart.

### Claims

C01 (affine inset); C02, C03 (visible local match).

### Scope

One 2D slice; a visual, not a global proof.

### Data source

Script `experiments/fig02_ellipse.py`. Run
`outputs/FIG-02/20260714T205637Z_4ee0e10/` (commit `4ee0e10`, **dirty=false**).
SIR contour panel: fresh 2D loss grid (121²) around
the EXP-005 smooth-SIR fit in the (γ, I₀) plane — the one cleanly readable plane
(every other pair is dominated by the single ultra-stiff λ₁≈2.5×10⁴ direction).
Affine panel: a fresh linear-Gaussian benchmark via `benchmarks.linear_gaussian`
(the EXP-001 construction, cond 1e2, P=6), AD vs analytic AᵀWA.

### Result

SIR (γ, I₀) block eigs [2.12×10³, 126] (anisotropy ~17, tilted valley); GGN ellipses
coincide with the true loss contours at L=0.5 and 2 and separate visibly at L=8 (the
honest onset of nonlinear/residual departure, foreshadowing FIG-03). Affine: AD GGN
matches analytic to **0** (≤ float64 ε) across the spectrum; scalar-gradient outer
product **≡ 0** at the exact fit while the GGN is full-rank.

### Design note (kept)

Genuine 2D overlay the eye verifies — not an error-metric bar chart. The affine
panel is an eigenvalue overlay (analytic ○ vs AD ×), not bars, because the errors
are machine-zero and would vanish on a log bar axis.

### Status

Candidate.

## FIG-03 — Where it breaks

### Question

Over what radius does the GGN quadratic predict the true loss change, and how does
residual curvature spoil it?

### Panels

1. Along a stiff and a sloppy eigenvector: true `ΔL(α)` vs the GGN quadratic
   `½α²λ` — agreement over a radius near a good fit.
2. `H`, `G`, `R = H−G` near an exact fit vs. a case with non-negligible residual
   curvature: a constant small-α bias appears in the latter.
3. Validity radius as valley curvature grows; directional curvature bias
   `e_curv(v)`.

### Main message

Two distinct failure modes — residual-curvature bias (nonzero as the perturbation
shrinks) and higher-order nonlinear error (grows with distance) — bound where the
local quadratic can be trusted. The method states its own limits.

### Claims

C03, C09 (conditional).

### Scope

Controlled nonlinear residual benchmarks (float64).

### Data source

EXP-002 · `4118f4e` (review `0d1eb4b`); real-ABM corroboration EXP-004 · `6ff8031`.

### Status

Candidate.

## FIG-04 — Real-ABM validation and the horizon boundary (Brock–Hommes)

### Question

In an actual, nonlinear, chaotic-capable ABM, does the GGN track the exact Hessian
— and where does pathwise-differentiated local geometry stop being meaningful?

### Panels

1. `H`, `G`, `R = H−G` at and away from the fit (δ ∈ {0, 0.3, 0.8}): `H=G` at the
   fit to `10⁻¹⁶`, `R/‖H‖` growing off-fit.
2. Predictive validity: GGN quadratic vs actual `ΔL(α)` along the stiff direction;
   tiny validity radius (≈0.003 at β=50, →0 at β=80).
3. Horizon sweep: feature-Jacobian norm and `λ₁` vs `grad_horizon` (explosion to
   `λ₁≈1.5×10⁸` at β=80); leading-eigenvector angle vs full horizon.
4. Representation control: the same explosion, larger, under a robust-summary
   representation — the effect is intrinsic to `∂X/∂θ`, not the summary choice.

### Main message

The GGN equals the exact Hessian at the fit in a real ABM and inherits the same
residual-curvature bias off-fit. In the chaotic regime the exact full-horizon
geometry is dominated by intrinsic pathwise sensitivity and depends materially on
the differentiation horizon — an honest boundary of the method, shown
representation-independent.

### Claims

C03, C09 (extended to a real ABM). Horizon/chaos dependence recorded as a scope
limitation, not a registered claim.

### Scope

Brock–Hommes, R=1.01, σ=0.04, β ∈ {50, 80}, finite-RFF mean embedding, float64.

### Data source

EXP-004 · `6ff8031` (+ EXP-004b run); reviews `docs/experiments/EXP-004.md`,
`EXP-004b.md`.

### Status

Candidate. **Kept deliberately compact** — validation + boundary, not co-equal with
the SIR chapters.

## FIG-05 — Posterior overlay (SIR)

### Question

Does the prior-relative GGN match a true, independently computed local posterior?

### Panels

1. GGN-Laplace covariance ellipses overplotted on reference-posterior contours
   (importance sampling) in 2D marginals; leading axes agree to 0.68°/1.0°,
   covariance relative error 25%.
2. Profiled generalized-posterior energy vs the GGN quadratic `½α²(λ+1)` along a
   stiff and a sloppy direction; match <0.5%.
3. Per-parameter marginal-width agreement (≤5% for 4/5; t_lock 20%, mild
   non-Gaussianity).

### Main message

On smooth SIR the local GGN is not a proxy for the posterior geometry — it *is* the
posterior's local geometry: orientation to ~1° and profiled energy to <0.5%. The
positive inferential result.

### Design note

The persuasion is the visible ellipse-on-contours overlay. Keep it a genuine 2D
overlay.

### Claims

C10, C11 (conditional).

### Scope

Smooth deterministic mean-field SIR, near-Gaussian posterior, float64.

### Data source

EXP-005 · `7ba9447`; review `docs/experiments/EXP-005.md`.

### Status

Candidate.

## FIG-06 — Reading the geometry

### Question

How does a practitioner read which parameter combinations the data constrains?

### Panels

1. Prior-relative eigenvalue spectrum on a log axis with the λ=1 prior line marked;
   the sloppy tail crosses below it — `d_data` (data-dominant local dimension).
   Example spectrum `[2.5×10⁴, 135, 2.78, 5.3×10⁻³, 8.9×10⁻⁶]`, `d_data=3`.
2. **Eigenvector-participation heatmap**: |components| of each eigenvector across
   the physical parameters — showing, e.g., that the sloppiest direction is f_lock
   and t_lock trading off.

### Main message

The spectrum says *how many* combinations the data locally informs; the
participation heatmap says *which parameters* form each stiff or sloppy
combination. This is the figure that turns eigenvectors from abstract to concrete.

### Claims

C10 (spectrum interpretation), supports the reading used in C12/C13.

### Scope

Smooth deterministic SIR, this observation design and θ*.

### Data source

EXP-005 · `7ba9447` (spectrum); EXP-006 · `91fe277` (loadings). Heatmap is a **new
render** of existing run data.

### Status

Candidate.

## FIG-07 — The payoff: policy underdetermination

### Question

Does a locally weak (sloppy) GGN direction leave a policy-relevant quantity
undetermined even at an excellent fit?

### Panels

1. Profiled fit cost `P(q) = min_{z:Q(z)=q} L(z)` vs policy value `Q`, for observed
   functionals (peak, total cases — nearly vertical, pinned to ±0.2–0.3%) and the
   counterfactual intervention value (nearly flat: 0→2180 cases within a 2-nat
   budget).
2. Loading of the achievable-policy direction onto the prior-relative eigenvectors
   — it aligns with the sloppiest direction (λ≈8.9×10⁻⁶, f_lock).

### Main message

The data pins the observed epidemic curve but *not* the counterfactual value of the
intervention; that freedom is exactly the sloppiest GGN direction. The diagnostic
separates data-determined from data-underdetermined policy conclusions. **The
central figure of the paper.**

### Claims

C12 (conditional).

### Scope

Smooth deterministic SIR, this observation design and θ*, float64.

### Data source

EXP-006 · `91fe277`; review `docs/experiments/EXP-006.md`.

### Status

Candidate.

## FIG-08 — Observation design

### Question

Which added observation recovers information in the weak direction and collapses the
policy uncertainty?

### Panels

1. Prior-relative spectra for three designs (incidence; +prevalence; +compliance);
   smallest λ 8.9×10⁻⁶ → 1.3×10⁻⁵ → **10**; d_data 3 → 3 → **5**.
2. In-fit intervention-value range per design: [0, 2181] → [0, 1821] →
   **[69, 108]** (57× collapse only under compliance).

### Main message

Only the observation that measures what incidence is *blind to* — transmission
across the lockdown transition — recovers the weak direction; redundant data does
not. The geometry prescribes *which* observation to add.

### Claims

C13 (conditional).

### Scope

Smooth deterministic SIR, this θ* and noise scales (compliance σ=0.02), float64.

### Data source

EXP-007 · `4adb5c6`; review `docs/experiments/EXP-007.md`.

### Status

Candidate. **May merge with FIG-07** into one two-panel "scientific consequence"
figure.

## FIG-09 — Calibration-path stability

### Question

Is the informative geometry available *during* calibration, or only as an offline
snapshot at the optimum?

### Panels

1. Loss vs iteration (context strip).
2. Prior-relative eigenvalue spectrum at checkpoints along the descent (λ₁…λ_P vs
   iteration) — the spread emerges and settles.
3. Principal angle between the current leading-k subspace and its final at-fit
   value vs iteration — the geometry "locks in" before exact convergence.
4. (optional) `d_data` vs iteration.

### Main message

The prior-relative geometry is a stable feature of the *whole calibration basin*,
not just the optimum: the data-dominant dimension is pinned at 3 and the leading-3
stiff eigenspace stays within ~10° of its converged orientation from the first
displaced iterate (loss 5051× the converged value), tightening to <0.1° at the fit.
So the diagnostic is usable throughout calibration, from the gradients already being
computed — unlike finite-difference and surrogate-network Hessians, which are
offline single snapshots at the fit. Directly closes the offline-snapshot contrast
set up in Background.

### Result (EXP-010)

d_data = 3 constant along the path; leading-2 angle-to-final 7.9° at step 0 →
<5° by loss 926× final → <0.1° at the fit; the three data-dominant eigenvalues
stable, only the deep-sloppy tail moves. A displacement scan corroborates stability
out to a 4-prior-std displacement (loss ~10⁴× final). Honest note: there is no
"lock-in event" — the finding is path-wide stability; the endpoint reference is
itself displaced in the sloppy directions (data-underdetermined), consistent with
the policy chapter.

### Claims

Supports C20 (usability). Path-wide stability could be promoted to a small dedicated
conditional claim — deferred to the gatekeeper.

### Scope

One model (smooth SIR), one descent, one z_init; a supporting figure, not a pillar.
A single trajectory does not establish path-independence.

### Data source

EXP-010 · `docs/experiments/EXP-010.md`. Run
`outputs/EXP-010/20260714T205630Z_4ee0e10/` (commit `4ee0e10`, **dirty=false**).
Script `experiments/exp010_calibration_path.py`.

### Status

Candidate.

## FIG-10 — Robustness under stochasticity, discreteness, and surrogate gradients

### Question

On a genuinely stochastic, discrete, network-structured SIR under an estimated MMD
geometry, is the leading GGN eigenspace robust to the surrogate gradient, and does
truncated differentiation destroy it?

### Panels

1. Cross-surrogate principal angles (Gumbel-sigmoid vs straight-through,
   leading-1/2/3) at two operating points; Gumbel finite-difference validation
   (median rel-Jacobian error 3.0×10⁻⁵ / 3.4×10⁻⁵).
2. Eigenvalue-scale ratio (straight-through / Gumbel) across leading directions —
   direction surrogate-invariant, scale not (0.22–0.48×).
3. Horizon sweep: leading-eigenspace angle vs full-horizon reference and
   top-eigenvalue ratio at `grad_horizon` ∈ {40, 20, 10, 5}, both operating points.

### Main message

The stiff/sloppy directions are a property of the geometry, not of the surrogate or
the regime — stable across two surrogate constructions and two operating points,
with the smooth surrogate FD-validated. Horizon truncation rotates the leading
eigenspace ~80° and collapses its curvature to 1–15% of full — the Brock–Hommes
horizon boundary reappearing in a stochastic, discrete, non-chaotic setting.

### Claims

C14, C15 (conditional).

### Scope

Two operating points (θ*, graph, graph seed), one model family (network-SIR),
D=128 RFF / median bandwidth, M=128, Gumbel-vs-straight-through only.

### Data source

EXP-008 · `98a8a19` (two operating points; predecessor `62ee88e`); runs
`outputs/EXP-008/20260714T154440Z_98a8a19/`,
`outputs/EXP-008/20260714T152944Z_62ee88e/`; review `docs/experiments/EXP-008.md`.

### Status

Candidate.

---

## Appendix figures

- **App A — gradient-outer-product / empirical-Fisher clarification (old FIG-04).**
  A general methodological warning: the per-seed gradient outer product is a
  different, residual-weighted object from the GGN and collapses at an exact fit;
  demonstrated on Brock–Hommes (`F_OPG ≈ C_g`, nearly orthogonal to the true GGN,
  86° at β=80). EXP-000 · `02d6bd6`. Not a motivating pillar.
- **App B — MMD estimator convergence (old FIG-03).** Plug-in `O(1/M)` bias,
  cross-seed unbiased-but-indefinite, leading-subspace angle vs M, finite-RFF →
  exact-RBF. EXP-003 · `ce45c43`.
- **App B — affine analytic recovery detail** (the full old FIG-01 panels, if the
  FIG-02 inset needs backing).

---

## Lean main-text set

In belief order: **FIG-01** (schematic) → **FIG-02** (ellipse is the valley) →
**FIG-03** (where it breaks) → **FIG-04** (real-ABM validation + horizon boundary,
compact) → **FIG-05** (posterior overlay) → **FIG-06** (reading the geometry) →
**FIG-07** (the payoff) → **FIG-08** (observation design) → **FIG-09** (calibration-
path stability) → **FIG-10** (robustness).

Tightening options: **FIG-07 + FIG-08** may merge into one two-panel "scientific
consequence" figure; **FIG-05 + FIG-06** may merge into one "SIR local geometry"
figure. Either merge brings the main text to ~8 figures. Appendix carries the OPG
clarification, MMD estimator detail, and affine recovery detail.

## Figure approval checklist

A figure becomes approved only if:

- it answers one scientific question;
- its claim is supported;
- its experiment is reviewed;
- commit and run provenance are known;
- uncertainty is represented appropriately;
- the caption states the scope;
- it does not imply a broader claim than the evidence supports.

All ten main figures are currently **candidate or proposed**; **none is approved**.
Figure approval is the one remaining gate before manuscript prose begins
(`WRITING_BRIEF.md` → manuscript-writing threshold). FIG-02 and FIG-09 additionally
need their new renders/run before they can be approved.

## New figure template

### FIG-XXX — Title

**Question:**
**Experiment:**
**Main message:**
**Data source:**
**Code commit:**
**Run ID:**
**Panels:**
**Claim supported:**
**Limitations:**
**Status:**
