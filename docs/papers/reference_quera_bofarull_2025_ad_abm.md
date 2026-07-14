---
name: reference-quera-bofarull-2025-ad-abm
description: Quera-Bofarull et al. 2025 (arXiv:2509.03303) — AD through discrete/stochastic ABMs enables VI calibration and one-shot first-order sensitivity; the canonical diff-ABM reference our project builds the OPG eigenstructure view on top of.
metadata:
  type: reference
---

## Citation

Arnau Quera-Bofarull, Nicholas Bishop, Joel Dyer, Daniel Jarne Ornia, Anisoara Calinescu,
Doyne Farmer, Michael Wooldridge. **"Automatic Differentiation of Agent-Based Models."**
arXiv:2509.03303 (2025). Submitted 3 Sep 2025, revised 18 Nov 2025.

## Core contribution

Makes agent-based simulators end-to-end differentiable via automatic differentiation (AD),
so that gradients of simulator outputs w.r.t. parameters are available "for free" from a
single forward pass. Two downstream uses are demonstrated:

1. **Calibration** — gradients drive generalised variational inference (VI) over ABM
   parameters, replacing likelihood-free / simulation-based-inference approaches that need
   many expensive forward simulations per calibration step.
2. **Sensitivity analysis** — the gradient ∇_θΦ(g(ε; θ)) of an emergent-property functional
   Φ w.r.t. parameters θ is a first-order local sensitivity, obtained in one simulation run
   instead of the many perturbed re-runs classical (finite-difference / Sobol-type) sensitivity
   analysis needs. This is the paper's "one-shot" sensitivity claim, framed as a special case
   of the same AD machinery used for calibration, not a separate method.

## How they differentiate through discrete decisions and stochastic transitions

ABMs are full of non-differentiable control flow (if/else branching on agent state) and
discrete/stochastic sampling (categorical choices, Bernoulli transitions). The paper handles
these with distinct techniques for each source of non-differentiability:

- **Discrete control flow (if/else, thresholds):** replace hard branches with smooth
  surrogate masks — sigmoid, Gaussian CDF, or piecewise-linear relaxations of the branch
  condition — and, where there are more than two branches, a softmax relaxation of arg-max
  over branches. In practice both branches are evaluated on the forward pass and blended by
  the smoothed indicator, so gradients flow through the mask weight instead of being killed
  by a hard conditional.
- **Discrete random sampling**, three alternative estimators, compared empirically rather
  than a single default:
  - **Straight-Through (ST) estimator** — forward pass uses the true discrete sample, but the
    backward pass replaces the (zero) gradient of the sampling op with the gradient of the
    identity/expectation, i.e. gradients pass "straight through" as if the discretisation were
    transparent. Biased but cheap; works best when the downstream function is close to linear
    in the sample (this is where they use it for parts of the Axtell firm model).
  - **Gumbel-Softmax (GS) estimator** — reparameterises categorical sampling via the
    Gumbel-Max trick, then relaxes the arg-max with a temperature-controlled softmax. Lower
    temperature τ tracks the true discrete distribution more closely (lower bias) at the cost
    of higher-variance gradients; τ is a tunable bias–variance knob.
  - **Stochastic derivatives via Smoothed Perturbation Analysis (SPA)** — an unbiased pathwise
    gradient estimator built from "stochastic triples" (δ, w, Y) that track infinitesimal
    smooth parameter perturbations together with the probability-weighted contribution of
    discrete jumps (e.g. an S→I transition flipping on/off). This is the most accurate but
    most complex of the three, and is what they use for the SIR state transitions.
  - The three estimators are validated against finite-difference gradients on all three ABMs;
    no single estimator dominates — the paper reports that SPA gives the lowest bias for SIR
    across network densities, GS gives a tunable bias/variance trade-off, and ST is adequate
    where the model is close to mean-field / near-linear (Axtell).

## Variational-inference calibration

Calibration uses **generalised VI**: a normalizing flow parameterises an approximate
posterior over θ, bijector transforms handle constrained parameter domains, and the
AD-supplied gradients (via a "hybrid AD strategy" combining the above discrete-gradient
techniques with standard reverse-mode AD through the smooth parts of the simulator) train the
flow. This is reported as more robust to model misspecification than standard Bayesian
inference and yields calibration with substantial performance improvements and computational
savings relative to non-gradient (e.g. simulation-based-inference / ABC-style) calibration
baselines, because each gradient step reuses information a zeroth-order method would need
many extra simulator calls to obtain.

## Three demonstration ABMs

- **Axtell's model of firms (AMOF):** agents choose effort (continuous) and which firm to
  join (discrete arg-max over firms) to maximise utility. Exercises the discrete-choice /
  arg-max relaxation machinery; ST estimator adequate here because the model is close to
  mean-field.
- **Sugarscape:** spatial grid model; agents move to the grid cell with max sugar within a
  vision radius (discrete arg-max over cells) and die when sugar is depleted (discrete
  life/death threshold). Exercises spatial discrete optimisation and threshold-triggered
  discontinuities.
- **SIR epidemiological model (network-based):** agents transition between discrete
  {S, I, R} states via stochastic events on a contact network, with discrete policy
  interventions (quarantine, social distancing) creating additional discontinuities.
  Exercises stochastic discrete-transition AD (SPA) and policy-discontinuity smoothing
  together. **This is the same network-SIR family our project's "Secondary" benchmark model
  is drawn from** (see `project_overview.md`) — our P=5 network-SIR parameterisation
  (`beta, gamma, I0_frac, t_lock_norm, f_lock`) reuses this paper's surrogate-gradient
  machinery (Gumbel-Sigmoid / smoothed-step surrogates) and its gradient-horizon truncation
  concerns directly.

Headline results: gradients from all three estimator families match finite-difference
gradients closely on all three ABMs (bias depends on estimator/model as above); VI
calibration driven by these gradients gives "substantial performance improvements and
computational savings" over non-gradient calibration; sensitivity analysis becomes a
single-run, one-shot computation instead of requiring many perturbed re-simulations, and the
paper additionally shows *temporal* sensitivity profiles — how ∂Φ/∂θ_i varies across
simulation time steps, something a scalar finite-difference sensitivity cannot show without
re-running at every time point.

## Verified §-level implementation map (arXiv:2509.03303v1 HTML, checked 2026-07-14 — for EXP-008)

Pulled directly from the paper to anchor the discrete-SIR AD consistency check:

- **§2.3 SIR is discrete per-agent / network-based:** each agent has state
  `s_i(t) ∈ {S, I, R}` on a contact network — genuinely discrete per-agent, **not**
  a relaxed continuous compartment. Policies (quarantine, social distancing) are
  time-window interventions `t ∈ {Q_start,…,Q_end}`.
- **§3.3 discrete control-flow smoothing:** parameter-dependent predicates are
  relaxed with a **Gaussian CDF, sigmoid, or piecewise-linear** surrogate (both
  branches evaluated, blended by the smoothed indicator).
- **§3.4 discrete-randomness estimators, compared not defaulted:** §3.4.1
  Straight-Through (ST); §3.4.2 Gumbel-Softmax (GS), temperature τ (Fig. 5 uses
  τ=0.1/0.5/1.0 illustratively); §3.4.3 stochastic derivatives / StochasticAD
  (unbiased, the direction they advocate for lower-variance discrete gradients).
- **§5.3.1 estimator comparison** is validated against a **finite-difference**
  gradient baseline; SPA/stochastic-derivative is the unbiased/low-bias reference
  for SIR.

**Consistency of our repo (`models/network_sir.py`, `models/surrogates.py`):**
- We implement **two** of their three estimators — `gumbel_sigmoid` (their GS,
  τ=0.5 default) and `straight_through_bernoulli` (their ST, `custom_jvp` passing
  `p_dot`). We do **not** implement StochasticAD / stochastic-derivatives (§3.4.3);
  that is Julia (`StochasticAD.jl`) in their work and has no JAX equivalent here.
- Control flow matches §3.3: we use **sigmoid** relaxations (`k_sig=20` for the
  lockdown `beta_eff`, `k_init=50` for infection seeding).
- Our per-node `(S,I,R)` state is continuous-relaxed under Gumbel but collapses to
  one-hot per node under ST, so with ST each node behaves as a discrete agent —
  reasonably aligned with their per-agent §2.3 model; under Gumbel it is a relaxed
  version. The smoothing sharpness (`k_sig`, `k_init`, `gumbel_tau`) is the
  discrete→smooth bias knob and must be a controlled variable in EXP-008.
- **Our novel angle, not theirs:** the `grad_horizon` truncation / gradient-variance
  -over-length concern (RQ4, DEC-011) is absent from their SIR treatment — it is our
  contribution, not something to reconcile against them.

## Relation to our project

Quera-Bofarull et al. 2025 (2509.03303) is our project's **canonical diff-ABM reference**: it
establishes that AD through discrete/stochastic ABMs is feasible and gives two uses of the
resulting gradients — VI calibration and **first-order** sensitivity, i.e. the per-parameter
Jacobian ∇_θΦ. Their §5.4 is exactly this first-order sensitivity analysis; §8.3 explicitly
flags second-order (curvature) analysis as future work.

Our project starts from that same gradient supply chain — per-seed simulator gradients
g_m = ∂L_m/∂θ under an MMD calibration loss — but asks a different question of them. Instead
of reading off first-order per-parameter sensitivities (their §5.4), we form the
**second-moment / OPG matrix** F̂ = (1/M) Σ_m g_m g_mᵀ (the outer-product-of-gradients matrix
— see `framing_kunstner_opg_not_fisher.md` for the mandatory OPG-not-Fisher terminology rule) and
study its **eigenstructure**. A per-parameter Jacobian can only ever say "parameter i matters
this much in isolation"; it cannot see that two parameters move the loss almost identically
in combination (a sloppy direction) while some *combination* of them is tightly constrained
(a stiff direction). F̂'s eigenvectors expose exactly those combinations, and its eigenvalue
spectrum gives an effective dimension / sloppiness diagnosis that first-order sensitivities
structurally cannot surface. Where their sensitivity view is "how much does output i move per
parameter," ours is "what is the actual identifiable rank of the parameter space, and along
which directions." We show (paper_story_arc.md §4.2, `honest_appraisal.md`) an explicit gap
in dynamic range (~10 000×) between what F̂'s worst-conditioned direction reveals and what
their per-parameter Jacobian would suggest.

The network-SIR overlap is direct and load-bearing, not incidental: our secondary benchmark
model is the same network-SIR family used here, reusing its surrogate-gradient techniques
(Gumbel-Sigmoid / smoothed-step) and inheriting the same gradient-horizon-truncation
sensitivity that motivated our Phase 1 killswitch experiment
(`phase1_horizon_bias_killswitch.md`).

**Not to be confused with:** Quera-Bofarull et al. 2023 (AAMAS), *"Don't Simulate Twice:
One-Shot Sensitivity Analyses via Automatic Differentiation"* — an earlier, narrower paper by
an overlapping author set that introduced the one-shot Jacobian-sensitivity idea on its own
(no VI calibration, no discrete-randomness estimator comparison, no three-ABM demonstration
suite). 2509.03303 is the later, broader paper: it subsumes the one-shot-sensitivity idea as
one component (§5.4) inside a larger contribution centred on VI calibration across three ABMs
and a systematic comparison of discrete-gradient estimators (ST / GS / SPA). When our own
`literature_positioning.md` cites "Quera-Bofarull et al. 2025 §5.4 Jacobian sensitivities," it
is referring to a section *within this same 2509.03303 paper*, not a separate work.
