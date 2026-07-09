# Network-SIR diagnostic-in-motion suite — design

**Date:** 2026-07-09
**Branch:** feat/visualization-booklets
**Status:** approved (design), pending implementation plan

## Goal

Replicate, for the **network-SIR** model, the three Brock–Hommes viz experiments that
already exist under `scripts/viz/` + `outputs/viz/`:

1. **the diagnostic in motion** (fig2) — eigenvalue trajectories + `d_eff(t)` across calibration,
2. **subspace rotation** (fig2b) — cumulative principal-angle drift `Θ_k(t)` of the identifiable subspace,
3. **falsification** (fig6, already exists for SIR) — sloppy-vs-stiff under non-MMD discrepancies.

Report everything in a structured MD log. Separately, write a `reference`-type memory
doc describing the ARNAU 2025 paper (arXiv:2509.03303), delegated to a Sonnet subagent
(user explicitly authorised Sonnet for this).

Locked decisions: **interventionist θ\*** (lockdown fires during the epidemic so the sloppy
direction is a genuine t_lock×f_lock combination), **3 R₀ regimes**, network-SIR only,
**trimmed runtime**.

## Deliverables

### 1. Reference memory doc (delegated to Sonnet, in parallel)

- Subagent reads arXiv:2509.03303 ("Automatic Differentiation of Agent-Based Models",
  Quera-Bofarull, Bishop, Dyer, Jarne Ornia, Calinescu, Farmer, Wooldridge, 2025) in full.
- Writes `docs/memory/reference_quera_bofarull_2025_ad_abm.md` (`metadata.type: reference`)
  + one index line in `docs/memory/MEMORY.md`.
- Content: core contribution (AD through ABMs → gradients → variational-inference calibration
  + first-order sensitivity analysis), the three demo ABMs (Axtell firms, Sugarscape, SIR),
  and — critically for us — the relation to our gap: **they use first-order gradient
  sensitivities / VI; we use the second-moment (OPG) eigenstructure of the same gradients**
  to expose parameter *combinations* and identifiability. Must obey the
  [[framing_kunstner_opg_not_fisher]] rule (call F̂ the OPG matrix, never "empirical Fisher").
- Note the distinction from the other Quera-Bofarull works already cited (2023 AAMAS
  "Don't Simulate Twice"; the §5.4 first-order Jacobian-sensitivity reference).

### 2. Shared SIR data loader — `scripts/viz/_sirdata.py`

Mirrors `scripts/viz/_bhdata.py` structure and caching convention.

- **3 R₀ regimes** at the interventionist θ\*. R₀ = β·⟨k⟩/γ with ⟨k⟩ = mean_degree = 6, γ = 0.10:
  - `slow`     R₀ ≈ 1.3 → β ≈ 0.022
  - `moderate` R₀ ≈ 2.5 → β ≈ 0.042  (≈ the state.md interventionist operating point)
  - `fast`     R₀ ≈ 5.0 → β ≈ 0.083
  - Shared across regimes: γ=0.10, I0_frac≈0.01, **t_lock_norm≈0.025** (lockdown fires early,
    during the epidemic, so it bends the curve), f_lock≈0.50.
- Per regime: calibrate from θ₀ = θ\* + δ, with **δ along a stiff direction** (empirically
  chosen — memories say v₁ ≈ I₀ is stiff on SIR; validate in the smoke run) so there is
  descent signal. Record per-iterate `eigvals_traj (n_iter,P)`, `eigvecs_traj (n_iter,P,P)`,
  bootstrap CI lo/hi, precision floor τ_t = REL_FLOOR·λ₁ (REL_FLOOR=1e-8), and
  `d_eff(t) = #{boot_lo_k > τ_t}`.
- Cache to `outputs/viz/_cache/sir_motion.npz` with a stale-cache guard (must contain
  `eigvecs_traj`), same idiom as `_bhdata.load()`.

**Trimmed runtime parameters** (vs BH's N=300 / n_iter=60 / N_BOOT=500):
- N = 250, mean_degree = 6, T = 200, M = 64, M_REF = 96, n_iter = 40, N_BOOT = 300.
- Network-SIR uses the Gumbel-Sigmoid surrogate (τ=0.5), fixed graph_seed.

### 3. Two new figures (paper_style idiom)

- `scripts/viz/fig7_sir_motion.py` — mirror of fig2: `log10 λ_k(t)` for P=5 eigenvalues with
  bootstrap bands + τ_t accent line, `d_eff(t)` step panel below, 3 R₀ columns, shared axes,
  single-hue rank ramp, direct right-margin line labels.
- `scripts/viz/fig7b_sir_subspace_rotation.py` — mirror of fig2b: `Θ_k(t)` largest principal
  angle between S_k(0) and S_k(t) for nested k=1..P-1, linear axis in degrees, solid =
  identifiable (k≤d_eff) / dashed-faint = into noise floor, 3 R₀ columns.
- **Falsification:** regenerate the existing `scripts/viz/fig6_sir_falsification.py`. Verify it
  uses the interventionist θ\* (fix if it is on the inert-lockdown point); no new script.

Outputs: `outputs/viz/fig7_sir_motion.{pdf,png}`, `fig7b_sir_subspace_rotation.{pdf,png}`,
refreshed `fig6_sir_falsification.{pdf,png}`.

### 4. Structured MD log — `outputs/viz/SIR_DIAGNOSTIC_LOG.md`

- (i) Network-SIR model + differentiation: discrete-state SIR on a fixed Erdős–Rényi contact
  graph; per-node S→I / I→R Bernoulli transitions relaxed by Gumbel-Sigmoid (τ=0.5) so the
  simulator is differentiable; FoI = A·I; sigmoid lockdown modulates β_eff; grad_horizon
  truncation via stop_gradient on the pre-window.
- (ii) The 3 R₀ operating points (θ\* table, R₀ derivation).
- (iii) Each experiment — motion, subspace rotation, falsification — figure reference + key
  numbers (spectrum span OOM, d_eff per regime, Θ_k drift, falsification stiff/sloppy ratios)
  + one-line finding each.
- (iv) Pointer to the ARNAU memory doc + one paragraph relating this suite to the paper's
  first-order sensitivity analysis (we add the second-moment / identifiability view).

## Approach / non-goals

- **Parallel `_sirdata.py`, not a refactor of `_bhdata.py`** (YAGNI — do not disturb working BH code).
- **Smoke run first:** short n_iter (~5) + small N_BOOT (~50) to validate the pipeline and the
  stiff-δ init produces descent, before the full cached run.
- Non-goals: no mean-field-vs-network contrast panel; no new falsification script; no changes to
  `src/curvature_calib/`; no paper prose.

## Risks

- Network-SIR calibration cost even trimmed; smoke run gates the full run.
- Stiff δ must be validated empirically — a random/sloppy init leaves the optimiser at the MMD
  noise floor with no signal (same failure mode documented for BH in state.md).
- fig6 may currently sit on the inert-lockdown θ\*; check before quoting its numbers in the log.
