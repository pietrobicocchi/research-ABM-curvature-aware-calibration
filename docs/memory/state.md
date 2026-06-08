---
name: state
description: "Live snapshot — what's implemented, what's verified, what's next. Update when the codebase changes."
metadata: 
  node_type: memory
  type: project
  originSessionId: 6531d326-bc4a-45fa-a786-e0860c232df1
---

Last updated 2026-06-05. **Phase 1 complete** (yellow light, see [[phase1-horizon-bias-result]]). **Phase 2 first pass complete** (H1 speedup hypothesis falsified, but result actually strengthens the diagnostic claim — see [[phase2-convergence-result]]). **Phase 3 complete** (Tier A + Tier B, see [[sir-generalization]]). **Paper draft v1 complete** (commit `08722a3`, 8 main pages + 1 appendix). **Paper v2 naturalness pass complete** (tip `2dfe3e7`; 9 pages; zero undefined references; banned-word grep clean except the allowed `empirical Fisher` meta-disclaimer in `03_method.tex`). **Codebase refactor complete** (2026-06-05): calibration internals split into focused modules, backwards-compat shims in place, 75 tests pass.

**NEXT SESSION**: Phase 4 — write the AI4ABM 2026 paper sections using [[paper-story-arc]] (structure locked 2026-06-08). Paper is 6 sections: §1 Intro, §2 Background, §3 Diagnostic (5 subsections inc. §3.5 Falsification Protocol), §4 Experiments (§4.1 BH + §4.2 Network-SIR), §5 Discussion, §6 Future Work; Appendix A (preconditioning) + Appendix B (technical). Working title: *Identifiability Geometry of MMD Calibration in Differentiable Agent-Based Models*. Locked numbers: 8.4 OOM, 20.7 OOM, 9.4×10⁴, 489×, 25/25, 9.3×10⁵. Mean-field SIR is a stepping stone (Fig 2 data + bridge paragraph in §4.2), not a standalone section.

## Codebase (`src/curvature_calib/`)

| Module | Role |
|---|---|
| `models/brock_hommes.py` | JAX scan simulator, canonical 5-param (β, g₁, b₁, g₂, b₂), differentiable, supports `x_init` perturbation |
| `models/sir.py` | Mean-field SIR with lockdown surrogate, differentiable |
| `models/network_sir.py` | Network-SIR with Erdős-Rényi contact graph, differentiable |
| `models/surrogates.py` | `gumbel_sigmoid`, `straight_through_bernoulli` — differentiable relaxations for discrete events |
| `losses/mmd.py` | unbiased squared MMD with Gaussian RBF; median-heuristic bandwidth via `stop_gradient` |
| `calibration/per_seed_grads.py` | VJP-based per-seed gradients; `CalibStats(loss, mean_grad, per_seed_grads, opg)`; also exports `loss_only` (shared by `calibrate.py` and `baselines.py`, deduplicates the old `_loss_only`) |
| `calibration/opg.py` | **12-line backwards-compat re-export shim** — all functions now live in `diagnostic.py` and `bootstrap.py`; kept so existing scripts do not break |
| `calibration/diagnostic.py` | `EigDecomp`, `eigendecompose`, `opg_from_grads`, `principal_angles`, `effective_dimension`, `d_eff_from_bootstrap` |
| `calibration/bootstrap.py` | `bootstrap_eigvals`, `eigenvalue_cis`, `noise_threshold`, `bootstrap_subspace_cis` |
| `calibration/falsification.py` | `FalsificationResult`, `perturbed_parameters`, `moments_difference`, `acf_difference`, `quantile_difference`, `run_falsification` |
| `calibration/preconditioner.py` | damped Cholesky solver + Levenberg-Marquardt damping (Martens-Grosse style) |
| `calibration/calibrate.py` | LM-adaptive calibration loop, logs every iterate; imports `loss_only` from `per_seed_grads` |
| `calibration/baselines.py` | SGD + Adam, share per-seed-grad backbone; imports `loss_only` from `per_seed_grads` |
| `viz/style.py` | shared palette + rcParams |

**Tests:** `uv run pytest -q` → **75 pass, ~112s**. Full coverage of surrogates, diagnostic, bootstrap, falsification, per-seed-grads, SIR models, network-SIR, MMD, calibration loop, and smoke import.

## Visualization scripts (`scripts/`) and outputs

| Script | Output PNG | Content |
|---|---|---|
| `02_simulation_gallery.py` | `02_simulation_gallery.png` | five regimes, returns, variance-vs-β |
| `03_phase_portraits.py` | `03_phase_portraits.png` | fixed point → limit cycle → chaos, lag plots |
| `04_mmd_landscape.py` | `04_mmd_landscape.png` | MMD² over 2D θ slice (linear + log heatmap) |
| `05_gradient_cloud.py` | `05_gradient_cloud.png` | gradient cloud + 1σ OPG ellipse + spectrum + \|V\| heatmap |
| `06_calibration_dashboard.py` | `06a_calibration_diagnostic.png`, `06b_calibration_distributions.png` | Split headline diagnostic: 06a OPG (loss, iterates, eigenvalue trajectory, \|V_T\|, bootstrap CI, subspace drift); 06b distributions (returns, ACF, quantiles, sample trajectories) at θ₀/θ_T/θ\* |
| `07_optimizer_comparison.py` | `07_optimizer_comparison.png` | OPG vs Adam vs SGD; losses, distance-to-truth, eigenbasis trajectories |
| `08_falsification.py` | `08_falsification.png` | §5.4 sloppy-vs-stiff under three non-MMD discrepancies |
| `09_horizon_bias.py` | `09_horizon_bias.png`, `09_horizon_bias.npz` | Phase 1 killswitch: F̂ eigenstructure across H ∈ {5,10,20,40,80,200}. **Result: yellow light** ([[phase1-horizon-bias-result]]) — hierarchy stable, magnitudes biased |
| `10_phase2_convergence.py` | `10_phase2_convergence.png`, `10_phase2_convergence.npz` | Phase 2 convergence race: 3 difficulty × 5 pairs × 3 optimizers (OPG/Adam/SGD). **Result: H1 fails, diagnostic strengthened** ([[phase2-convergence-result]]) — nobody recovers θ\*; Adam actively diverges; the sloppy spectrum predicted this |
| `11_stiff_sloppy_decomposition.py` | `11_stiff_sloppy_decomposition.png` | **The "diagnostic predicting itself" panel.** Decompose recovery error along F̂(θ\*) eigenbasis. Stiff/sloppy error ratio: 450× for OPG/SGD, **280 000× for Adam**. Adam's noise adaptation specifically amplifies sloppy directions, exactly as Kunstner 2019 predicted |
| `12_adam_lr_sweep.py` | `12_adam_lr_sweep.png` | Adam-lr sanity check. Verdict: Adam's divergence is STRUCTURAL (no lr matches OPG/SGD). Diverges at lr ≥ 1e-3, stops moving at lr ≤ 1e-4 |
| `13_jacobian_comparison.py` | `13_jacobian_comparison.png`, `13_jacobian_comparison.npz` | Phase 3 Objective (d): per-parameter Jacobian sensitivity (Quera-Bofarull 2025 §5.4 style) vs OPG eigendecomposition. **OPG has 10 000× more dynamic range** (5e6 vs 5e2) because off-diagonal correlations (e.g. \|ρ(g₁,g₂)\| = 0.999) are what eigendecomposition exploits. v₁ ≈ (b₁+b₂)/√2 — a combination the per-parameter view cannot surface |
| `14_multiseed_far_from_eq.py` | `14_multiseed_far_from_eq.{png,npz}` | Multi-seed (N=15) far-from-eq race. **OPG and SGD each beat Adam 15/15**; OPG vs SGD 8-7 (tied). Resolves [[reviewer-concerns-audit]] #1 |
| `15_hyperparam_robustness.py` | `15_hyperparam_robustness.png` | Adam β₁ ∈ {0.5, 0.9, 0.95, 0.99} + SGD lr ∈ {1e-2, 1e-3, 1e-4} sweeps. **Adam diverges at the standard β₁=0.9 default and worse for β₁ > 0.9; β₁=0.5 mitigates but is still 8× worse than OPG. SGD only works at lr=1e-3 (beats OPG 0.008 vs 0.012); at lr=1e-2 or 1e-4 it is 10-18× worse than OPG.** Honest reframe: OPG with default init_damping=100 needs no tuning; SGD requires problem-specific lr. Resolves [[reviewer-concerns-audit]] #6, #7 |
| `16_sir_diagnostic.py` | `16_sir_diagnostic.png` | **Phase 3 Tier A — mean-field SIR diagnostic**. OPG spectrum spans **13 OOM** (vs BH's 7); §5.4 falsification ratios stiff/sloppy = 5971× / 54 822× / 17 023× — even larger than BH. Eigenvectors are epidemiologically meaningful (v₁ = I₀, v₂ ≈ R₀, v_P = lockdown strength/timing). Diagnostic generalises. See [[sir-generalization]] |
| `17_sir_calibration_race.py` | `17_sir_calibration_race.{png,npz}` | **Phase 3 Tier A — SIR optimiser race** (mirrors BH scripts 06+07+14). 6-panel figure: loss + parameter recovery + trajectory in (v₁, v_P) plane + per-eigendir error + multi-seed (N=10). **OPG beats Adam 10/10 seeds; SGD has 2/10 catastrophic failures (err 28.67)**. OPG: max err 1.5e-3. Adam: median 0.13. SGD median 1.5e-3 but heavy-tailed. **Strongest optimiser claim**: OPG robust without tuning on two models; SGD overshoots on SIR's high-curvature directions. See [[sir-generalization]] |
| `18_network_sir_diagnostic.py` | `18_network_sir_diagnostic.{png,npz}` | **Phase 3 Tier B — Network-SIR with Gumbel-Sigmoid surrogate gradients**. The hard regime the project plan was designed for. **GREEN LIGHT**: same eigenvector structure as mean-field (I₀ stiff, f_lock sloppy); falsification ratios 185k×/281k×/**11.9M×** — *larger* than mean-field, under surrogate bias. 16-OOM spectrum span. The diagnostic survives the regime. See [[sir-generalization]] |
| `21_fig1_spectrum.py` | `outputs/paper/figures/fig1_spectrum.{png,npz}` | **Paper Figure 1** — float64 polish of script 05. BH OPG spectrum (8.4 OOM span, condition 2.4e8) + gradient cloud + \|V\| heatmap. Paper title, panel labels (a)(b)(c). |
| `22_fig3_predicts_adam.py` | `outputs/paper/figures/fig3_predicts_adam.{png,npz}` | **Paper Figure 3** — float64 polish of script 11, two panels (was 9). Self-contained: embeds a slim 5-pair medium-difficulty Phase-2 run. Cross-opt headline: Adam v_P / OPG v_1 = 9.4e4. |
| `23_fig4_network_sir.py` | `outputs/paper/figures/fig4_network_sir.{png,npz}` | **Paper Figure 4** — float64 polish of script 18 (drops falsification panel; covered by Fig 2). 4 panels: trajectories, spectrum (20.7 OOM span — up from float32's 16 OOM), \|V\|, MF-vs-network comparison. |
| `24_surrogate_comparison.py` | `outputs/network_sir/` | Gumbel-sigmoid vs straight-through surrogate gradient comparison on network-SIR |
| `25_eigenvalue_trajectory.py` | `outputs/brock_hommes/` | Eigenvalue trajectory across BH regimes (β sweep) |
| `26_horizon_sensitivity_sir.py` | `outputs/sir/` | Gradient-horizon sensitivity for mean-field SIR (mirrors script 09 for BH) |
| `27_jacobian_comparison_sir.py` | `outputs/sir/` | Jacobian vs OPG comparison for SIR (mirrors script 13 for BH) |

Calibration log dumped to `outputs/brock_hommes/calibration_log.npz` for downstream scripts.

## Project layout (per-model subdirs)

```
outputs/
  brock_hommes/      # all BH figures + npz (scripts 02-15, 25)
  sir/               # SIR figures (scripts 16-17, 26-27)
  network_sir/       # Network-SIR figures (script 18, 24)
  paper/
    figures/         # fig1_spectrum, fig3_predicts_adam, fig4_network_sir
    20_merged_falsification.{png,npz}   # Paper Figure 2 (canonical)
scripts/
  02_..._23_*.py     # BH + SIR + paper-figure scripts
  24_..._27_*.py     # New 2026-06-05 scripts (surrogates, BH traj, SIR horizon, SIR Jacobian)
```

Note: `notebooks/` directory, `paper/` directory, `scripts/build_*.py`, and `scripts/01-04_*.py` were deleted in the 2026-06-05 refactor. `docs/memory/paper_style_guide.md` was also removed.

### Calibration visualization improvements (2026-06-01)

The calibration loss plots used to look noisy / featureless. Two infrastructure changes + one experimental-design fix:

1. **`val_losses` tracking** (in `calibrate.py` + `baselines.py`): every iteration evaluates MMD² on a *fixed* held-out seed set (`val_seed=999999`, `val_M` seeds). The optimiser itself still sees the noisy fresh-seed MMD² (`losses`); the val track is monotone-style for plots.
2. **best-so-far display**: plots now show `np.minimum.accumulate(val_losses)` as the primary line, with raw val faint underneath. Standard convention for stochastic optimisation.
3. **Stiff-direction init**: θ₀ = θ\* + (0, 0, +0.1, 0, +0.1) — both biases shift the same way along v₁ ≈ (b₁+b₂)/√2. A naive random-direction perturbation almost entirely lands along the *sloppy* spectrum where MMD² is already at the noise floor, leaving the optimiser no signal to descend. (Verified at d₀=1.10 random: MMD²=-0.0007; at d₀=0.14 stiff: MMD²=+0.84.)
4. **`init_damping=100`** for the OPG calibration: at this stiff init the gradient norm is ~3.5 and the default damping=1 makes the first LM-Newton step overshoot into the unstable BH regime (g/R > 1.5). Higher damping bounds the first step; LM then adapts down.
5. **NaN guard** in `calibrate.py`: if a step produces non-finite L_prop, force damping ×10 and treat as hard reject. Prevents the optimiser from getting stuck when the proposed step explodes.

### Far-from-equilibrium calibration results (paper notebook, d₀ = 0.14 stiff)

- **OPG (LM)**: loss 0.84 → 1e-6 in 10 iter; **err 0.14 → 0.012** (recovered to ~1%).
- **SGD**: loss 0.84 → 1e-6 in 30 iter; **err 0.14 → 0.008** (recovered).
- **Adam (lr=1e-2)**: loss 0.84 → 1e-3 plateau (above noise floor!); **err 0.14 → 0.323** (DIVERGED).
- Stiff/sloppy error ratio: OPG 11×, SGD 59×, **Adam 15 000×**. Adam recovers the stiff direction but wanders 5+ OOM in sloppy.
- The Figure 2 (c) trajectory in (v₁, v₅) plane shows Adam loop-spiralling along the sloppy axis to (0, 0.31) while OPG sits at the origin (θ\*).

### Multi-seed far-from-equilibrium robustness (script 14, N=15 random unit-vector inits, d₀=0.14)

Resolves the "single anecdote" reviewer concern. Each seed is a random unit-vector init on the sphere of radius 0.14, run through OPG/Adam/SGD for 60 iter, M=64.

| Optimizer | median err | IQR | max |
|---|---|---|---|
| OPG (LM) | **0.130** | [0.118, 0.135] | 0.138 |
| Adam | 0.325 | [0.307, 0.392] | 0.447 |
| SGD | 0.130 | [0.118, 0.136] | 0.139 |

Pairwise wins:
- **OPG < Adam: 15/15 seeds** (universal)
- **SGD < Adam: 15/15 seeds** (universal)
- OPG vs SGD: 8 vs 7 (statistically tied)

Interpretation: the "Adam diverges along sloppy directions" result is **robust to initialisation** — Adam loses in every seed. OPG and SGD are statistically indistinguishable on this metric, confirming the diagnostic-not-preconditioner framing of the paper.

Outputs: `outputs/brock_hommes/14_multiseed_far_from_eq.{png,npz}`.

## Key empirical findings (canonical θ* = (3.0, 1.2, 0.2, 1.2, -0.2), R=1.1, σ=0.05, T=200, M=64)

1. **Sloppy spectrum confirmed.** OPG eigenvalues at calibrated θ_T span ~7 orders of magnitude (λ₁ ≈ 2.5e-2, λ₅ ≈ 2.3e-9). Bootstrap CI on λ₅ touches zero — smallest eigenvalue statistically indistinguishable from zero.
2. **Substantive identifiability content.** At theta near truth: sloppiest direction `v_P` dominated by β (component 1.00) — **β is non-identifiable by MMD at this θ.** Stiffest direction is the symmetric-bias combination (b₁ + b₂). Trend coefficients (g₁, g₂) intermediate.
3. **Falsification protocol works.** With α = 1e-2, sloppy-direction perturbations produce discrepancies ~3-4 orders of magnitude smaller than stiff-direction perturbations across all three non-MMD channels (moments, ACF, tail quantiles). Aggregate ratio stiff/sloppy ≈ 600-7500×.
4. **Optimizer comparison: Adam competitive.** OPG-preconditioned LM oscillates at the MMD² noise floor; Adam matches/beats it on parameter recovery. **Aligns with project Risk 1**: speedup small or absent here. The diagnostic is the contribution, not the speedup.
5. **Horizon-bias killswitch: yellow light.** Eigenvalue *hierarchy* (which direction is stiff vs sloppy) is stable across H ∈ {5..200}; top-1 subspace drifts 3-8°; top-2 subspace drifts 10-15°. Eigenvalue *magnitudes* shift by up to 70× from short to full horizon. Verdict: diagnostic supports qualitative identifiability claims; quantitative claims need matched H. See [[phase1-horizon-bias-result]].
6. **Phase 2 convergence: H1 fails, diagnostic strengthened.** OPG/Adam/SGD all reach MMD² noise floor (~10⁻³) but none recovers θ\* at any difficulty. OPG and SGD stay at θ₀; Adam *actively diverges* (e.g. d=0.05 → final 0.31). Exactly what the Phase 1 sloppy spectrum predicted: the MMD basin is too sloppy to navigate. Paper claim becomes "the diagnostic predicts the failure of first-order methods to recover non-identifiable parameter combinations." See [[phase2-convergence-result]].
7. **Diagnostic-predicts-itself (script 11)**. Decompose ‖θ_T − θ\*‖² along the OPG eigenbasis at θ\*. Stiff (v_1, λ=2.6e-2) error: 10⁻⁶–10⁻⁴. Sloppy (v_5, λ=4.8e-10) error: 10⁻³–10⁻¹. Ratio sloppy/stiff = 450× for OPG/SGD, **280 000× for Adam**. The diagnostic eigenbasis correctly predicts in which directions optimizers fail.
8. **Adam-lr sweep (script 12)**: Adam's divergence is STRUCTURAL, not hyperparameter. Tested lr ∈ {1e-1, 1e-2, 1e-3, 1e-4, 1e-5} at medium d=0.15: divergence at lr ≥ 1e-3, frozen at lr ≤ 1e-4. No lr matches OPG/SGD's modest improvement.
9. **Jacobian vs OPG comparison (script 13)** [Phase 3 Objective (d) ✓]: per-parameter Jacobian sensitivity spans only ~500×; OPG eigenvalues span ~5e6 — 10 000× more dynamic range because of off-diagonal coupling (|ρ(g₁,g₂)| = 0.999, |ρ(b₁,b₂)| = 0.96). Stiffest eigenvector v₁ ≈ (b₁+b₂)/√2, a combination no per-parameter view can surface. The explicit comparison promised by the project plan.

## Environment

- Python 3.12, `uv`-managed.
- `jax==jaxlib==0.4.30` pinned (Intel-Mac x86_64 wheel constraint). Unpin on Apple Silicon.
- `pyproject.toml` defines deps; `uv sync --extra dev` installs.

## Deferred / known gaps

- Demos use *small* initial perturbation (loss starts near noise floor). For more compelling demos, push θ₀ further, but stay below g/R > ~1.5 explosion threshold (so g₁, g₂ < ~1.5).
- The β=3 regime has weak MMD signal; β > 15 may give richer demos but needs gradient-stability check.

## Recommended next steps (in order)

1. **Regenerate Paper Figure 2** (`scripts/20_merged_falsification.py`): requires `outputs/brock_hommes/calibration_log.npz`. Run once to refresh `outputs/paper/20_merged_falsification.{png,npz}` if the source NPZ was affected by the refactor.
2. **Script 26 confidence intervals**: `scripts/26_horizon_sensitivity_sir.py` — `bootstrap_eigvals` is not yet wired up; add eigenvalue CIs to the horizon-sensitivity plot.
3. **Phase 4 — write AI4ABM 2026 paper sections**: canonical structure locked in [[paper-story-arc]] (2026-06-08). Draft order: §1 Intro → §2 Background → §3 Diagnostic → §4 Experiments → §5 Discussion → §6 Future Work → Appendices. Working title: *Identifiability Geometry of MMD Calibration in Differentiable Agent-Based Models*.
4. **Real data**: S&P 500 daily returns calibration of Brock-Hommes under deliberate misspecification — *strengthens, not blocking*.
5. N=50+ multi-seed (Appendix A aspirational target: 240 runs) — *strengthens, not blocking*.
6. Pearson-normalised spectrum reporting — *strengthens, not blocking* (flag in §5 Limitations if skipped).

**How to apply:** This doc is the canonical "where we are." Update it when the codebase changes. Don't write a new dated status file unless capturing a milestone — keep this current instead.
