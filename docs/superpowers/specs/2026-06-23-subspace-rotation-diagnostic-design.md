# Subspace-rotation diagnostic for the calibration trajectory

Date: 2026-06-23
Branch: feat/visualization-booklets

## Motivation

The camera-ready trajectory figure `scripts/viz/fig2_bh_motion.py` tracks the
eigenvalue trajectories λ_k(t) and d_eff(t) over the calibration trajectory but
never tracks **eigenvector rotation**. Individual eigenvectors are not robust
diagnostics across iterates: when two eigenvalues are close in magnitude, small
perturbations to F̂_t can interchange the corresponding eigenvectors — an
artefact of the spectral decomposition, not a genuine change in identifiability
geometry.

The robust object is the eigenvector **subspace**. For each k, the k-dimensional
subspace S_k(t) = span(v_1^(t), …, v_k^(t)). Its rotation between iterates is
measured by the principal angles between S_k(t) and S_k(t+1), recovered from the
singular values σ_1 ≥ … ≥ σ_k of V_t^(k)ᵀ V_{t+1}^(k) via θ_i = arccos(σ_i)
(Transtrum et al., 2011). Sustained large principal angles ⇒ the identifiable
subspace is genuinely reorienting as the optimiser moves; small angles ⇒ local
geometric stability.

`principal_angles(V1, V2)` in `src/curvature_calib/calibration/diagnostic.py`
already implements exactly this (QR-orthonormalise → SVD of Q1ᵀQ2 → arccos),
and is already used in scripts 06/09/24/26/27 — but **not** in the camera-ready
`scripts/viz/` set. This is the gap.

## What measures — CUMULATIVE drift, LINEAR axis (final)

For each regime (stable / periodic / chaotic) and each nested subspace
k ∈ {1, …, P-1}, the cumulative reorientation of S_k away from its **initial**
geometry:

    Θ_k(t) = max principal angle between S_k(0) and S_k(t)
           = principal_angles(V_0[:, :k], V_t[:, :k]).max()   [degrees]

so **Θ_k(0) = 0 for every k** by construction. This is "how far the identifiable
subspace has drifted by iterate t," not the per-step rotation *rate*
(∠(S_k(t),S_k(t+1)), which is largest at the start when the optimiser takes its
biggest LM steps and is therefore non-zero at the origin — the confusing object
in the first draft). Reporting the **largest** of the k angles is the sign- and
degeneracy-robust summary; it matches scripts 06/09/24.

**All subspaces.** Plot all nested k=1..P-1. Subspaces reaching into the
numerical noise floor (k > d_eff) contain unconstrained directions that drift
*freely* toward 90°. Encode the boundary: solid for the certified identifiable
subspace (k ≤ d_eff at convergence), dashed/faint for k > d_eff.
d_eff per regime = 2 / 4 / 4 (matches Fig 2).

**Scale.** Cumulative drift is bounded in [0°, 90°], so a **linear** y-axis
(shared across regimes) is the natural choice — and "0 at the beginning" reads
correctly. (The earlier consecutive-rate object spanned ~5 OOM and forced a log
axis; both were rendered and rejected during review in favour of cumulative +
linear.)

## Changes

### 1. `scripts/viz/_bhdata.py` — cache per-iterate eigenvectors

- In `_run_regime`, add `eigvecs_traj = np.asarray(log.eigvecs)` (shape
  `(n_iter, P, P)`) to the returned dict. Today only `V_T = log.eigvecs[-1]`
  is stored.
- In `load`, guard against a stale cache: if a cached regime lacks the
  `eigvecs_traj` key, force a full recompute (the existing `bh_motion.npz`
  predates this field).
- fig2 and fig3 read the same cache and are otherwise untouched.

### 2. New `scripts/viz/fig2b_subspace_rotation.py`

- 1 row × 3 cols (stable / periodic / chaotic), shared **linear** axis, x =
  calibration iteration (t = 0..n-1). P-1 lines per panel (k=1..4) from the rank
  ramp (dark stiff → pale sloppy); solid for k ≤ d_eff, dashed/faint for
  k > d_eff; d_eff annotated per panel; one-line key for the solid/dashed
  encoding. Direct line labels at the right edge; regime name as the quiet
  small-caps marker — same idiom as fig2.
- **Linear y-axis in degrees**, Θ_k(0)=0; shared [≈0, 90°].
- Saved via `ps.save` → PDF + PNG in `outputs/viz/`.

## Out of scope / non-goals

- No change to fig2 or fig3.
- No new `src/` logic; `principal_angles` is already implemented and tested,
  so no new unit tests. Re-run the existing suite to confirm no regression.
- Cumulative drift (vs θ_0 or θ_T) is deliberately not added here; the
  Transtrum framing in the motivation is about consecutive-iterate rotation.

## Verification

- `uv run python scripts/viz/_bhdata.py` recomputes the cache with the new field.
- `uv run python scripts/viz/fig2b_subspace_rotation.py` produces the figure.
- `uv run pytest -q` still passes.
- Record the per-regime finding (subspace stabilises vs reorients) in
  `outputs/viz/FIGURE_NOTES.md`.
