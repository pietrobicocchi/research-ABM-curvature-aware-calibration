# Design: Thesis-Grade Visualization Booklets

**Date:** 2026-06-08
**Status:** Approved (design); pending spec review
**Author:** Pietro (with Claude)

## Purpose

Produce two polished, academic-quality visualization booklets for internal reports and
the thesis:

1. **Booklet 1 — The Models.** Explains the Brock–Hommes and (mean-field / network) SIR
   simulators: what the agents do, the governing equations, and how the dynamics behave.
2. **Booklet 2 — The Methodology.** Explains the OPG identifiability diagnostic pipeline:
   from per-seed gradients, through the OPG matrix and its Gauss-Newton reading, to the
   eigendecomposition, trajectory tracking, and falsification validation.

These are not the AI4ABM paper's hero figures (those are locked in
`paper_story_arc` memory). They are a pedagogical/portfolio asset that will feed both
internal reports and thesis chapters.

## Key decisions (locked during brainstorming)

| Decision | Choice |
|---|---|
| Figure character | **Hybrid** — hand-built concept schematics + polished re-renders of existing result figures |
| Audience / depth | **Expert, thesis-grade.** Captions carry the load; minimal hand-holding |
| Output | **Both** — standalone per-figure vector files (PDF + PNG) AND compiled, captioned booklet PDFs |
| Visual style | **Journal-clean base + explainer annotations** — serif labels, muted blue/teal palette, faint dashed grid; in-figure callouts (stiff/sloppy tags, braces) where they earn their place |
| Toolchain | **All matplotlib** — concept schematics use matplotlib patches/annotations; no TikZ/Illustrator. One reproducible pipeline |
| Captions | Live at **assembly time**, not baked into figures. Standalone figures are caption-free for LaTeX reuse; the booklet build adds a caption band per page |

## Architecture

### New style module: `src/curvature_calib/viz/booklet_style.py`
Extends the existing `viz/style.py`. Provides:
- `apply_booklet_style()` — serif labels, muted palette, faint dashed grid (Journal-clean).
- Annotation toolkit: `callout(ax, xy, text, ...)`, `brace(ax, ...)`, `tag_stiff_sloppy(...)`
  — the B-style in-figure guidance, used sparingly.
- A consistent figure size / DPI / vector-friendly rcParams for one-figure-per-page layout.
- Re-uses `QUAL`, `SEQ`, `DIV`, `REGIME` from `style.py`; does not duplicate the palette.

### Caption source of truth: `captions.yaml`
One entry per figure (keyed by output filename), holding the caption text used at booklet
assembly. Single place to edit caption prose without touching figure code.

### Figure scripts
```
scripts/booklets/
  models/         b1_01_bh_agents.py … b1_10_mf_vs_network.py
  methodology/    b2_01_pipeline.py … b2_12_jacobian_vs_opg.py
  build_booklet.py
```
- Each script is runnable standalone (`uv run python scripts/booklets/<area>/<name>.py`)
  and writes its `.pdf` + `.png` into `outputs/booklets/<area>/`.
- Result-figure scripts re-render existing computations through `booklet_style`; where the
  underlying data is cheap to recompute they do so, otherwise they load the existing
  `outputs/**/*.npz`.
- `build_booklet.py` collects the per-figure PDFs for one area, adds a caption band per
  page from `captions.yaml`, and writes the assembled `*_booklet.pdf` via `PdfPages`.

### Output layout
```
outputs/booklets/
  models/          fig_01_*.pdf … (+ .png mirrors)
  methodology/     fig_01_*.pdf …
  models_booklet.pdf          # assembled, captioned
  methodology_booklet.pdf     # assembled, captioned
```
(`outputs/` is gitignored — booklets are regenerable, consistent with all project figures.)

## Booklet contents

### Booklet 1 — The Models (10 pages)
**Brock–Hommes (financial)**
1. Agent schematic — belief types, fitness, switching mechanism · CONCEPT
2. Annotated model equations · CONCEPT
3. Simulation gallery: three dynamical regimes · RESULT (from script 02)
4. Phase portraits: fixed point → limit cycle → chaos · RESULT (from script 03)
5. Bifurcation map vs β · HYBRID (one net-new computation)

**SIR / Network-SIR (epidemic)**
6. Compartment schematic S→I→R + lockdown · CONCEPT
7. Annotated mean-field equations · CONCEPT
8. Contact-network illustration (Erdős–Rényi) · CONCEPT
9. Example trajectories, lockdown on/off · RESULT (from scripts 18/23)
10. Mean-field vs network dynamics comparison · RESULT (from script 23)

### Booklet 2 — The Methodology (12 pages)
**The pipeline**
1. Calibration loop flow diagram (per-seed grads → mean + F̂) · CONCEPT
2. MMD as RKHS residual norm · CONCEPT
3. Surrogate gradient: discrete event → Gumbel-sigmoid · CONCEPT

**The diagnostic**
4. Gradient cloud → outer products → F̂ · CONCEPT
5. GGN bridge: F̂ ≈ JᵀJ (why it is curvature) · CONCEPT
6. Stiff/sloppy: the two-parameter worked example · CONCEPT
7. Gradient cloud + OPG ellipse + spectrum · RESULT (from scripts 05/21)
8. Eigenvector heatmap |V| · RESULT (from script 21)

**Validation**
9. Trajectory tracking: λ(t), d_eff, principal angles · HYBRID (from scripts 19/25)
10. Bootstrap CIs + noise floor · RESULT
11. Falsification protocol schematic + result · HYBRID (from script 20)
12. Jacobian vs OPG contrast · RESULT (from script 13)

**Cross-booklet notes**
- Surrogate-gradient page lives in Booklet 2 (methodology / differentiability detail).
- Terminology: F̂ is always the "OPG matrix"; Kunstner framing is contributory
  (see `framing_kunstner_opg_not_fisher` and `paper_story_arc` memories).

## Testing / verification
- Each figure script runs standalone without error and produces its PDF + PNG.
- `build_booklet.py` regenerates both booklet PDFs.
- One smoke-import test added for `booklet_style`; no further unit tests (figure scripts).
- Visual verification is manual (review the assembled PDFs).

## Out of scope
- Changes to the AI4ABM paper hero figures or `paper_story_arc` scope.
- New scientific results beyond the single bifurcation-map computation (B1 #5).
- Interactive / web figures.

## Open defaults (confirmed during brainstorming, change if needed)
- Surrogate page in Booklet 2 (not Booklet 1).
- Bifurcation map (B1 #5) kept as a HYBRID build.
