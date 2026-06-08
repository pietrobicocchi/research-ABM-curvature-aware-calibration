# Visualization Booklets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build two thesis-grade matplotlib visualization booklets (The Models, The Methodology) as standalone vector figures plus compiled, captioned booklet PDFs.

**Architecture:** A shared `booklet_style` module (extends `viz/style.py`) gives every figure one visual language and an in-figure annotation toolkit. Each figure is a standalone script writing PDF+PNG into `outputs/booklets/<area>/`. A `captions.yaml` holds caption prose; `build_booklet.py` concatenates per-figure PDFs and stamps a caption band per page via matplotlib `PdfPages`.

**Tech Stack:** Python 3.12, `uv`, matplotlib (vector PDF backend), JAX (existing models), PyYAML, pytest.

---

## File Structure

```
src/curvature_calib/viz/
  booklet_style.py        # apply_booklet_style(), save_vector(), palette re-export
  booklet_annotate.py     # callout(), brace(), tag_stiff_sloppy()
scripts/booklets/
  captions.yaml           # caption text keyed by output filename
  build_booklet.py        # assemble per-figure PDFs + caption band -> booklet PDF
  models/
    b1_01_bh_agents.py … b1_10_mf_vs_network.py
  methodology/
    b2_01_pipeline.py … b2_12_jacobian_vs_opg.py
tests/
  test_booklet_style.py   # style + save_vector + annotation toolkit
  test_booklet_build.py   # caption loader + assembler
  test_booklet_scripts.py # every figure script exposes a callable main()
outputs/booklets/         # generated (gitignored): per-area PDFs/PNGs + 2 booklet PDFs
```

Pre-existing API the figures reuse (verified in `scripts/21_fig1_spectrum.py`):
- `curvature_calib.viz.style`: `QUAL`, `SEQ`, `DIV`, `REGIME`, `apply_style`, `save`.
- `curvature_calib.models.brock_hommes.simulate(theta, key, T, sigma, R, x_init)`.
- `curvature_calib.calibration.opg`: `eigendecompose`, `bootstrap_eigvals`.
- `curvature_calib.calibration.per_seed_grads`: `per_seed_loss_and_grads`, `vmap_simulate`.
- SIR / network-SIR `simulate` signatures: copy exactly from `scripts/16_sir_diagnostic.py` and `scripts/18_network_sir_diagnostic.py` respectively (do not guess; open those files for the call pattern).

---

## Phase 0 — Infrastructure

### Task 1: Booklet style module

**Files:**
- Create: `src/curvature_calib/viz/booklet_style.py`
- Test: `tests/test_booklet_style.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_booklet_style.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from curvature_calib.viz import booklet_style


def test_apply_sets_serif_and_grid():
    booklet_style.apply_booklet_style()
    assert plt.rcParams["axes.grid"] is True
    assert "serif" in plt.rcParams["font.family"][0] or plt.rcParams["font.family"] == ["serif"]


def test_save_vector_writes_pdf_and_png(tmp_path):
    booklet_style.apply_booklet_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = booklet_style.save_vector(fig, "demo", out_dir=tmp_path)
    assert (tmp_path / "demo.pdf").exists()
    assert (tmp_path / "demo.png").exists()
    assert paths["pdf"] == tmp_path / "demo.pdf"
    plt.close(fig)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_booklet_style.py -v`
Expected: FAIL with `ModuleNotFoundError`/`AttributeError` (module/functions not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# src/curvature_calib/viz/booklet_style.py
"""Shared style for the thesis visualization booklets.

Journal-clean base (serif labels, muted palette, faint dashed grid) layered on
top of viz/style.py, plus a vector-friendly save helper. Annotation helpers live
in booklet_annotate.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from curvature_calib.viz.style import DIV, QUAL, REGIME, SEQ  # re-export palette

__all__ = ["apply_booklet_style", "save_vector", "QUAL", "SEQ", "DIV", "REGIME"]


def apply_booklet_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#2c3e50",
        "axes.labelcolor": "#2c3e50",
        "xtick.color": "#2c3e50",
        "ytick.color": "#2c3e50",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia"],
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.6,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,      # editable text in vector PDF
        "ps.fonttype": 42,
        "axes.prop_cycle": mpl.cycler(color=QUAL),
    })


def save_vector(fig: mpl.figure.Figure, name: str,
                out_dir: str | Path = "outputs/booklets") -> dict[str, Path]:
    """Save a figure as both vector PDF and raster PNG. `name` is stem only."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    return {"pdf": pdf, "png": png}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_booklet_style.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/curvature_calib/viz/booklet_style.py tests/test_booklet_style.py
git commit -m "feat: booklet_style module (journal-clean base + vector save)"
```

---

### Task 2: Annotation toolkit

**Files:**
- Create: `src/curvature_calib/viz/booklet_annotate.py`
- Modify: `tests/test_booklet_style.py` (append annotation tests)

- [ ] **Step 1: Write the failing tests (append)**

```python
# append to tests/test_booklet_style.py
from curvature_calib.viz import booklet_annotate


def test_callout_adds_annotation():
    booklet_style.apply_booklet_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    n_before = len(ax.texts)
    booklet_annotate.callout(ax, (0.5, 0.5), "stiff", (0.7, 0.2))
    assert len(ax.texts) == n_before + 1
    plt.close(fig)


def test_tag_stiff_sloppy_adds_two_texts():
    booklet_style.apply_booklet_style()
    fig, ax = plt.subplots()
    ax.bar(range(5), [5, 4, 3, 2, 1])
    n_before = len(ax.texts)
    booklet_annotate.tag_stiff_sloppy(ax, stiff_xy=(0, 5), sloppy_xy=(4, 1))
    assert len(ax.texts) == n_before + 2
    plt.close(fig)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_booklet_style.py -k "callout or stiff_sloppy" -v`
Expected: FAIL (`ModuleNotFoundError: booklet_annotate`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/curvature_calib/viz/booklet_annotate.py
"""In-figure annotation helpers for the booklets (explainer-style callouts)."""
from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from curvature_calib.viz.style import QUAL

STIFF_COLOR = QUAL[1]   # warm red
SLOPPY_COLOR = "#7f8c8d"  # grey

__all__ = ["callout", "brace", "tag_stiff_sloppy"]


def callout(ax, xy, text, xytext, *, color="#2c3e50", fontsize=8.5):
    """Labelled arrow pointing at data coordinate `xy`, text placed at `xytext`."""
    return ax.annotate(
        text, xy=xy, xytext=xytext, textcoords="data",
        fontsize=fontsize, color=color, fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=0.8, alpha=0.95),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
    )


def brace(ax, x0, x1, y, text, *, color="#2c3e50", fontsize=8.5, dy=0.04):
    """Horizontal annotation bracket from x0..x1 at height y with a label."""
    ax.annotate("", xy=(x0, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-", color=color, lw=1.0))
    ax.text((x0 + x1) / 2, y + dy, text, ha="center", va="bottom",
            color=color, fontsize=fontsize, fontweight="bold")


def tag_stiff_sloppy(ax, stiff_xy, sloppy_xy):
    """Convenience: tag a stiff and a sloppy point with the canonical colors."""
    callout(ax, stiff_xy, "stiff", (stiff_xy[0] + 0.6, stiff_xy[1]),
            color=STIFF_COLOR)
    callout(ax, sloppy_xy, "sloppy", (sloppy_xy[0] - 0.6, sloppy_xy[1]),
            color=SLOPPY_COLOR)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_booklet_style.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/curvature_calib/viz/booklet_annotate.py tests/test_booklet_style.py
git commit -m "feat: booklet annotation toolkit (callout/brace/tag_stiff_sloppy)"
```

---

### Task 3: Captions file + loader

**Files:**
- Create: `scripts/booklets/captions.yaml`
- Create: `scripts/booklets/build_booklet.py` (loader part only this task)
- Test: `tests/test_booklet_build.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_booklet_build.py
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_booklet", ROOT / "scripts" / "booklets" / "build_booklet.py")
build_booklet = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_booklet)


def test_load_captions_returns_dict():
    caps = build_booklet.load_captions()
    assert isinstance(caps, dict)
    # every value is a non-empty string
    assert all(isinstance(v, str) and v for v in caps.values())


def test_caption_for_known_key():
    caps = build_booklet.load_captions()
    assert "fig_01_bh_agents" in caps
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_booklet_build.py -v`
Expected: FAIL (file `build_booklet.py` does not exist).

- [ ] **Step 3: Create captions.yaml (seed with all keys)**

```yaml
# scripts/booklets/captions.yaml
# Caption prose stamped under each figure at booklet-assembly time.
# Keys are output filename stems (no extension). One line each.
fig_01_bh_agents: "Brock–Hommes agents choose among belief types by past fitness; the discrete-choice intensity beta governs how sharply they switch."
fig_02_bh_equations: "The Brock–Hommes pricing map and the fitness-weighted type dynamics, with the five calibrated parameters highlighted."
fig_03_bh_gallery: "Representative price/return series across the three dynamical regimes selected by beta: fundamental, periodic, and chaotic."
fig_04_bh_phase: "Phase portraits and lag plots tracing the route from a stable fixed point through a limit cycle to chaos."
fig_05_bh_bifurcation: "Bifurcation diagram of the long-run dynamics as beta is swept, locating the three regimes used in calibration."
fig_06_sir_compartments: "The mean-field SIR compartments with the lockdown control that modulates the transmission rate."
fig_07_sir_equations: "The mean-field SIR ODEs and the smoothed lockdown surrogate, with calibrated parameters highlighted."
fig_08_sir_network: "An Erdős–Rényi contact graph: infection spreads along edges, replacing the mean-field mixing assumption."
fig_09_sir_trajectories: "Network-SIR epidemic curves with and without lockdown, showing the policy effect the calibration must recover."
fig_10_mf_vs_network: "Mean-field versus network-SIR dynamics under matched parameters, motivating the surrogate-gradient regime."
fig_01_pipeline: "The differentiable calibration loop: per-seed gradients drive the mean update and, at zero extra cost, form the discarded OPG matrix F."
fig_02_mmd_residual: "MMD as the squared RKHS norm of a kernel-mean residual, the structure that licenses the Gauss-Newton reading of F."
fig_03_surrogate: "A discrete event replaced by a Gumbel-sigmoid relaxation, yielding the biased but usable surrogate gradients."
fig_04_opg_construction: "From the per-seed gradient cloud to outer products to the averaged OPG matrix F."
fig_05_ggn_bridge: "Why F approximates J^T J for MMD losses: the residual-structured generalised Gauss-Newton decomposition."
fig_06_stiff_sloppy_toy: "The two-parameter worked example: the sum direction is stiff, the difference direction sloppy."
fig_07_gradient_cloud: "Per-seed gradient cloud with the 1-sigma OPG ellipse and the eigenvalue spectrum for Brock–Hommes."
fig_08_eigenvector_heatmap: "Eigenvector content |V|: which parameter combinations the calibration data constrains."
fig_09_trajectory: "Tracking the OPG spectrum, effective dimension, and eigenvector principal angles along the calibration trajectory."
fig_10_bootstrap: "Bootstrap confidence intervals on the eigenvalues and the noise floor that defines the effective dimension."
fig_11_falsification: "The falsification protocol and its result: sloppy-direction perturbations are indistinguishable under three non-MMD discrepancies."
fig_12_jacobian_vs_opg: "Per-parameter Jacobian sensitivity versus OPG eigenvalues: F surfaces parameter combinations the Jacobian misses."
```

- [ ] **Step 4: Write the loader**

```python
# scripts/booklets/build_booklet.py
"""Assemble standalone booklet figures into captioned booklet PDFs."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CAPTIONS_PATH = Path(__file__).resolve().parent / "captions.yaml"


def load_captions(path: Path = CAPTIONS_PATH) -> dict[str, str]:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return dict(data)
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/test_booklet_build.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add scripts/booklets/captions.yaml scripts/booklets/build_booklet.py tests/test_booklet_build.py
git commit -m "feat: booklet captions.yaml + loader"
```

---

### Task 4: Booklet assembler (caption band + PdfPages)

**Files:**
- Modify: `scripts/booklets/build_booklet.py`
- Modify: `tests/test_booklet_build.py` (append)

- [ ] **Step 1: Write the failing test (append)**

```python
# append to tests/test_booklet_build.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _make_pdf(path, label):
    fig, ax = plt.subplots()
    ax.set_title(label)
    fig.savefig(path)
    plt.close(fig)


def test_assemble_booklet_creates_pdf(tmp_path):
    fig_dir = tmp_path / "models"
    fig_dir.mkdir()
    _make_pdf(fig_dir / "fig_01_bh_agents.pdf", "one")
    _make_pdf(fig_dir / "fig_02_bh_equations.pdf", "two")
    out = tmp_path / "models_booklet.pdf"
    n = build_booklet.assemble_booklet(fig_dir, out, build_booklet.load_captions())
    assert out.exists()
    assert n == 2  # two pages assembled
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_booklet_build.py::test_assemble_booklet_creates_pdf -v`
Expected: FAIL (`assemble_booklet` not defined).

- [ ] **Step 3: Implement assembler + CLI**

```python
# append to scripts/booklets/build_booklet.py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def assemble_booklet(fig_dir: Path, out_pdf: Path, captions: dict[str, str]) -> int:
    """Render each fig_*.pdf's PNG sibling (or rasterised PDF) as a booklet page
    with a caption band. Returns the page count."""
    pdfs = sorted(fig_dir.glob("fig_*.pdf"))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pp:
        for pdf in pdfs:
            stem = pdf.stem
            png = pdf.with_suffix(".png")
            page = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
            ax_img = page.add_axes([0.06, 0.18, 0.88, 0.76])
            ax_img.axis("off")
            if png.exists():
                ax_img.imshow(mpimg.imread(png))
            cap = captions.get(stem, "")
            page.text(0.06, 0.10, cap, ha="left", va="top", wrap=True,
                      fontsize=10, family="serif")
            page.text(0.94, 0.04, stem, ha="right", va="bottom",
                      fontsize=7, color="#7f8c8d", family="serif")
            pp.savefig(page)
            plt.close(page)
    return len(pdfs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(ROOT / "outputs" / "booklets"))
    args = ap.parse_args()
    out_root = Path(args.out_root)
    caps = load_captions()
    for area, name in [("models", "models_booklet.pdf"),
                       ("methodology", "methodology_booklet.pdf")]:
        fig_dir = out_root / area
        if not fig_dir.exists():
            print(f"skip {area}: {fig_dir} missing")
            continue
        n = assemble_booklet(fig_dir, out_root / name, caps)
        print(f"assembled {area}: {n} pages -> {out_root / name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_booklet_build.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/booklets/build_booklet.py tests/test_booklet_build.py
git commit -m "feat: booklet assembler (PdfPages caption band) + CLI"
```

---

### Task 5: Figure-script contract + runner test

Every figure script must expose a no-arg `main()` and an `OUT_AREA`/`OUT_NAME`. The runner test only checks the contract (not full execution — figures are slow).

**Files:**
- Create: `scripts/booklets/_figbase.py`
- Test: `tests/test_booklet_scripts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_booklet_scripts.py
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts" / "booklets"
FIG_SCRIPTS = sorted(
    [p for p in (SCRIPT_DIR / "models").glob("b1_*.py")]
    + [p for p in (SCRIPT_DIR / "methodology").glob("b2_*.py")]
)


@pytest.mark.parametrize("path", FIG_SCRIPTS, ids=lambda p: p.stem)
def test_script_exposes_main_and_names(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod.main)
    assert isinstance(mod.OUT_AREA, str) and mod.OUT_AREA in {"models", "methodology"}
    assert isinstance(mod.OUT_NAME, str) and mod.OUT_NAME.startswith("fig_")
```

- [ ] **Step 2: Run to verify it fails (or passes vacuously)**

Run: `uv run pytest tests/test_booklet_scripts.py -v`
Expected: PASS with 0 selected (no scripts yet) — acceptable. After each figure task, re-run; it must stay green.

- [ ] **Step 3: Create the shared figure base**

```python
# scripts/booklets/_figbase.py
"""Shared helpers for booklet figure scripts. Import-safe (no heavy work at import)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "outputs" / "booklets"


def out_dir(area: str) -> Path:
    d = OUT_ROOT / area
    d.mkdir(parents=True, exist_ok=True)
    return d
```

- [ ] **Step 4: Commit**

```bash
git add scripts/booklets/_figbase.py tests/test_booklet_scripts.py
git commit -m "test: figure-script contract + runner; add _figbase helper"
```

---

## Phase 1 — Booklet 1: The Models

Each figure task follows the same recipe. **Recipe (apply to every Phase 1/2 figure task):**
1. Create `scripts/booklets/<area>/<file>.py` with this skeleton:

```python
"""<one-line description>."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from curvature_calib.viz.booklet_style import apply_booklet_style, save_vector
from scripts.booklets._figbase import out_dir  # if run as module; else inline path

OUT_AREA = "models"          # or "methodology"
OUT_NAME = "fig_NN_xxx"


def main() -> None:
    apply_booklet_style()
    fig = ...                 # build per the panel spec below
    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)


if __name__ == "__main__":
    main()
```
(If `from scripts...` import fails when run directly, replace with the two lines from `_figbase.py` inlined: compute `OUT_ROOT` from `Path(__file__).resolve().parents[2]`.)

2. Run `uv run python scripts/booklets/<area>/<file>.py`; confirm `outputs/booklets/<area>/<OUT_NAME>.pdf` and `.png` exist.
3. Run `uv run pytest tests/test_booklet_scripts.py -k <OUT_NAME or stem> -v`; expect PASS.
4. Commit.

Result-rerender figures additionally: open the cited source script and reuse its computation verbatim, swapping `apply_style`→`apply_booklet_style` and `save`→`save_vector`, and add explainer annotations from `booklet_annotate` where noted.

---

### Task 6: B1-01 BH agent schematic (CONCEPT)

**Files:** Create `scripts/booklets/models/b1_01_bh_agents.py`

- [ ] **Step 1: Build.** `OUT_NAME="fig_01_bh_agents"`. Single axis, `ax.axis("off")`. Draw with `matplotlib.patches`: (left) a box "Past performance / profits" → (middle) two rounded boxes "Type 1: trend (g₁,b₁)" and "Type 2: contrarian (g₂,b₂)" → (right) a "Market price xₜ" box. Connect with `FancyArrowPatch`. Add a curved arrow from price back to performance labelled "fitness feedback". Place a callout via `booklet_annotate.callout` on the switching arrow: text `r"switch ∝ $e^{\beta U}$"`. Use `QUAL` colors for the two type boxes.
- [ ] **Step 2: Run script; confirm PDF+PNG written.**

Run: `uv run python scripts/booklets/models/b1_01_bh_agents.py`
Expected: prints/creates `outputs/booklets/models/fig_01_bh_agents.{pdf,png}`.

- [ ] **Step 3: Contract test.**

Run: `uv run pytest tests/test_booklet_scripts.py -k b1_01 -v`
Expected: PASS.

- [ ] **Step 4: Commit.**

```bash
git add scripts/booklets/models/b1_01_bh_agents.py
git commit -m "feat: booklet1 fig01 BH agent schematic"
```

---

### Task 7: B1-02 BH annotated equations (CONCEPT)

**Files:** Create `scripts/booklets/models/b1_02_bh_equations.py`

- [ ] **Step 1: Build.** `OUT_NAME="fig_02_bh_equations"`. `ax.axis("off")`. Render the BH equations with `ax.text(..., usetex=False)` mathtext: the pricing equation `r"$R x_{t+1} = \sum_h n_{h,t}(g_h x_t + b_h)+\epsilon_t$"`, the fitness `r"$U_{h,t}=(x_t-Rx_{t-1})(g_h x_{t-1}+b_h-Rx_{t-1})$"`, and the discrete choice `r"$n_{h,t}=\dfrac{e^{\beta U_{h,t-1}}}{\sum_k e^{\beta U_{k,t-1}}}$"`. Below, a small legend row mapping the 5 calibrated params β,g₁,b₁,g₂,b₂ to one-line meanings; highlight each symbol in `QUAL[0]` using a colored `bbox` behind the symbol via `booklet_annotate.callout` or `ax.text` with bbox. Confirm symbol names match `PARAM_NAMES` in script 21.
- [ ] **Step 2–4:** Run script; contract test `-k b1_02`; commit `"feat: booklet1 fig02 BH equations"`.

---

### Task 8: B1-03 BH simulation gallery (RESULT, from script 02)

**Files:** Create `scripts/booklets/models/b1_03_bh_gallery.py`

- [ ] **Step 1: Build.** `OUT_NAME="fig_03_bh_gallery"`. Open `scripts/02_simulation_gallery.py`; reuse its `simulate` calls for the three regime β values. Lay out a 3-row figure: each row = price/return series for one regime, colored by `REGIME["fundamental"|"periodic"|"chaotic"]`. Row titles name the regime + β. Use `booklet_annotate.brace` on the chaotic row to mark a representative burst if helpful.
- [ ] **Step 2–4:** Run; contract test `-k b1_03`; commit `"feat: booklet1 fig03 BH simulation gallery"`.

---

### Task 9: B1-04 BH phase portraits (RESULT, from script 03)

**Files:** Create `scripts/booklets/models/b1_04_bh_phase.py`

- [ ] **Step 1: Build.** `OUT_NAME="fig_04_bh_phase"`. Reuse the lag-plot/phase-portrait computation from `scripts/03_phase_portraits.py`. Three columns: fixed point, limit cycle, chaotic attractor (lag plot xₜ vs xₜ₋₁), each colored by `REGIME`. Title `"(a) fixed point"` etc.
- [ ] **Step 2–4:** Run; contract test `-k b1_04`; commit `"feat: booklet1 fig04 BH phase portraits"`.

---

### Task 10: B1-05 BH bifurcation map (HYBRID, net-new computation)

**Files:** Create `scripts/booklets/models/b1_05_bh_bifurcation.py`

- [ ] **Step 1: Build.** `OUT_NAME="fig_05_bh_bifurcation"`. New computation: sweep `beta` over e.g. `np.linspace(0, 10, 400)` (keep g₁,b₁,g₂,b₂ at `THETA_STAR` values from script 21); for each β simulate `T=400` with the BH `simulate` (fixed key, x_init=0.0), discard a 200-step transient, scatter the remaining `x_t` values at that β (`s=0.3, alpha=0.3, color=QUAL[0]`). Mark the three regime β's used elsewhere with vertical lines colored by `REGIME` and `booklet_annotate.callout` labels. Cap runtime: if >30 s, reduce β grid to 250. x-axis β, y-axis `x_t` (long-run).
- [ ] **Step 2: Run; confirm it completes under ~60 s and writes outputs.**
- [ ] **Step 3: Contract test `-k b1_05`.**
- [ ] **Step 4: Commit `"feat: booklet1 fig05 BH bifurcation map"`.**

---

### Task 11: B1-06 SIR compartment schematic (CONCEPT)

**Files:** Create `scripts/booklets/models/b1_06_sir_compartments.py`

- [ ] **Step 1: Build.** `OUT_NAME="fig_06_sir_compartments"`. `ax.axis("off")`. Three circles S, I, R (`mpatches.Circle`) left→right, arrows S→I labelled `r"$\beta SI/N$"` and I→R labelled `r"$\gamma I$"`. Add a "lockdown" block above the S→I arrow with a downward arrow modulating β, `booklet_annotate.callout` text "lockdown scales β by (1−f·σ(t))". Colors: S `REGIME["fundamental"]`, I `REGIME["chaotic"]`, R `REGIME["periodic"]`.
- [ ] **Step 2–4:** Run; contract test `-k b1_06`; commit `"feat: booklet1 fig06 SIR compartments"`.

---

### Task 12: B1-07 SIR annotated equations (CONCEPT)

**Files:** Create `scripts/booklets/models/b1_07_sir_equations.py`

- [ ] **Step 1: Build.** `OUT_NAME="fig_07_sir_equations"`. `ax.axis("off")`. Render mean-field ODEs `dS/dt=-βSI/N`, `dI/dt=βSI/N−γI`, `dR/dt=γI` and the smoothed lockdown surrogate (copy exact functional form from `src/curvature_calib/models/sir.py`). Highlight calibrated parameters (use the param list from `scripts/16_sir_diagnostic.py`). One-line meaning per parameter beneath.
- [ ] **Step 2–4:** Run; contract test `-k b1_07`; commit `"feat: booklet1 fig07 SIR equations"`.

---

### Task 13: B1-08 contact-network illustration (CONCEPT)

**Files:** Create `scripts/booklets/models/b1_08_sir_network.py`

- [ ] **Step 1: Build.** `OUT_NAME="fig_08_sir_network"`. Draw a small Erdős–Rényi graph by hand with matplotlib (no networkx dependency required): sample `N=40` node positions via `np.random.default_rng(0)` on the unit square; connect pairs with probability `p=0.08`; draw edges as thin grey lines, nodes as scatter colored by a frozen S/I/R assignment (mostly S, a few I in `REGIME["chaotic"]`, some R). Caption-free; `booklet_annotate.callout` on one infected node "infection spreads along edges".
- [ ] **Step 2–4:** Run; contract test `-k b1_08`; commit `"feat: booklet1 fig08 contact network"`.

---

### Task 14: B1-09 network-SIR trajectories (RESULT, from scripts 18/23)

**Files:** Create `scripts/booklets/models/b1_09_sir_trajectories.py`

- [ ] **Step 1: Build.** `OUT_NAME="fig_09_sir_trajectories"`. Reuse the network-SIR `simulate` call from `scripts/18_network_sir_diagnostic.py`. Plot S/I/R curves for two scenarios (lockdown on vs off) in two panels or overlaid with linestyle; color compartments consistently with Task 11. Annotate the peak-flattening with `booklet_annotate.brace`.
- [ ] **Step 2–4:** Run; contract test `-k b1_09`; commit `"feat: booklet1 fig09 network-SIR trajectories"`.

---

### Task 15: B1-10 mean-field vs network comparison (RESULT, from script 23)

**Files:** Create `scripts/booklets/models/b1_10_mf_vs_network.py`

- [ ] **Step 1: Build.** `OUT_NAME="fig_10_mf_vs_network"`. Reuse the MF-vs-network comparison computation from `scripts/23_fig4_network_sir.py` (the comparison panel). Overlay mean-field I(t) vs network I(t) under matched parameters; annotate the divergence with `booklet_annotate.callout`.
- [ ] **Step 2–4:** Run; contract test `-k b1_10`; commit `"feat: booklet1 fig10 MF vs network"`.

---

## Phase 2 — Booklet 2: The Methodology

(Use the same per-figure recipe; `OUT_AREA="methodology"`.)

### Task 16: B2-01 calibration-loop flow diagram (CONCEPT)

**Files:** Create `scripts/booklets/methodology/b2_01_pipeline.py`

- [ ] **Step 1: Build.** `OUT_NAME="fig_01_pipeline"`. `ax.axis("off")`. Boxes left→right: "θ_t" → "simulate M seeds (differentiable ABM)" → "per-seed gradients {g_m}". Then branch: top arrow → "mean ḡ → gradient step θ_{t+1}"; bottom arrow → "OPG  F = (1/M)Σ gₘgₘᵀ" with a red `booklet_annotate.callout` "normally discarded — we keep it". Loop arrow back to θ_t.
- [ ] **Step 2–4:** Run; contract test `-k b2_01`; commit `"feat: booklet2 fig01 pipeline diagram"`.

---

### Task 17: B2-02 MMD as RKHS residual (CONCEPT)

**Files:** Create `scripts/booklets/methodology/b2_02_mmd_residual.py`

- [ ] **Step 1: Build.** `OUT_NAME="fig_02_mmd_residual"`. Left: two small sample clouds (sim vs reference) via scatter. Middle: arrows into kernel mean embeddings μ_P, μ_ref (two dots in an abstract "RKHS" ellipse). Right: the residual vector between them with label `r"$\mathcal{L}=\|\mu_{\mathbb{P}_\theta}-\mu_{\mathrm{ref}}\|_\mathcal{H}^2$"`. `booklet_annotate.callout` "residual structure → Gauss-Newton reading".
- [ ] **Step 2–4:** Run; contract test `-k b2_02`; commit `"feat: booklet2 fig02 MMD residual"`.

---

### Task 18: B2-03 surrogate gradient (CONCEPT)

**Files:** Create `scripts/booklets/methodology/b2_03_surrogate.py`

- [ ] **Step 1: Build.** `OUT_NAME="fig_03_surrogate"`. Plot a Heaviside step vs the Gumbel-sigmoid relaxation across temperatures τ (compute with `src/curvature_calib/models/surrogates.py` `gumbel_sigmoid`; reuse call pattern from `scripts/24_surrogate_comparison.py`). Twin: the derivative (zero a.e. for the step, bell-shaped for the surrogate). `booklet_annotate.callout` "usable gradient where the step has none".
- [ ] **Step 2–4:** Run; contract test `-k b2_03`; commit `"feat: booklet2 fig03 surrogate gradient"`.

---

### Task 19: B2-04 OPG construction (CONCEPT)

**Files:** Create `scripts/booklets/methodology/b2_04_opg_construction.py`

- [ ] **Step 1: Build.** `OUT_NAME="fig_04_opg_construction"`. Three panels: (a) a 2-D gradient cloud scatter, (b) a couple of outer-product matrices gₘgₘᵀ shown as small imshows, (c) the averaged F as an imshow with colorbar. Arrows between panels via `fig`-level annotations. Synthetic 2-D gradients (`np.random.default_rng`) suffice — this is a concept figure.
- [ ] **Step 2–4:** Run; contract test `-k b2_04`; commit `"feat: booklet2 fig04 OPG construction"`.

---

### Task 20: B2-05 GGN bridge (CONCEPT)

**Files:** Create `scripts/booklets/methodology/b2_05_ggn_bridge.py`

- [ ] **Step 1: Build.** `OUT_NAME="fig_05_ggn_bridge"`. `ax.axis("off")`. Render the decomposition `r"$\nabla^2\mathcal{L}=2J^\top J+\underbrace{(\text{residual curvature})}_{\approx 0\ \text{near optimum}}$"` and the identity that the per-seed gradient is the residual-contracted Jacobian, concluding `r"$\hat{F}\approx J^\top J$"`. Use `booklet_annotate.callout` to tie F̂ to "curvature, not a Fisher claim (cf. Kunstner 2019)". Keep terminology: "OPG matrix".
- [ ] **Step 2–4:** Run; contract test `-k b2_05`; commit `"feat: booklet2 fig05 GGN bridge"`.

---

### Task 21: B2-06 stiff/sloppy toy example (CONCEPT)

**Files:** Create `scripts/booklets/methodology/b2_06_stiff_sloppy_toy.py`

- [ ] **Step 1: Build.** `OUT_NAME="fig_06_stiff_sloppy_toy"`. A 2-parameter loss `L(a,b)=(a+b)²` (stiff along a+b, flat along a−b). Left: filled contour of L over a grid. Overlay the two eigenvectors as arrows; `booklet_annotate.tag_stiff_sloppy` on the stiff (sum) and sloppy (difference) directions. Right: 1-D slices along each direction showing steep vs flat. This is the canonical worked example from `paper_story_arc` §3.2.
- [ ] **Step 2–4:** Run; contract test `-k b2_06`; commit `"feat: booklet2 fig06 stiff/sloppy toy"`.

---

### Task 22: B2-07 gradient cloud + spectrum (RESULT, from scripts 05/21)

**Files:** Create `scripts/booklets/methodology/b2_07_gradient_cloud.py`

- [ ] **Step 1: Build.** `OUT_NAME="fig_07_gradient_cloud"`. Reuse panels (a) gradient cloud + 1σ ellipse and (b) spectrum from `scripts/21_fig1_spectrum.py` verbatim (same constants, float64, `confidence_ellipse`, `bootstrap_eigvals`). Swap to `apply_booklet_style`/`save_vector`. Add `booklet_annotate.tag_stiff_sloppy` on the spectrum's λ₁ and λ_P bars.
- [ ] **Step 2–4:** Run (note: JAX float64, ~tens of seconds); contract test `-k b2_07`; commit `"feat: booklet2 fig07 gradient cloud + spectrum"`.

---

### Task 23: B2-08 eigenvector heatmap (RESULT, from script 21)

**Files:** Create `scripts/booklets/methodology/b2_08_eigenvector_heatmap.py`

- [ ] **Step 1: Build.** `OUT_NAME="fig_08_eigenvector_heatmap"`. Reuse panel (c) |V| heatmap from script 21 (same computation; `PARAM_NAMES`; cell text). Annotate the stiff column (v₁) and sloppy column (v_P) with `booklet_annotate.callout` naming the recovered combination (e.g. "b₁+b₂ symmetric bias").
- [ ] **Step 2–4:** Run; contract test `-k b2_08`; commit `"feat: booklet2 fig08 eigenvector heatmap"`.

---

### Task 24: B2-09 trajectory tracking (HYBRID, from scripts 19/25)

**Files:** Create `scripts/booklets/methodology/b2_09_trajectory.py`

- [ ] **Step 1: Build.** `OUT_NAME="fig_09_trajectory"`. Reuse trajectory data from `scripts/19_trajectory_bootstrap.py` (and/or `25_eigenvalue_trajectory.py`). Three stacked panels sharing x=iteration: (a) log₁₀λ_k(t) lines, (b) d_eff(t) step line, (c) principal angle of v₁ and v_P to V(θ*). Annotate "within 2° by iter 4" with `booklet_annotate.callout` (number from `paper_story_arc` scope rule).
- [ ] **Step 2–4:** Run; contract test `-k b2_09`; commit `"feat: booklet2 fig09 trajectory tracking"`.

---

### Task 25: B2-10 bootstrap CIs (RESULT)

**Files:** Create `scripts/booklets/methodology/b2_10_bootstrap.py`

- [ ] **Step 1: Build.** `OUT_NAME="fig_10_bootstrap"`. Reuse `bootstrap_eigvals` exactly as in script 21 to get the boot distribution; plot eigenvalues with 95% CI error bars (log y), draw a horizontal "noise floor" line at the largest CI-upper that includes zero, and shade the region below. `booklet_annotate.brace` over the eigenvalues whose CI lower bound exceeds the floor → label "effective dimension d_eff".
- [ ] **Step 2–4:** Run; contract test `-k b2_10`; commit `"feat: booklet2 fig10 bootstrap CIs"`.

---

### Task 26: B2-11 falsification protocol (HYBRID, from script 20)

**Files:** Create `scripts/booklets/methodology/b2_11_falsification.py`

- [ ] **Step 1: Build.** `OUT_NAME="fig_11_falsification"`. Top: a small schematic row (perturb along v_P vs v₁ → three non-MMD discrepancy meters: moments, ACF, tail quantiles), `ax.axis("off")`, drawn with boxes/arrows. Bottom: reuse the result panel from `scripts/20_merged_falsification.py` (load `outputs/paper/20_merged_falsification.npz` if present; else recompute via the script's functions) showing stiff/sloppy ratios. `booklet_annotate.callout` the BH ratio (489×, from `paper_story_arc`).
- [ ] **Step 2–4:** Run; contract test `-k b2_11`; commit `"feat: booklet2 fig11 falsification protocol"`.

---

### Task 27: B2-12 Jacobian vs OPG (RESULT, from script 13)

**Files:** Create `scripts/booklets/methodology/b2_12_jacobian_vs_opg.py`

- [ ] **Step 1: Build.** `OUT_NAME="fig_12_jacobian_vs_opg"`. Reuse the computation from `scripts/13_jacobian_comparison.py`. Two panels: (a) per-parameter Jacobian sensitivity bars (~500× span), (b) OPG eigenvalues (~5e6 span); shared log y. `booklet_annotate.callout` "10 000× more dynamic range — OPG sees combinations". 
- [ ] **Step 2–4:** Run; contract test `-k b2_12`; commit `"feat: booklet2 fig12 Jacobian vs OPG"`.

---

## Phase 3 — Assemble & verify

### Task 28: Generate all figures and build both booklets

- [ ] **Step 1: Run every figure script.**

Run:
```bash
for f in scripts/booklets/models/b1_*.py scripts/booklets/methodology/b2_*.py; do
  echo "== $f"; uv run python "$f" || exit 1
done
```
Expected: each writes `outputs/booklets/<area>/fig_*.{pdf,png}` with no errors.

- [ ] **Step 2: Assemble booklets.**

Run: `uv run python scripts/booklets/build_booklet.py`
Expected: prints `assembled models: 10 pages …` and `assembled methodology: 12 pages …`; both `outputs/booklets/*_booklet.pdf` exist.

- [ ] **Step 3: Full test suite stays green.**

Run: `uv run pytest -q`
Expected: previous 75 pass + new booklet tests pass.

- [ ] **Step 4: Manual visual verification.** Open both booklet PDFs; confirm: consistent serif/style across pages, captions present and correct, annotations legible, no clipped text, vector-sharp on zoom. Note any figure needing rework and loop back to its task.

- [ ] **Step 5: Commit any caption tweaks.**

```bash
git add scripts/booklets/captions.yaml
git commit -m "docs: caption polish after visual review"
```

---

## Self-Review

**Spec coverage:** booklet_style (T1), annotation toolkit (T2), captions.yaml + loader (T3), assembler/PdfPages caption band (T4), figure-script contract (T5), all 10 Booklet-1 pages (T6–15), all 12 Booklet-2 pages (T16–27), both standalone PDFs+PNGs (save_vector everywhere) and compiled booklets (T28), outputs under `outputs/booklets/` (gitignored), surrogate page in methodology (T18), bifurcation as the one net-new computation (T10). All spec sections covered.

**Placeholder scan:** No "TBD/TODO". Figure tasks give exact filenames, OUT_NAME/OUT_AREA, data source (named source script or explicit synthetic generation), panel layout, specific annotation calls, and run/test/commit steps. Result figures point to the exact existing script whose computation is reused (DRY — not re-pasting 150 lines that already exist in the repo).

**Type consistency:** `apply_booklet_style()`, `save_vector(fig, name, out_dir) -> dict{"pdf","png"}`, `callout(ax, xy, text, xytext, ...)`, `brace(ax, x0, x1, y, text, ...)`, `tag_stiff_sloppy(ax, stiff_xy, sloppy_xy)`, `load_captions() -> dict[str,str]`, `assemble_booklet(fig_dir, out_pdf, captions) -> int`, `out_dir(area) -> Path`, module-level `OUT_AREA`/`OUT_NAME` — names used consistently across tasks and the contract test.
