# Local Information Geometry of Differentiable ABMs

Front door for the project's documentation. Start here.

## What this project is

Calibration tells us *which parameter values* fit data. This project answers the
deeper question: **which combinations of parameters are actually informed by the
observations?**

The tool is the local **generalized Gauss–Newton (GGN) geometry** of the
calibration objective. For a calibrated representation `m(z)` and loss
`L(z) = ½‖m(z) − m_y‖²_W`,

    G(z) = Dm(z)* W Dm(z).

Its eigenvectors are parameter *combinations*; its eigenvalues say how strongly
the calibrated representation moves along each. The exact Hessian is `H = G + R`,
so `G` is trustworthy only where the residual-curvature `R` is small. We work in
prior-scaled coordinates `z` so eigenvalues are comparable to prior precision.

**Why not gradients?** A gradient says which direction reduces the loss; it does
not reveal which *combinations* are locally constrained. The earlier project used
a per-seed scalar-gradient second moment (`F_OPG`) and mistook it for curvature —
this project rejects that identification (see `DECISIONS.md` DEC-001) and studies
the true GGN.

**Models:** Brock–Hommes (financial ABM), SIR / network-SIR (epidemic).

## How the docs are organized

| File | Role | Changes |
|---|---|---|
| `README.md` (this) | Orientation + the recording rule | rarely |
| `STATUS.md` | Where we are: phase, what's proven, what's next | every experiment |
| `CLAIMS.md` | Each claim, its status, evidence link | every experiment |
| `MATH.md` | The corrected mathematical formulation (authoritative) | rarely |
| `DECISIONS.md` | Dated decisions + rejections (DEC-001 …) | on a real decision |
| `experiments/EXP-*.md` | One short file per experiment: question · method · result · claim impact | when run |
| `papers/` | Literature the project relies on | when a paper is studied |
| `archive/` | Superseded/verbose material (old plans, reviews, retired vault). Frozen. | never |

**Authority order** when sources disagree: `MATH.md` → `DECISIONS.md` →
`CLAIMS.md` → `STATUS.md` / `experiments/`.

## The recording rule (definition of done)

An experiment updates **exactly three places** — nothing else:

1. **`experiments/EXP-xxx.md`** — fill in the result (the file already exists from
   when the experiment was planned; use `experiments/TEMPLATE.md`).
2. **`CLAIMS.md`** — set the affected claim's status + one-line evidence link.
3. **`STATUS.md`** — advance the phase / set the next experiment.

Anything longer (derivations, full run logs) belongs in the experiment file or in
the machine run record under `outputs/EXP-xxx/<run-id>/` (gitignored) — never a
new top-level doc. Studying a paper adds one file under `papers/` and a line in
`papers/README.md`.

## Status vocabulary

`proposed` · `in progress` · `supported` · `supported conditionally` ·
`not supported` · `rejected`.

## Code & runs

- Library: `src/curvature_calib/`. Experiments: `experiments/exp0XX_*.py`.
- Tests: `uv run pytest`. Precision: float64 (`config.enable_x64()` at entry).
- Every experiment writes a provenance-stamped run record to
  `outputs/EXP-xxx/<run-id>/` (gitignored). Authoritative runs come from a clean
  commit (`git_dirty: false`).
