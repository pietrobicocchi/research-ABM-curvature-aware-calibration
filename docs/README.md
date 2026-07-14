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

## Two layers, two agents

The project runs on **two agents + a gatekeeper** (full map:
`paper/OPERATING_MODEL.md`):

- **Implementation specialist** (the coding agent, driven by repo-root `CLAUDE.md`)
  — writes code, runs experiments, manages provenance. Edits **Layer 1 only**.
- **Scientific lead & editor** (a strategic conversation) — question → experiment
  design → interpretation → paper argument. Owns **Layer 2**.
- **Pietro** — the gatekeeper who carries results between them and promotes accepted
  conclusions into Layer 2.

The docs mirror this split:

- **Layer 1 — internal research memory** (`docs/` root): may contain corrections,
  false starts, negative results. The full history of how decisions were reached.
- **Layer 2 — paper-facing memory** (`docs/paper/`): only the *accepted* scientific
  state, from which the manuscript is written. A curated projection of Layer 1 — it
  references Layer-1 facts by ID, never invents them, never carries the correction
  chronology.

## How the docs are organized

### Layer 1 — internal research memory (`docs/` root)

| File | Role | Changes |
|---|---|---|
| `README.md` (this) | Orientation + the recording rule | rarely |
| `STATUS.md` | **Bridge** — where we are: phase, proven, next | every experiment |
| `CLAIMS.md` | The one claims ledger — each claim, status, evidence | every experiment |
| `MATH.md` | The corrected mathematical formulation (authoritative) | rarely |
| `DECISIONS.md` | Dated decisions + rejections (DEC-001 …) | on a real decision |
| `experiments/EXP-*.md` | One short file per experiment: question · method · result · claim impact | when run |
| `papers/` | Literature the project relies on | when a paper is studied |

### Layer 2 — paper-facing memory (`docs/paper/`)

| File | Role | Changes |
|---|---|---|
| `paper/OPERATING_MODEL.md` | The two-agent model, layers, working cycle | rarely |
| `paper/STRATEGIC_HANDOFF.md` | Fresh strategic-session entrypoint: thesis, interpretation, next | per accepted result |
| `paper/PAPER_ARCHITECTURE.md` | Claim–evidence manuscript outline | on structural change |
| `paper/WRITING_BRIEF.md` | Accepted narrative · terminology · limitations · style | per accepted result |
| `paper/EVIDENCE_MAP.md` | Claim → evidence → figure → section | per accepted result |
| `paper/FIGURE_LEDGER.md` | Approved figures + provenance | per approved figure |

`archive/` is superseded/verbose material (old plans, reviews, retired vault) —
**frozen, never cited as current state.**

**Authority order** when sources disagree: `MATH.md` → `DECISIONS.md` →
`CLAIMS.md` → `STATUS.md` / `experiments/`. Layer 2 (`paper/`) is **derived** — if
it ever disagrees with Layer 1, Layer 1 wins and Layer 2 is corrected.

## The recording rule (definition of done)

Each agent has its own definition of done, so the layers stay in sync.

**Implementation specialist — per experiment, update exactly three Layer-1 places:**

1. **`experiments/EXP-xxx.md`** — fill in the result (the file already exists from
   when the experiment was planned; use `experiments/TEMPLATE.md`).
2. **`CLAIMS.md`** — set the affected claim's status + one-line evidence link.
3. **`STATUS.md`** — advance the phase / set the next experiment.

Nothing in `paper/`; no narrative, no claim promotion, no figure approval.

**Scientific lead — per accepted result:** promote wording in `paper/EVIDENCE_MAP.md`,
approve the figure in `paper/FIGURE_LEDGER.md`, advance `paper/STRATEGIC_HANDOFF.md`
(occasionally `paper/PAPER_ARCHITECTURE.md` / `paper/WRITING_BRIEF.md`).

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
