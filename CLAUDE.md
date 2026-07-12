# CLAUDE.md

Project standards and orientation for Claude Code. Loaded every session — keep it concise and current.

## What this is

Research code studying the local generalized Gauss–Newton (GGN) geometry of
calibration objectives for differentiable ABMs — a diagnostic for which parameter
*combinations* are locally informed by the data. The GGN is `G = Dm* W Dm`; the
per-seed scalar-gradient OPG is **not** the GGN (DEC-001). Read `docs/README.md`
first, then `docs/STATUS.md`.

## Setup & tests

```bash
uv sync --extra dev      # install (Python 3.12, uv-managed)
uv run pytest            # ~2 min (JAX)
```

## Technical invariants (do not silently break)

- **float64 everywhere for diagnostics:** set `jax.config.update("jax_enable_x64", True)` before
  importing `jax.numpy` in any script that computes an OPG spectrum. SIR condition numbers reach
  ~10¹³; float32 corrupts the sloppy tail.
- **JAX pin:** `jax==jaxlib==0.4.30` (Intel-Mac x86_64 wheel constraint). Unpin on Apple Silicon.
- **Run via `uv run`.** Deps live in `pyproject.toml`.


## Package layout (the surviving skeleton)

```
src/curvature_calib/
  models/       brock_hommes.py, sir.py, network_sir.py, surrogates.py
  losses/       mmd.py                     # unbiased MMD² + median-heuristic bandwidth
  calibration/  per_seed_grads.py          # VJP per-seed grads -> CalibStats(loss, mean_grad, per_seed_grads, opg)
                diagnostic.py              # eigendecompose, principal_angles, effective_dimension
                bootstrap.py               # bootstrap_eigvals, eigenvalue_cis, noise_threshold
                falsification.py           # moments/acf/quantile differences, run_falsification
                calibrate.py, baselines.py, preconditioner.py, jacobian_sensitivity.py
                opg.py                     # backwards-compat re-export shim
  viz/          style.py                   # shared palette + rcParams
tests/          one test_*.py per module
```

For a pedagogical end-to-end walkthrough, see `docs/papers/brock_hommes_code_guide.md`.

## Docs & recording discipline (READ THIS)

Documentation is deliberately lean. The living docs are: `docs/README.md` (front
door + this rule), `docs/STATUS.md` (where we are), `docs/CLAIMS.md` (what's
proven), `docs/MATH.md` (authoritative formulation), `docs/DECISIONS.md`
(decisions/rejections), `docs/experiments/EXP-*.md` (one short file per
experiment), `docs/papers/` (literature). `docs/archive/` is frozen — never cite it
as current state.

**Definition of done for an experiment — update exactly three places, nothing else:**
1. `docs/experiments/EXP-xxx.md` (result; use `experiments/TEMPLATE.md`);
2. `docs/CLAIMS.md` (claim status + evidence link);
3. `docs/STATUS.md` (phase / next).
Long detail → the experiment file or the gitignored `outputs/EXP-xxx/<run-id>/`
run record. Studying a paper → one file in `docs/papers/` + a line in its README.
Do **not** create new top-level docs or an `inbox/`.

## Memory vault

Durable session memory lives in `docs/memory/` (gitignored; mirrored to
`~/.claude/.../memory/`); `docs/memory/MEMORY.md` is the index. Add focused
single-fact files (`type: user | feedback | project | reference`). Durable facts
only — no transient run logs.
