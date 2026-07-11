# CLAUDE.md

Project standards and orientation for Claude Code. Loaded every session — keep it concise and current.

## What this is

Research code for the paper *"Curvature-Aware Calibration of Differentiable Agent-Based Models."*
The central contribution is a **diagnostic** for identifiability in differentiable ABMs: the
eigenstructure of the curvature matrix exposes which
parameter *combinations* are stiff (identifiable) vs sloppy (non-identifiable).

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

## Repo status (2026-07-11)

Directory was **cleared back to the library skeleton**: `src/` + `tests/` + packaging kept; all
`scripts/`, experiment outputs, `docs/superpowers/` plans+specs, booklets, and the old memory vault
were removed pending a substantial rewrite and a fresh standard. Do **not** assume old scripts or
figures exist. `main` and `origin/main` are at the pre-clean commit `e1b3a6f`; the clean is
uncommitted working-tree state.

## The Claude memory vault

Durable session memory lives in `docs/memory/` (gitignored; mirrored to `~/.claude/.../memory/`).
`docs/memory/MEMORY.md` is the index loaded each session — one line per memory. Add memories as
focused single-fact files with frontmatter (`type: user | feedback | project | reference`); link
related ones with `[[slug]]`. Keep durable facts; don't record transient run logs.
