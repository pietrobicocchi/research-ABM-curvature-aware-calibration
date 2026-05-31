# curvature-aware-calibration

Curvature-aware calibration of differentiable agent-based models via the eigenstructure of the per-seed gradient outer-product matrix.

## Setup

```bash
uv sync --extra dev
```

## Run tests

```bash
uv run pytest
```

## Layout

- `src/curvature_calib/` — package
- `tests/` — pytest suite
- `docs/memory/` — Claude session notes (gitignored, symlinked from `~/.claude/`)
