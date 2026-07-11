"""Precision configuration for curvature diagnostics.

float64 is required for all GGN / Hessian spectra (CLAUDE.md invariant: SIR
condition numbers reach ~1e13 and float32 corrupts the sloppy tail). x64 is a
global JAX flag and MUST be enabled before any significant array is constructed
or any function is traced/JIT-compiled (see EXP001_IMPLEMENTATION_PLAN §2).

The library does not force x64 at import (that would surprise importers).
Experiment entry points and precision-sensitive test fixtures call `enable_x64()`
first, then `require_x64()` as a guard.
"""
from __future__ import annotations

import jax


def enable_x64() -> None:
    """Enable float64 globally. Idempotent; call before creating arrays/tracing."""
    jax.config.update("jax_enable_x64", True)


def x64_enabled() -> bool:
    """True if JAX float64 is currently active."""
    return bool(jax.config.jax_enable_x64)


def require_x64() -> None:
    """Raise if float64 is not active. Guard for precision-sensitive entry points."""
    if not x64_enabled():
        raise RuntimeError(
            "float64 is required for curvature diagnostics but jax_enable_x64 is "
            "off. Call curvature_calib.config.enable_x64() before constructing "
            "arrays or tracing functions."
        )
