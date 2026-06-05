"""Backwards-compatible re-exports from diagnostic and bootstrap.

Prefer importing directly from:
    curvature_calib.calibration.diagnostic
    curvature_calib.calibration.bootstrap
"""
from curvature_calib.calibration.bootstrap import bootstrap_eigvals  # noqa: F401
from curvature_calib.calibration.diagnostic import (  # noqa: F401
    EigDecomp,
    eigendecompose,
    opg_from_grads,
    principal_angles,
)
