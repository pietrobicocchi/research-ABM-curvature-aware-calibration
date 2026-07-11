"""Backwards-compatible re-exports from diagnostic and bootstrap.

Prefer importing directly from:
    curvature_calib.calibration.diagnostic
    curvature_calib.calibration.bootstrap

Naming note (DEC-001): the per-seed scalar-gradient second moment is
`scalar_gradient_opg`; `opg_from_grads` is a deprecated alias. Neither is the
GGN — see diagnostic.scalar_gradient_opg / geometry.ggn.
"""
from curvature_calib.calibration.bootstrap import bootstrap_eigvals  # noqa: F401
from curvature_calib.calibration.diagnostic import (  # noqa: F401
    EigDecomp,
    eigendecompose,
    opg_from_grads,  # deprecated alias
    principal_angles,
    scalar_gradient_covariance,
    scalar_gradient_opg,
)
