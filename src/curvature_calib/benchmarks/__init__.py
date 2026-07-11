"""Analytic benchmarks with known geometry, for validating the GGN machinery."""
from curvature_calib.benchmarks.linear_gaussian import (  # noqa: F401
    LinearGaussian,
    analytic_ggn,
    analytic_gradient,
    make_loss,
    make_representation,
    random_A,
)
