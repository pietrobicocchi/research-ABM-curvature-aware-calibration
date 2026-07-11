"""Local Gauss-Newton geometry of calibration objectives.

The generalized Gauss-Newton (GGN) matrix of the calibrated representation
(Math-Spec §8):

    G(z) = J_m(z)* W J_m(z),    J_m(z) = D_z m(z).

See geometry/ggn.py. This is the project's primary object; the historical
per-seed scalar-loss OPG (calibration/per_seed_grads.py) is a distinct
comparison object and is NOT a GGN estimator (DEC-001, Math-Spec §14).
"""
from curvature_calib.geometry.ggn import (  # noqa: F401
    exact_hessian,
    finite_difference_hessian,
    ggn_dense,
    ggn_linear_operator,
    ggn_matvec,
    scalar_gradient_outer_product,
    symmetrize,
)
