"""Generalized Gauss-Newton (GGN) construction for calibration objectives.

Primary object (Math-Spec §8), computed in unconstrained, prior-scaled
coordinates z (DEC-003); the caller supplies a representation map already
expressed in z:

    G(z) = J_m(z)^T W J_m(z),    J_m(z) = D_z m(z) in R^{K x P}.

Public API (minimal — EXP001_IMPLEMENTATION_PLAN §4):
    ggn_dense(representation_fn, z, weight=None)          -> (P, P)
    ggn_matvec(representation_fn, z, vector, weight=None) -> (P,)
    exact_hessian(loss_fn, z)                             -> (P, P)
    finite_difference_hessian(loss_fn, z, step_size)      -> (P, P)
    scalar_gradient_outer_product(loss_fn, z)             -> (P, P)   [COMPARISON object]
    ggn_linear_operator(representation_fn, z, weight=None) -> Callable  [thin wrapper]

`weight` (the data metric W) is one of:
    None       -> identity;
    (K,) array -> diagonal;
    (K, K)     -> dense symmetric PSD.

The residual curvature R = H - G (Math-Spec §7) is computed inline by callers
(exact_hessian - ggn_dense); it is zero for affine m.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

Array = jax.Array

# Tolerance for structural checks on the weight matrix (symmetry / PSD).
_WEIGHT_TOL = 1e-8


def symmetrize(A: Array) -> Array:
    """0.5 * (A + A^T)."""
    return 0.5 * (A + A.T)


def _validate_weight(weight: Array | None, K: int, dtype) -> None:
    """Validate shape, dtype, finiteness, symmetry, and PSD of the data metric W."""
    if weight is None:
        return
    if weight.dtype != dtype:
        raise ValueError(
            f"weight dtype {weight.dtype} does not match representation dtype {dtype}"
        )
    if not bool(jnp.all(jnp.isfinite(weight))):
        raise ValueError("weight contains non-finite entries")
    if weight.ndim == 1:
        if weight.shape != (K,):
            raise ValueError(f"diagonal weight must have shape ({K},), got {weight.shape}")
        if bool(jnp.any(weight < -_WEIGHT_TOL)):
            raise ValueError("diagonal weight must be PSD (non-negative entries)")
    elif weight.ndim == 2:
        if weight.shape != (K, K):
            raise ValueError(f"dense weight must have shape ({K}, {K}), got {weight.shape}")
        asym = float(jnp.max(jnp.abs(weight - weight.T)))
        if asym > _WEIGHT_TOL:
            raise ValueError(f"dense weight must be symmetric (max |W - W^T| = {asym:.2e})")
        min_eig = float(jnp.linalg.eigvalsh(symmetrize(weight))[0])
        if min_eig < -_WEIGHT_TOL:
            raise ValueError(f"dense weight must be PSD (min eigenvalue = {min_eig:.2e})")
    else:
        raise ValueError(f"weight must be None, (K,), or (K, K); got ndim {weight.ndim}")


def _apply_weight(weight: Array | None, u: Array) -> Array:
    """Apply W to u, where u is (K,) or (K, P). None -> identity."""
    if weight is None:
        return u
    if weight.ndim == 1:
        return weight * u if u.ndim == 1 else weight[:, None] * u
    return weight @ u


def _representation_jacobian(representation_fn: Callable[[Array], Array], z: Array) -> Array:
    """D_z m(z) as an explicit (K, P) matrix. jacfwd if K>=P else jacrev."""
    m0 = representation_fn(z)
    K = m0.shape[0]
    P = z.shape[0]
    jac = jax.jacfwd if K >= P else jax.jacrev
    return jac(representation_fn)(z)


def ggn_dense(representation_fn: Callable[[Array], Array], z: Array,
              weight: Array | None = None) -> Array:
    """Explicit GGN  G = J^T W J  in R^{P x P}. PSD; symmetrized on return."""
    J = _representation_jacobian(representation_fn, z)  # (K, P)
    _validate_weight(weight, J.shape[0], z.dtype)
    return symmetrize(J.T @ _apply_weight(weight, J))


def ggn_matvec(representation_fn: Callable[[Array], Array], z: Array, vector: Array,
               weight: Array | None = None) -> Array:
    """Matrix-free GGN product  G v = J^T W (J v)  without materializing J."""
    m0, Jv = jax.jvp(representation_fn, (z,), (vector,))   # Jv: (K,)
    _validate_weight(weight, m0.shape[0], z.dtype)
    WJv = _apply_weight(weight, Jv)                        # (K,)
    _, vjp_fn = jax.vjp(representation_fn, z)
    (out,) = vjp_fn(WJv)                                   # (P,)
    return out


def ggn_linear_operator(representation_fn: Callable[[Array], Array], z: Array,
                        weight: Array | None = None) -> Callable[[Array], Array]:
    """Thin wrapper: returns v -> ggn_matvec(...). For Lanczos/eigsh later."""
    return lambda vector: ggn_matvec(representation_fn, z, vector, weight)


def exact_hessian(loss_fn: Callable[[Array], Array], z: Array) -> Array:
    """Exact Hessian of the scalar loss, symmetrized. Equals GGN for affine m."""
    return symmetrize(jax.hessian(loss_fn)(z))


def finite_difference_hessian(loss_fn: Callable[[Array], Array], z: Array,
                              step_size: float) -> Array:
    """Central-difference Hessian from AD gradients. `step_size` is required.

    Separate, looser accuracy regime than AD (EXP001_IMPLEMENTATION_PLAN §3):
    do not apply the AD tolerance to this.
    """
    grad = jax.grad(loss_fn)
    P = z.shape[0]
    eye = jnp.eye(P, dtype=z.dtype)

    def column(i):
        gp = grad(z + step_size * eye[i])
        gm = grad(z - step_size * eye[i])
        return (gp - gm) / (2.0 * step_size)

    H = jax.vmap(column)(jnp.arange(P))  # (P, P); row i is d grad / d z_i
    return symmetrize(H)


def scalar_gradient_outer_product(loss_fn: Callable[[Array], Array], z: Array) -> Array:
    """COMPARISON object (DEC-001), NOT a curvature estimate.

    g g^T for the scalar-loss gradient g = grad L(z). At an exact fit g = 0 so
    this is the zero matrix, while the GGN G = J^T W J may remain nonzero. This
    is the concrete counterexample to identifying the OPG with the GGN
    (Math-Spec §14).
    """
    g = jax.grad(loss_fn)(z)
    return jnp.outer(g, g)
