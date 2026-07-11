"""Nonlinear residual benchmarks for EXP-002 (local GGN validity).

Both use the squared-residual loss L(z)=½‖r(z)‖² (representation m=r, m_y=0,
W=I), so G=J_r^T J_r and the residual curvature R=Σ_k r_k ∇²r_k is analytic.
Coordinates are z∈R² directly (T=Id); float64.

Case A — Rosenbrock-type exact-fit curved valley:
    r1 = a(z2 − z1²),  r2 = b(1 − z1);   optimum ẑ=(1,1), r=0 ⇒ R=0 there.

Case B — irreducible overdetermined residual (nonzero-residual minimizer):
    r1 = z1 − 1,  r2 = z2 − 1,  r3 = λ(z1² + z2² − c);
    r1=r2=0 ⇒ z=(1,1) where r3=λ(2−c)≠0 for c≠2, so the minimizer has R≠0.
    Analytic R(z) = 2λ·r3(z)·I.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


# --------------------------------------------------------------------------- #
# Case A — Rosenbrock-type curved valley
# --------------------------------------------------------------------------- #
class RosenbrockResidual(NamedTuple):
    a: float = 5.0
    b: float = 1.0


def rosen_residual_fn(model: RosenbrockResidual) -> Callable[[Array], Array]:
    a, b = model.a, model.b
    return lambda z: jnp.array([a * (z[1] - z[0] ** 2), b * (1.0 - z[0])])


def rosen_loss_fn(model: RosenbrockResidual) -> Callable[[Array], Array]:
    r = rosen_residual_fn(model)
    return lambda z: 0.5 * jnp.dot(r(z), r(z))


def rosen_jacobian(model: RosenbrockResidual, z: Array) -> Array:
    a, b = model.a, model.b
    return jnp.array([[-2.0 * a * z[0], a], [-b, 0.0]], dtype=z.dtype)


def rosen_ggn(model: RosenbrockResidual, z: Array) -> Array:
    J = rosen_jacobian(model, z)
    return J.T @ J


def rosen_residual_curvature(model: RosenbrockResidual, z: Array) -> Array:
    a = model.a
    r1 = a * (z[1] - z[0] ** 2)
    hess_m1 = jnp.array([[-2.0 * a, 0.0], [0.0, 0.0]], dtype=z.dtype)  # ∇²r2 = 0
    return r1 * hess_m1


def rosen_hessian(model: RosenbrockResidual, z: Array) -> Array:
    return rosen_ggn(model, z) + rosen_residual_curvature(model, z)


def rosen_optimum(model: RosenbrockResidual, dtype=jnp.float64) -> Array:
    return jnp.array([1.0, 1.0], dtype=dtype)


# --------------------------------------------------------------------------- #
# Case B — irreducible nonlinear residual
# --------------------------------------------------------------------------- #
class IrreducibleResidual(NamedTuple):
    lam: float = 1.0
    c: float = 0.0


def irr_residual_fn(model: IrreducibleResidual) -> Callable[[Array], Array]:
    lam, c = model.lam, model.c
    return lambda z: jnp.array([z[0] - 1.0, z[1] - 1.0, lam * (z[0] ** 2 + z[1] ** 2 - c)])


def irr_loss_fn(model: IrreducibleResidual) -> Callable[[Array], Array]:
    r = irr_residual_fn(model)
    return lambda z: 0.5 * jnp.dot(r(z), r(z))


def irr_jacobian(model: IrreducibleResidual, z: Array) -> Array:
    lam = model.lam
    return jnp.array([[1.0, 0.0], [0.0, 1.0], [2.0 * lam * z[0], 2.0 * lam * z[1]]], dtype=z.dtype)


def irr_ggn(model: IrreducibleResidual, z: Array) -> Array:
    J = irr_jacobian(model, z)
    return J.T @ J


def irr_residual_curvature(model: IrreducibleResidual, z: Array) -> Array:
    lam, c = model.lam, model.c
    r3 = lam * (z[0] ** 2 + z[1] ** 2 - c)
    return 2.0 * lam * r3 * jnp.eye(2, dtype=z.dtype)  # ∇²r3 = 2λI; ∇²r1=∇²r2=0


def irr_hessian(model: IrreducibleResidual, z: Array) -> Array:
    return irr_ggn(model, z) + irr_residual_curvature(model, z)


# --------------------------------------------------------------------------- #
# Minimizer solver (small dense Newton on ∇L)
# --------------------------------------------------------------------------- #
def newton_minimize(loss_fn: Callable[[Array], Array], z0: Array,
                    n_iter: int = 100, tol: float = 1e-13) -> Array:
    """Damped-free Newton on the 2-D loss. Returns the stationary point."""
    grad = jax.grad(loss_fn)
    hess = jax.hessian(loss_fn)
    z = z0
    for _ in range(n_iter):
        g = grad(z)
        if float(jnp.linalg.norm(g)) < tol:
            break
        dz = jnp.linalg.solve(hess(z), g)
        z = z - dz
    return z
