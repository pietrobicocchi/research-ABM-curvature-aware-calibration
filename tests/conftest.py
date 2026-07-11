"""Test-time precision scoping.

float64 (x64) is a PROCESS-GLOBAL JAX flag. The geometry tests need it, but the
historical Brock-Hommes / SIR / calibrate / MMD tests were written for the
float32 default and change behavior under x64. To avoid cross-test
contamination, enable x64 ONLY for the geometry-test modules and restore the
previous value after each of their tests.
"""
import jax
import pytest

# Modules that require float64. Everything else runs in the default precision.
_X64_MODULES = {
    "test_config",
    "test_metrics",
    "test_linear_gaussian",
    "test_ggn",
}


@pytest.fixture(autouse=True)
def _x64_for_geometry_tests(request):
    name = request.module.__name__.split(".")[-1]
    if name not in _X64_MODULES:
        yield
        return
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)
