import jax
import pytest

from curvature_calib.config import enable_x64, require_x64, x64_enabled


def test_enable_x64_sets_flag():
    enable_x64()
    assert x64_enabled() is True


def test_default_dtype_is_float64_after_enable():
    enable_x64()
    import jax.numpy as jnp
    assert jnp.zeros(()).dtype == jnp.float64


def test_require_x64_raises_when_off():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError):
            require_x64()
    finally:
        jax.config.update("jax_enable_x64", prev)
