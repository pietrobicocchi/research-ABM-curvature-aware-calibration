"""The forward-mode Jacobian J_m of the mean-field SIR incidence agrees with central
finite differences."""
from gndiag.config import enable_x64, load_config

enable_x64()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from gndiag.models.mfsir import MeanFieldSIR  # noqa: E402


def test_jacobian_matches_central_differences():
    cfg = load_config("mfsir_model")
    cfg["horizon"] = 60
    sir = MeanFieldSIR(cfg)
    z = jnp.array([0.3, -0.2, 0.1, 0.4, -0.5])
    J = np.asarray(jax.jacfwd(sir.incidence)(z))
    h = 1e-5
    J_fd = np.stack([(np.asarray(sir.incidence(z + h * e)) - np.asarray(sir.incidence(z - h * e)))
                     / (2 * h) for e in jnp.eye(5)], axis=1)
    assert J.shape == (60, 5)
    np.testing.assert_allclose(J, J_fd, rtol=1e-6, atol=1e-6 * np.abs(J).max())
