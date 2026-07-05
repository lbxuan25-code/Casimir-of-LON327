import numpy as np
import pytest

from lno327.response.config import KuboConfig
from lno327.numerics.weights import k_weights
from lno327.numerics.grids import uniform_bz_mesh
from lno327.response.normal_density_current import (
    normal_density_current_response_imag_axis as new_normal_density_current_response,
    normal_physical_density_current_response_components_imag_axis as new_physical_components,
    normal_physical_density_current_response_imag_axis as new_physical_response,
)


def _inputs():
    points = uniform_bz_mesh(2)
    return (
        points,
        k_weights(points),
        KuboConfig.from_kelvin(omega_eV=0.01, temperature_K=10.0, eta_eV=1e-8, output_si=False),
    )


@pytest.mark.parametrize("q", (np.array([0.0, 0.0]), np.array([0.03, -0.02])))
@pytest.mark.parametrize(
    "kwargs",
    (
        {"vertex_scheme": "midpoint", "contact_scheme": "none"},
        {"vertex_scheme": "peierls", "contact_scheme": "none"},
        {"vertex_scheme": "peierls", "contact_scheme": "finite_q_peierls", "contact_sign_convention": "plus"},
        {"vertex_scheme": "peierls", "contact_scheme": "finite_q_peierls", "contact_sign_convention": "minus"},
    ),
)
def test_normal_density_current_response_is_finite(q, kwargs):
    points, weights, config = _inputs()

    new = new_normal_density_current_response(points, config, q, weights, **kwargs)

    assert new.shape == (3, 3)
    assert np.all(np.isfinite(new))


@pytest.mark.parametrize("q", (np.array([0.0, 0.0]), np.array([0.03, -0.02])))
def test_normal_physical_response_and_components_are_consistent(q):
    points, weights, config = _inputs()

    new_response = new_physical_response(points, config, q, weights)
    new_components = new_physical_components(points, config, q, weights)
    for key in ("bubble", "direct", "total"):
        assert new_components[key].shape == (3, 3)
        assert np.all(np.isfinite(new_components[key]))
    np.testing.assert_allclose(new_response, new_components["total"])


def test_normal_density_current_rejects_invalid_inputs():
    points, weights, config = _inputs()
    q = np.array([0.03, -0.02])
    with pytest.raises(ValueError, match="vertex_scheme"):
        new_normal_density_current_response(points, config, q, weights, vertex_scheme="bad")
    with pytest.raises(ValueError, match="contact_scheme"):
        new_normal_density_current_response(points, config, q, weights, contact_scheme="bad")
    with pytest.raises(ValueError, match="contact_sign_convention"):
        new_normal_density_current_response(points, config, q, weights, contact_sign_convention="bad")
    with pytest.raises(ValueError, match="q must have shape"):
        new_normal_density_current_response(points, config, np.array([0.0]), weights)
    with pytest.raises(ValueError, match="k_points must have shape"):
        new_normal_density_current_response(np.empty((0, 2)), config, q, np.empty((0,)))


def test_physical_response_rejects_explicit_hamiltonian_fallback():
    points, weights, config = _inputs()

    def hamiltonian(kx, ky):
        return np.eye(4) * (kx + ky)

    with pytest.raises(ValueError, match="explicit hamiltonian fallback is not supported"):
        new_physical_components(points, config, np.array([0.03, -0.02]), weights, hamiltonian=hamiltonian)

    with pytest.raises(ValueError, match="explicit hamiltonian fallback is not supported"):
        new_physical_response(points, config, np.array([0.03, -0.02]), weights, hamiltonian=hamiltonian)
