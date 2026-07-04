import numpy as np
import pytest

from lno327.conductivity import KuboConfig, k_weights, uniform_bz_mesh
from lno327.response.normal_density_current import (
    normal_density_current_response_imag_axis as new_normal_density_current_response,
    normal_physical_density_current_response_components_imag_axis as new_physical_components,
    normal_physical_density_current_response_imag_axis as new_physical_response,
)
from lno327.ward_response import (
    normal_density_current_response_imag_axis as old_normal_density_current_response,
    normal_physical_density_current_response_components_imag_axis as old_physical_components,
    normal_physical_density_current_response_imag_axis as old_physical_response,
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
def test_normal_density_current_response_matches_old_reference(q, kwargs):
    points, weights, config = _inputs()

    new = new_normal_density_current_response(points, config, q, weights, **kwargs)
    old = old_normal_density_current_response(points, config, q, weights, **kwargs)

    np.testing.assert_allclose(new, old, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("q", (np.array([0.0, 0.0]), np.array([0.03, -0.02])))
def test_normal_physical_response_and_components_match_old_reference(q):
    points, weights, config = _inputs()

    new_response = new_physical_response(points, config, q, weights)
    old_response = old_physical_response(points, config, q, weights)
    np.testing.assert_allclose(new_response, old_response, rtol=1e-12, atol=1e-12)

    new_components = new_physical_components(points, config, q, weights)
    old_components = old_physical_components(points, config, q, weights)
    for key in ("bubble", "direct", "total"):
        np.testing.assert_allclose(new_components[key], old_components[key], rtol=1e-12, atol=1e-12)


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
