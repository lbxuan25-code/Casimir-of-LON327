import numpy as np
import pytest

from lno327.numerics import bosonic_matsubara_energy_eV, k_weights, uniform_bz_mesh


def test_bosonic_matsubara_energy_uses_ev_units():
    np.testing.assert_allclose(
        bosonic_matsubara_energy_eV(3, 12.5),
        2.0 * np.pi * 3 * 12.5 * 8.617333262145e-5,
    )


@pytest.mark.parametrize("n, temperature", [(-1, 10.0), (0, -1.0)])
def test_bosonic_matsubara_energy_exceptions(n, temperature):
    with pytest.raises(ValueError):
        bosonic_matsubara_energy_eV(n, temperature)


def test_uniform_bz_mesh_shape_and_bounds():
    mesh = uniform_bz_mesh(3, 2)
    assert mesh.shape == (6, 2)
    assert np.all(mesh >= -np.pi)
    assert np.all(mesh < np.pi)


@pytest.mark.parametrize("nkx, nky", [(0, None), (2, 0)])
def test_uniform_bz_mesh_exceptions(nkx, nky):
    with pytest.raises(ValueError):
        uniform_bz_mesh(nkx, nky)


def test_k_weights_are_normalized():
    points = uniform_bz_mesh(2)

    np.testing.assert_allclose(k_weights(points), np.full(4, 0.25))


def test_k_weights_exception():
    bad_points = np.array([0.0, 1.0])

    with pytest.raises(ValueError):
        k_weights(bad_points)
