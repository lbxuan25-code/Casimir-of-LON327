import numpy as np
import pytest

from lno327.conductivity import (
    bosonic_matsubara_energy_eV as old_matsubara,
    k_weights as old_k_weights,
    uniform_bz_mesh as old_mesh,
)
from lno327.numerics import bosonic_matsubara_energy_eV, k_weights, uniform_bz_mesh


def test_bosonic_matsubara_energy_matches_legacy():
    assert bosonic_matsubara_energy_eV(3, 12.5) == old_matsubara(3, 12.5)


@pytest.mark.parametrize("n, temperature", [(-1, 10.0), (0, -1.0)])
def test_bosonic_matsubara_energy_exceptions_match_legacy(n, temperature):
    with pytest.raises(ValueError):
        old_matsubara(n, temperature)
    with pytest.raises(ValueError):
        bosonic_matsubara_energy_eV(n, temperature)


def test_uniform_bz_mesh_matches_legacy():
    np.testing.assert_allclose(uniform_bz_mesh(3, 2), old_mesh(3, 2))
    np.testing.assert_allclose(uniform_bz_mesh(3), old_mesh(3))


@pytest.mark.parametrize("nkx, nky", [(0, None), (2, 0)])
def test_uniform_bz_mesh_exceptions_match_legacy(nkx, nky):
    with pytest.raises(ValueError):
        old_mesh(nkx, nky)
    with pytest.raises(ValueError):
        uniform_bz_mesh(nkx, nky)


def test_k_weights_matches_legacy():
    points = uniform_bz_mesh(2)

    np.testing.assert_allclose(k_weights(points), old_k_weights(points))


def test_k_weights_exception_matches_legacy():
    bad_points = np.array([0.0, 1.0])

    with pytest.raises(ValueError):
        old_k_weights(bad_points)
    with pytest.raises(ValueError):
        k_weights(bad_points)
