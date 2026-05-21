import numpy as np

from lno327 import ORBITAL_BASIS, qiu_bilayer_hamiltonian


def test_qiu_basis_order():
    assert ORBITAL_BASIS == ("dz1", "dx1", "dz2", "dx2")


def test_qiu_hamiltonian_is_hermitian_and_four_band():
    h = qiu_bilayer_hamiltonian(0.37, -0.21)

    assert h.shape == (4, 4)
    np.testing.assert_allclose(h, h.conjugate().T)


def test_qiu_gamma_point_matches_appendix_coefficients_with_mu():
    h = qiu_bilayer_hamiltonian(0.0, 0.0)

    assert np.isclose(h[0, 0], -0.128 - 0.05)
    assert np.isclose(h[1, 1], -0.928 - 0.05)
    assert np.isclose(h[0, 2], -0.468)
    assert np.isclose(h[1, 3], 0.005)
    assert np.isclose(h[0, 1], 0.0)
