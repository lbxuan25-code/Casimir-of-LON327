import numpy as np

from lno327 import ORBITAL_BASIS, ground_state_hamiltonian, ground_state_velocity_operator


def test_ground_state_basis_order():
    assert ORBITAL_BASIS == ("dz1", "dx1", "dz2", "dx2")


def test_ground_state_hamiltonian_is_hermitian_and_four_band():
    h = ground_state_hamiltonian(0.37, -0.21)

    assert h.shape == (4, 4)
    np.testing.assert_allclose(h, h.conjugate().T)


def test_gamma_point_matches_adopted_coefficients_with_mu():
    h = ground_state_hamiltonian(0.0, 0.0)

    assert np.isclose(h[0, 0], -0.128 - 0.05)
    assert np.isclose(h[1, 1], -0.928 - 0.05)
    assert np.isclose(h[0, 2], -0.468)
    assert np.isclose(h[1, 3], 0.005)
    assert np.isclose(h[0, 1], 0.0)


def test_velocity_operator_matches_finite_difference():
    kx, ky = 0.41, -0.23
    step = 1e-6
    finite_difference = (
        ground_state_hamiltonian(kx + step, ky) - ground_state_hamiltonian(kx - step, ky)
    ) / (2.0 * step)

    velocity = ground_state_velocity_operator(kx, ky, "x")

    np.testing.assert_allclose(velocity, finite_difference, rtol=1e-8, atol=1e-8)
