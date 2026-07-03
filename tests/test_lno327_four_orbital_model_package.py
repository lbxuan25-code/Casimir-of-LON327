import numpy as np

from lno327.models.lno327_four_orbital import (
    ORBITAL_BASIS,
    LNO327FourOrbitalSpec,
    normal_state_hamiltonian,
    normal_state_mass_operator,
    normal_state_velocity_operator,
    pairing_matrix,
)


def test_four_orbital_package_exports_existing_model_pieces():
    assert ORBITAL_BASIS == ("dz1", "dx1", "dz2", "dx2")
    assert isinstance(LNO327FourOrbitalSpec(), LNO327FourOrbitalSpec)


def test_four_orbital_package_functions_have_expected_shapes():
    kx, ky = 0.31, -0.27

    assert normal_state_hamiltonian(kx, ky).shape == (4, 4)
    assert pairing_matrix("spm", kx, ky).shape == (4, 4)
    assert pairing_matrix("dwave", kx, ky).shape == (4, 4)
    assert normal_state_velocity_operator(kx, ky, "x").shape == (4, 4)
    assert normal_state_mass_operator(kx, ky, "x", "y").shape == (4, 4)


def test_four_orbital_package_normal_hamiltonian_is_hermitian():
    h = normal_state_hamiltonian(0.31, -0.27)

    np.testing.assert_allclose(h, h.conjugate().T)
