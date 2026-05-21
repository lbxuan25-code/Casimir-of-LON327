import numpy as np

from lno327 import bdg_hamiltonian, dwave_pairing_matrix, qiu_bilayer_hamiltonian, spm_pairing_matrix


def test_spm_pairing_is_symmetric():
    delta = spm_pairing_matrix(0.4, 0.1)

    assert delta.shape == (4, 4)
    np.testing.assert_allclose(delta, delta.T)


def test_dwave_pairing_vanishes_on_zone_diagonal():
    delta = dwave_pairing_matrix(0.3, 0.3)

    np.testing.assert_allclose(delta, np.zeros((4, 4)), atol=1e-14)


def test_bdg_hamiltonian_is_hermitian():
    kx, ky = 0.2, -0.5
    h = qiu_bilayer_hamiltonian(kx, ky)
    delta = spm_pairing_matrix(kx, ky)
    bdg = bdg_hamiltonian(kx, ky, delta, h)

    assert bdg.shape == (8, 8)
    np.testing.assert_allclose(bdg, bdg.conjugate().T)
