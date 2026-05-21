from unittest.mock import Mock

import numpy as np

from lno327 import bdg_hamiltonian, dwave_pairing_matrix, spm_pairing_matrix


def test_spm_pairing_is_symmetric():
    delta = spm_pairing_matrix(0.4, 0.1)

    assert delta.shape == (4, 4)
    np.testing.assert_allclose(delta, delta.T)


def test_dwave_pairing_vanishes_on_zone_diagonal():
    delta = dwave_pairing_matrix(0.3, 0.3)

    np.testing.assert_allclose(delta, np.zeros((4, 4)), atol=1e-14)


def test_bdg_hamiltonian_is_hermitian():
    kx, ky = 0.2, -0.5
    delta = spm_pairing_matrix(kx, ky)
    bdg = bdg_hamiltonian(kx, ky, delta)

    assert bdg.shape == (8, 8)
    np.testing.assert_allclose(bdg, bdg.conjugate().T)


def test_bdg_hamiltonian_explicitly_builds_k_and_minus_k():
    normal_state = Mock(
        side_effect=[
            np.diag([1.0, 2.0, 3.0, 4.0]),
            np.diag([5.0, 6.0, 7.0, 8.0]),
        ]
    )
    delta = np.zeros((4, 4))

    bdg = bdg_hamiltonian(0.2, -0.5, delta, normal_state=normal_state)

    assert normal_state.call_args_list[0].args == (0.2, -0.5)
    assert normal_state.call_args_list[1].args == (-0.2, 0.5)
    np.testing.assert_allclose(np.diag(bdg[:4, :4]), [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(np.diag(-bdg[4:, 4:]), [5.0, 6.0, 7.0, 8.0])
