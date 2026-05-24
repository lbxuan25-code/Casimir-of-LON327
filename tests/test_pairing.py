from unittest.mock import Mock

import numpy as np
import pytest

from lno327 import (
    PairingAmplitudes,
    bdg_hamiltonian,
    dwave_pairing_matrix,
    pairing_matrix,
    spm_pairing_matrix,
)


def test_spm_pairing_is_interlayer_dz2_symmetric_and_in_ev():
    delta = spm_pairing_matrix(0.4, 0.1, PairingAmplitudes(delta0=0.025))

    assert delta.shape == (4, 4)
    np.testing.assert_allclose(
        delta,
        [
            [0.0, 0.0, 0.025, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.025, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
    )


def test_dwave_pairing_is_same_layer_interorbital_b1g():
    kx, ky = 0.2, -0.5
    delta0 = 0.03
    delta = dwave_pairing_matrix(kx, ky, PairingAmplitudes(delta0=delta0))
    expected_scale = delta0 * (np.cos(kx) + np.cos(ky))

    assert delta.shape == (4, 4)
    np.testing.assert_allclose(
        delta,
        expected_scale
        * np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ),
    )


def test_pairing_matrices_satisfy_even_spin_singlet_condition():
    kx, ky = 0.2, -0.5

    np.testing.assert_allclose(spm_pairing_matrix(kx, ky), spm_pairing_matrix(-kx, -ky).T)
    np.testing.assert_allclose(dwave_pairing_matrix(kx, ky), dwave_pairing_matrix(-kx, -ky).T)


def test_pairing_matrix_dispatches_supported_channels_only():
    kx, ky = 0.1, -0.2

    np.testing.assert_allclose(pairing_matrix("spm", kx, ky), spm_pairing_matrix(kx, ky))
    np.testing.assert_allclose(pairing_matrix("dwave", kx, ky), dwave_pairing_matrix(kx, ky))
    with pytest.raises(ValueError, match="pairing kind must be 'spm' or 'dwave'"):
        pairing_matrix("extended_s", kx, ky)  # type: ignore[arg-type]


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


def test_bdg_hamiltonian_rejects_invalid_pairing_shape():
    with pytest.raises(ValueError, match="pairing must be a 4x4 matrix"):
        bdg_hamiltonian(0.0, 0.0, np.zeros((2, 2)))
