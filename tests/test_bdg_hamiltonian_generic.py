import numpy as np
import pytest

from lno327.bdg.hamiltonian import bdg_hamiltonian_from_blocks, bdg_hamiltonian_from_model_pairing
from lno327.models.lno327_four_orbital.bdg import bdg_hamiltonian as old_four_bdg
from lno327.models.lno327_four_orbital.normal import normal_state_hamiltonian
from lno327.models.lno327_four_orbital.pairing import pairing_matrix
from lno327.models.lno327_four_orbital.parameters import PairingAmplitudes
from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.models.symmetry_bdg_2band.spec import SymmetryBdG2BandSpec


def test_bdg_hamiltonian_from_blocks_matches_legacy_four_orbital():
    kx, ky = 0.21, -0.34
    delta = pairing_matrix("spm", kx, ky, PairingAmplitudes(delta0_eV=0.04))

    actual = bdg_hamiltonian_from_blocks(
        normal_state_hamiltonian(kx, ky),
        normal_state_hamiltonian(-kx, -ky),
        delta,
    )

    np.testing.assert_allclose(actual, old_four_bdg(kx, ky, delta))


def test_bdg_hamiltonian_from_model_pairing_matches_legacy_four_orbital():
    kx, ky = 0.21, -0.34
    amp = PairingAmplitudes(delta0_eV=0.04)
    spec = LNO327FourOrbitalSpec(pairing_amplitudes=amp)
    delta = spec.pairing_matrix(kx, ky, "dwave")

    actual = bdg_hamiltonian_from_model_pairing(spec, kx, ky, delta)

    np.testing.assert_allclose(actual, old_four_bdg(kx, ky, delta))


def test_bdg_hamiltonian_from_model_pairing_matches_two_band_spec():
    kx, ky = 0.21, -0.34
    spec = SymmetryBdG2BandSpec()
    delta = spec.pairing_matrix(kx, ky, "spm")

    actual = bdg_hamiltonian_from_model_pairing(spec, kx, ky, delta)

    np.testing.assert_allclose(actual, spec.bdg_hamiltonian(kx, ky, "spm"))


def test_bdg_hamiltonian_from_blocks_rejects_bad_shapes_and_nonhermitian_normal_blocks():
    h = np.eye(2, dtype=complex)
    delta = np.eye(2, dtype=complex)

    with pytest.raises(ValueError, match="same square shape"):
        bdg_hamiltonian_from_blocks(h, h, np.eye(3, dtype=complex))

    bad_normal = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    with pytest.raises(ValueError, match="normal_k must be Hermitian"):
        bdg_hamiltonian_from_blocks(bad_normal, h, delta)

    with pytest.raises(ValueError, match="normal_minus_k must be Hermitian"):
        bdg_hamiltonian_from_blocks(h, bad_normal, delta)
