import numpy as np

from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec


def test_lno327_four_orbital_spec_constructs_and_reports_channels():
    spec = LNO327FourOrbitalSpec()

    assert spec.metadata().basis == ("dz1", "dx1", "dz2", "dx2")
    assert tuple(channel.name for channel in spec.channels()) == ("normal", "spm", "dwave")


def test_lno327_four_orbital_spec_matrix_shapes_and_zero_normal_pairing():
    spec = LNO327FourOrbitalSpec()
    kx, ky = 0.31, -0.27

    assert spec.normal_hamiltonian(kx, ky).shape == (4, 4)
    assert spec.pairing_matrix(kx, ky, "spm").shape == (4, 4)
    assert spec.pairing_matrix(kx, ky, "dwave").shape == (4, 4)
    np.testing.assert_allclose(spec.pairing_matrix(kx, ky, "normal"), np.zeros((4, 4), dtype=complex))
    assert spec.bdg_hamiltonian(kx, ky, "spm").shape == (8, 8)
    assert spec.bdg_hamiltonian(kx, ky, "dwave").shape == (8, 8)
    assert spec.velocity_operator(kx, ky, "x").shape == (4, 4)
    assert spec.mass_operator(kx, ky, "x", "y").shape == (4, 4)


def test_lno327_four_orbital_spec_hermiticity():
    spec = LNO327FourOrbitalSpec()
    kx, ky = 0.31, -0.27

    h = spec.normal_hamiltonian(kx, ky)
    bdg_spm = spec.bdg_hamiltonian(kx, ky, "spm")
    bdg_dwave = spec.bdg_hamiltonian(kx, ky, "dwave")

    np.testing.assert_allclose(h, h.conjugate().T)
    np.testing.assert_allclose(bdg_spm, bdg_spm.conjugate().T)
    np.testing.assert_allclose(bdg_dwave, bdg_dwave.conjugate().T)
