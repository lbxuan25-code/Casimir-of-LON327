import numpy as np

from lno327.models.symmetry_bdg_2band.observables import (
    band_energies_on_path,
    bdg_energies,
    gap_value,
    min_positive_bdg_energy,
    normal_band_energies,
)
from lno327.models.symmetry_bdg_2band.spec import SymmetryBdG2BandSpec


def test_symmetry_bdg_2band_spec_metadata_and_channels():
    spec = SymmetryBdG2BandSpec()

    metadata = spec.metadata()
    assert metadata.name == "symmetry_bdg_2band"
    assert metadata.basis == ("a", "b")
    assert metadata.description == "Symmetry-focused two-band BdG model"
    assert tuple(channel.name for channel in spec.channels()) == ("normal", "spp", "spm", "dwave")


def test_symmetry_bdg_2band_spec_shapes_and_hermiticity():
    spec = SymmetryBdG2BandSpec()
    kx, ky = 0.37, -0.22

    h = spec.normal_hamiltonian(kx, ky)
    assert h.shape == (2, 2)
    np.testing.assert_allclose(h, h.conjugate().T)
    for channel in ("normal", "spp", "spm", "dwave"):
        delta = spec.pairing_matrix(kx, ky, channel)
        bdg = spec.bdg_hamiltonian(kx, ky, channel)
        assert delta.shape == (2, 2)
        assert bdg.shape == (4, 4)
        np.testing.assert_allclose(bdg, bdg.conjugate().T)
    assert spec.velocity_operator(kx, ky, "x").shape == (2, 2)
    assert spec.mass_operator(kx, ky, "x", "y").shape == (2, 2)


def test_symmetry_bdg_2band_observables_return_expected_shapes():
    spec = SymmetryBdG2BandSpec()
    kx, ky = 0.37, -0.22
    k_path = np.array([[0.0, 0.0], [kx, ky], [np.pi, 0.0]])

    assert normal_band_energies(kx, ky, spec).shape == (2,)
    assert band_energies_on_path(spec, k_path).shape == (3, 2)
    assert bdg_energies(kx, ky, "spp", spec).shape == (4,)
    assert min_positive_bdg_energy(kx, ky, "spp", spec) >= 0.0
    assert gap_value(kx, ky, "dwave", spec).shape == (2, 2)
