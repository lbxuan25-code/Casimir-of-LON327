import numpy as np

from lno327.bdg.spectrum import (
    bdg_eigensystem_from_model,
    normal_eigensystem_from_model,
    transform_operator_to_band_basis,
)
from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.models.symmetry_bdg_2band.spec import SymmetryBdG2BandSpec


def _assert_particle_hole_symmetric(energies: np.ndarray) -> None:
    np.testing.assert_allclose(energies, -energies[::-1], atol=1e-10)


def test_symmetry_bdg_2band_normal_and_bdg_spectra_are_finite():
    spec = SymmetryBdG2BandSpec()
    kx, ky = 0.23, -0.35

    normal = normal_eigensystem_from_model(spec, kx, ky)
    assert normal.energies.shape == (2,)
    assert normal.states.shape == (2, 2)
    assert np.all(np.isfinite(normal.energies))

    for channel in ("spm", "dwave"):
        bdg = bdg_eigensystem_from_model(spec, kx, ky, channel)
        assert bdg.energies.shape == (4,)
        assert bdg.states.shape == (4, 4)
        assert np.all(np.isfinite(bdg.energies))
        _assert_particle_hole_symmetric(bdg.energies)


def test_lno327_four_orbital_normal_and_bdg_spectra_are_finite():
    spec = LNO327FourOrbitalSpec()
    kx, ky = 0.23, -0.35

    normal = normal_eigensystem_from_model(spec, kx, ky)
    assert normal.energies.shape == (4,)
    assert normal.states.shape == (4, 4)
    assert np.all(np.isfinite(normal.energies))

    for channel in ("spm", "dwave"):
        bdg = bdg_eigensystem_from_model(spec, kx, ky, channel)
        assert bdg.energies.shape == (8,)
        assert bdg.states.shape == (8, 8)
        assert np.all(np.isfinite(bdg.energies))
        _assert_particle_hole_symmetric(bdg.energies)


def test_transform_operator_to_band_basis_matches_direct_formula():
    spec = SymmetryBdG2BandSpec()
    bands = normal_eigensystem_from_model(spec, 0.23, -0.35)
    operator = spec.velocity_operator(0.23, -0.35, "x")

    transformed = transform_operator_to_band_basis(bands.states, operator)

    np.testing.assert_allclose(transformed, bands.states.conjugate().T @ operator @ bands.states)
