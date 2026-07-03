import numpy as np

from lno327.conductivity import (
    KuboConfig as OldKuboConfig,
    kubo_conductivity_imag_axis as old_imag,
    kubo_conductivity_real_axis as old_real,
)
from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.models.symmetry_bdg_2band.spec import SymmetryBdG2BandSpec
from lno327.response.config import KuboConfig
from lno327.response.local_normal import (
    kubo_conductivity_imag_axis_from_model,
    kubo_conductivity_real_axis_from_model,
)


def _k_points() -> np.ndarray:
    return np.array([[0.1, -0.2], [0.4, 0.3], [-0.5, 0.25]], dtype=float)


def _k_weights() -> np.ndarray:
    return np.array([0.2, 0.3, 0.5], dtype=float)


def test_four_orbital_imag_axis_matches_legacy_with_si_output():
    old_config = OldKuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4, output_si=True)
    new_config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4, output_si=True)

    old = old_imag(_k_points(), old_config, _k_weights()).matrix()
    new = kubo_conductivity_imag_axis_from_model(
        LNO327FourOrbitalSpec(),
        _k_points(),
        new_config,
        _k_weights(),
    ).matrix()

    np.testing.assert_allclose(new, old)


def test_four_orbital_real_axis_matches_legacy_without_si_output():
    old_config = OldKuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4, output_si=False)
    new_config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4, output_si=False)

    old = old_real(_k_points(), old_config, _k_weights()).matrix()
    new = kubo_conductivity_real_axis_from_model(
        LNO327FourOrbitalSpec(),
        _k_points(),
        new_config,
        _k_weights(),
    ).matrix()

    np.testing.assert_allclose(new, old)


def test_symmetry_bdg_2band_normal_local_response_smoke():
    config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4, output_si=False)
    spec = SymmetryBdG2BandSpec()

    imag = kubo_conductivity_imag_axis_from_model(spec, _k_points(), config, _k_weights()).matrix()
    real = kubo_conductivity_real_axis_from_model(spec, _k_points(), config, _k_weights()).matrix()

    assert imag.shape == (2, 2)
    assert real.shape == (2, 2)
    assert np.all(np.isfinite(imag))
    assert np.all(np.isfinite(real))
