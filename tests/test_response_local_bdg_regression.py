import numpy as np

from lno327.bdg_response import (
    bdg_diamagnetic_kernel as old_dia,
    bdg_paramagnetic_kernel_imag_axis as old_para,
    bdg_superconducting_response_imag_axis as old_sc,
    bdg_total_kernel_imag_axis as old_total,
)
from lno327.conductivity import KuboConfig as OldKuboConfig
from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.models.symmetry_bdg_2band.spec import SymmetryBdG2BandSpec
from lno327.response.config import KuboConfig
from lno327.response.local_bdg import (
    bdg_local_diamagnetic_kernel,
    bdg_local_paramagnetic_kernel_imag_axis,
    bdg_local_superconducting_response_imag_axis,
    bdg_local_total_kernel_imag_axis,
)


def _k_points() -> np.ndarray:
    return np.array([[0.1, -0.2], [0.4, 0.3], [-0.5, 0.25]], dtype=float)


def _k_weights() -> np.ndarray:
    return np.array([0.2, 0.3, 0.5], dtype=float)


def test_four_orbital_bdg_local_response_matches_legacy_for_spm_and_dwave():
    old_config = OldKuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4)
    new_config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4)
    spec = LNO327FourOrbitalSpec()

    for channel in ("spm", "dwave"):
        np.testing.assert_allclose(
            bdg_local_paramagnetic_kernel_imag_axis(
                spec,
                channel,
                _k_points(),
                new_config,
                _k_weights(),
            ),
            old_para(_k_points(), old_config, channel, None, _k_weights()),
        )
        np.testing.assert_allclose(
            bdg_local_diamagnetic_kernel(
                spec,
                channel,
                _k_points(),
                new_config,
                _k_weights(),
            ),
            old_dia(channel, None, _k_points(), old_config, _k_weights()),
        )

        old_components = old_total(_k_points(), old_config, channel, None, _k_weights())
        new_components = bdg_local_total_kernel_imag_axis(
            spec,
            channel,
            _k_points(),
            new_config,
            _k_weights(),
        )
        np.testing.assert_allclose(new_components.paramagnetic, old_components.paramagnetic)
        np.testing.assert_allclose(new_components.diamagnetic, old_components.diamagnetic)
        np.testing.assert_allclose(new_components.total, old_components.total)

        old_response = old_sc(_k_points(), old_config, channel, None, _k_weights())
        new_response = bdg_local_superconducting_response_imag_axis(
            spec,
            channel,
            _k_points(),
            new_config,
            _k_weights(),
        )
        np.testing.assert_allclose(new_response.sigma_like_response, old_response.sigma_like_response)


def test_symmetry_bdg_2band_bdg_local_response_smoke_for_spm_and_dwave():
    config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4)
    spec = SymmetryBdG2BandSpec()

    for channel in ("spm", "dwave"):
        para = bdg_local_paramagnetic_kernel_imag_axis(spec, channel, _k_points(), config, _k_weights())
        dia = bdg_local_diamagnetic_kernel(spec, channel, _k_points(), config, _k_weights())
        total = bdg_local_total_kernel_imag_axis(spec, channel, _k_points(), config, _k_weights())
        response = bdg_local_superconducting_response_imag_axis(
            spec,
            channel,
            _k_points(),
            config,
            _k_weights(),
        )

        assert para.shape == (2, 2)
        assert dia.shape == (2, 2)
        assert total.total.shape == (2, 2)
        assert response.sigma_like_response.shape == (2, 2)
        assert np.all(np.isfinite(para))
        assert np.all(np.isfinite(dia))
        assert np.all(np.isfinite(total.total))
        assert np.all(np.isfinite(response.sigma_like_response))
        np.testing.assert_allclose(total.total, dia - para)
