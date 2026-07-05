import numpy as np

from lno327 import (
    bdg_diamagnetic_kernel as facade_dia,
    bdg_paramagnetic_kernel_imag_axis as facade_para,
    bdg_superconducting_response_imag_axis as facade_sc,
    bdg_total_kernel_imag_axis as facade_total,
)
from lno327.response.config import KuboConfig
from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.models.symmetry_bdg_2band.spec import SymmetryBdG2BandSpec
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


def test_four_orbital_bdg_local_response_matches_root_facade_for_spm_and_dwave():
    config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4)
    spec = LNO327FourOrbitalSpec()

    for channel in ("spm", "dwave"):
        np.testing.assert_allclose(
            bdg_local_paramagnetic_kernel_imag_axis(
                spec,
                channel,
                _k_points(),
                config,
                _k_weights(),
            ),
            facade_para(_k_points(), config, channel, None, _k_weights()),
        )
        np.testing.assert_allclose(
            bdg_local_diamagnetic_kernel(
                spec,
                channel,
                _k_points(),
                config,
                _k_weights(),
            ),
            facade_dia(channel, None, _k_points(), config, _k_weights()),
        )

        facade_components = facade_total(_k_points(), config, channel, None, _k_weights())
        new_components = bdg_local_total_kernel_imag_axis(
            spec,
            channel,
            _k_points(),
            config,
            _k_weights(),
        )
        np.testing.assert_allclose(new_components.paramagnetic, facade_components.paramagnetic)
        np.testing.assert_allclose(new_components.diamagnetic, facade_components.diamagnetic)
        np.testing.assert_allclose(new_components.total, facade_components.total)

        facade_response = facade_sc(_k_points(), config, channel, None, _k_weights())
        new_response = bdg_local_superconducting_response_imag_axis(
            spec,
            channel,
            _k_points(),
            config,
            _k_weights(),
        )
        np.testing.assert_allclose(new_response.sigma_like_response, facade_response.sigma_like_response)


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


def test_bdg_local_total_kernel_shared_path_matches_separate_para_and_dia():
    config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4)
    spec = LNO327FourOrbitalSpec()

    for channel in ("spm", "dwave"):
        para = bdg_local_paramagnetic_kernel_imag_axis(spec, channel, _k_points(), config, _k_weights())
        dia = bdg_local_diamagnetic_kernel(spec, channel, _k_points(), config, _k_weights())
        total = bdg_local_total_kernel_imag_axis(spec, channel, _k_points(), config, _k_weights())

        np.testing.assert_allclose(total.paramagnetic, para)
        np.testing.assert_allclose(total.diamagnetic, dia)
        np.testing.assert_allclose(total.total, dia - para)
