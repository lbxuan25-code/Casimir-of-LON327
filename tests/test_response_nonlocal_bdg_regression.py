import numpy as np
import pytest

from lno327.bdg_nonlocal_response import (
    bdg_current_current_kernel_imag_axis as old_kernel,
    midpoint_bdg_current_vertex as old_midpoint_vertex,
    shifted_bdg_eigensystem as old_shifted,
)
from lno327.conductivity import KuboConfig as OldKuboConfig
from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.models.symmetry_bdg_2band.spec import SymmetryBdG2BandSpec
from lno327.response.config import KuboConfig
from lno327.response.nonlocal_bdg import (
    bdg_current_current_kernel_imag_axis_from_model,
    midpoint_bdg_current_vertex_from_model,
    shifted_bdg_eigensystem_from_model,
)


def _k_points() -> np.ndarray:
    return np.array([[0.1, -0.2], [0.4, 0.3], [-0.5, 0.25], [0.0, 0.6]], dtype=float)


def _k_weights() -> np.ndarray:
    return np.array([0.15, 0.25, 0.35, 0.25], dtype=float)


@pytest.mark.parametrize("channel", ("spm", "dwave"))
def test_four_orbital_q_zero_bdg_nonlocal_kernel_matches_legacy_with_si_output(channel):
    old_config = OldKuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4, output_si=True)
    new_config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4, output_si=True)

    old = old_kernel(_k_points(), old_config, np.array([0.0, 0.0]), channel, None, _k_weights())
    new = bdg_current_current_kernel_imag_axis_from_model(
        LNO327FourOrbitalSpec(),
        _k_points(),
        new_config,
        np.array([0.0, 0.0]),
        channel,
        _k_weights(),
    )

    np.testing.assert_allclose(new, old)


@pytest.mark.parametrize("channel", ("spm", "dwave"))
def test_four_orbital_finite_q_bdg_nonlocal_kernel_matches_legacy_without_si_output(channel):
    old_config = OldKuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4, output_si=False)
    new_config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4, output_si=False)
    q = np.array([0.17, -0.09])

    old = old_kernel(_k_points(), old_config, q, channel, None, _k_weights())
    new = bdg_current_current_kernel_imag_axis_from_model(
        LNO327FourOrbitalSpec(),
        _k_points(),
        new_config,
        q,
        channel,
        _k_weights(),
    )

    np.testing.assert_allclose(new, old)


def test_four_orbital_shifted_bdg_eigensystem_matches_legacy():
    old_config = OldKuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4)
    new_config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4)
    spec = LNO327FourOrbitalSpec()
    kx, ky = 0.21, -0.34
    qx, qy = 0.17, -0.09

    old = old_shifted(kx, ky, qx, qy, "spm", None, old_config)
    new = shifted_bdg_eigensystem_from_model(spec, kx, ky, qx, qy, "spm", new_config)

    np.testing.assert_allclose(new.energies_minus_eV, old.energies_minus_eV)
    np.testing.assert_allclose(new.states_minus, old.states_minus)
    np.testing.assert_allclose(new.occupations_minus, old.occupations_minus)
    np.testing.assert_allclose(new.energies_plus_eV, old.energies_plus_eV)
    np.testing.assert_allclose(new.states_plus, old.states_plus)
    np.testing.assert_allclose(new.occupations_plus, old.occupations_plus)


def test_q_zero_shifted_bdg_eigensystem_reuses_shared_eigenbasis():
    config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4)
    bands = shifted_bdg_eigensystem_from_model(
        LNO327FourOrbitalSpec(),
        0.21,
        -0.34,
        0.0,
        0.0,
        "spm",
        config,
    )

    np.testing.assert_allclose(bands.energies_minus_eV, bands.energies_plus_eV)
    np.testing.assert_allclose(bands.states_minus, bands.states_plus)
    np.testing.assert_allclose(bands.occupations_minus, bands.occupations_plus)
    assert bands.states_minus is bands.states_plus


@pytest.mark.parametrize("direction", ("x", "y"))
def test_four_orbital_midpoint_bdg_current_vertex_matches_legacy(direction):
    old_config = OldKuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4)
    spec = LNO327FourOrbitalSpec()
    kx, ky = 0.21, -0.34
    qx, qy = 0.17, -0.09
    old_bands = old_shifted(kx, ky, qx, qy, "spm", None, old_config)

    old = old_midpoint_vertex(kx, ky, direction, old_bands.states_minus, old_bands.states_plus)
    new = midpoint_bdg_current_vertex_from_model(
        spec,
        kx,
        ky,
        direction,
        old_bands.states_minus,
        old_bands.states_plus,
    )

    np.testing.assert_allclose(new, old)


def test_midpoint_bdg_current_vertex_rejects_invalid_direction():
    states = np.eye(8, dtype=complex)

    with pytest.raises(ValueError, match="direction must be"):
        midpoint_bdg_current_vertex_from_model(
            LNO327FourOrbitalSpec(),
            0.1,
            0.2,
            "z",
            states,
            states,
        )


def test_bdg_nonlocal_rejects_bad_q_shape():
    config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4)

    with pytest.raises(ValueError, match="q must have shape \\(2,\\)"):
        bdg_current_current_kernel_imag_axis_from_model(
            LNO327FourOrbitalSpec(),
            _k_points(),
            config,
            np.array([0.1, 0.2, 0.3]),
            "spm",
            _k_weights(),
        )


def test_symmetry_bdg_2band_bdg_nonlocal_smoke_q_zero_and_finite_q_for_pairing_channels():
    config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4, output_si=False)
    spec = SymmetryBdG2BandSpec()

    for channel in ("spm", "dwave"):
        q_zero = bdg_current_current_kernel_imag_axis_from_model(
            spec,
            _k_points(),
            config,
            np.array([0.0, 0.0]),
            channel,
            _k_weights(),
        )
        q_finite = bdg_current_current_kernel_imag_axis_from_model(
            spec,
            _k_points(),
            config,
            np.array([0.17, -0.09]),
            channel,
            _k_weights(),
        )

        assert q_zero.shape == (2, 2)
        assert q_finite.shape == (2, 2)
        assert np.all(np.isfinite(q_zero))
        assert np.all(np.isfinite(q_finite))
