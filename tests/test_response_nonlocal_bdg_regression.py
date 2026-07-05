import numpy as np
import pytest

from lno327.response.config import KuboConfig
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
def test_four_orbital_q_zero_bdg_nonlocal_kernel_is_finite_with_si_output(channel):
    config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4, output_si=True)

    new = bdg_current_current_kernel_imag_axis_from_model(
        LNO327FourOrbitalSpec(),
        _k_points(),
        config,
        np.array([0.0, 0.0]),
        channel,
        _k_weights(),
    )

    assert new.shape == (2, 2)
    assert np.all(np.isfinite(new))


@pytest.mark.parametrize("channel", ("spm", "dwave"))
def test_four_orbital_finite_q_bdg_nonlocal_kernel_is_finite_without_si_output(channel):
    config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4, output_si=False)
    q = np.array([0.17, -0.09])

    new = bdg_current_current_kernel_imag_axis_from_model(
        LNO327FourOrbitalSpec(),
        _k_points(),
        config,
        q,
        channel,
        _k_weights(),
    )

    assert new.shape == (2, 2)
    assert np.all(np.isfinite(new))


def test_four_orbital_shifted_bdg_eigensystem_is_well_formed():
    new_config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4)
    spec = LNO327FourOrbitalSpec()
    kx, ky = 0.21, -0.34
    qx, qy = 0.17, -0.09

    new = shifted_bdg_eigensystem_from_model(spec, kx, ky, qx, qy, "spm", new_config)

    assert new.energies_minus_eV.shape == (8,)
    assert new.states_minus.shape == (8, 8)
    assert new.occupations_minus.shape == (8,)
    assert new.energies_plus_eV.shape == (8,)
    assert new.states_plus.shape == (8, 8)
    assert new.occupations_plus.shape == (8,)


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
def test_four_orbital_midpoint_bdg_current_vertex_is_well_formed(direction):
    config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4)
    spec = LNO327FourOrbitalSpec()
    kx, ky = 0.21, -0.34
    qx, qy = 0.17, -0.09
    bands = shifted_bdg_eigensystem_from_model(spec, kx, ky, qx, qy, "spm", config)

    new = midpoint_bdg_current_vertex_from_model(
        spec,
        kx,
        ky,
        direction,
        bands.states_minus,
        bands.states_plus,
    )

    assert new.shape == (8, 8)
    assert np.all(np.isfinite(new))


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
