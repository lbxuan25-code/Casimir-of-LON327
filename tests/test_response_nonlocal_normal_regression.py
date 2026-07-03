import numpy as np
import pytest

from lno327.conductivity import KuboConfig as OldKuboConfig
from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.models.symmetry_bdg_2band.spec import SymmetryBdG2BandSpec
from lno327.nonlocal_response import (
    c4_covariance_error as old_c4_error,
    midpoint_velocity_vertex as old_midpoint_vertex,
    normal_current_current_kernel_imag_axis as old_kernel,
    shifted_normal_eigensystem as old_shifted,
)
from lno327.response.config import KuboConfig
from lno327.response.nonlocal_normal import (
    c4_covariance_error,
    midpoint_velocity_vertex_from_model,
    normal_current_current_kernel_imag_axis_from_model,
    shifted_normal_eigensystem_from_model,
)


def _k_points() -> np.ndarray:
    return np.array([[0.1, -0.2], [0.4, 0.3], [-0.5, 0.25], [0.0, 0.6]], dtype=float)


def _k_weights() -> np.ndarray:
    return np.array([0.15, 0.25, 0.35, 0.25], dtype=float)


def test_four_orbital_q_zero_nonlocal_kernel_matches_legacy_with_si_output():
    old_config = OldKuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4, output_si=True)
    new_config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4, output_si=True)

    old = old_kernel(_k_points(), old_config, np.array([0.0, 0.0]), _k_weights())
    new = normal_current_current_kernel_imag_axis_from_model(
        LNO327FourOrbitalSpec(),
        _k_points(),
        new_config,
        np.array([0.0, 0.0]),
        _k_weights(),
    )

    np.testing.assert_allclose(new, old)


def test_four_orbital_finite_q_nonlocal_kernel_matches_legacy_without_si_output():
    old_config = OldKuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4, output_si=False)
    new_config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4, output_si=False)
    q = np.array([0.17, -0.09])

    old = old_kernel(_k_points(), old_config, q, _k_weights())
    new = normal_current_current_kernel_imag_axis_from_model(
        LNO327FourOrbitalSpec(),
        _k_points(),
        new_config,
        q,
        _k_weights(),
    )

    np.testing.assert_allclose(new, old)


def test_four_orbital_shifted_eigensystem_and_midpoint_vertex_match_legacy():
    old_config = OldKuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4)
    new_config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4)
    spec = LNO327FourOrbitalSpec()
    kx, ky = 0.21, -0.34
    qx, qy = 0.17, -0.09

    old = old_shifted(kx, ky, qx, qy, old_config)
    new = shifted_normal_eigensystem_from_model(spec, kx, ky, qx, qy, new_config)

    np.testing.assert_allclose(new.energies_minus_eV, old.energies_minus_eV)
    np.testing.assert_allclose(new.states_minus, old.states_minus)
    np.testing.assert_allclose(new.occupations_minus, old.occupations_minus)
    np.testing.assert_allclose(new.energies_plus_eV, old.energies_plus_eV)
    np.testing.assert_allclose(new.states_plus, old.states_plus)
    np.testing.assert_allclose(new.occupations_plus, old.occupations_plus)

    old_vertex = old_midpoint_vertex(kx, ky, "x", old.states_minus, old.states_plus)
    new_vertex = midpoint_velocity_vertex_from_model(
        spec,
        kx,
        ky,
        "x",
        new.states_minus,
        new.states_plus,
    )
    np.testing.assert_allclose(new_vertex, old_vertex)


def test_c4_covariance_error_matches_legacy():
    matrix_q = np.array([[1.0, 0.2j], [-0.2j, 0.4]], dtype=complex)
    matrix_rotated_q = np.array([[0.3, -0.1], [-0.1, 1.2]], dtype=complex)

    assert c4_covariance_error(matrix_q, matrix_rotated_q) == old_c4_error(
        matrix_q,
        matrix_rotated_q,
    )


def test_nonlocal_normal_rejects_bad_q_shape():
    config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4)

    with pytest.raises(ValueError, match="q must have shape \\(2,\\)"):
        normal_current_current_kernel_imag_axis_from_model(
            LNO327FourOrbitalSpec(),
            _k_points(),
            config,
            np.array([0.1, 0.2, 0.3]),
            _k_weights(),
        )


def test_symmetry_bdg_2band_nonlocal_normal_smoke_q_zero_and_finite_q():
    config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4, output_si=False)
    spec = SymmetryBdG2BandSpec()

    q_zero = normal_current_current_kernel_imag_axis_from_model(
        spec,
        _k_points(),
        config,
        np.array([0.0, 0.0]),
        _k_weights(),
    )
    q_finite = normal_current_current_kernel_imag_axis_from_model(
        spec,
        _k_points(),
        config,
        np.array([0.17, -0.09]),
        _k_weights(),
    )

    assert q_zero.shape == (2, 2)
    assert q_finite.shape == (2, 2)
    assert np.all(np.isfinite(q_zero))
    assert np.all(np.isfinite(q_finite))
