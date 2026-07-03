import numpy as np

from lno327.conductivity import fermi_function as old_fermi
from lno327.conductivity import negative_fermi_derivative as old_minus_df
from lno327.response.occupations import (
    fermi_function as new_fermi,
    negative_fermi_derivative as new_minus_df,
    occupation_difference,
)


def test_fermi_function_matches_legacy_at_finite_temperature():
    energies = np.array([-0.3, -0.01, 0.0, 0.02, 0.4])

    np.testing.assert_allclose(
        new_fermi(energies, fermi_level_eV=0.01, temperature_eV=0.025),
        old_fermi(energies, fermi_level_eV=0.01, temperature_eV=0.025),
    )


def test_fermi_function_matches_legacy_at_zero_temperature():
    energies = np.array([-0.3, 0.0, 0.4])

    np.testing.assert_allclose(
        new_fermi(energies, fermi_level_eV=0.0, temperature_eV=0.0),
        old_fermi(energies, fermi_level_eV=0.0, temperature_eV=0.0),
    )


def test_negative_fermi_derivative_matches_legacy_with_eta_fallback():
    energies = np.array([-0.1, 0.0, 0.1])

    np.testing.assert_allclose(
        new_minus_df(energies, fermi_level_eV=0.0, temperature_eV=0.0, eta_eV=1e-4),
        old_minus_df(energies, fermi_level_eV=0.0, temperature_eV=0.0, eta_eV=1e-4),
    )


def test_negative_fermi_derivative_matches_legacy_at_finite_temperature():
    energies = np.array([-0.1, 0.0, 0.1])

    np.testing.assert_allclose(
        new_minus_df(energies, fermi_level_eV=0.01, temperature_eV=0.02, eta_eV=1e-5),
        old_minus_df(energies, fermi_level_eV=0.01, temperature_eV=0.02, eta_eV=1e-5),
    )


def test_occupation_difference_is_pairwise_difference():
    occupations = np.array([1.0, 0.25, 0.0])

    diff = occupation_difference(occupations)

    np.testing.assert_allclose(diff, occupations[:, None] - occupations[None, :])
