import numpy as np

from lno327.response.occupations import (
    fermi_function as new_fermi,
    negative_fermi_derivative as new_minus_df,
    occupation_difference,
)


def test_fermi_function_is_bounded_at_finite_temperature():
    energies = np.array([-0.3, -0.01, 0.0, 0.02, 0.4])

    values = new_fermi(energies, fermi_level_eV=0.01, temperature_eV=0.025)
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)
    assert np.all(np.diff(values) <= 0.0)


def test_fermi_function_zero_temperature_step():
    energies = np.array([-0.3, 0.0, 0.4])

    np.testing.assert_allclose(new_fermi(energies, fermi_level_eV=0.0, temperature_eV=0.0), [1.0, 0.0, 0.0])


def test_negative_fermi_derivative_with_eta_fallback_is_nonnegative():
    energies = np.array([-0.1, 0.0, 0.1])

    values = new_minus_df(energies, fermi_level_eV=0.0, temperature_eV=0.0, eta_eV=1e-4)
    assert np.all(values >= 0.0)


def test_negative_fermi_derivative_at_finite_temperature_is_nonnegative():
    energies = np.array([-0.1, 0.0, 0.1])

    values = new_minus_df(energies, fermi_level_eV=0.01, temperature_eV=0.02, eta_eV=1e-5)
    assert np.all(values >= 0.0)


def test_occupation_difference_is_pairwise_difference():
    occupations = np.array([1.0, 0.25, 0.0])

    diff = occupation_difference(occupations)

    np.testing.assert_allclose(diff, occupations[:, None] - occupations[None, :])
