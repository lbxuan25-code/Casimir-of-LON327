import numpy as np

from lno327.response.bubble import (
    two_sided_band_basis_bubble_imag_axis,
    two_sided_response_factor_imag_axis,
)


def _legacy_two_sided_factor(
    energy_left: float,
    energy_right: float,
    occupation_left: float,
    occupation_right: float,
    omega_eV: float,
    eta_eV: float,
) -> float:
    delta_energy = energy_left - energy_right
    delta_occupation = occupation_left - occupation_right
    denominator = delta_energy**2 + omega_eV**2
    if denominator <= eta_eV**2:
        return 0.0
    return -delta_occupation * delta_energy / denominator


def test_two_sided_response_factor_matches_legacy_logic_without_eta_shifted_omega():
    actual = two_sided_response_factor_imag_axis(-0.2, 0.3, 0.9, 0.1, 0.07, 1e-2)
    expected = _legacy_two_sided_factor(-0.2, 0.3, 0.9, 0.1, 0.07, 1e-2)
    wrong_eta_shifted = _legacy_two_sided_factor(-0.2, 0.3, 0.9, 0.1, 0.08, 1e-2)

    assert actual == expected
    assert actual != wrong_eta_shifted


def test_two_sided_bubble_shape_finite_and_prefactor_scaling():
    energies_left = np.array([-0.2, 0.4])
    energies_right = np.array([-0.1, 0.3, 0.8])
    occupations_left = np.array([0.9, 0.1])
    occupations_right = np.array([0.8, 0.2, 0.05])
    vx = np.array([[1.0, 0.2j, 0.3], [-0.1j, 0.4, 0.5j]], dtype=complex)
    vy = np.array([[0.2, 0.1 + 0.4j, -0.3j], [0.7, -0.2j, 0.6]], dtype=complex)

    full = two_sided_band_basis_bubble_imag_axis(
        energies_left,
        energies_right,
        occupations_left,
        occupations_right,
        (vx, vy),
        0.07,
        1e-4,
    )
    half = two_sided_band_basis_bubble_imag_axis(
        energies_left,
        energies_right,
        occupations_left,
        occupations_right,
        (vx, vy),
        0.07,
        1e-4,
        prefactor=0.5,
    )

    assert full.shape == (2, 2)
    assert np.all(np.isfinite(full))
    np.testing.assert_allclose(half, 0.5 * full)


def test_two_sided_bubble_matches_reference_with_conjugated_second_vertex():
    energies_left = np.array([-0.2, 0.4])
    energies_right = np.array([-0.1, 0.3])
    occupations_left = np.array([0.9, 0.1])
    occupations_right = np.array([0.8, 0.2])
    vertices = (
        np.array([[1.0, 2.0 + 0.5j], [3.0 - 0.2j, 4.0]], dtype=complex),
        np.array([[0.7j, 0.1 - 0.3j], [2.0 + 0.4j, -0.6j]], dtype=complex),
    )
    reference = np.zeros((2, 2), dtype=complex)
    wrong_transposed_reference = np.zeros((2, 2), dtype=complex)

    for m, energy_left in enumerate(energies_left):
        for n, energy_right in enumerate(energies_right):
            factor = _legacy_two_sided_factor(
                float(energy_left),
                float(energy_right),
                float(occupations_left[m]),
                float(occupations_right[n]),
                0.07,
                1e-4,
            )
            for alpha, vertex_alpha in enumerate(vertices):
                for beta, vertex_beta in enumerate(vertices):
                    reference[alpha, beta] += (
                        factor * vertex_alpha[m, n] * np.conjugate(vertex_beta[m, n])
                    )
                    wrong_transposed_reference[alpha, beta] += (
                        factor * vertex_alpha[m, n] * vertex_beta[n, m]
                    )

    actual = two_sided_band_basis_bubble_imag_axis(
        energies_left,
        energies_right,
        occupations_left,
        occupations_right,
        vertices,
        0.07,
        1e-4,
    )

    np.testing.assert_allclose(actual, reference)
    assert not np.allclose(actual, wrong_transposed_reference)
