import numpy as np

from lno327.response.bubble import band_basis_bubble_imag_axis, response_factor_imag_axis


def _legacy_reference_factor(
    energy_m: float,
    energy_n: float,
    occupation_m: float,
    occupation_n: float,
    negative_df_m: float,
    omega_eV: float,
    eta_eV: float,
    same_state: bool,
) -> float:
    if same_state:
        return negative_df_m
    occupation_diff = occupation_m - occupation_n
    if np.isclose(occupation_diff, 0.0):
        return 0.0
    energy_diff = energy_m - energy_n
    if abs(energy_diff) < eta_eV:
        return 0.0
    omega = omega_eV + eta_eV
    return -occupation_diff * energy_diff / (energy_diff**2 + omega**2)


def test_response_factor_matches_legacy_logic():
    args = (-0.2, 0.3, 0.9, 0.1, 2.5, 0.07, 1e-4)

    assert response_factor_imag_axis(*args, same_state=False) == _legacy_reference_factor(
        *args, same_state=False
    )
    assert response_factor_imag_axis(*args, same_state=True) == _legacy_reference_factor(
        *args, same_state=True
    )


def test_band_basis_bubble_shape_finite_and_prefactor_scaling():
    energies = np.array([-0.2, 0.3])
    occupations = np.array([0.9, 0.1])
    minus_df = np.array([0.2, 0.3])
    vx = np.array([[1.0, 0.2j], [-0.2j, 0.4]], dtype=complex)
    vy = np.array([[0.1, 0.3], [0.3, -0.2]], dtype=complex)

    full = band_basis_bubble_imag_axis(energies, occupations, minus_df, (vx, vy), 0.07, 1e-4)
    half = band_basis_bubble_imag_axis(
        energies, occupations, minus_df, (vx, vy), 0.07, 1e-4, prefactor=0.5
    )

    assert full.shape == (2, 2)
    assert np.all(np.isfinite(full))
    np.testing.assert_allclose(half, 0.5 * full)
    np.testing.assert_allclose(full, full.conjugate().T)


def test_band_basis_bubble_matches_handwritten_legacy_reference():
    energies = np.array([-0.2, 0.3])
    occupations = np.array([0.9, 0.1])
    minus_df = np.array([0.2, 0.3])
    vertices = (
        np.array([[1.0, 0.25], [0.25, 0.4]], dtype=complex),
        np.array([[0.1, -0.2j], [0.2j, -0.3]], dtype=complex),
    )
    omega_eV = 0.07
    eta_eV = 1e-4
    reference = np.zeros((2, 2), dtype=complex)

    for m, energy_m in enumerate(energies):
        for n, energy_n in enumerate(energies):
            factor = _legacy_reference_factor(
                float(energy_m),
                float(energy_n),
                float(occupations[m]),
                float(occupations[n]),
                float(minus_df[m]),
                omega_eV,
                eta_eV,
                same_state=m == n,
            )
            for alpha, vertex_alpha in enumerate(vertices):
                for beta, vertex_beta in enumerate(vertices):
                    reference[alpha, beta] += factor * vertex_alpha[m, n] * vertex_beta[n, m]

    actual = band_basis_bubble_imag_axis(
        energies,
        occupations,
        minus_df,
        vertices,
        omega_eV,
        eta_eV,
    )

    np.testing.assert_allclose(actual, reference)
