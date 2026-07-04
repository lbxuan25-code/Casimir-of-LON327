from __future__ import annotations

import numpy as np

from lno327.conductivity import KuboConfig, bosonic_matsubara_energy_eV, fermi_function, k_weights, uniform_bz_mesh
from lno327.tb_fourier import normal_state_hopping_terms, peierls_hamiltonian_vector_vertex
from lno327.response.normal_density_current import (
    _finite_q_band_bubble_imag_axis,
    normal_density_current_response_imag_axis,
    normal_physical_density_current_response_imag_axis,
)


def test_finite_q_band_bubble_matches_corrected_positive_factor_and_matrix_order():
    # The linear-response minus sign is canceled by the fermion-loop minus sign.
    energies_minus = np.array([-0.17, 0.23], dtype=float)
    energies_plus = np.array([-0.11, 0.31], dtype=float)
    states_minus = np.eye(2, dtype=complex)
    states_plus = np.eye(2, dtype=complex)
    observable = np.array([[0.2, 0.3 + 0.4j], [0.5 - 0.1j, -0.7]], dtype=complex)
    source = np.array([[0.6, -0.2 + 0.9j], [0.8 + 0.3j, 0.4]], dtype=complex)
    config = KuboConfig.from_kelvin(
        omega_eV=bosonic_matsubara_energy_eV(2, 30.0),
        temperature_K=30.0,
        eta_eV=1e-10,
        output_si=False,
    )

    bubble = _finite_q_band_bubble_imag_axis(
        energies_minus,
        states_minus,
        energies_plus,
        states_plus,
        (observable,),
        (source,),
        config,
    )

    occupations_minus = fermi_function(energies_minus, config.fermi_level_eV, config.temperature_eV)
    occupations_plus = fermi_function(energies_plus, config.fermi_level_eV, config.temperature_eV)
    expected = 0.0j
    for m, energy_minus in enumerate(energies_minus):
        for n, energy_plus in enumerate(energies_plus):
            occupation_diff = occupations_minus[m] - occupations_plus[n]
            denominator = 1j * config.omega_eV + energy_minus - energy_plus
            factor = occupation_diff / denominator
            expected += factor * observable[m, n] * np.conjugate(source[m, n])

    np.testing.assert_allclose(bubble[0, 0], expected, atol=1e-15, rtol=1e-13)


def test_peierls_vector_vertex_finite_q_reverse_element_relation():
    kx, ky = 0.31, -0.22
    qx, qy = 0.07, 0.03
    terms = normal_state_hopping_terms()
    vector_q = peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, "x", hopping_terms=terms)
    vector_minus_q = peierls_hamiltonian_vector_vertex(kx, ky, -qx, -qy, "x", hopping_terms=terms)

    np.testing.assert_allclose(vector_q.conjugate().T, vector_minus_q, atol=1e-14, rtol=1e-12)


def test_physical_response_uses_observable_and_source_vertices_with_direct_contact():
    mesh = uniform_bz_mesh(4)
    weights = k_weights(mesh)
    q = np.array([0.01, 0.007], dtype=float)
    config = KuboConfig.from_kelvin(
        omega_eV=bosonic_matsubara_energy_eV(1, 30.0),
        temperature_K=30.0,
        eta_eV=1e-10,
        output_si=False,
    )

    physical = normal_physical_density_current_response_imag_axis(mesh, config, q, weights)
    source_source_bubble = normal_density_current_response_imag_axis(
        mesh,
        config,
        q,
        weights,
        vertex_scheme="peierls",
        contact_scheme="none",
    )
    source_contact_plus = normal_density_current_response_imag_axis(
        mesh,
        config,
        q,
        weights,
        vertex_scheme="peierls",
        contact_scheme="finite_q_peierls",
        contact_sign_convention="plus",
    )
    code_contact = source_contact_plus - source_source_bubble

    manual = np.array(source_source_bubble, copy=True)
    manual[0, 0] = source_source_bubble[0, 0]
    manual[0, 1:] = source_source_bubble[0, 1:]
    manual[1:, 0] = -source_source_bubble[1:, 0]
    manual[1:, 1:] = -source_source_bubble[1:, 1:] - code_contact[1:, 1:]

    np.testing.assert_allclose(physical, manual, atol=1e-13, rtol=1e-11)
