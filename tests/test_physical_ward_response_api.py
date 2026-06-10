from __future__ import annotations

import inspect

import numpy as np

from lno327.conductivity import KuboConfig, bosonic_matsubara_energy_eV, k_weights, uniform_bz_mesh
from lno327.model import normal_state_mass_operator
from lno327.tb_fourier import (
    normal_state_hamiltonian_from_hoppings,
    normal_state_hopping_terms,
    peierls_hamiltonian_contact_vertex,
    peierls_hamiltonian_vector_vertex,
)
from lno327.ward_response import (
    normal_density_current_response_imag_axis,
    normal_physical_density_current_response_imag_axis,
)


def test_hamiltonian_vector_vertex_identity():
    kx, ky = 0.37, -0.42
    qx, qy = 0.031, -0.017
    terms = normal_state_hopping_terms()

    vector_x = peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, "x", hopping_terms=terms)
    vector_y = peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, "y", hopping_terms=terms)
    lhs = qx * vector_x + qy * vector_y
    rhs = normal_state_hamiltonian_from_hoppings(
        kx + 0.5 * qx,
        ky + 0.5 * qy,
        hopping_terms=terms,
    ) - normal_state_hamiltonian_from_hoppings(
        kx - 0.5 * qx,
        ky - 0.5 * qy,
        hopping_terms=terms,
    )

    np.testing.assert_allclose(lhs, rhs, atol=1e-14, rtol=1e-12)


def test_hamiltonian_contact_vertex_q0_mass_limit():
    kx, ky = 0.21, -0.31
    terms = normal_state_hopping_terms()
    for direction_i in ("x", "y"):
        for direction_j in ("x", "y"):
            contact = peierls_hamiltonian_contact_vertex(
                kx,
                ky,
                0.0,
                0.0,
                direction_i,
                direction_j,
                hopping_terms=terms,
            )
            mass = normal_state_mass_operator(kx, ky, direction_i, direction_j)
            np.testing.assert_allclose(contact, mass, atol=1e-14, rtol=1e-12)


def test_physical_response_candidate_matches_manual_code_piece_combination():
    mesh = uniform_bz_mesh(4)
    weights = k_weights(mesh)
    q = np.array([0.01, 0.007], dtype=float)
    omega_eV = bosonic_matsubara_energy_eV(1, 30.0)
    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=30.0,
        eta_eV=1e-10,
        output_si=False,
    )

    physical = normal_physical_density_current_response_imag_axis(mesh, config, q, weights)
    code_bubble = normal_density_current_response_imag_axis(
        mesh,
        config,
        q,
        weights,
        vertex_scheme="peierls",
        contact_scheme="none",
    )
    code_contact_plus = normal_density_current_response_imag_axis(
        mesh,
        config,
        q,
        weights,
        vertex_scheme="peierls",
        contact_scheme="finite_q_peierls",
        contact_sign_convention="plus",
    )
    code_contact = code_contact_plus - code_bubble

    manual = np.array(code_bubble, copy=True)
    manual[0, 1:] *= -1.0
    manual[1:, 0] *= -1.0
    manual[1:, 1:] = code_bubble[1:, 1:] - code_contact[1:, 1:]

    np.testing.assert_allclose(physical, manual, atol=1e-13, rtol=1e-11)


def test_main_apis_do_not_expose_sign_scan_parameters():
    vector_signature = inspect.signature(peierls_hamiltonian_vector_vertex)
    physical_signature = inspect.signature(normal_physical_density_current_response_imag_axis)

    assert "sign_convention" not in vector_signature.parameters
    for forbidden in (
        "sign_convention",
        "contact_sign_convention",
        "current_vertex_multiplier",
        "ward_q_sign",
    ):
        assert forbidden not in physical_signature.parameters


def test_old_convention_scanner_docstring_is_diagnostic_only():
    assert "Diagnostic-only" in (normal_density_current_response_imag_axis.__doc__ or "")
