from __future__ import annotations

import inspect

import numpy as np
import pytest

from lno327.models.lno327_four_orbital.vertices import normal_state_mass_operator
from lno327.models.lno327_four_orbital.peierls import (
    normal_state_hamiltonian_from_hoppings,
    normal_state_hopping_terms,
    peierls_hamiltonian_contact_vertex,
    peierls_hamiltonian_vector_vertex,
)


def test_hamiltonian_vector_vertex_has_no_sign_convention():
    signature = inspect.signature(peierls_hamiltonian_vector_vertex)
    assert "sign_convention" not in signature.parameters


def test_hamiltonian_contact_vertex_has_no_physical_contact_switches():
    signature = inspect.signature(peierls_hamiltonian_contact_vertex)
    for forbidden in ("contact_sign", "contact_sign_convention", "physical", "minus"):
        assert forbidden not in signature.parameters


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


def test_hamiltonian_vertex_functions_return_four_by_four_matrices():
    kx, ky = 0.11, -0.09
    qx, qy = 0.01, 0.02
    assert peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, "x").shape == (4, 4)
    assert peierls_hamiltonian_contact_vertex(kx, ky, qx, qy, "x", "x").shape == (4, 4)
