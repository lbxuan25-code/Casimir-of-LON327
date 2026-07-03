"""Two-band hopping representation and Peierls finite-q vertices."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from lno327.models.hopping import (
    HoppingTerm,
    normal_state_hamiltonian_from_hoppings as core_normal_state_hamiltonian_from_hoppings,
    peierls_hamiltonian_contact_vertex_from_hoppings,
    peierls_hamiltonian_vector_vertex_from_hoppings,
    peierls_vector_vertex_sign_audit_residual_from_hoppings,
    peierls_vertex_ward_residual_from_hoppings,
    sinc_stable,
    validate_hopping_hermiticity as core_validate_hopping_hermiticity,
)
from lno327.models.symmetry_bdg_2band.normal import TAU0, TAUX, TAUZ
from lno327.models.symmetry_bdg_2band.parameters import TwoBandParameters


def normal_state_hopping_terms(params: TwoBandParameters | None = None) -> tuple[HoppingTerm, ...]:
    """Return the finite hopping representation of the two-band normal-state model."""

    params = params or TwoBandParameters()
    onsite = -params.mu * TAU0 + params.t_perp * TAUX + params.m * TAUZ
    nearest = -params.t * TAU0 + params.t_perp_p * TAUX - params.t_z * TAUZ
    diagonal = -params.tp * TAU0
    terms = [
        ((0, 0), onsite),
        ((1, 0), nearest),
        ((-1, 0), nearest),
        ((0, 1), nearest),
        ((0, -1), nearest),
        ((1, 1), diagonal),
        ((1, -1), diagonal),
        ((-1, 1), diagonal),
        ((-1, -1), diagonal),
    ]
    return tuple((r, np.asarray(matrix, dtype=complex)) for r, matrix in sorted(terms))


def normal_state_hamiltonian_from_hoppings(
    kx: float,
    ky: float,
    params: TwoBandParameters | None = None,
    hopping_terms: Iterable[HoppingTerm] | None = None,
) -> np.ndarray:
    terms = normal_state_hopping_terms(params) if hopping_terms is None else hopping_terms
    return core_normal_state_hamiltonian_from_hoppings(kx, ky, terms)


def peierls_hamiltonian_vector_vertex(
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    direction: str,
    params: TwoBandParameters | None = None,
    hopping_terms: Iterable[HoppingTerm] | None = None,
) -> np.ndarray:
    terms = normal_state_hopping_terms(params) if hopping_terms is None else hopping_terms
    return peierls_hamiltonian_vector_vertex_from_hoppings(kx, ky, qx, qy, direction, terms)


def peierls_hamiltonian_contact_vertex(
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    direction_i: str,
    direction_j: str,
    params: TwoBandParameters | None = None,
    hopping_terms: Iterable[HoppingTerm] | None = None,
) -> np.ndarray:
    terms = normal_state_hopping_terms(params) if hopping_terms is None else hopping_terms
    return peierls_hamiltonian_contact_vertex_from_hoppings(
        kx,
        ky,
        qx,
        qy,
        direction_i,
        direction_j,
        terms,
    )


def peierls_vertex_ward_residual(
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    params: TwoBandParameters | None = None,
    hopping_terms: Iterable[HoppingTerm] | None = None,
) -> tuple[float, float, float, float]:
    terms = normal_state_hopping_terms(params) if hopping_terms is None else hopping_terms
    return peierls_vertex_ward_residual_from_hoppings(kx, ky, qx, qy, terms)


def peierls_vector_vertex_sign_audit_residual(
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    params: TwoBandParameters | None = None,
    hopping_terms: Iterable[HoppingTerm] | None = None,
    sign_convention: str = "plus",
) -> tuple[float, float, float, float]:
    terms = normal_state_hopping_terms(params) if hopping_terms is None else hopping_terms
    return peierls_vector_vertex_sign_audit_residual_from_hoppings(
        kx,
        ky,
        qx,
        qy,
        terms,
        sign_convention,
    )


def validate_hopping_hermiticity(
    hopping_terms: Iterable[HoppingTerm] | None = None,
    *,
    atol: float = 1e-12,
) -> float:
    terms = normal_state_hopping_terms() if hopping_terms is None else hopping_terms
    return core_validate_hopping_hermiticity(terms, atol=atol)
