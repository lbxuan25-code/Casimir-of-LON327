"""Four-orbital hopping representation and Peierls finite-q vertices."""

from __future__ import annotations

from collections import defaultdict
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
from lno327.models.lno327_four_orbital.parameters import ORBITAL_BASIS, NormalStateParameters


def _orbital_dim() -> int:
    return len(ORBITAL_BASIS)


def _zero_matrix() -> np.ndarray:
    return np.zeros((_orbital_dim(), _orbital_dim()), dtype=complex)


def _basis_matrices() -> dict[str, np.ndarray]:
    matrices = {name: _zero_matrix() for name in ("tz", "tx", "tz_perp", "tx_perp", "vxz", "vxz_perp")}
    matrices["tz"][0, 0] = 1.0
    matrices["tz"][2, 2] = 1.0
    matrices["tx"][1, 1] = 1.0
    matrices["tx"][3, 3] = 1.0
    matrices["tz_perp"][0, 2] = 1.0
    matrices["tz_perp"][2, 0] = 1.0
    matrices["tx_perp"][1, 3] = 1.0
    matrices["tx_perp"][3, 1] = 1.0
    matrices["vxz"][0, 1] = 1.0
    matrices["vxz"][1, 0] = 1.0
    matrices["vxz"][2, 3] = 1.0
    matrices["vxz"][3, 2] = 1.0
    matrices["vxz_perp"][0, 3] = 1.0
    matrices["vxz_perp"][3, 0] = 1.0
    matrices["vxz_perp"][1, 2] = 1.0
    matrices["vxz_perp"][2, 1] = 1.0
    return matrices


def _add_term(terms: dict[tuple[int, int], np.ndarray], r: tuple[int, int], matrix: np.ndarray) -> None:
    terms[r] += np.asarray(matrix, dtype=complex)


def _add_constant(terms: dict[tuple[int, int], np.ndarray], coefficient: float, matrix: np.ndarray) -> None:
    _add_term(terms, (0, 0), coefficient * matrix)


def _add_cos_axis(
    terms: dict[tuple[int, int], np.ndarray],
    coefficient: float,
    harmonic: int,
    axis: str,
    matrix: np.ndarray,
) -> None:
    if axis == "x":
        vectors = ((harmonic, 0), (-harmonic, 0))
    elif axis == "y":
        vectors = ((0, harmonic), (0, -harmonic))
    else:
        raise ValueError("axis must be 'x' or 'y'")
    for r in vectors:
        _add_term(terms, r, 0.5 * coefficient * matrix)


def _add_cos_x_plus_cos_y(
    terms: dict[tuple[int, int], np.ndarray],
    coefficient: float,
    harmonic: int,
    matrix: np.ndarray,
) -> None:
    _add_cos_axis(terms, coefficient, harmonic, "x", matrix)
    _add_cos_axis(terms, coefficient, harmonic, "y", matrix)


def _add_cos_x_minus_cos_y(
    terms: dict[tuple[int, int], np.ndarray],
    coefficient: float,
    harmonic: int,
    matrix: np.ndarray,
) -> None:
    _add_cos_axis(terms, coefficient, harmonic, "x", matrix)
    _add_cos_axis(terms, -coefficient, harmonic, "y", matrix)


def _add_cos_x_cos_y(terms: dict[tuple[int, int], np.ndarray], coefficient: float, matrix: np.ndarray) -> None:
    for rx in (-1, 1):
        for ry in (-1, 1):
            _add_term(terms, (rx, ry), 0.25 * coefficient * matrix)


def normal_state_hopping_terms(params: NormalStateParameters | None = None) -> list[HoppingTerm]:
    """Return hopping terms t_R for the four-orbital normal-state H0(k)."""

    params = params or NormalStateParameters()
    basis = _basis_matrices()
    terms: dict[tuple[int, int], np.ndarray] = defaultdict(_zero_matrix)

    _add_constant(terms, params.tz_0, basis["tz"])
    _add_constant(terms, params.tx_0, basis["tx"])
    _add_constant(terms, params.tz_perp_0, basis["tz_perp"])
    _add_constant(terms, params.tx_perp_0, basis["tx_perp"])
    _add_constant(terms, -params.chemical_potential, np.eye(_orbital_dim(), dtype=complex))

    _add_cos_x_plus_cos_y(terms, params.tz_1, 1, basis["tz"])
    _add_cos_x_plus_cos_y(terms, params.tz_3, 2, basis["tz"])
    _add_cos_x_plus_cos_y(terms, params.tz_4, 3, basis["tz"])
    _add_cos_x_cos_y(terms, params.tz_2, basis["tz"])

    _add_cos_x_plus_cos_y(terms, params.tx_1, 1, basis["tx"])
    _add_cos_x_plus_cos_y(terms, params.tx_3, 2, basis["tx"])
    _add_cos_x_plus_cos_y(terms, params.tx_4, 3, basis["tx"])
    _add_cos_x_cos_y(terms, params.tx_2, basis["tx"])

    _add_cos_x_plus_cos_y(terms, params.tz_perp_1, 1, basis["tz_perp"])
    _add_cos_x_minus_cos_y(terms, params.vxz_1, 1, basis["vxz"])
    _add_cos_x_minus_cos_y(terms, params.vxz_2, 2, basis["vxz"])
    _add_cos_x_minus_cos_y(terms, params.vxz_perp_1, 1, basis["vxz_perp"])

    return [(r, terms[r]) for r in sorted(terms)]


def normal_state_hamiltonian_from_hoppings(
    kx: float,
    ky: float,
    params: NormalStateParameters | None = None,
    hopping_terms: Iterable[HoppingTerm] | None = None,
) -> np.ndarray:
    """Reconstruct H0(k) from Fourier hopping terms."""

    terms = list(normal_state_hopping_terms(params) if hopping_terms is None else hopping_terms)
    return core_normal_state_hamiltonian_from_hoppings(kx, ky, terms)


def _peierls_vector_vertex_core(
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    direction: str,
    sign: float,
    params: NormalStateParameters | None = None,
    hopping_terms: Iterable[HoppingTerm] | None = None,
) -> np.ndarray:
    if direction not in {"x", "y"}:
        raise ValueError("direction must be 'x' or 'y'")
    terms = list(normal_state_hopping_terms(params) if hopping_terms is None else hopping_terms)
    if sign == 1.0:
        return peierls_hamiltonian_vector_vertex_from_hoppings(kx, ky, qx, qy, direction, terms)
    return -peierls_hamiltonian_vector_vertex_from_hoppings(kx, ky, qx, qy, direction, terms)


def _peierls_vector_vertex_with_sign(
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    direction: str,
    params: NormalStateParameters | None = None,
    hopping_terms: Iterable[HoppingTerm] | None = None,
    sign_convention: str = "plus",
) -> np.ndarray:
    """Diagnostic-only helper for historical sign audits."""

    if sign_convention not in {"plus", "minus"}:
        raise ValueError("sign_convention must be 'plus' or 'minus'")
    sign = 1.0 if sign_convention == "plus" else -1.0
    return _peierls_vector_vertex_core(
        kx,
        ky,
        qx,
        qy,
        direction,
        sign,
        params=params,
        hopping_terms=hopping_terms,
    )


def peierls_hamiltonian_vector_vertex(
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    direction: str,
    params: NormalStateParameters | None = None,
    hopping_terms: Iterable[HoppingTerm] | None = None,
) -> np.ndarray:
    """Return the Hamiltonian vector vertex V_i = delta H / delta A_i.

    This is not the physical current vertex. The physical current vertex is
    j_i = -V_i.
    """

    return _peierls_vector_vertex_core(
        kx,
        ky,
        qx,
        qy,
        direction,
        1.0,
        params=params,
        hopping_terms=hopping_terms,
    )


def peierls_hamiltonian_contact_vertex(
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    direction_i: str,
    direction_j: str,
    params: NormalStateParameters | None = None,
    hopping_terms: Iterable[HoppingTerm] | None = None,
) -> np.ndarray:
    """Return the Hamiltonian contact vertex M_ij = delta^2 H / delta A_i delta A_j."""

    terms = list(normal_state_hopping_terms(params) if hopping_terms is None else hopping_terms)
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
    params: NormalStateParameters | None = None,
    hopping_terms: Iterable[HoppingTerm] | None = None,
) -> tuple[float, float, float, float]:
    """Return vertex-level residual errors for q_i V_i = H(k+q/2)-H(k-q/2)."""

    terms = list(normal_state_hopping_terms(params) if hopping_terms is None else hopping_terms)
    return peierls_vertex_ward_residual_from_hoppings(kx, ky, qx, qy, terms)


def peierls_vector_vertex_sign_audit_residual(
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    params: NormalStateParameters | None = None,
    hopping_terms: Iterable[HoppingTerm] | None = None,
    sign_convention: str = "plus",
) -> tuple[float, float, float, float]:
    """Diagnostic-only historical sign audit for the Peierls vector vertex."""

    terms = list(normal_state_hopping_terms(params) if hopping_terms is None else hopping_terms)
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
    """Return max ||t_-R - t_R^dagger|| over hopping terms."""

    terms = normal_state_hopping_terms() if hopping_terms is None else hopping_terms
    return core_validate_hopping_hermiticity(terms, atol=atol)
