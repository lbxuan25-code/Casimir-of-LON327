"""Generic hopping-representation and Peierls-vertex algebra."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

HoppingTerm = tuple[tuple[int, int], np.ndarray]
EPS = 1e-300


def sinc_stable(x):
    """Return sin(x)/x with the stable x->0 limit."""

    values = np.asarray(x, dtype=float)
    result = np.ones_like(values, dtype=float)
    mask = np.abs(values) > 1e-12
    result[mask] = np.sin(values[mask]) / values[mask]
    if np.isscalar(x):
        return float(result)
    return result


def normal_state_hamiltonian_from_hoppings(
    kx: float,
    ky: float,
    hopping_terms: Iterable[HoppingTerm],
) -> np.ndarray:
    """Return H(k) = sum_R t_R exp(i k.R)."""

    terms = list(hopping_terms)
    if not terms:
        raise ValueError("hopping_terms must not be empty")
    dim = np.asarray(terms[0][1]).shape[0]
    hamiltonian = np.zeros((dim, dim), dtype=complex)
    for (rx, ry), hopping in terms:
        phase = np.exp(1j * (kx * rx + ky * ry))
        hamiltonian += np.asarray(hopping, dtype=complex) * phase
    return hamiltonian


def _component(direction: str, rx: int, ry: int) -> int:
    if direction == "x":
        return rx
    if direction == "y":
        return ry
    raise ValueError("direction must be 'x' or 'y'")


def _peierls_vector_vertex_from_hoppings_with_sign(
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    direction: str,
    hopping_terms: Iterable[HoppingTerm],
    sign: float,
) -> np.ndarray:
    terms = list(hopping_terms)
    if not terms:
        raise ValueError("hopping_terms must not be empty")
    dim = np.asarray(terms[0][1]).shape[0]
    vertex = np.zeros((dim, dim), dtype=complex)
    for (rx, ry), hopping in terms:
        component = _component(direction, rx, ry)
        if component == 0:
            continue
        q_dot_r = qx * rx + qy * ry
        phase = np.exp(1j * (kx * rx + ky * ry))
        vertex += 1j * sign * component * hopping * phase * sinc_stable(0.5 * q_dot_r)
    return vertex


def peierls_hamiltonian_vector_vertex_from_hoppings(
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    direction: str,
    hopping_terms: Iterable[HoppingTerm],
) -> np.ndarray:
    """Return the Hamiltonian vector vertex V_i = delta H / delta A_i."""

    return _peierls_vector_vertex_from_hoppings_with_sign(
        kx,
        ky,
        qx,
        qy,
        direction,
        hopping_terms,
        1.0,
    )


def peierls_hamiltonian_contact_vertex_from_hoppings(
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    direction_i: str,
    direction_j: str,
    hopping_terms: Iterable[HoppingTerm],
) -> np.ndarray:
    """Return the Hamiltonian contact vertex M_ij = delta^2 H / delta A_i delta A_j."""

    terms = list(hopping_terms)
    if not terms:
        raise ValueError("hopping_terms must not be empty")
    if direction_i not in {"x", "y"} or direction_j not in {"x", "y"}:
        raise ValueError("directions must be 'x' or 'y'")
    dim = np.asarray(terms[0][1]).shape[0]
    vertex = np.zeros((dim, dim), dtype=complex)
    for (rx, ry), hopping in terms:
        component_i = rx if direction_i == "x" else ry
        component_j = rx if direction_j == "x" else ry
        if component_i == 0 or component_j == 0:
            continue
        q_dot_r = qx * rx + qy * ry
        phase = np.exp(1j * (kx * rx + ky * ry))
        sinc_factor = sinc_stable(0.5 * q_dot_r)
        vertex += -component_i * component_j * hopping * phase * sinc_factor * sinc_factor
    return vertex


def peierls_vertex_ward_residual_from_hoppings(
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    hopping_terms: Iterable[HoppingTerm],
) -> tuple[float, float, float, float]:
    """Return vertex-level residual errors for q_i V_i = H(k+q/2)-H(k-q/2)."""

    terms = list(hopping_terms)
    vector_x = peierls_hamiltonian_vector_vertex_from_hoppings(kx, ky, qx, qy, "x", terms)
    vector_y = peierls_hamiltonian_vector_vertex_from_hoppings(kx, ky, qx, qy, "y", terms)
    lhs = qx * vector_x + qy * vector_y
    h_plus = normal_state_hamiltonian_from_hoppings(kx + 0.5 * qx, ky + 0.5 * qy, terms)
    h_minus = normal_state_hamiltonian_from_hoppings(kx - 0.5 * qx, ky - 0.5 * qy, terms)
    rhs = h_plus - h_minus
    abs_error = float(np.linalg.norm(lhs - rhs))
    rhs_norm = float(np.linalg.norm(rhs))
    lhs_norm = float(np.linalg.norm(lhs))
    rel_error = abs_error / max(rhs_norm, EPS)
    return abs_error, rel_error, lhs_norm, rhs_norm


def peierls_vector_vertex_sign_audit_residual_from_hoppings(
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    hopping_terms: Iterable[HoppingTerm],
    sign_convention: str = "plus",
) -> tuple[float, float, float, float]:
    """Diagnostic-only historical sign audit for the Peierls vector vertex."""

    if sign_convention not in {"plus", "minus"}:
        raise ValueError("sign_convention must be 'plus' or 'minus'")
    sign = 1.0 if sign_convention == "plus" else -1.0
    terms = list(hopping_terms)
    vector_x = _peierls_vector_vertex_from_hoppings_with_sign(kx, ky, qx, qy, "x", terms, sign)
    vector_y = _peierls_vector_vertex_from_hoppings_with_sign(kx, ky, qx, qy, "y", terms, sign)
    lhs = qx * vector_x + qy * vector_y
    h_plus = normal_state_hamiltonian_from_hoppings(kx + 0.5 * qx, ky + 0.5 * qy, terms)
    h_minus = normal_state_hamiltonian_from_hoppings(kx - 0.5 * qx, ky - 0.5 * qy, terms)
    rhs = h_plus - h_minus
    abs_error = float(np.linalg.norm(lhs - rhs))
    rhs_norm = float(np.linalg.norm(rhs))
    lhs_norm = float(np.linalg.norm(lhs))
    rel_error = abs_error / max(rhs_norm, EPS)
    return abs_error, rel_error, lhs_norm, rhs_norm


def validate_hopping_hermiticity(
    hopping_terms: Iterable[HoppingTerm],
    *,
    atol: float = 1e-12,
) -> float:
    """Return max ||t_-R - t_R^dagger|| over hopping terms."""

    terms = dict(hopping_terms)
    max_error = 0.0
    for (rx, ry), hopping in terms.items():
        partner_key = (-rx, -ry)
        if partner_key not in terms:
            raise ValueError(f"missing Hermitian partner for R={(rx, ry)}")
        error = float(np.linalg.norm(terms[partner_key] - hopping.conjugate().T))
        max_error = max(max_error, error)
    if max_error > atol:
        raise ValueError(f"hopping Hermiticity error {max_error:.6g} exceeds atol {atol:.6g}")
    return max_error
