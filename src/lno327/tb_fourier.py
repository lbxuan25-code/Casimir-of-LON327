"""Fourier/hopping representation of the normal-state Hamiltonian.

This module reconstructs the existing trigonometric H0(k) from hopping terms.
It is not a new model and is not wired into response, Ward, BdG, reflection, or
Casimir calculations.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from .model import NormalStateParameters

HoppingTerm = tuple[tuple[int, int], np.ndarray]
EPS = 1e-300


def _zero_matrix() -> np.ndarray:
    return np.zeros((4, 4), dtype=complex)


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
    """Return hopping terms t_R for the existing normal-state H0(k).

    The returned representation satisfies H0(k) = sum_R t_R exp(i k.R), using
    the same coefficients as ``model.normal_state_hamiltonian``.  The chemical
    potential is included as the R=(0,0) term ``-mu I``.
    """

    params = params or NormalStateParameters()
    basis = _basis_matrices()
    terms: dict[tuple[int, int], np.ndarray] = defaultdict(_zero_matrix)

    _add_constant(terms, params.tz_0, basis["tz"])
    _add_constant(terms, params.tx_0, basis["tx"])
    _add_constant(terms, params.tz_perp_0, basis["tz_perp"])
    _add_constant(terms, params.tx_perp_0, basis["tx_perp"])
    _add_constant(terms, -params.chemical_potential, np.eye(4, dtype=complex))

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
    hamiltonian = np.zeros((4, 4), dtype=complex)
    for (rx, ry), hopping in terms:
        phase = np.exp(1j * (kx * rx + ky * ry))
        hamiltonian += hopping * phase
    return hamiltonian


def sinc_stable(x):
    """Return sin(x)/x with the stable x->0 limit.

    ``x`` may be a scalar or a NumPy array.  Scalar input returns a float.
    """

    values = np.asarray(x, dtype=float)
    result = np.ones_like(values, dtype=float)
    mask = np.abs(values) > 1e-12
    result[mask] = np.sin(values[mask]) / values[mask]
    if np.isscalar(x):
        return float(result)
    return result


def peierls_current_vertex(
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    direction: str,
    params: NormalStateParameters | None = None,
    hopping_terms: Iterable[HoppingTerm] | None = None,
    sign_convention: str = "plus",
) -> np.ndarray:
    """Return the Peierls finite-q current vertex from hopping terms.

    The implemented convention is
    Gamma_i^P(k,q) = +/- i sum_R R_i t_R exp(i k.R) sinc(q.R/2).
    This helper is not connected to any response, conductivity, Ward-response,
    reflection, or Casimir calculation path.
    """

    if direction not in {"x", "y"}:
        raise ValueError("direction must be 'x' or 'y'")
    if sign_convention not in {"plus", "minus"}:
        raise ValueError("sign_convention must be 'plus' or 'minus'")
    sign = 1.0 if sign_convention == "plus" else -1.0
    terms = list(normal_state_hopping_terms(params) if hopping_terms is None else hopping_terms)
    vertex = np.zeros((4, 4), dtype=complex)
    for (rx, ry), hopping in terms:
        component = rx if direction == "x" else ry
        if component == 0:
            continue
        q_dot_r = qx * rx + qy * ry
        phase = np.exp(1j * (kx * rx + ky * ry))
        vertex += 1j * sign * component * hopping * phase * sinc_stable(0.5 * q_dot_r)
    return vertex


def peierls_vertex_ward_residual(
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    params: NormalStateParameters | None = None,
    hopping_terms: Iterable[HoppingTerm] | None = None,
    sign_convention: str = "plus",
) -> tuple[float, float, float, float]:
    """Return vertex-level Ward residual errors.

    H0(k+q/2)-H0(k-q/2) is evaluated with
    ``normal_state_hamiltonian_from_hoppings`` so the check stays within the
    same Fourier/hopping representation chain.
    """

    terms = list(normal_state_hopping_terms(params) if hopping_terms is None else hopping_terms)
    gamma_x = peierls_current_vertex(kx, ky, qx, qy, "x", params, terms, sign_convention)
    gamma_y = peierls_current_vertex(kx, ky, qx, qy, "y", params, terms, sign_convention)
    lhs = qx * gamma_x + qy * gamma_y
    h_plus = normal_state_hamiltonian_from_hoppings(kx + 0.5 * qx, ky + 0.5 * qy, params, terms)
    h_minus = normal_state_hamiltonian_from_hoppings(kx - 0.5 * qx, ky - 0.5 * qy, params, terms)
    rhs = h_plus - h_minus
    abs_error = float(np.linalg.norm(lhs - rhs))
    rhs_norm = float(np.linalg.norm(rhs))
    lhs_norm = float(np.linalg.norm(lhs))
    rel_error = abs_error / max(rhs_norm, EPS)
    return abs_error, rel_error, lhs_norm, rhs_norm


def validate_hopping_hermiticity(
    hopping_terms: Iterable[HoppingTerm] | None = None,
    *,
    atol: float = 1e-12,
) -> float:
    """Return max ||t_-R - t_R^dagger|| over hopping terms.

    A ``ValueError`` is raised if a required partner term is absent or the max
    error exceeds ``atol``.
    """

    terms = dict(normal_state_hopping_terms() if hopping_terms is None else hopping_terms)
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
