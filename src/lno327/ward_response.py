"""Normal-state density/current response prototypes for Ward diagnostics.

This module is a normal-state prototype for Pi_{mu nu}(i omega_n, q), with
mu,nu = 0,x,y.  It is not a gauge-closed conductivity implementation and is
not a reflection or Casimir input.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .conductivity import KuboConfig, fermi_function
from .model import normal_state_hamiltonian, normal_state_mass_operator, normal_state_velocity_operator
from .tb_fourier import HoppingTerm, normal_state_hopping_terms, peierls_contact_vertex, peierls_current_vertex

HamiltonianBuilder = Callable[[float, float], np.ndarray]
VelocityBuilder = Callable[[float, float, str], np.ndarray]
MassBuilder = Callable[[float, float, str, str], np.ndarray]


def _validate_inputs(
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    q: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(k_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("k_points must have shape (n, 2) and must not be empty")
    if config.omega_eV < 0.0:
        raise ValueError("omega_eV must be non-negative")
    q_vector = np.asarray(q, dtype=float)
    if q_vector.shape != (2,):
        raise ValueError("q must have shape (2,)")
    if weights is None:
        normalized_weights = np.full(points.shape[0], 1.0 / points.shape[0])
    else:
        normalized_weights = np.asarray(weights, dtype=float)
        if normalized_weights.shape != (points.shape[0],):
            raise ValueError("k_weights must have shape (n,)")
    return points, normalized_weights, q_vector


def normal_density_current_response_imag_axis(
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    q: Sequence[float] | np.ndarray,
    k_weights: Sequence[float] | np.ndarray | None = None,
    hamiltonian: HamiltonianBuilder = normal_state_hamiltonian,
    velocity: VelocityBuilder = normal_state_velocity_operator,
    mass_operator: MassBuilder = normal_state_mass_operator,
    vertex_scheme: str = "midpoint",
    hopping_terms: Sequence[HoppingTerm] | None = None,
    contact_scheme: str = "none",
    contact_sign_convention: str = "plus",
) -> np.ndarray:
    """Return the normal-state 3x3 prototype Pi_{mu nu}(i omega_n, q).

    Vertex order is (density, current_x, current_y).  The density vertex is
    identity(4).  ``vertex_scheme="midpoint"`` uses midpoint normal-state
    velocity operators, preserving the original prototype behavior.
    ``vertex_scheme="peierls"`` uses the plus-sign Peierls current vertex from
    the Fourier/hopping representation.

    ``contact_scheme="none"`` keeps the original bubble-only behavior.
    ``contact_scheme="q0_mass_diagnostic"`` adds a q=0 mass/contact
    approximation only to the spatial-spatial block.  ``contact_scheme=
    "finite_q_peierls"`` adds the finite-q contact vertex from the second-order
    Peierls phase expansion.  This remains a normal-state Ward diagnostic
    prototype, not final finite-q conductivity and not a reflection/Casimir
    input; response-level signs, equal-time conventions, and closure still need
    to be fixed by a later derivation.
    """

    points, weights, q_vector = _validate_inputs(k_points, config, q, k_weights)
    qx, qy = (float(q_vector[0]), float(q_vector[1]))
    if vertex_scheme not in {"midpoint", "peierls"}:
        raise ValueError("vertex_scheme must be 'midpoint' or 'peierls'")
    if contact_scheme not in {"none", "q0_mass_diagnostic", "finite_q_peierls"}:
        raise ValueError("contact_scheme must be 'none', 'q0_mass_diagnostic', or 'finite_q_peierls'")
    if contact_sign_convention not in {"plus", "minus"}:
        raise ValueError("contact_sign_convention must be 'plus' or 'minus'")
    peierls_terms = None
    if vertex_scheme == "peierls" or contact_scheme == "finite_q_peierls":
        peierls_terms = list(normal_state_hopping_terms() if hopping_terms is None else hopping_terms)
    density_vertex = np.eye(4, dtype=complex)
    response = np.zeros((3, 3), dtype=complex)

    for weight, (kx_value, ky_value) in zip(weights, points, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        h_minus = hamiltonian(kx - 0.5 * qx, ky - 0.5 * qy)
        h_plus = hamiltonian(kx + 0.5 * qx, ky + 0.5 * qy)
        energies_minus, states_minus = np.linalg.eigh(h_minus)
        energies_plus, states_plus = np.linalg.eigh(h_plus)
        occupations_minus = fermi_function(
            energies_minus,
            config.fermi_level_eV,
            config.temperature_eV,
        )
        occupations_plus = fermi_function(
            energies_plus,
            config.fermi_level_eV,
            config.temperature_eV,
        )
        if vertex_scheme == "midpoint":
            current_x = velocity(kx, ky, "x")
            current_y = velocity(kx, ky, "y")
        else:
            current_x = peierls_current_vertex(
                kx,
                ky,
                qx,
                qy,
                "x",
                hopping_terms=peierls_terms,
                sign_convention="plus",
            )
            current_y = peierls_current_vertex(
                kx,
                ky,
                qx,
                qy,
                "y",
                hopping_terms=peierls_terms,
                sign_convention="plus",
            )
        vertices = (
            states_minus.conjugate().T @ density_vertex @ states_plus,
            states_minus.conjugate().T @ current_x @ states_plus,
            states_minus.conjugate().T @ current_y @ states_plus,
        )
        for m, energy_minus in enumerate(energies_minus):
            for n, energy_plus in enumerate(energies_plus):
                occupation_diff = float(occupations_minus[m] - occupations_plus[n])
                if occupation_diff == 0.0:
                    continue
                denominator = 1j * config.omega_eV + float(energy_minus - energy_plus)
                factor = -occupation_diff / denominator
                for mu in range(3):
                    for nu in range(3):
                        response[mu, nu] += (
                            weight
                            * factor
                            * vertices[mu][m, n]
                            * np.conjugate(vertices[nu][m, n])
                        )
        if contact_scheme in {"q0_mass_diagnostic", "finite_q_peierls"}:
            h_midpoint = hamiltonian(kx, ky)
            energies_midpoint, states_midpoint = np.linalg.eigh(h_midpoint)
            occupations_midpoint = fermi_function(
                energies_midpoint,
                config.fermi_level_eV,
                config.temperature_eV,
            )
            sign = 1.0 if contact_sign_convention == "plus" else -1.0
            directions = ("x", "y")
            for i, direction_i in enumerate(directions):
                for j, direction_j in enumerate(directions):
                    if contact_scheme == "q0_mass_diagnostic":
                        # q=0 local diamagnetic/contact approximation for small-q diagnostics only.
                        contact_matrix = mass_operator(kx, ky, direction_i, direction_j)
                    else:
                        # Finite-q Peierls contact vertex from the same hopping phase expansion.
                        contact_matrix = peierls_contact_vertex(
                            kx,
                            ky,
                            qx,
                            qy,
                            direction_i,
                            direction_j,
                            hopping_terms=peierls_terms,
                        )
                    band_contact = states_midpoint.conjugate().T @ contact_matrix @ states_midpoint
                    contact_value = np.sum(occupations_midpoint * np.diag(band_contact))
                    response[1 + i, 1 + j] += sign * weight * contact_value
    return response


def ward_residuals(response: np.ndarray, omega_eV: float, q: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return left and right Ward residual vectors for Pi_{mu nu}."""

    matrix = np.asarray(response, dtype=complex)
    if matrix.shape != (3, 3):
        raise ValueError("response must have shape (3, 3)")
    q_vector = np.asarray(q, dtype=float)
    if q_vector.shape != (2,):
        raise ValueError("q must have shape (2,)")
    qx, qy = (float(q_vector[0]), float(q_vector[1]))
    left = 1j * omega_eV * matrix[0, :] + qx * matrix[1, :] + qy * matrix[2, :]
    right = 1j * omega_eV * matrix[:, 0] + matrix[:, 1] * qx + matrix[:, 2] * qy
    return left, right


def ward_errors(response: np.ndarray, omega_eV: float, q: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    """Return normalized left, right, and max Ward residual errors."""

    left, right = ward_residuals(response, omega_eV, q)
    scale = max(float(np.linalg.norm(response)), 1e-300)
    left_error = float(np.linalg.norm(left) / scale)
    right_error = float(np.linalg.norm(right) / scale)
    return left_error, right_error, max(left_error, right_error)
