"""Model-agnostic diagnostic normal-state nonlocal response."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lno327.bdg.kinematics import shifted_momenta
from lno327.bdg.spectrum import diagonalize_hermitian
from lno327.constants import E2_OVER_HBAR
from lno327.response.bubble import two_sided_band_basis_bubble_imag_axis
from lno327.response.config import KuboConfig
from lno327.response.occupations import fermi_function
from lno327.response.validation import validate_k_points_and_weights


@dataclass(frozen=True)
class ShiftedNormalEigensystem:
    """Normal-state eigensystems at k-q/2 and k+q/2."""

    energies_minus_eV: np.ndarray
    states_minus: np.ndarray
    occupations_minus: np.ndarray
    energies_plus_eV: np.ndarray
    states_plus: np.ndarray
    occupations_plus: np.ndarray


@dataclass(frozen=True)
class NormalNonlocalWorkspaceEntry:
    weight: float
    eigensystem: ShiftedNormalEigensystem
    vertices: tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class NormalNonlocalWorkspace:
    k_points: np.ndarray
    k_weights: np.ndarray
    q: np.ndarray
    config: KuboConfig
    shared_eigenbasis_q0: bool
    entries: tuple[NormalNonlocalWorkspaceEntry, ...]


def shifted_normal_eigensystem_from_model(
    spec,
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    config: KuboConfig,
) -> ShiftedNormalEigensystem:
    plus_momentum, minus_momentum = shifted_momenta(kx, ky, qx, qy)
    kx_plus, ky_plus = plus_momentum
    kx_minus, ky_minus = minus_momentum

    minus_bands = diagonalize_hermitian(spec.normal_hamiltonian(kx_minus, ky_minus))
    plus_bands = diagonalize_hermitian(spec.normal_hamiltonian(kx_plus, ky_plus))
    occupations_minus = fermi_function(
        minus_bands.energies,
        config.fermi_level_eV,
        config.temperature_eV,
    )
    occupations_plus = fermi_function(
        plus_bands.energies,
        config.fermi_level_eV,
        config.temperature_eV,
    )
    return ShiftedNormalEigensystem(
        minus_bands.energies,
        minus_bands.states,
        occupations_minus,
        plus_bands.energies,
        plus_bands.states,
        occupations_plus,
    )


def _q_is_zero(q_vector: np.ndarray) -> bool:
    return bool(np.all(q_vector == 0.0))


def midpoint_velocity_vertex_from_model(
    spec,
    kx: float,
    ky: float,
    direction: str,
    states_minus: np.ndarray,
    states_plus: np.ndarray,
) -> np.ndarray:
    if direction not in {"x", "y"}:
        raise ValueError("direction must be 'x' or 'y'")
    vertex = spec.velocity_operator(kx, ky, direction)
    return states_minus.conjugate().T @ vertex @ states_plus


def normal_current_current_kernel_imag_axis_from_model(
    spec,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    q: Sequence[float] | np.ndarray,
    k_weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    points, weights = validate_k_points_and_weights(k_points, config, k_weights)
    q_vector = np.asarray(q, dtype=float)
    if q_vector.shape != (2,):
        raise ValueError("q must have shape (2,)")
    workspace = precompute_normal_nonlocal_workspace_from_model(spec, points, config, q_vector, weights)
    return normal_current_current_kernel_imag_axis_from_workspace(workspace, config)


def precompute_normal_nonlocal_workspace_from_model(
    spec,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    q: Sequence[float] | np.ndarray,
    k_weights: Sequence[float] | np.ndarray | None = None,
) -> NormalNonlocalWorkspace:
    points, weights = validate_k_points_and_weights(k_points, config, k_weights)
    q_vector = np.asarray(q, dtype=float)
    if q_vector.shape != (2,):
        raise ValueError("q must have shape (2,)")
    qx, qy = (float(q_vector[0]), float(q_vector[1]))
    shared = _q_is_zero(q_vector)
    entries = []
    for weight, (kx_value, ky_value) in zip(weights, points, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        if shared:
            bands = diagonalize_hermitian(spec.normal_hamiltonian(kx, ky))
            occupations = fermi_function(
                bands.energies,
                config.fermi_level_eV,
                config.temperature_eV,
            )
            vertices = (
                bands.states.conjugate().T @ spec.velocity_operator(kx, ky, "x") @ bands.states,
                bands.states.conjugate().T @ spec.velocity_operator(kx, ky, "y") @ bands.states,
            )
            shifted = ShiftedNormalEigensystem(
                bands.energies,
                bands.states,
                occupations,
                bands.energies,
                bands.states,
                occupations,
            )
        else:
            shifted = shifted_normal_eigensystem_from_model(spec, kx, ky, qx, qy, config)
            vertices = (
                midpoint_velocity_vertex_from_model(
                    spec,
                    kx,
                    ky,
                    "x",
                    shifted.states_minus,
                    shifted.states_plus,
                ),
                midpoint_velocity_vertex_from_model(
                    spec,
                    kx,
                    ky,
                    "y",
                    shifted.states_minus,
                    shifted.states_plus,
                ),
            )
        entries.append(NormalNonlocalWorkspaceEntry(float(weight), shifted, vertices))
    return NormalNonlocalWorkspace(points, weights, q_vector, config, shared, tuple(entries))


def normal_current_current_kernel_imag_axis_from_workspace(
    workspace: NormalNonlocalWorkspace,
    config: KuboConfig | None = None,
) -> np.ndarray:
    eval_config = config or workspace.config
    response = np.zeros((2, 2), dtype=complex)
    for entry in workspace.entries:
        bands = entry.eigensystem
        response += entry.weight * two_sided_band_basis_bubble_imag_axis(
            bands.energies_minus_eV,
            bands.energies_plus_eV,
            bands.occupations_minus,
            bands.occupations_plus,
            entry.vertices,
            eval_config.omega_eV,
            eval_config.eta_eV,
        )

    if eval_config.output_si:
        response *= E2_OVER_HBAR
    return response


def c4_covariance_error(matrix_q: np.ndarray, matrix_rotated_q: np.ndarray) -> float:
    """Return ||K(Rq)-R K(q) R^T|| / max(||K(Rq)||, ||K(q)||, eps)."""

    matrix_q = np.asarray(matrix_q, dtype=complex)
    matrix_rotated_q = np.asarray(matrix_rotated_q, dtype=complex)
    if matrix_q.shape != (2, 2) or matrix_rotated_q.shape != (2, 2):
        raise ValueError("both response matrices must have shape (2, 2)")
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    expected = rotation @ matrix_q @ rotation.T
    scale = max(float(np.linalg.norm(matrix_q)), float(np.linalg.norm(matrix_rotated_q)), 1e-300)
    return float(np.linalg.norm(matrix_rotated_q - expected) / scale)
