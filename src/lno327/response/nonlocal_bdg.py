"""Model-agnostic diagnostic BdG nonlocal current-current response."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lno327.bdg.kinematics import shifted_momenta
from lno327.bdg.nambu import charge_current_vertex_from_model
from lno327.bdg.spectrum import bdg_eigensystem_from_model
from lno327.constants import E2_OVER_HBAR
from lno327.response.bubble import two_sided_band_basis_bubble_imag_axis
from lno327.response.config import KuboConfig
from lno327.response.occupations import fermi_function
from lno327.response.validation import validate_k_points_and_weights


@dataclass(frozen=True)
class ShiftedBdGEigensystem:
    """BdG eigensystems at k-q/2 and k+q/2."""

    energies_minus_eV: np.ndarray
    states_minus: np.ndarray
    occupations_minus: np.ndarray
    energies_plus_eV: np.ndarray
    states_plus: np.ndarray
    occupations_plus: np.ndarray


def shifted_bdg_eigensystem_from_model(
    spec,
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    channel: str,
    config: KuboConfig,
) -> ShiftedBdGEigensystem:
    plus_momentum, minus_momentum = shifted_momenta(kx, ky, qx, qy)
    kx_plus, ky_plus = plus_momentum
    kx_minus, ky_minus = minus_momentum

    minus_bands = bdg_eigensystem_from_model(spec, kx_minus, ky_minus, channel)
    plus_bands = bdg_eigensystem_from_model(spec, kx_plus, ky_plus, channel)
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
    return ShiftedBdGEigensystem(
        minus_bands.energies,
        minus_bands.states,
        occupations_minus,
        plus_bands.energies,
        plus_bands.states,
        occupations_plus,
    )


def midpoint_bdg_current_vertex_from_model(
    spec,
    kx: float,
    ky: float,
    direction: str,
    states_minus: np.ndarray,
    states_plus: np.ndarray,
) -> np.ndarray:
    if direction not in {"x", "y"}:
        raise ValueError("direction must be 'x' or 'y'")
    vertex = charge_current_vertex_from_model(spec, kx, ky, direction)
    return states_minus.conjugate().T @ vertex @ states_plus


def bdg_current_current_kernel_imag_axis_from_model(
    spec,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    q: Sequence[float] | np.ndarray,
    channel: str,
    k_weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    points, weights = validate_k_points_and_weights(k_points, config, k_weights)
    q_vector = np.asarray(q, dtype=float)
    if q_vector.shape != (2,):
        raise ValueError("q must have shape (2,)")

    qx, qy = (float(q_vector[0]), float(q_vector[1]))
    response = np.zeros((2, 2), dtype=complex)
    for weight, (kx_value, ky_value) in zip(weights, points, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        bands = shifted_bdg_eigensystem_from_model(spec, kx, ky, qx, qy, channel, config)
        vertices = (
            midpoint_bdg_current_vertex_from_model(
                spec,
                kx,
                ky,
                "x",
                bands.states_minus,
                bands.states_plus,
            ),
            midpoint_bdg_current_vertex_from_model(
                spec,
                kx,
                ky,
                "y",
                bands.states_minus,
                bands.states_plus,
            ),
        )

        response += weight * two_sided_band_basis_bubble_imag_axis(
            bands.energies_minus_eV,
            bands.energies_plus_eV,
            bands.occupations_minus,
            bands.occupations_plus,
            vertices,
            config.omega_eV,
            config.eta_eV,
            prefactor=0.5,
        )

    if config.output_si:
        response *= E2_OVER_HBAR
    return response
