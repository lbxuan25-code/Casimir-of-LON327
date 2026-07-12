"""Fast chunk evaluator for the exact-static d-wave primitive contract.

The 48-component primitive definition is unchanged.  This evaluator batches the
three BdG diagonalization passes for each chunk, reuses one midpoint thermal
density matrix per point, and replaces repeated dynamic ``einsum`` planning with
fixed matrix contractions.
"""

from __future__ import annotations

import numpy as np

from lno327.bdg.finite_q import phase_phase_direct_vertex
from lno327.bdg.hamiltonian import bdg_hamiltonian_from_model_pairing
from lno327.response.finite_q_bdg import (
    bdg_contact_vertex_from_spec,
    bdg_vector_vertex_from_spec,
)
from lno327.response.occupations import fermi_function
from validation.lib.dwave_iterated_adaptive import (
    DWaveStaticIntegrandContext,
    _COLLECTIVE_CHANNELS,
    _COMPLEX_WIDTH,
    _COUNTERTERM_SLICE,
    _DIRECT_SLICE,
    _EM_CHANNELS,
    _EM_OBSERVABLE_SIGNS,
    _PHASE_DIRECT_SLICE,
    _UNIFIED_SLICE,
    _WARD_CONTACT_SLICE,
    _WARD_DELTA_V_SLICE,
    _WARD_EQUAL_SLICE,
    _static_factor_matrix,
    build_dwave_static_integrand_context as _build_reference_context,
)


def _batched_eigensystems(
    spec: object,
    ansatz: object,
    pairing_params: object,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrices = np.stack(
        [
            bdg_hamiltonian_from_model_pairing(
                spec,
                float(kx),
                float(ky),
                ansatz.mean_pairing(float(kx), float(ky), pairing_params),
            )
            for kx, ky in np.asarray(points, dtype=float)
        ],
        axis=0,
    )
    energies, states = np.linalg.eigh(matrices)
    return np.asarray(energies, dtype=float), np.asarray(states, dtype=complex)


def _thermal_density(states: np.ndarray, occupations: np.ndarray) -> np.ndarray:
    vectors = np.asarray(states, dtype=complex)
    weights = np.asarray(occupations, dtype=float)
    return 0.5 * (vectors * weights[np.newaxis, :]) @ vectors.conjugate().T


def _thermal_expectation_from_density(
    thermal_density: np.ndarray,
    vertex: np.ndarray,
) -> complex:
    return complex(
        np.sum(
            np.asarray(thermal_density, dtype=complex)
            * np.asarray(vertex, dtype=complex).T
        )
    )


def _band_transform_stack(
    states_minus: np.ndarray,
    vertices: np.ndarray,
    states_plus: np.ndarray,
) -> np.ndarray:
    """Match ``vertex_band`` minus-plus storage for a stack of vertices.

    ``vertex_band`` stores ``(U_plus^dagger V U_minus).T`` so loop indices remain
    ordered by minus-band then plus-band energies.  Preserving this transpose is
    essential for imaginary amplitude/phase cross channels.
    """

    plus_left = np.asarray(states_plus, dtype=complex).conjugate().T[
        np.newaxis, :, :
    ]
    middle = np.asarray(vertices, dtype=complex)
    minus_right = np.asarray(states_minus, dtype=complex)[np.newaxis, :, :]
    forward = (plus_left @ middle) @ minus_right
    return np.asarray(np.swapaxes(forward, -1, -2), dtype=complex)


def _unified_contraction(
    factor: np.ndarray,
    left_band: np.ndarray,
    right_band: np.ndarray,
) -> np.ndarray:
    left = np.asarray(left_band, dtype=complex).reshape(left_band.shape[0], -1)
    right = np.asarray(right_band, dtype=complex).reshape(right_band.shape[0], -1)
    weighted_left = left * np.asarray(factor, dtype=complex).reshape(1, -1)
    return np.asarray(0.5 * weighted_left @ right.conjugate().T, dtype=complex)


def _ward_equal_contraction(
    occupation_difference: np.ndarray,
    source_band: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_band, dtype=complex)
    weighted_density = (
        np.asarray(occupation_difference, dtype=float) * source[0]
    ).reshape(-1)
    return np.asarray(
        0.5 * source.conjugate().reshape(source.shape[0], -1) @ weighted_density,
        dtype=complex,
    )


class FastDWaveStaticIntegrandContext(DWaveStaticIntegrandContext):
    """Drop-in context with batched eigensystems and reused thermal density."""

    def evaluate_complex(self, k_points: np.ndarray) -> np.ndarray:
        points = np.asarray(k_points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("k_points must have shape (n, 2) with finite values")

        qx, qy = float(self.q_model[0]), float(self.q_model[1])
        q_half = 0.5 * np.asarray([qx, qy], dtype=float)
        spec, ansatz = self.spec, self.ansatz
        amp, opts = self.pairing_params, self.options

        energies_mid, states_mid = _batched_eigensystems(spec, ansatz, amp, points)
        energies_minus, states_minus = _batched_eigensystems(
            spec, ansatz, amp, points - q_half
        )
        energies_plus, states_plus = _batched_eigensystems(
            spec, ansatz, amp, points + q_half
        )
        occupations_mid = fermi_function(
            energies_mid, self.config.fermi_level_eV, self.config.temperature_eV
        )
        occupations_minus = fermi_function(
            energies_minus, self.config.fermi_level_eV, self.config.temperature_eV
        )
        occupations_plus = fermi_function(
            energies_plus, self.config.fermi_level_eV, self.config.temperature_eV
        )

        result = np.zeros((points.shape[0], _COMPLEX_WIDTH), dtype=complex)
        for index, (kx_value, ky_value) in enumerate(points):
            kx, ky = float(kx_value), float(ky_value)
            mid_states = states_mid[index]
            minus_states = states_minus[index]
            plus_states = states_plus[index]
            mid_occupations = occupations_mid[index]
            minus_occupations = occupations_minus[index]
            plus_occupations = occupations_plus[index]
            thermal_density = _thermal_density(mid_states, mid_occupations)

            vx = bdg_vector_vertex_from_spec(
                spec, kx, ky, qx, qy, "x", opts.current_vertex
            )
            vy = bdg_vector_vertex_from_spec(
                spec, kx, ky, qx, qy, "y", opts.current_vertex
            )
            collective_vertices = tuple(
                ansatz.collective_vertices(kx, ky, qx, qy, amp)
            )
            if len(collective_vertices) != _COLLECTIVE_CHANNELS:
                raise ValueError(
                    "adaptive integrand requires exactly two collective channels"
                )
            all_vertices = np.stack(
                (self.density, vx, vy, *collective_vertices), axis=0
            )
            all_band = _band_transform_stack(
                minus_states, all_vertices, plus_states
            )
            source_band = all_band[:_EM_CHANNELS]
            collective_band = all_band[_EM_CHANNELS:]
            observable_band = _EM_OBSERVABLE_SIGNS[:, None, None] * source_band
            left_band = np.concatenate((observable_band, collective_band), axis=0)
            right_band = np.concatenate((source_band, collective_band), axis=0)
            factor = _static_factor_matrix(
                energies_minus[index],
                minus_occupations,
                energies_plus[index],
                plus_occupations,
                self.config,
            )
            unified = _unified_contraction(factor, left_band, right_band)

            direct = np.zeros((_EM_CHANNELS, _EM_CHANNELS), dtype=complex)
            for i, direction_i in enumerate(("x", "y")):
                for j, direction_j in enumerate(("x", "y")):
                    contact = bdg_contact_vertex_from_spec(
                        spec,
                        kx,
                        ky,
                        qx,
                        qy,
                        direction_i,
                        direction_j,
                        opts.current_vertex,
                    )
                    direct[1 + i, 1 + j] = -_thermal_expectation_from_density(
                        thermal_density, contact
                    )

            delta_theta = ansatz.phase_pairing_matrix(kx, ky, qx, qy, amp)
            phase_direct_plus = _thermal_expectation_from_density(
                thermal_density, phase_phase_direct_vertex(delta_theta)
            )

            collective_zero = tuple(
                ansatz.collective_vertices(kx, ky, 0.0, 0.0, amp)
            )
            if len(collective_zero) != _COLLECTIVE_CHANNELS:
                raise ValueError(
                    "adaptive counterterm requires exactly two collective channels"
                )
            eta2_band = (
                mid_states.conjugate().T
                @ np.asarray(collective_zero[1], dtype=complex)
                @ mid_states
            )
            midpoint_factor = _static_factor_matrix(
                energies_mid[index],
                mid_occupations,
                energies_mid[index],
                mid_occupations,
                self.config,
            )
            eta2_bubble = 0.5 * np.sum(
                midpoint_factor * eta2_band * np.conjugate(eta2_band)
            )
            counterterm = -complex(eta2_bubble) * np.eye(
                _COLLECTIVE_CHANNELS, dtype=complex
            )

            occupation_difference = (
                np.asarray(minus_occupations, dtype=float)[:, None]
                - np.asarray(plus_occupations, dtype=float)[None, :]
            )
            ward_equal = _ward_equal_contraction(
                occupation_difference, source_band
            )
            ward_delta_v = np.zeros(_EM_CHANNELS, dtype=complex)
            for j, direction in enumerate(("x", "y"), start=1):
                vertex_plus = bdg_vector_vertex_from_spec(
                    spec,
                    kx + 0.5 * qx,
                    ky + 0.5 * qy,
                    qx,
                    qy,
                    direction,
                    opts.current_vertex,
                )
                vertex_minus = bdg_vector_vertex_from_spec(
                    spec,
                    kx - 0.5 * qx,
                    ky - 0.5 * qy,
                    qx,
                    qy,
                    direction,
                    opts.current_vertex,
                )
                ward_delta_v[j] = _thermal_expectation_from_density(
                    thermal_density, vertex_plus - vertex_minus
                )

            ward_contact = np.zeros(_EM_CHANNELS, dtype=complex)
            for i, qi in enumerate((qx, qy)):
                for j in range(2):
                    ward_contact[1 + j] += qi * direct[1 + i, 1 + j]

            row = result[index]
            row[_UNIFIED_SLICE] = unified.reshape(-1)
            row[_DIRECT_SLICE] = direct.reshape(-1)
            row[_PHASE_DIRECT_SLICE] = phase_direct_plus
            row[_COUNTERTERM_SLICE] = counterterm.reshape(-1)
            row[_WARD_EQUAL_SLICE] = ward_equal
            row[_WARD_DELTA_V_SLICE] = ward_delta_v
            row[_WARD_CONTACT_SLICE] = ward_contact

        return result


def build_dwave_static_integrand_context(
    spec: object,
    ansatz: object,
    q_model: np.ndarray,
    config: object,
    pairing_params: object | None = None,
    options: object | None = None,
) -> FastDWaveStaticIntegrandContext:
    reference = _build_reference_context(
        spec, ansatz, q_model, config, pairing_params, options
    )
    return FastDWaveStaticIntegrandContext(
        spec=reference.spec,
        ansatz=reference.ansatz,
        q_model=reference.q_model,
        config=reference.config,
        pairing_params=reference.pairing_params,
        options=reference.options,
        density=reference.density,
        delta0_eV=reference.delta0_eV,
    )


__all__ = [
    "FastDWaveStaticIntegrandContext",
    "build_dwave_static_integrand_context",
]
