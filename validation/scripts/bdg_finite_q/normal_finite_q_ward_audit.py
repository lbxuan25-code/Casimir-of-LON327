#!/usr/bin/env python3
"""Normal-state finite-q Ward residual audit for diagnostic output only."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from lno327.conductivity import (  # noqa: E402
    KuboConfig,
    fermi_function,
    k_weights,
    negative_fermi_derivative,
    uniform_bz_mesh,
)
from lno327.model import normal_state_hamiltonian  # noqa: E402
from lno327.tb_fourier import (  # noqa: E402
    normal_state_hopping_terms,
    peierls_hamiltonian_contact_vertex,
    peierls_hamiltonian_vector_vertex,
    peierls_vertex_ward_residual,
)
from lno327.ward_response import (  # noqa: E402
    normal_physical_density_current_response_components_imag_axis,
    physical_ward_residuals,
)

WARD_COMPONENT_LABELS = ("density", "current_x", "current_y")
RESPONSE_NAMES = ("bubble", "direct", "total")
DIRECTION_VECTORS = {
    "x": (1.0, 0.0),
    "y": (0.0, 1.0),
    "diagonal": (1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)),
}
SUPPORTED_TWIST_COUNTS = (1, 4, 8, 16, 32)
SUPPORTED_ACTUAL_TWIST_COUNTS = (4, 8, 12, 16, 24)
MAX_JSON_SIZE_MB = 50.0
TARGET_JSON_SIZE_MB = 10.0
DEFAULT_TEMPERATURE_K = 30.0
KB_EV_PER_K_LOCAL = 8.617333262145e-5
PROGRESS_GREEN = "\033[92m"
PROGRESS_RESET = "\033[0m"


def _render_dot_progress(completed: int, total: int, width: int = 30) -> str:
    if total <= 0:
        filled = width
    else:
        filled = min(width, max(0, int(round(width * completed / total))))
    completed_dots = f"{PROGRESS_GREEN}{'●' * filled}{PROGRESS_RESET}"
    remaining_dots = "○" * (width - filled)
    return f"[{completed_dots}{remaining_dots}] {completed}/{total}"


def _print_progress(completed: int, total: int, enabled: bool = True) -> None:
    if not enabled:
        return
    print(f"\r{_render_dot_progress(completed, total)}", end="", flush=True)
    if completed >= total:
        print(flush=True)


def _radical_inverse(index: int, base: int) -> float:
    value = 0.0
    inverse_base = 1.0 / base
    factor = inverse_base
    while index > 0:
        digit = index % base
        value += digit * factor
        index //= base
        factor *= inverse_base
    return value


def _wrap_unit(value: float) -> float:
    return float(value % 1.0)


def _symmetry_paired_offsets(seed_offsets: list[tuple[float, float]]) -> list[tuple[float, float]]:
    offsets: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for tau_x, tau_y in seed_offsets:
        candidates = (
            (tau_x, tau_y),
            (1.0 - tau_x, 1.0 - tau_y),
            (tau_y, tau_x),
            (1.0 - tau_y, 1.0 - tau_x),
        )
        for candidate_x, candidate_y in candidates:
            pair = (round(_wrap_unit(candidate_x), 15), round(_wrap_unit(candidate_y), 15))
            if pair not in seen:
                seen.add(pair)
                offsets.append(pair)
    return offsets


def twist_offsets(twist_count: int, twist_mode: str = "halton") -> list[tuple[float, float]]:
    if twist_count not in SUPPORTED_TWIST_COUNTS:
        raise ValueError(f"twist_count must be one of {SUPPORTED_TWIST_COUNTS}")
    if twist_count == 1:
        seed_offsets = [(0.0, 0.0)]
    elif twist_count == 4:
        seed_offsets = [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)]
    else:
        seed_offsets = [(_radical_inverse(index, 2), _radical_inverse(index, 3)) for index in range(1, twist_count + 1)]
    if twist_mode == "halton":
        return seed_offsets
    if twist_mode == "symmetry_paired":
        return _symmetry_paired_offsets(seed_offsets)
    raise ValueError("twist_mode must be 'halton' or 'symmetry_paired'")


def actual_twist_offsets(actual_twist_count: int, twist_mode: str = "symmetry_paired") -> list[tuple[float, float]]:
    if actual_twist_count not in SUPPORTED_ACTUAL_TWIST_COUNTS:
        raise ValueError(f"actual_twist_count must be one of {SUPPORTED_ACTUAL_TWIST_COUNTS}")
    if twist_mode == "halton":
        return [(_radical_inverse(index, 2), _radical_inverse(index, 3)) for index in range(1, actual_twist_count + 1)]
    if twist_mode != "symmetry_paired":
        raise ValueError("twist_mode must be 'halton' or 'symmetry_paired'")
    offsets: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    seed_index = 1
    while len(offsets) < actual_twist_count:
        orbit = _symmetry_paired_offsets([(_radical_inverse(seed_index, 2), _radical_inverse(seed_index, 3))])
        new_orbit = [pair for pair in orbit if pair not in seen]
        if len(offsets) + len(new_orbit) > actual_twist_count:
            raise ValueError(
                f"cannot construct exactly {actual_twist_count} symmetry-paired twists from whole deterministic orbits"
            )
        for pair in new_orbit:
            seen.add(pair)
            offsets.append(pair)
        seed_index += 1
    return offsets


def uniform_bz_mesh_twisted(nk: int, twist_offset: tuple[float, float]) -> np.ndarray:
    if nk <= 0:
        raise ValueError("nk must be positive")
    tau_x, tau_y = twist_offset
    if not (0.0 <= tau_x < 1.0 and 0.0 <= tau_y < 1.0):
        raise ValueError("twist offsets must lie in [0, 1)")
    kx_values = -np.pi + 2.0 * np.pi * (np.arange(nk) + tau_x) / nk
    ky_values = -np.pi + 2.0 * np.pi * (np.arange(nk) + tau_y) / nk
    return np.array([(kx, ky) for kx in kx_values for ky in ky_values], dtype=float)


def _complex_vector_components(vector: np.ndarray) -> list[dict[str, float | str]]:
    array = np.asarray(vector, dtype=complex)
    if array.shape != (3,):
        raise ValueError("Ward residual vector must have shape (3,)")
    return [
        {
            "component": label,
            "real": float(np.real(value)),
            "imag": float(np.imag(value)),
        }
        for label, value in zip(WARD_COMPONENT_LABELS, array, strict=True)
    ]


def _complex_value(value: complex) -> dict[str, float]:
    return {
        "real": float(np.real(value)),
        "imag": float(np.imag(value)),
        "abs": float(abs(value)),
    }


def _component_vector(values: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "component": label,
            **_complex_value(value),
        }
        for label, value in zip(WARD_COMPONENT_LABELS, np.asarray(values, dtype=complex), strict=True)
    ]


def _vector_from_component_rows(rows: list[dict[str, Any]]) -> np.ndarray:
    values = [complex(row["real"], row["imag"]) for row in rows]
    return np.asarray(values, dtype=complex)


def _complex_matrix_entries(matrix: np.ndarray) -> list[list[dict[str, float]]]:
    array = np.asarray(matrix, dtype=complex)
    return [[_complex_value(value) for value in row] for row in array]


def _ward_contraction_decomposition(matrix: np.ndarray, omega_eV: float, q: np.ndarray) -> dict[str, Any]:
    response = np.asarray(matrix, dtype=complex)
    qx, qy = float(q[0]), float(q[1])
    left_rows = []
    right_rows = []
    for idx, label in enumerate(WARD_COMPONENT_LABELS):
        left_terms = {
            "iomega_Pi_0nu": 1j * omega_eV * response[0, idx],
            "qx_Pi_xnu": qx * response[1, idx],
            "qy_Pi_ynu": qy * response[2, idx],
        }
        right_terms = {
            "iomega_Pi_mu0": 1j * omega_eV * response[idx, 0],
            "minus_qx_Pi_mux": -qx * response[idx, 1],
            "minus_qy_Pi_muy": -qy * response[idx, 2],
        }
        left_rows.append(
            {
                "component": label,
                "terms": {name: _complex_value(value) for name, value in left_terms.items()},
                "residual": _complex_value(sum(left_terms.values())),
            }
        )
        right_rows.append(
            {
                "component": label,
                "terms": {name: _complex_value(value) for name, value in right_terms.items()},
                "residual": _complex_value(sum(right_terms.values())),
            }
        )
    return {
        "left_contraction": left_rows,
        "right_contraction": right_rows,
        "left_formula": "R_left[nu] = iomega*Pi[0,nu] + qx*Pi[x,nu] + qy*Pi[y,nu]",
        "right_formula": "R_right[mu] = iomega*Pi[mu,0] - qx*Pi[mu,x] - qy*Pi[mu,y]",
    }


def _response_residual_row(response_name: str, matrix: np.ndarray, omega_eV: float, q: np.ndarray) -> dict[str, Any]:
    left, right = physical_ward_residuals(matrix, omega_eV, q)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    return {
        "response_name": response_name,
        "residual_kind": "response_level",
        "residual_component_labels": list(WARD_COMPONENT_LABELS),
        "left_ward_residual_vector": _complex_vector_components(left),
        "right_ward_residual_vector": _complex_vector_components(right),
        "ward_contraction_decomposition": _ward_contraction_decomposition(matrix, omega_eV, q),
        "left_ward_residual_norm": left_norm,
        "right_ward_residual_norm": right_norm,
        "max_ward_residual_norm": float(max(left_norm, right_norm)),
        "valid_for_casimir_input": False,
    }


def _longitudinal_current_component(vector: np.ndarray, q: np.ndarray) -> complex:
    q_norm = float(np.linalg.norm(q))
    if q_norm <= 0.0:
        raise ValueError("q must be nonzero for longitudinal current projection")
    q_hat = np.asarray(q, dtype=float) / q_norm
    residual = np.asarray(vector, dtype=complex)
    return complex(q_hat[0] * residual[1] + q_hat[1] * residual[2])


def _longitudinal_current_scaling(
    response_rows: list[dict[str, Any]],
    q: np.ndarray,
) -> dict[str, Any]:
    by_name = {str(row["response_name"]): row for row in response_rows}
    q_norm = float(np.linalg.norm(q))
    output: dict[str, Any] = {
        "component": "longitudinal_current",
        "projection": {
            "qx_hat": float(q[0] / q_norm),
            "qy_hat": float(q[1] / q_norm),
            "definition": "qhat_x * current_x_residual + qhat_y * current_y_residual",
        },
    }
    for side, vector_key in (
        ("left", "left_ward_residual_vector"),
        ("right", "right_ward_residual_vector"),
    ):
        values = {}
        for response_name in RESPONSE_NAMES:
            vector = np.array(
                [
                    complex(component["real"], component["imag"])
                    for component in by_name[response_name][vector_key]
                ],
                dtype=complex,
            )
            values[response_name] = _longitudinal_current_component(vector, q)
        total = values["total"]
        output[f"{side}_contraction"] = {
            "bubble_residual": _complex_value(values["bubble"]),
            "direct_residual": _complex_value(values["direct"]),
            "total_residual": _complex_value(total),
            "total_residual_over_q": _complex_value(total / q_norm),
            "total_residual_over_q2": _complex_value(total / (q_norm * q_norm)),
        }
    return output


def _ward_residual_payload(matrix: np.ndarray, omega_eV: float, q: np.ndarray) -> dict[str, Any]:
    left, right = physical_ward_residuals(matrix, omega_eV, q)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    q_norm = float(np.linalg.norm(q))
    return {
        "diagnostic_only": True,
        "valid_for_casimir_input": False,
        "left_ward_residual_vector": _component_vector(left),
        "right_ward_residual_vector": _component_vector(right),
        "left_ward_residual_norm": left_norm,
        "right_ward_residual_norm": right_norm,
        "max_ward_residual_norm": float(max(left_norm, right_norm)),
        "left_ward_residual_over_q": _component_vector(left / q_norm),
        "left_ward_residual_over_q2": _component_vector(left / (q_norm * q_norm)),
        "left_ward_residual_over_q_norm": float(left_norm / q_norm),
        "left_ward_residual_over_q2_norm": float(left_norm / (q_norm * q_norm)),
    }


class RuntimeProfiler:
    def __init__(self) -> None:
        self.diagonalization_time_seconds = 0.0
        self.vertex_time_seconds = 0.0
        self.response_accumulation_time_seconds = 0.0
        self.adaptive_refinement_time_seconds = 0.0
        self.json_write_time_seconds = 0.0

    def merge(self, other: "RuntimeProfiler") -> None:
        self.diagonalization_time_seconds += other.diagonalization_time_seconds
        self.vertex_time_seconds += other.vertex_time_seconds
        self.response_accumulation_time_seconds += other.response_accumulation_time_seconds
        self.adaptive_refinement_time_seconds += other.adaptive_refinement_time_seconds
        self.json_write_time_seconds += other.json_write_time_seconds


class SpectralCache:
    def __init__(self) -> None:
        self.eigensystem_cache: dict[tuple[float, float, float, float, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.vector_vertex_cache: dict[tuple[float, float, float, float, str], np.ndarray] = {}
        self.contact_vertex_cache: dict[tuple[float, float, float, float, str, str], np.ndarray] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.hopping_terms = normal_state_hopping_terms()

    @staticmethod
    def _float_key(value: float) -> float:
        return round(float(value), 14)

    def eigensystem(self, kx: float, ky: float, config: KuboConfig, profiler: RuntimeProfiler) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (
            self._float_key(kx),
            self._float_key(ky),
            self._float_key(config.fermi_level_eV),
            self._float_key(config.temperature_eV),
            self._float_key(config.eta_eV),
        )
        if key in self.eigensystem_cache:
            self.cache_hits += 1
            return self.eigensystem_cache[key]
        self.cache_misses += 1
        start = time.perf_counter()
        energies, states = np.linalg.eigh(normal_state_hamiltonian(kx, ky))
        occupations = fermi_function(energies, config.fermi_level_eV, config.temperature_eV)
        profiler.diagonalization_time_seconds += time.perf_counter() - start
        self.eigensystem_cache[key] = (energies, states, occupations)
        return energies, states, occupations

    def vector_vertex(self, kx: float, ky: float, q: np.ndarray, direction: str, profiler: RuntimeProfiler) -> np.ndarray:
        key = (
            self._float_key(kx),
            self._float_key(ky),
            self._float_key(q[0]),
            self._float_key(q[1]),
            direction,
        )
        if key in self.vector_vertex_cache:
            self.cache_hits += 1
            return self.vector_vertex_cache[key]
        self.cache_misses += 1
        start = time.perf_counter()
        vertex = peierls_hamiltonian_vector_vertex(
            kx,
            ky,
            float(q[0]),
            float(q[1]),
            direction,
            hopping_terms=self.hopping_terms,
        )
        profiler.vertex_time_seconds += time.perf_counter() - start
        self.vector_vertex_cache[key] = vertex
        return vertex

    def contact_vertex(self, kx: float, ky: float, q: np.ndarray, direction_i: str, direction_j: str, profiler: RuntimeProfiler) -> np.ndarray:
        key = (
            self._float_key(kx),
            self._float_key(ky),
            self._float_key(q[0]),
            self._float_key(q[1]),
            direction_i,
            direction_j,
        )
        if key in self.contact_vertex_cache:
            self.cache_hits += 1
            return self.contact_vertex_cache[key]
        self.cache_misses += 1
        start = time.perf_counter()
        vertex = peierls_hamiltonian_contact_vertex(
            kx,
            ky,
            float(q[0]),
            float(q[1]),
            direction_i,
            direction_j,
            hopping_terms=self.hopping_terms,
        )
        profiler.vertex_time_seconds += time.perf_counter() - start
        self.contact_vertex_cache[key] = vertex
        return vertex


def _cached_normal_components_and_translation(
    points: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    q: np.ndarray,
    cache: SpectralCache,
    profiler: RuntimeProfiler,
) -> dict[str, Any]:
    qx, qy = float(q[0]), float(q[1])
    rho = np.eye(4, dtype=complex)
    bubble = np.zeros((3, 3), dtype=complex)
    direct = np.zeros((3, 3), dtype=complex)
    actual_equal_time = np.zeros(3, dtype=complex)
    shifted_equal_time_reference = np.zeros(3, dtype=complex)
    contact_contraction = np.zeros(3, dtype=complex)

    for weight, (kx_value, ky_value) in zip(weights, points, strict=True):
        start = time.perf_counter()
        kx = float(kx_value)
        ky = float(ky_value)
        energies_minus, states_minus, occupations_minus = cache.eigensystem(
            kx - 0.5 * qx,
            ky - 0.5 * qy,
            config,
            profiler,
        )
        energies_plus, states_plus, occupations_plus = cache.eigensystem(
            kx + 0.5 * qx,
            ky + 0.5 * qy,
            config,
            profiler,
        )
        energies_midpoint, states_midpoint, occupations_midpoint = cache.eigensystem(kx, ky, config, profiler)
        vector_x = cache.vector_vertex(kx, ky, q, "x", profiler)
        vector_y = cache.vector_vertex(kx, ky, q, "y", profiler)
        observable_vertices = (rho, -vector_x, -vector_y)
        source_vertices = (rho, vector_x, vector_y)
        observable_matrices = tuple(
            states_minus.conjugate().T @ vertex @ states_plus for vertex in observable_vertices
        )
        source_matrices = tuple(states_minus.conjugate().T @ vertex @ states_plus for vertex in source_vertices)
        rho_band = states_minus.conjugate().T @ rho @ states_plus
        for m, energy_minus in enumerate(energies_minus):
            for n, energy_plus in enumerate(energies_plus):
                occupation_diff = float(occupations_minus[m] - occupations_plus[n])
                if occupation_diff == 0.0:
                    continue
                denominator = 1j * config.omega_eV + float(energy_minus - energy_plus)
                factor = occupation_diff / denominator
                for mu, observable_matrix in enumerate(observable_matrices):
                    for nu, source_matrix in enumerate(source_matrices):
                        bubble[mu, nu] += (
                            weight
                            * factor
                            * observable_matrix[m, n]
                            * np.conjugate(source_matrix[m, n])
                        )
                actual_equal_time += weight * np.array(
                    [
                        occupation_diff * rho_band[m, n] * np.conjugate(source_matrix[m, n])
                        for source_matrix in source_matrices
                    ],
                    dtype=complex,
                )
        for source_index, source_direction in enumerate(("x", "y"), start=1):
            shifted_vertex_reference = cache.vector_vertex(
                kx + 0.5 * qx,
                ky + 0.5 * qy,
                q,
                source_direction,
                profiler,
            ) - cache.vector_vertex(
                kx - 0.5 * qx,
                ky - 0.5 * qy,
                q,
                source_direction,
                profiler,
            )
            band_shifted_reference = states_midpoint.conjugate().T @ shifted_vertex_reference @ states_midpoint
            shifted_equal_time_reference[source_index] += weight * np.sum(
                occupations_midpoint * np.diag(band_shifted_reference)
            )
        for i, direction_i in enumerate(("x", "y")):
            for j, direction_j in enumerate(("x", "y")):
                contact_matrix = cache.contact_vertex(kx, ky, q, direction_i, direction_j, profiler)
                band_contact = states_midpoint.conjugate().T @ contact_matrix @ states_midpoint
                physical_direct_contact = -np.sum(occupations_midpoint * np.diag(band_contact))
                direct[1 + i, 1 + j] += weight * physical_direct_contact
                contact_contraction[1 + j] += weight * (qx if direction_i == "x" else qy) * physical_direct_contact
        profiler.response_accumulation_time_seconds += time.perf_counter() - start
    total = bubble + direct
    total_residual, _ = physical_ward_residuals(total, config.omega_eV, q)
    shifted_equal_time_plus_contact = shifted_equal_time_reference + contact_contraction
    translation_error = actual_equal_time - shifted_equal_time_reference
    return {
        "response_components": {"bubble": bubble, "direct": direct, "total": total},
        "actual_equal_time": actual_equal_time,
        "shifted_equal_time_reference": shifted_equal_time_reference,
        "contact_contraction": contact_contraction,
        "shifted_equal_time_plus_contact": shifted_equal_time_plus_contact,
        "translation_error": translation_error,
        "translation_error_minus_total_residual": translation_error - total_residual,
        "total_ward_residual": total_residual,
    }


def _fs_window_eV(config: KuboConfig, q: np.ndarray, fs_window_factor: float) -> float:
    thermal = fs_window_factor * config.temperature_eV
    vf_placeholder = 1.0
    vq_placeholder = vf_placeholder * float(np.linalg.norm(q))
    return max(thermal, vq_placeholder, config.eta_eV, 1e-5)


def _adaptive_refined_quadrature(
    nk: int,
    twist_offset: tuple[float, float],
    config: KuboConfig,
    q: np.ndarray,
    refine_level: int,
    fs_window_factor: float,
    adaptive_mode: str,
    cache: SpectralCache,
    profiler: RuntimeProfiler,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    start = time.perf_counter()
    base_points = uniform_bz_mesh_twisted(nk, twist_offset)
    parent_cell_weight = 1.0 / float(base_points.shape[0])
    if refine_level <= 0:
        weights = np.full(base_points.shape[0], parent_cell_weight, dtype=float)
        weight_sum = float(np.sum(weights))
        if abs(weight_sum - 1.0) >= 1e-12 or np.any(weights <= 0.0):
            raise ValueError("non-adaptive quadrature weights failed sanity checks")
        profiler.adaptive_refinement_time_seconds += time.perf_counter() - start
        return base_points, weights, {
            "number_of_base_nodes": int(base_points.shape[0]),
            "number_of_refined_nodes": 0,
            "effective_total_nodes": int(base_points.shape[0]),
            "number_of_unrefined_parent_cells": int(base_points.shape[0]),
            "number_of_refined_parent_cells": 0,
            "number_of_q_partner_refined_cells": 0,
            "children_per_refined_cell": 0,
            "parent_cell_weight": parent_cell_weight,
            "refined_child_weight": 0.0,
            "weight_sum": weight_sum,
            "abs_weight_sum_minus_one": float(abs(weight_sum - 1.0)),
            "adaptive_quadrature_rule": (
                "unrefined parent cells use one representative node with the full parent cell weight"
            ),
            "adaptive_mode": adaptive_mode,
            "q_covariant_partner_rule": "not_applied_for_refine_level_0",
            "fs_window_eV": _fs_window_eV(config, q, fs_window_factor),
            "vf_estimate_note": "placeholder_vF_estimate_1_model_unit_used_for_diagnostic_only",
        }
    window = _fs_window_eV(config, q, fs_window_factor)
    cell_step = 2.0 * np.pi / nk
    quadrature_points: list[tuple[float, float]] = []
    quadrature_weights: list[float] = []
    refined_node_count = 0
    refined_parent_count = 0
    q_partner_refined_count = 0
    unrefined_parent_count = 0
    offsets_1d = (np.arange(2**refine_level) + 0.5) / (2**refine_level) - 0.5
    children_per_refined_cell = int(len(offsets_1d) * len(offsets_1d))
    refined_child_weight = parent_cell_weight / children_per_refined_cell
    for kx, ky in base_points:
        energies, _, _ = cache.eigensystem(float(kx), float(ky), config, profiler)
        center_trigger = float(np.min(np.abs(energies - config.fermi_level_eV))) < window
        partner_trigger = False
        if adaptive_mode == "q_covariant":
            for partner_kx, partner_ky in (
                (float(kx + 0.5 * q[0]), float(ky + 0.5 * q[1])),
                (float(kx - 0.5 * q[0]), float(ky - 0.5 * q[1])),
            ):
                partner_energies, _, _ = cache.eigensystem(partner_kx, partner_ky, config, profiler)
                if float(np.min(np.abs(partner_energies - config.fermi_level_eV))) < window:
                    partner_trigger = True
                    break
        should_refine = center_trigger or partner_trigger
        if not should_refine:
            quadrature_points.append((float(kx), float(ky)))
            quadrature_weights.append(parent_cell_weight)
            unrefined_parent_count += 1
            continue
        refined_parent_count += 1
        if partner_trigger and not center_trigger:
            q_partner_refined_count += 1
        child_weight_sum = 0.0
        for dx in offsets_1d:
            for dy in offsets_1d:
                quadrature_points.append((float(kx + dx * cell_step), float(ky + dy * cell_step)))
                quadrature_weights.append(refined_child_weight)
                child_weight_sum += refined_child_weight
                refined_node_count += 1
        if abs(child_weight_sum - parent_cell_weight) >= 1e-12:
            raise ValueError("adaptive child weights do not sum to parent cell weight")
    points = np.asarray(quadrature_points, dtype=float)
    weights = np.asarray(quadrature_weights, dtype=float)
    weight_sum = float(np.sum(weights))
    if abs(weight_sum - 1.0) >= 1e-12:
        raise ValueError(f"adaptive quadrature weights sum to {weight_sum}, not 1")
    if np.any(weights <= 0.0):
        raise ValueError("adaptive quadrature weights must be positive")
    profiler.adaptive_refinement_time_seconds += time.perf_counter() - start
    return points, weights, {
        "number_of_base_nodes": int(base_points.shape[0]),
        "number_of_refined_nodes": int(refined_node_count),
        "effective_total_nodes": int(points.shape[0]),
        "number_of_unrefined_parent_cells": int(unrefined_parent_count),
        "number_of_refined_parent_cells": int(refined_parent_count),
        "number_of_q_partner_refined_cells": int(q_partner_refined_count),
        "children_per_refined_cell": int(children_per_refined_cell),
        "parent_cell_weight": float(parent_cell_weight),
        "refined_child_weight": float(refined_child_weight),
        "weight_sum": weight_sum,
        "abs_weight_sum_minus_one": float(abs(weight_sum - 1.0)),
        "adaptive_quadrature_rule": (
            "refined parent cells are replaced by subcell quadrature nodes; "
            "parent and children are not double-counted"
        ),
        "adaptive_mode": adaptive_mode,
        "q_covariant_partner_rule": (
            "refine a parent cell if the midpoint k or either q-related point k+q/2, k-q/2 lies in the FS window"
        ),
        "fs_window_eV": window,
        "vf_estimate_note": "placeholder_vF_estimate_1_model_unit_used_for_diagnostic_only",
    }


def _average_compact_evaluations(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    inverse_count = 1.0 / len(evaluations)
    response = {name: np.zeros((3, 3), dtype=complex) for name in RESPONSE_NAMES}
    vectors = {
        "actual_equal_time": np.zeros(3, dtype=complex),
        "shifted_equal_time_reference": np.zeros(3, dtype=complex),
        "contact_contraction": np.zeros(3, dtype=complex),
        "shifted_equal_time_plus_contact": np.zeros(3, dtype=complex),
        "translation_error": np.zeros(3, dtype=complex),
        "translation_error_minus_total_residual": np.zeros(3, dtype=complex),
        "total_ward_residual": np.zeros(3, dtype=complex),
    }
    for evaluation in evaluations:
        for name in RESPONSE_NAMES:
            response[name] += evaluation["response_components"][name]
        for name in vectors:
            vectors[name] += evaluation[name]
    return {
        "response_components": {name: response[name] * inverse_count for name in RESPONSE_NAMES},
        **{name: vectors[name] * inverse_count for name in vectors},
    }


def _compact_summary_row(
    *,
    nk: int,
    q_direction: str,
    q: np.ndarray,
    twist_count: int,
    twist_mode: str,
    adaptive_refine_level: int,
    node_counts: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    q_norm = float(np.linalg.norm(q))
    response = evaluation["response_components"]
    residual_norm = float(np.linalg.norm(evaluation["total_ward_residual"]))
    translation_error_norm = float(np.linalg.norm(evaluation["translation_error"]))
    return {
        "diagnostic_only": True,
        "valid_for_casimir_input": False,
        "nk": int(nk),
        "q_direction": q_direction,
        "q_norm": q_norm,
        "twist_count": int(twist_count),
        "twist_mode": twist_mode,
        "adaptive_refine_levels": int(adaptive_refine_level),
        "number_of_base_nodes": int(node_counts["number_of_base_nodes"]),
        "number_of_refined_nodes": int(node_counts["number_of_refined_nodes"]),
        "effective_total_nodes": int(node_counts["effective_total_nodes"]),
        "weight_sum": float(node_counts.get("weight_sum", 1.0)),
        "abs_weight_sum_minus_one": float(node_counts.get("abs_weight_sum_minus_one", 0.0)),
        "number_of_unrefined_parent_cells": int(node_counts.get("number_of_unrefined_parent_cells", 0)),
        "number_of_refined_parent_cells": int(node_counts.get("number_of_refined_parent_cells", 0)),
        "number_of_q_partner_refined_cells": int(node_counts.get("number_of_q_partner_refined_cells", 0)),
        "children_per_refined_cell": int(node_counts.get("children_per_refined_cell", 0)),
        "parent_cell_weight": float(node_counts.get("parent_cell_weight", 0.0)),
        "refined_child_weight": float(node_counts.get("refined_child_weight", 0.0)),
        "total_ward_residual_norm": residual_norm,
        "total_ward_residual_over_q_norm": float(residual_norm / q_norm),
        "translation_error_norm": translation_error_norm,
        "translation_error_over_q_norm": float(translation_error_norm / q_norm),
        "shifted_equal_time_plus_contact_norm": float(np.linalg.norm(evaluation["shifted_equal_time_plus_contact"])),
        "translation_error_minus_total_residual_norm": float(
            np.linalg.norm(evaluation["translation_error_minus_total_residual"])
        ),
        "total_response_norm": float(np.linalg.norm(response["total"])),
        "current_current_block_norm": float(np.linalg.norm(response["total"][1:3, 1:3])),
    }


def _compact_case_worker(args: tuple[Any, ...]) -> dict[str, Any]:
    (
        nk,
        q_direction,
        q_value,
        direction,
        twist_counts,
        use_actual_twist_counts,
        twist_mode,
        adaptive_mode,
        adaptive_refine_levels,
        fs_window_factor,
        omega_eV,
        temperature_K,
        eta_eV,
    ) = args
    profiler = RuntimeProfiler()
    cache = SpectralCache()
    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        output_si=False,
    )
    q = float(q_value) * np.asarray(direction, dtype=float)
    twist_rows: list[dict[str, Any]] = []
    adaptive_rows: list[dict[str, Any]] = []
    convergence_items: list[dict[str, Any]] = []
    for twist_count in twist_counts:
        offsets = (
            actual_twist_offsets(int(twist_count), twist_mode)
            if use_actual_twist_counts
            else twist_offsets(int(twist_count), twist_mode)
        )
        evaluations = []
        total_base_nodes = 0
        for offset in offsets:
            points = uniform_bz_mesh_twisted(int(nk), offset)
            weights = k_weights(points)
            total_base_nodes += int(points.shape[0])
            evaluations.append(_cached_normal_components_and_translation(points, weights, config, q, cache, profiler))
        averaged = _average_compact_evaluations(evaluations)
        node_counts = {
            "number_of_base_nodes": total_base_nodes,
            "number_of_refined_nodes": 0,
            "effective_total_nodes": total_base_nodes,
        }
        row = _compact_summary_row(
            nk=int(nk),
            q_direction=str(q_direction),
            q=q,
            twist_count=len(offsets),
            twist_mode=twist_mode,
            adaptive_refine_level=0,
            node_counts=node_counts,
            evaluation=averaged,
        )
        row["requested_twist_count"] = int(twist_count)
        row["requested_actual_twist_count"] = int(twist_count) if use_actual_twist_counts else int(len(offsets))
        row["actual_twist_count"] = int(len(offsets))
        row["twist_offset_rule"] = (
            "actual_twist_count deterministic symmetry-paired orbits"
            if use_actual_twist_counts and twist_mode == "symmetry_paired"
            else "legacy seed twist count expansion"
        )
        row["adaptive_mode"] = "none"
        row["temperature_K"] = float(temperature_K)
        row["eta_eV"] = float(eta_eV)
        twist_rows.append(row)
        convergence_items.append(
            {
                "q_direction": str(q_direction),
                "q_norm": float(np.linalg.norm(q)),
                "temperature_K": float(temperature_K),
                "eta_eV": float(eta_eV),
                "nk": int(nk),
                "twist_count": int(len(offsets)),
                "total_response": averaged["response_components"]["total"],
                "current_current_block": averaged["response_components"]["total"][1:3, 1:3],
                "ward_residual": averaged["total_ward_residual"],
                "translation_error": averaged["translation_error"],
                "effective_total_nodes": int(total_base_nodes),
            }
        )
        for refine_level in (() if adaptive_mode == "none" else adaptive_refine_levels):
            adaptive_evaluations = []
            total_counts = {
                "number_of_base_nodes": 0,
                "number_of_refined_nodes": 0,
                "effective_total_nodes": 0,
                "number_of_unrefined_parent_cells": 0,
                "number_of_refined_parent_cells": 0,
                "number_of_q_partner_refined_cells": 0,
                "children_per_refined_cell": 0,
                "parent_cell_weight": 0.0,
                "refined_child_weight": 0.0,
                "weight_sum": 0.0,
                "abs_weight_sum_minus_one": 0.0,
            }
            adaptive_rule = ""
            for offset in offsets:
                points, weights, counts = _adaptive_refined_quadrature(
                    int(nk),
                    offset,
                    config,
                    q,
                    int(refine_level),
                    float(fs_window_factor),
                    adaptive_mode,
                    cache,
                    profiler,
                )
                for key in (
                    "number_of_base_nodes",
                    "number_of_refined_nodes",
                    "effective_total_nodes",
                    "number_of_unrefined_parent_cells",
                    "number_of_refined_parent_cells",
                    "number_of_q_partner_refined_cells",
                ):
                    total_counts[key] += int(counts[key])
                total_counts["children_per_refined_cell"] = int(counts["children_per_refined_cell"])
                total_counts["parent_cell_weight"] = float(counts["parent_cell_weight"])
                total_counts["refined_child_weight"] = float(counts["refined_child_weight"])
                total_counts["weight_sum"] += float(counts["weight_sum"])
                total_counts["abs_weight_sum_minus_one"] = max(
                    float(total_counts["abs_weight_sum_minus_one"]),
                    float(counts["abs_weight_sum_minus_one"]),
                )
                adaptive_rule = str(counts["adaptive_quadrature_rule"])
                adaptive_evaluations.append(
                    _cached_normal_components_and_translation(points, weights, config, q, cache, profiler)
                )
            total_counts["weight_sum"] = float(total_counts["weight_sum"] / max(len(offsets), 1))
            if abs(float(total_counts["weight_sum"]) - 1.0) >= 1e-12:
                raise ValueError("twist-averaged adaptive quadrature weight sum sanity check failed")
            averaged_adaptive = _average_compact_evaluations(adaptive_evaluations)
            adaptive_row = _compact_summary_row(
                nk=int(nk),
                q_direction=str(q_direction),
                q=q,
                twist_count=len(offsets),
                twist_mode=twist_mode,
                adaptive_refine_level=int(refine_level),
                node_counts=total_counts,
                evaluation=averaged_adaptive,
            )
            adaptive_row["requested_twist_count"] = int(twist_count)
            adaptive_row["requested_actual_twist_count"] = int(twist_count) if use_actual_twist_counts else int(len(offsets))
            adaptive_row["actual_twist_count"] = int(len(offsets))
            adaptive_row["twist_offset_rule"] = (
                "actual_twist_count deterministic symmetry-paired orbits"
                if use_actual_twist_counts and twist_mode == "symmetry_paired"
                else "legacy seed twist count expansion"
            )
            adaptive_row["adaptive_mode"] = adaptive_mode
            adaptive_row["temperature_K"] = float(temperature_K)
            adaptive_row["eta_eV"] = float(eta_eV)
            adaptive_row["fs_window_factor"] = float(fs_window_factor)
            adaptive_row["fs_window_note"] = "E_window=max(fs_window_factor*kBT, placeholder_vF*|q|, eta_eff, E_min)"
            adaptive_row["adaptive_quadrature_rule"] = adaptive_rule
            adaptive_rows.append(adaptive_row)
    return {
        "twist_rows": twist_rows,
        "adaptive_rows": adaptive_rows,
        "convergence_items": convergence_items,
        "profile": {
            "diagonalization_time_seconds": profiler.diagonalization_time_seconds,
            "vertex_time_seconds": profiler.vertex_time_seconds,
            "response_accumulation_time_seconds": profiler.response_accumulation_time_seconds,
            "adaptive_refinement_time_seconds": profiler.adaptive_refinement_time_seconds,
            "cache_hits": cache.cache_hits,
            "cache_misses": cache.cache_misses,
        },
    }


def _operator_level_rows(points: np.ndarray, q: np.ndarray) -> list[dict[str, Any]]:
    qx, qy = float(q[0]), float(q[1])
    rows: list[dict[str, Any]] = []
    for kx_value, ky_value in points:
        kx = float(kx_value)
        ky = float(ky_value)
        abs_error, rel_error, lhs_norm, rhs_norm = peierls_vertex_ward_residual(kx, ky, qx, qy)
        rows.append(
            {
                "residual_kind": "operator_level",
                "identity": "q_x V_x(k,q) + q_y V_y(k,q) = H(k+q/2)-H(k-q/2)",
                "k_model": [kx, ky],
                "absolute_error_norm": float(abs_error),
                "relative_error_norm": float(rel_error),
                "lhs_norm": float(lhs_norm),
                "rhs_norm": float(rhs_norm),
            }
        )
    return rows


def _operator_level_second_order_contact_ward(points: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    qx, qy = float(q[0]), float(q[1])
    peierls_terms = normal_state_hopping_terms()
    rows: list[dict[str, Any]] = []
    max_absolute_error = -1.0
    max_relative_error = -1.0
    max_error_k_model: list[float] | None = None
    max_error_component = ""

    for kx_value, ky_value in points:
        kx = float(kx_value)
        ky = float(ky_value)
        component_rows: list[dict[str, Any]] = []
        for component_label, source_direction in (("current_x", "x"), ("current_y", "y")):
            implemented_contact_contraction = np.zeros((4, 4), dtype=complex)
            hessian_q0_reference = np.zeros((4, 4), dtype=complex)
            for q_component, observable_direction in ((qx, "x"), (qy, "y")):
                implemented_contact_contraction += q_component * peierls_hamiltonian_contact_vertex(
                    kx,
                    ky,
                    qx,
                    qy,
                    observable_direction,
                    source_direction,
                    hopping_terms=peierls_terms,
                )
                hessian_q0_reference += q_component * peierls_hamiltonian_contact_vertex(
                    kx,
                    ky,
                    0.0,
                    0.0,
                    observable_direction,
                    source_direction,
                    hopping_terms=peierls_terms,
                )

            finite_difference_current_vertex_reference = peierls_hamiltonian_vector_vertex(
                kx + 0.5 * qx,
                ky + 0.5 * qy,
                qx,
                qy,
                source_direction,
                hopping_terms=peierls_terms,
            ) - peierls_hamiltonian_vector_vertex(
                kx - 0.5 * qx,
                ky - 0.5 * qy,
                qx,
                qy,
                source_direction,
                hopping_terms=peierls_terms,
            )
            residual_vs_finite_difference = (
                implemented_contact_contraction - finite_difference_current_vertex_reference
            )
            residual_vs_hessian_q0 = implemented_contact_contraction - hessian_q0_reference
            absolute_error = float(np.linalg.norm(residual_vs_finite_difference))
            reference_norm = float(np.linalg.norm(finite_difference_current_vertex_reference))
            relative_error = absolute_error / max(reference_norm, 1e-300)
            if absolute_error > max_absolute_error:
                max_absolute_error = absolute_error
                max_error_k_model = [kx, ky]
                max_error_component = component_label
            max_relative_error = max(max_relative_error, relative_error)
            component_rows.append(
                {
                    "component": component_label,
                    "implemented_contact_contraction": _complex_matrix_entries(
                        implemented_contact_contraction
                    ),
                    "finite_difference_current_vertex_reference": _complex_matrix_entries(
                        finite_difference_current_vertex_reference
                    ),
                    "hessian_q0_reference": _complex_matrix_entries(hessian_q0_reference),
                    "residual_vs_finite_difference_reference": {
                        "matrix": _complex_matrix_entries(residual_vs_finite_difference),
                        "norm": absolute_error,
                        "reference_norm": reference_norm,
                        "relative_error_norm": float(relative_error),
                    },
                    "residual_vs_hessian_q0_reference": {
                        "matrix": _complex_matrix_entries(residual_vs_hessian_q0),
                        "norm": float(np.linalg.norm(residual_vs_hessian_q0)),
                        "reference_norm": float(np.linalg.norm(hessian_q0_reference)),
                    },
                }
            )
        rows.append(
            {
                "k_model": [kx, ky],
                "components": component_rows,
            }
        )

    return {
        "diagnostic_only": True,
        "valid_for_casimir_input": False,
        "residual_kind": "operator_level",
        "identity": (
            "Hamiltonian Peierls convention check: "
            "q_i M_ij(k,q) = V_j(k+q/2,q) - V_j(k-q/2,q). "
            "The physical current is -V_j, so this block tracks the implemented "
            "Hamiltonian contact/source-vertex convention without changing response formulas."
        ),
        "implemented_contact_contraction": "qx * M[x,j](k,q) + qy * M[y,j](k,q)",
        "finite_difference_current_vertex_reference": "V_j(k+q/2,q) - V_j(k-q/2,q)",
        "hessian_q0_reference": "qx * M[x,j](k,0) + qy * M[y,j](k,0)",
        "residual_vs_finite_difference_reference": (
            "implemented_contact_contraction - finite_difference_current_vertex_reference"
        ),
        "residual_vs_hessian_q0_reference": "implemented_contact_contraction - hessian_q0_reference",
        "max_absolute_error_norm": float(max_absolute_error),
        "max_relative_error_norm": float(max_relative_error),
        "max_error_k_model": max_error_k_model,
        "max_error_component": max_error_component,
        "per_k_residuals": rows,
    }


def _shifted_pair_response_components(
    points: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    q: np.ndarray,
) -> dict[str, np.ndarray]:
    qx, qy = float(q[0]), float(q[1])
    peierls_terms = normal_state_hopping_terms()
    rho = np.eye(4, dtype=complex)
    bubble = np.zeros((3, 3), dtype=complex)
    direct = np.zeros((3, 3), dtype=complex)

    for weight, (kx_value, ky_value) in zip(weights, points, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        h_minus = normal_state_hamiltonian(kx - 0.5 * qx, ky - 0.5 * qy)
        h_plus = normal_state_hamiltonian(kx + 0.5 * qx, ky + 0.5 * qy)
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

        vector_x = peierls_hamiltonian_vector_vertex(
            kx,
            ky,
            qx,
            qy,
            "x",
            hopping_terms=peierls_terms,
        )
        vector_y = peierls_hamiltonian_vector_vertex(
            kx,
            ky,
            qx,
            qy,
            "y",
            hopping_terms=peierls_terms,
        )
        observable_vertices = (rho, -vector_x, -vector_y)
        source_vertices = (rho, vector_x, vector_y)
        observable_matrices = tuple(
            states_minus.conjugate().T @ vertex @ states_plus for vertex in observable_vertices
        )
        source_matrices = tuple(states_minus.conjugate().T @ vertex @ states_plus for vertex in source_vertices)
        for m, energy_minus in enumerate(energies_minus):
            for n, energy_plus in enumerate(energies_plus):
                occupation_diff = float(occupations_minus[m] - occupations_plus[n])
                if occupation_diff == 0.0:
                    continue
                denominator = 1j * config.omega_eV + float(energy_minus - energy_plus)
                factor = occupation_diff / denominator
                for mu, observable_matrix in enumerate(observable_matrices):
                    for nu, source_matrix in enumerate(source_matrices):
                        bubble[mu, nu] += (
                            weight
                            * factor
                            * observable_matrix[m, n]
                            * np.conjugate(source_matrix[m, n])
                        )

        h_midpoint = normal_state_hamiltonian(kx, ky)
        energies_midpoint, states_midpoint = np.linalg.eigh(h_midpoint)
        occupations_midpoint = fermi_function(
            energies_midpoint,
            config.fermi_level_eV,
            config.temperature_eV,
        )
        for i, direction_i in enumerate(("x", "y")):
            for j, direction_j in enumerate(("x", "y")):
                contact_matrix = peierls_hamiltonian_contact_vertex(
                    kx,
                    ky,
                    qx,
                    qy,
                    direction_i,
                    direction_j,
                    hopping_terms=peierls_terms,
                )
                band_contact = states_midpoint.conjugate().T @ contact_matrix @ states_midpoint
                physical_direct_contact = -np.sum(occupations_midpoint * np.diag(band_contact))
                direct[1 + i, 1 + j] += weight * physical_direct_contact
    return {"bubble": bubble, "direct": direct, "total": bubble + direct}


def _ward_compatible_shifted_pair_quadrature_audit(
    points: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    q: np.ndarray,
    raw_components: dict[str, np.ndarray],
) -> dict[str, Any]:
    q_norm = float(np.linalg.norm(q))
    shifted_pair_components = _shifted_pair_response_components(points, weights, config, q)
    shifted_pair_total_left, _ = physical_ward_residuals(shifted_pair_components["total"], config.omega_eV, q)
    raw_total_left, _ = physical_ward_residuals(raw_components["total"], config.omega_eV, q)
    residual_difference = shifted_pair_total_left - raw_total_left
    current_current_block_difference = (
        raw_components["total"][1:3, 1:3] - shifted_pair_components["total"][1:3, 1:3]
    )
    return {
        "diagnostic_only": True,
        "valid_for_casimir_input": False,
        "shifted_pair_response_is_raw_equivalent_diagnostic": True,
        "quadrature": "midpoint shifted pair raw-equivalent diagnostic",
        "mesh_definition": "k_mid = k, k_plus = k_mid + q/2, k_minus = k_mid - q/2",
        "weights": "bubble, equal-time diagnostics, and contact use the same midpoint mesh weights",
        "no_longitudinal_projection_completion": True,
        "shifted_pair_bubble_ward_residual": _ward_residual_payload(
            shifted_pair_components["bubble"],
            config.omega_eV,
            q,
        ),
        "shifted_pair_direct_ward_residual": _ward_residual_payload(
            shifted_pair_components["direct"],
            config.omega_eV,
            q,
        ),
        "shifted_pair_total_ward_residual": _ward_residual_payload(
            shifted_pair_components["total"],
            config.omega_eV,
            q,
        ),
        "shifted_pair_total_residual_over_q": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "components": _component_vector(shifted_pair_total_left / q_norm),
            "norm": float(np.linalg.norm(shifted_pair_total_left) / q_norm),
        },
        "shifted_pair_total_residual_over_q2": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "components": _component_vector(shifted_pair_total_left / (q_norm * q_norm)),
            "norm": float(np.linalg.norm(shifted_pair_total_left) / (q_norm * q_norm)),
        },
        "raw_total_ward_residual": _ward_residual_payload(raw_components["total"], config.omega_eV, q),
        "raw_total_residual_over_q": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "components": _component_vector(raw_total_left / q_norm),
            "norm": float(np.linalg.norm(raw_total_left) / q_norm),
        },
        "shifted_pair_minus_raw_response_norm": float(
            np.linalg.norm(shifted_pair_components["total"] - raw_components["total"])
        ),
        "shifted_pair_minus_raw_longitudinal_residual": _complex_value(
            _longitudinal_current_component(residual_difference, q)
        ),
        "raw_vs_shifted_pair_current_current_block_difference": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "definition": "raw total current-current block minus shifted-pair total current-current block",
            "matrix": _complex_matrix_entries(current_current_block_difference),
            "norm": float(np.linalg.norm(current_current_block_difference)),
        },
    }


def _finite_mesh_translation_error_audit(
    equal_time_audit: dict[str, Any],
    components: dict[str, np.ndarray],
    config: KuboConfig,
    q: np.ndarray,
) -> dict[str, Any]:
    shifted_grid = equal_time_audit["shifted_grid_equal_time_sum_rule"]
    actual_equal_time = _vector_from_component_rows(
        shifted_grid["actual_bubble_equal_time_term"]["components"]
    )
    shifted_equal_time_reference = _vector_from_component_rows(
        shifted_grid["shifted_equal_time_reference"]["components"]
    )
    contact_contraction = _vector_from_component_rows(shifted_grid["contact_contraction"]["components"])
    shifted_equal_time_plus_contact = shifted_equal_time_reference + contact_contraction
    translation_error = actual_equal_time - shifted_equal_time_reference
    total_left_residual, _ = physical_ward_residuals(components["total"], config.omega_eV, q)
    translation_error_minus_total_residual = translation_error - total_left_residual
    q_norm = float(np.linalg.norm(q))
    total_residual_norm = float(np.linalg.norm(total_left_residual))
    translation_error_norm = float(np.linalg.norm(translation_error))
    return {
        "diagnostic_only": True,
        "valid_for_casimir_input": False,
        "scope": "finite_k_mesh_translation_invariance_failure_at_equal_time_contact_level",
        "density_component_note": (
            "density has no current-vertex finite-difference reference in this diagnostic; "
            "the shifted reference density component is stored as zero"
        ),
        "actual_equal_time": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "source": "normal_current_equal_time_sum_rule_audit.bubble_equal_time_term",
            "components": _component_vector(actual_equal_time),
        },
        "shifted_equal_time_reference": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "definition": "Tr[f(H(k)) * (V_j(k+q/2,q) - V_j(k-q/2,q))]",
            "components": _component_vector(shifted_equal_time_reference),
        },
        "contact_contraction": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "definition": "C_j(q) = q_i D_ij(q)",
            "components": _component_vector(contact_contraction),
        },
        "shifted_equal_time_plus_contact": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "definition": "shifted_equal_time_reference + contact_contraction",
            "components": _component_vector(shifted_equal_time_plus_contact),
        },
        "translation_error": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "definition": "actual_equal_time - shifted_equal_time_reference",
            "components": _component_vector(translation_error),
        },
        "translation_error_minus_total_residual": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "definition": "translation_error - left_ward_residual(total)",
            "components": _component_vector(translation_error_minus_total_residual),
        },
        "actual_equal_time_norm": float(np.linalg.norm(actual_equal_time)),
        "shifted_equal_time_reference_norm": float(np.linalg.norm(shifted_equal_time_reference)),
        "contact_contraction_norm": float(np.linalg.norm(contact_contraction)),
        "shifted_equal_time_plus_contact_norm": float(np.linalg.norm(shifted_equal_time_plus_contact)),
        "translation_error_norm": translation_error_norm,
        "translation_error_minus_total_residual_norm": float(
            np.linalg.norm(translation_error_minus_total_residual)
        ),
        "total_ward_residual_norm": total_residual_norm,
        "translation_error_over_q_norm": float(translation_error_norm / q_norm),
        "total_ward_residual_over_q_norm": float(total_residual_norm / q_norm),
    }


def _twist_averaged_normal_response_components(
    *,
    nk: int,
    twist_count: int,
    config: KuboConfig,
    q: np.ndarray,
) -> dict[str, Any]:
    offsets = twist_offsets(twist_count)
    response_sums = {name: np.zeros((3, 3), dtype=complex) for name in RESPONSE_NAMES}
    vector_sums = {
        "actual_equal_time": np.zeros(3, dtype=complex),
        "shifted_equal_time_reference": np.zeros(3, dtype=complex),
        "contact_contraction": np.zeros(3, dtype=complex),
        "shifted_equal_time_plus_contact": np.zeros(3, dtype=complex),
        "translation_error": np.zeros(3, dtype=complex),
        "translation_error_minus_total_residual": np.zeros(3, dtype=complex),
    }
    one_photon_max = 0.0
    second_order_max = 0.0

    for offset in offsets:
        points = uniform_bz_mesh_twisted(nk, offset)
        weights = k_weights(points)
        components = normal_physical_density_current_response_components_imag_axis(points, config, q, weights)
        equal_time_audit = _normal_equal_time_sum_rule_audit(points, weights, config, q, components)
        translation_audit = _finite_mesh_translation_error_audit(equal_time_audit, components, config, q)
        for name in RESPONSE_NAMES:
            response_sums[name] += components[name]
        vector_sums["actual_equal_time"] += _vector_from_component_rows(
            translation_audit["actual_equal_time"]["components"]
        )
        vector_sums["shifted_equal_time_reference"] += _vector_from_component_rows(
            translation_audit["shifted_equal_time_reference"]["components"]
        )
        vector_sums["contact_contraction"] += _vector_from_component_rows(
            translation_audit["contact_contraction"]["components"]
        )
        vector_sums["shifted_equal_time_plus_contact"] += _vector_from_component_rows(
            translation_audit["shifted_equal_time_plus_contact"]["components"]
        )
        vector_sums["translation_error"] += _vector_from_component_rows(
            translation_audit["translation_error"]["components"]
        )
        vector_sums["translation_error_minus_total_residual"] += _vector_from_component_rows(
            translation_audit["translation_error_minus_total_residual"]["components"]
        )
        operator_rows = _operator_level_rows(points, q)
        one_photon_max = max(one_photon_max, max(row["absolute_error_norm"] for row in operator_rows))
        second_order_audit = _operator_level_second_order_contact_ward(points, q)
        second_order_max = max(second_order_max, float(second_order_audit["max_absolute_error_norm"]))

    inverse_twist_count = 1.0 / twist_count
    response_avg = {name: response_sums[name] * inverse_twist_count for name in RESPONSE_NAMES}
    vector_avg = {name: vector_sums[name] * inverse_twist_count for name in vector_sums}
    total_left_residual, _ = physical_ward_residuals(response_avg["total"], config.omega_eV, q)
    vector_avg["translation_error_minus_total_residual"] = (
        vector_avg["translation_error"] - total_left_residual
    )
    return {
        "diagnostic_only": True,
        "valid_for_casimir_input": False,
        "twist_offsets": [[float(x), float(y)] for x, y in offsets],
        "response_components": response_avg,
        "actual_equal_time_avg": vector_avg["actual_equal_time"],
        "shifted_equal_time_reference_avg": vector_avg["shifted_equal_time_reference"],
        "contact_contraction_avg": vector_avg["contact_contraction"],
        "shifted_equal_time_plus_contact_avg": vector_avg["shifted_equal_time_plus_contact"],
        "translation_error_avg": vector_avg["translation_error"],
        "translation_error_minus_total_residual_avg": vector_avg["translation_error_minus_total_residual"],
        "total_ward_residual": total_left_residual,
        "one_photon_peierls_ward_max_absolute_error_norm": float(one_photon_max),
        "second_order_contact_ward_max_absolute_error_norm": float(second_order_max),
    }


def _normal_equal_time_sum_rule_audit(
    points: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    q: np.ndarray,
    components: dict[str, np.ndarray],
) -> dict[str, Any]:
    qx, qy = float(q[0]), float(q[1])
    peierls_terms = normal_state_hopping_terms()
    rho = np.eye(4, dtype=complex)
    bubble_equal_time = np.zeros(3, dtype=complex)
    interband_contribution = np.zeros(3, dtype=complex)
    intraband_contribution = np.zeros(3, dtype=complex)
    intraband_finite_q_difference = np.zeros(3, dtype=complex)
    intraband_fprime_approximation = np.zeros(3, dtype=complex)
    direct_contact_contraction = np.zeros(3, dtype=complex)
    shifted_equal_time_reference = np.zeros(3, dtype=complex)

    for weight, (kx_value, ky_value) in zip(weights, points, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        h_minus = normal_state_hamiltonian(kx - 0.5 * qx, ky - 0.5 * qy)
        h_plus = normal_state_hamiltonian(kx + 0.5 * qx, ky + 0.5 * qy)
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

        vector_x = peierls_hamiltonian_vector_vertex(
            kx,
            ky,
            qx,
            qy,
            "x",
            hopping_terms=peierls_terms,
        )
        vector_y = peierls_hamiltonian_vector_vertex(
            kx,
            ky,
            qx,
            qy,
            "y",
            hopping_terms=peierls_terms,
        )
        source_vertices = (rho, vector_x, vector_y)
        rho_band = states_minus.conjugate().T @ rho @ states_plus
        source_matrices = tuple(states_minus.conjugate().T @ vertex @ states_plus for vertex in source_vertices)

        for m, energy_minus in enumerate(energies_minus):
            for n, energy_plus in enumerate(energies_plus):
                occupation_diff = float(occupations_minus[m] - occupations_plus[n])
                if occupation_diff == 0.0:
                    continue
                term = np.array(
                    [
                        occupation_diff * rho_band[m, n] * np.conjugate(source_matrix[m, n])
                        for source_matrix in source_matrices
                    ],
                    dtype=complex,
                )
                weighted_term = weight * term
                bubble_equal_time += weighted_term
                if m == n:
                    intraband_contribution += weighted_term
                    intraband_finite_q_difference += weighted_term
                else:
                    interband_contribution += weighted_term

        midpoint_energies = 0.5 * (energies_minus + energies_plus)
        finite_difference_delta = energies_plus - energies_minus
        fprime_occupation_diff = negative_fermi_derivative(
            midpoint_energies,
            config.fermi_level_eV,
            config.temperature_eV,
            config.eta_eV,
        ) * finite_difference_delta
        for band_index, occupation_diff_approx in enumerate(fprime_occupation_diff):
            intraband_fprime_approximation += weight * np.array(
                [
                    float(occupation_diff_approx)
                    * rho_band[band_index, band_index]
                    * np.conjugate(source_matrix[band_index, band_index])
                    for source_matrix in source_matrices
                ],
                dtype=complex,
            )

        h_midpoint = normal_state_hamiltonian(kx, ky)
        energies_midpoint, states_midpoint = np.linalg.eigh(h_midpoint)
        occupations_midpoint = fermi_function(
            energies_midpoint,
            config.fermi_level_eV,
            config.temperature_eV,
        )
        for source_index, source_direction in enumerate(("x", "y"), start=1):
            shifted_vertex_reference = peierls_hamiltonian_vector_vertex(
                kx + 0.5 * qx,
                ky + 0.5 * qy,
                qx,
                qy,
                source_direction,
                hopping_terms=peierls_terms,
            ) - peierls_hamiltonian_vector_vertex(
                kx - 0.5 * qx,
                ky - 0.5 * qy,
                qx,
                qy,
                source_direction,
                hopping_terms=peierls_terms,
            )
            band_shifted_reference = states_midpoint.conjugate().T @ shifted_vertex_reference @ states_midpoint
            shifted_equal_time_reference[source_index] += weight * np.sum(
                occupations_midpoint * np.diag(band_shifted_reference)
            )
        for source_index, source_direction in enumerate(("x", "y"), start=1):
            contraction_value = 0.0j
            for q_component, observable_direction in ((qx, "x"), (qy, "y")):
                contact_matrix = peierls_hamiltonian_contact_vertex(
                    kx,
                    ky,
                    qx,
                    qy,
                    observable_direction,
                    source_direction,
                    hopping_terms=peierls_terms,
                )
                band_contact = states_midpoint.conjugate().T @ contact_matrix @ states_midpoint
                physical_direct_contact = -np.sum(occupations_midpoint * np.diag(band_contact))
                contraction_value += q_component * physical_direct_contact
            direct_contact_contraction[source_index] += weight * contraction_value

    total_left_residual, _ = physical_ward_residuals(components["total"], config.omega_eV, q)
    equal_time_plus_contact = bubble_equal_time + direct_contact_contraction
    difference_from_total = equal_time_plus_contact - total_left_residual
    shifted_equal_time_plus_contact = shifted_equal_time_reference + direct_contact_contraction
    actual_minus_shifted_equal_time = bubble_equal_time - shifted_equal_time_reference
    actual_minus_shifted_vs_total = actual_minus_shifted_equal_time - total_left_residual
    return {
        "diagnostic_only": True,
        "valid_for_casimir_input": False,
        "side": "left_contraction",
        "component_labels": list(WARD_COMPONENT_LABELS),
        "bubble_equal_time_term": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "definition": (
                "Denominator-cancelled left bubble Ward term using the same finite-q band basis, "
                "source vertices, k weights, and Fermi occupations as the normal bubble."
            ),
            "components": _component_vector(bubble_equal_time),
            "interband_contribution": {
                "diagnostic_only": True,
                "valid_for_casimir_input": False,
                "band_partition": "m != n in the finite-q sorted band labels",
                "components": _component_vector(interband_contribution),
            },
            "intraband_contribution": {
                "diagnostic_only": True,
                "valid_for_casimir_input": False,
                "band_partition": "m == n in the finite-q sorted band labels",
                "components": _component_vector(intraband_contribution),
                "finite_q_difference_form": {
                    "diagnostic_only": True,
                    "valid_for_casimir_input": False,
                    "occupation_difference": "f(E_minus[m]) - f(E_plus[m])",
                    "components": _component_vector(intraband_finite_q_difference),
                },
                "fprime_approximation_form": {
                    "diagnostic_only": True,
                    "valid_for_casimir_input": False,
                    "occupation_difference": "(-df/dE at midpoint energy) * (E_plus[m] - E_minus[m])",
                    "components": _component_vector(intraband_fprime_approximation),
                },
            },
        },
        "direct_contact_contraction": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "definition": "qx * D[x,nu] + qy * D[y,nu] from the normal direct/contact response.",
            "components": _component_vector(direct_contact_contraction),
        },
        "equal_time_plus_contact": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "definition": "bubble_equal_time_term + direct_contact_contraction",
            "components": _component_vector(equal_time_plus_contact),
        },
        "difference_from_total_ward_residual": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "definition": "equal_time_plus_contact - left_ward_residual(total)",
            "components": _component_vector(difference_from_total),
            "norm": float(np.linalg.norm(difference_from_total)),
        },
        "shifted_grid_equal_time_sum_rule": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "purpose": (
                "Diagnose whether the remaining normal-state response residual is caused by finite-k-mesh "
                "failure of the k -> k +/- q/2 variable shift rather than by the implemented contact vertex."
            ),
            "actual_bubble_equal_time_term": {
                "diagnostic_only": True,
                "valid_for_casimir_input": False,
                "source": "same vector as bubble_equal_time_term.components",
                "components": _component_vector(bubble_equal_time),
            },
            "shifted_equal_time_reference": {
                "diagnostic_only": True,
                "valid_for_casimir_input": False,
                "definition": "Tr[f(H(k)) * (V_j(k+q/2,q) - V_j(k-q/2,q))]",
                "density_component_note": "not_applicable; stored as zero because there is no density current-vertex finite-difference reference",
                "components": _component_vector(shifted_equal_time_reference),
            },
            "contact_contraction": {
                "diagnostic_only": True,
                "valid_for_casimir_input": False,
                "source": "same vector as direct_contact_contraction.components",
                "components": _component_vector(direct_contact_contraction),
            },
            "shifted_equal_time_plus_contact": {
                "diagnostic_only": True,
                "valid_for_casimir_input": False,
                "definition": "shifted_equal_time_reference + direct_contact_contraction",
                "components": _component_vector(shifted_equal_time_plus_contact),
            },
            "actual_minus_shifted_equal_time": {
                "diagnostic_only": True,
                "valid_for_casimir_input": False,
                "definition": "actual_bubble_equal_time_term - shifted_equal_time_reference",
                "components": _component_vector(actual_minus_shifted_equal_time),
            },
            "actual_minus_shifted_vs_total_residual_difference": {
                "diagnostic_only": True,
                "valid_for_casimir_input": False,
                "definition": "actual_minus_shifted_equal_time - left_ward_residual(total)",
                "components": _component_vector(actual_minus_shifted_vs_total),
            },
            "shifted_equal_time_plus_contact_norm": float(np.linalg.norm(shifted_equal_time_plus_contact)),
            "actual_minus_shifted_equal_time_norm": float(np.linalg.norm(actual_minus_shifted_equal_time)),
            "difference_from_total_ward_residual_norm": float(np.linalg.norm(actual_minus_shifted_vs_total)),
        },
    }


def _convergence_summary_from_items(convergence_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    convergence_summary_rows: list[dict[str, Any]] = []
    grouped_items: dict[tuple[str, str, float, float], list[dict[str, Any]]] = {}
    for item in convergence_items:
        key = (
            str(item["q_direction"]),
            f"{float(item['q_norm']):.16g}",
            float(item["temperature_K"]),
            float(item["eta_eV"]),
        )
        grouped_items.setdefault(key, []).append(item)
    for items in grouped_items.values():
        ordered = sorted(items, key=lambda row: (int(row["nk"]), int(row["twist_count"])))
        for level_a, level_b in zip(ordered, ordered[1:], strict=False):
            response_b_norm = float(np.linalg.norm(level_b["total_response"]))
            block_b_norm = float(np.linalg.norm(level_b["current_current_block"]))
            convergence_summary_rows.append(
                {
                    "diagnostic_only": True,
                    "valid_for_casimir_input": False,
                    "level_a": {
                        "nk": int(level_a["nk"]),
                        "twist_count": int(level_a["twist_count"]),
                    },
                    "level_b": {
                        "nk": int(level_b["nk"]),
                        "twist_count": int(level_b["twist_count"]),
                    },
                    "q_direction": str(level_b["q_direction"]),
                    "q_norm": float(level_b["q_norm"]),
                    "temperature_K": float(level_b["temperature_K"]),
                    "eta_eV": float(level_b["eta_eV"]),
                    "response_relative_change_norm": float(
                        np.linalg.norm(level_b["total_response"] - level_a["total_response"])
                        / max(response_b_norm, 1e-300)
                    ),
                    "current_current_block_relative_change_norm": float(
                        np.linalg.norm(level_b["current_current_block"] - level_a["current_current_block"])
                        / max(block_b_norm, 1e-300)
                    ),
                    "ward_residual_change_norm": float(
                        np.linalg.norm(level_b["ward_residual"] - level_a["ward_residual"])
                    ),
                    "translation_error_change_norm": float(
                        np.linalg.norm(level_b["translation_error"] - level_a["translation_error"])
                    ),
                    "cost_ratio_effective_nodes": float(
                        int(level_b.get("effective_total_nodes", 0))
                        / max(int(level_a.get("effective_total_nodes", 0)), 1)
                    ),
                }
            )
    return convergence_summary_rows


def run_compact_summary_audit(
    *,
    omega_eV: float,
    q_values: tuple[float, ...],
    q_directions: tuple[str, ...],
    nk_values: tuple[int, ...],
    twist_counts: tuple[int, ...],
    actual_twist_counts: tuple[int, ...] | None,
    twist_mode: str,
    adaptive_mode: str,
    adaptive_refine_levels: tuple[int, ...],
    fs_window_factor: float,
    temperature_K: float,
    eta_eV: float,
    workers: int,
    progress_enabled: bool,
) -> dict[str, Any]:
    start_total = time.perf_counter()
    unknown_directions = sorted(set(q_directions) - set(DIRECTION_VECTORS))
    if unknown_directions:
        raise ValueError(f"unknown q direction(s): {unknown_directions}")
    use_actual_twist_counts = actual_twist_counts is not None
    requested_twist_counts = actual_twist_counts if actual_twist_counts is not None else twist_counts
    if use_actual_twist_counts:
        unknown_actual_twist_counts = sorted(set(requested_twist_counts) - set(SUPPORTED_ACTUAL_TWIST_COUNTS))
        if unknown_actual_twist_counts:
            raise ValueError(f"unsupported actual twist count(s): {unknown_actual_twist_counts}")
    else:
        unknown_twist_counts = sorted(set(requested_twist_counts) - set(SUPPORTED_TWIST_COUNTS))
        if unknown_twist_counts:
            raise ValueError(f"unsupported twist count(s): {unknown_twist_counts}")
    if twist_mode not in {"halton", "symmetry_paired"}:
        raise ValueError("twist_mode must be 'halton' or 'symmetry_paired'")
    if adaptive_mode not in {"none", "q_covariant"}:
        raise ValueError("adaptive_mode must be 'none' or 'q_covariant'")

    tasks = [
        (
            int(nk),
            direction_name,
            float(q_value),
            DIRECTION_VECTORS[direction_name],
            tuple(int(value) for value in requested_twist_counts),
            bool(use_actual_twist_counts),
            twist_mode,
            adaptive_mode,
            tuple(int(value) for value in adaptive_refine_levels),
            float(fs_window_factor),
            float(omega_eV),
            float(temperature_K),
            float(eta_eV),
        )
        for nk in nk_values
        for direction_name in q_directions
        for q_value in q_values
    ]
    total_tasks = len(tasks)
    progress_enabled = bool(progress_enabled and sys.stdout.isatty())
    _print_progress(0, total_tasks, enabled=progress_enabled)
    if workers > 1:
        results = []
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = [executor.submit(_compact_case_worker, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                _print_progress(completed, total_tasks, enabled=progress_enabled)
        parallel_backend = "concurrent.futures.ProcessPoolExecutor"
    else:
        results = []
        for completed, task in enumerate(tasks, start=1):
            results.append(_compact_case_worker(task))
            _print_progress(completed, total_tasks, enabled=progress_enabled)
        parallel_backend = "sequential"

    twist_rows: list[dict[str, Any]] = []
    adaptive_rows: list[dict[str, Any]] = []
    convergence_items: list[dict[str, Any]] = []
    profile = {
        "diagonalization_time_seconds": 0.0,
        "vertex_time_seconds": 0.0,
        "response_accumulation_time_seconds": 0.0,
        "adaptive_refinement_time_seconds": 0.0,
        "json_write_time_seconds": 0.0,
        "cache_hits": 0,
        "cache_misses": 0,
    }
    for result in results:
        twist_rows.extend(result["twist_rows"])
        adaptive_rows.extend(result["adaptive_rows"])
        convergence_items.extend(result["convergence_items"])
        for key in profile:
            profile[key] += result["profile"].get(key, 0)

    return {
        "audit_name": "normal_finite_q_ward_audit",
        "scope": "diagnostic_only_summary_normal_state_finite_q_ward_translation_error_convergence",
        "omega_eV": float(omega_eV),
        "temperature_K": float(temperature_K),
        "eta_eV": float(eta_eV),
        "nk_values": [int(value) for value in nk_values],
        "q_values": [float(value) for value in q_values],
        "q_directions": list(q_directions),
        "twist_counts": [int(value) for value in twist_counts],
        "actual_twist_counts": [int(value) for value in actual_twist_counts] if actual_twist_counts is not None else None,
        "twist_mode": twist_mode,
        "adaptive_mode": adaptive_mode,
        "adaptive_refine_levels": [int(value) for value in adaptive_refine_levels],
        "component_labels": list(WARD_COMPONENT_LABELS),
        "twist_averaged_quadrature_summary": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "rows": twist_rows,
        },
        "twist_averaged_convergence_summary": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "relative_change_definition": "norm(level_b - level_a) / max(norm(level_b), 1e-300)",
            "rows": _convergence_summary_from_items(convergence_items),
        },
        "adaptive_quadrature_summary": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "fs_window_definition": "E_window=max(fs_window_factor*kBT, placeholder_vF*|q|, eta_eff, E_min)",
            "rows": adaptive_rows,
        },
        "runtime_profile_summary": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "total_runtime_seconds": float(time.perf_counter() - start_total),
            **{key: float(value) for key, value in profile.items() if key.endswith("_seconds")},
            "cache_hits": int(profile["cache_hits"]),
            "cache_misses": int(profile["cache_misses"]),
            "workers": int(workers),
            "parallel_backend": parallel_backend,
        },
        "output_format": {
            "summary_only": True,
            "twist_summary_only": True,
            "adaptive_summary_only": True,
            "removed_large_fields": [
                "nk_reports",
                "q_reports",
                "per_k_residuals",
                "4x4_matrix_entries",
                "full_response_level_residuals",
                "full_operator_level_per_k_matrices",
                "full_band_basis_diagnostic_dump",
            ],
            "max_expected_file_size_mb": TARGET_JSON_SIZE_MB,
            "github_safe_output": True,
        },
        "ward_identity_closed": False,
        "valid_for_casimir_input": False,
    }


def run_normal_finite_q_ward_audit(
    *,
    omega_eV: float = 0.01,
    q_values: tuple[float, ...] = (0.001, 0.002, 0.005, 0.01, 0.02),
    q_directions: tuple[str, ...] = ("x", "y", "diagonal"),
    nk_values: tuple[int, ...] = (3,),
    twist_counts: tuple[int, ...] = (1,),
    twist_summary_only: bool = False,
    temperature_K: float = 10.0,
    eta_eV: float = 1e-8,
) -> dict[str, Any]:
    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        output_si=False,
    )
    unknown_directions = sorted(set(q_directions) - set(DIRECTION_VECTORS))
    if unknown_directions:
        raise ValueError(f"unknown q direction(s): {unknown_directions}")
    unknown_twist_counts = sorted(set(twist_counts) - set(SUPPORTED_TWIST_COUNTS))
    if unknown_twist_counts:
        raise ValueError(f"unsupported twist count(s): {unknown_twist_counts}")
    nk_reports: list[dict[str, Any]] = []
    quadrature_summary_rows: list[dict[str, Any]] = []
    translation_error_summary_rows: list[dict[str, Any]] = []
    twist_summary_rows: list[dict[str, Any]] = []
    convergence_items: list[dict[str, Any]] = []
    for nk in nk_values:
        points = uniform_bz_mesh(int(nk))
        weights = k_weights(points)
        q_reports: list[dict[str, Any]] = []
        for direction_name in q_directions:
            direction = np.asarray(DIRECTION_VECTORS[direction_name], dtype=float)
            for q_value in q_values:
                q = float(q_value) * direction
                components = normal_physical_density_current_response_components_imag_axis(points, config, q, weights)
                response_rows = [
                    _response_residual_row(response_name, components[response_name], config.omega_eV, q)
                    for response_name in RESPONSE_NAMES
                ]
                operator_rows = _operator_level_rows(points, q)
                equal_time_audit = _normal_equal_time_sum_rule_audit(
                    points,
                    weights,
                    config,
                    q,
                    components,
                )
                second_order_audit = _operator_level_second_order_contact_ward(points, q)
                shifted_pair_audit = _ward_compatible_shifted_pair_quadrature_audit(
                    points,
                    weights,
                    config,
                    q,
                    components,
                )
                finite_mesh_translation_error_audit = _finite_mesh_translation_error_audit(
                    equal_time_audit,
                    components,
                    config,
                    q,
                )
                single_origin_total_left, _ = physical_ward_residuals(components["total"], config.omega_eV, q)
                shifted_grid_summary = equal_time_audit["shifted_grid_equal_time_sum_rule"]
                raw_total_residual_norm = shifted_pair_audit["raw_total_ward_residual"]["left_ward_residual_norm"]
                shifted_pair_total_residual_norm = shifted_pair_audit["shifted_pair_total_ward_residual"][
                    "left_ward_residual_norm"
                ]
                q_norm = float(np.linalg.norm(q))
                compact_quadrature_summary = {
                    "diagnostic_only": True,
                    "valid_for_casimir_input": False,
                    "shifted_equal_time_plus_contact_norm": float(
                        shifted_grid_summary["shifted_equal_time_plus_contact_norm"]
                    ),
                    "actual_minus_shifted_equal_time_norm": float(
                        shifted_grid_summary["actual_minus_shifted_equal_time_norm"]
                    ),
                    "actual_minus_shifted_vs_total_residual_difference_norm": float(
                        shifted_grid_summary["difference_from_total_ward_residual_norm"]
                    ),
                }
                quadrature_summary_rows.append(
                    {
                        "diagnostic_only": True,
                        "valid_for_casimir_input": False,
                        "nk": int(nk),
                        "q_direction": direction_name,
                        "q_norm": q_norm,
                        "raw_total_residual_norm": float(raw_total_residual_norm),
                        "shifted_pair_total_residual_norm": float(shifted_pair_total_residual_norm),
                        "raw_total_residual_over_q_abs": float(raw_total_residual_norm / q_norm),
                        "shifted_pair_total_residual_over_q_abs": float(
                            shifted_pair_total_residual_norm / q_norm
                        ),
                        "second_order_contact_ward_max_absolute_error_norm": float(
                            second_order_audit["max_absolute_error_norm"]
                        ),
                        "shifted_equal_time_plus_contact_norm": compact_quadrature_summary[
                            "shifted_equal_time_plus_contact_norm"
                        ],
                        "actual_minus_shifted_vs_total_residual_difference_norm": compact_quadrature_summary[
                            "actual_minus_shifted_vs_total_residual_difference_norm"
                        ],
                    }
                )
                translation_error_summary_rows.append(
                    {
                        "diagnostic_only": True,
                        "valid_for_casimir_input": False,
                        "nk": int(nk),
                        "q_direction": direction_name,
                        "q_norm": q_norm,
                        "total_ward_residual_norm": finite_mesh_translation_error_audit[
                            "total_ward_residual_norm"
                        ],
                        "total_ward_residual_over_q_norm": finite_mesh_translation_error_audit[
                            "total_ward_residual_over_q_norm"
                        ],
                        "shifted_equal_time_plus_contact_norm": finite_mesh_translation_error_audit[
                            "shifted_equal_time_plus_contact_norm"
                        ],
                        "translation_error_norm": finite_mesh_translation_error_audit[
                            "translation_error_norm"
                        ],
                        "translation_error_over_q_norm": finite_mesh_translation_error_audit[
                            "translation_error_over_q_norm"
                        ],
                        "translation_error_minus_total_residual_norm": finite_mesh_translation_error_audit[
                            "translation_error_minus_total_residual_norm"
                        ],
                        "second_order_contact_ward_max_absolute_error_norm": float(
                            second_order_audit["max_absolute_error_norm"]
                        ),
                    }
                )
                for twist_count in twist_counts:
                    twist_average = _twist_averaged_normal_response_components(
                        nk=int(nk),
                        twist_count=int(twist_count),
                        config=config,
                        q=q,
                    )
                    twist_response = twist_average["response_components"]
                    twist_total_residual = np.asarray(twist_average["total_ward_residual"], dtype=complex)
                    twist_translation_error = np.asarray(
                        twist_average["translation_error_avg"],
                        dtype=complex,
                    )
                    twist_shifted_plus_contact = np.asarray(
                        twist_average["shifted_equal_time_plus_contact_avg"],
                        dtype=complex,
                    )
                    twist_translation_minus_total = np.asarray(
                        twist_average["translation_error_minus_total_residual_avg"],
                        dtype=complex,
                    )
                    twist_summary_rows.append(
                        {
                            "diagnostic_only": True,
                            "valid_for_casimir_input": False,
                            "nk": int(nk),
                            "twist_count": int(twist_count),
                            "q_direction": direction_name,
                            "q_norm": q_norm,
                            "temperature_K": float(temperature_K),
                            "eta_eV": float(eta_eV),
                            "single_origin_total_residual_norm": float(
                                finite_mesh_translation_error_audit["total_ward_residual_norm"]
                            ),
                            "single_origin_total_residual_over_q_norm": float(
                                finite_mesh_translation_error_audit["total_ward_residual_over_q_norm"]
                            ),
                            "twist_averaged_total_residual_norm": float(np.linalg.norm(twist_total_residual)),
                            "twist_averaged_total_residual_over_q_norm": float(
                                np.linalg.norm(twist_total_residual) / q_norm
                            ),
                            "twist_averaged_total_residual_over_q2_norm": float(
                                np.linalg.norm(twist_total_residual) / (q_norm * q_norm)
                            ),
                            "single_origin_translation_error_norm": float(
                                finite_mesh_translation_error_audit["translation_error_norm"]
                            ),
                            "twist_averaged_translation_error_norm": float(
                                np.linalg.norm(twist_translation_error)
                            ),
                            "twist_averaged_translation_error_over_q_norm": float(
                                np.linalg.norm(twist_translation_error) / q_norm
                            ),
                            "twist_averaged_shifted_equal_time_plus_contact_norm": float(
                                np.linalg.norm(twist_shifted_plus_contact)
                            ),
                            "twist_averaged_translation_error_minus_total_residual_norm": float(
                                np.linalg.norm(twist_translation_minus_total)
                            ),
                            "twist_averaged_total_response_norm": float(np.linalg.norm(twist_response["total"])),
                            "twist_averaged_bubble_response_norm": float(np.linalg.norm(twist_response["bubble"])),
                            "twist_averaged_direct_response_norm": float(np.linalg.norm(twist_response["direct"])),
                            "twist_avg_minus_single_origin_total_response_norm": float(
                                np.linalg.norm(twist_response["total"] - components["total"])
                            ),
                            "twist_avg_minus_single_origin_current_current_block_norm": float(
                                np.linalg.norm(twist_response["total"][1:3, 1:3] - components["total"][1:3, 1:3])
                            ),
                            "twist_avg_minus_single_origin_total_residual_norm": float(
                                np.linalg.norm(twist_total_residual - single_origin_total_left)
                            ),
                            "one_photon_peierls_ward_max_absolute_error_norm": float(
                                twist_average["one_photon_peierls_ward_max_absolute_error_norm"]
                            ),
                            "second_order_contact_ward_max_absolute_error_norm": float(
                                twist_average["second_order_contact_ward_max_absolute_error_norm"]
                            ),
                        }
                    )
                    convergence_items.append(
                        {
                            "q_direction": direction_name,
                            "q_norm": q_norm,
                            "temperature_K": float(temperature_K),
                            "eta_eV": float(eta_eV),
                            "nk": int(nk),
                            "twist_count": int(twist_count),
                            "total_response": twist_response["total"],
                            "current_current_block": twist_response["total"][1:3, 1:3],
                            "ward_residual": twist_total_residual,
                            "translation_error": twist_translation_error,
                        }
                    )
                q_reports.append(
                    {
                        "q_direction": direction_name,
                        "q_model": [float(q[0]), float(q[1])],
                        "q_norm": q_norm,
                        "response_level_residuals": response_rows,
                        "longitudinal_current_residual_scaling": _longitudinal_current_scaling(response_rows, q),
                        "normal_current_equal_time_sum_rule_audit": equal_time_audit,
                        "ward_compatible_shifted_pair_quadrature_audit": shifted_pair_audit,
                        "shifted_grid_equal_time_consistency_summary": compact_quadrature_summary,
                        "finite_mesh_translation_error_audit": finite_mesh_translation_error_audit,
                        "operator_level_peierls_ward": {
                            "residual_kind": "operator_level",
                            "identity": "q_x V_x(k,q) + q_y V_y(k,q) = H(k+q/2)-H(k-q/2)",
                            "max_absolute_error_norm": float(max(row["absolute_error_norm"] for row in operator_rows)),
                            "max_relative_error_norm": float(max(row["relative_error_norm"] for row in operator_rows)),
                            "per_k_residuals": operator_rows,
                        },
                        "operator_level_second_order_contact_ward": second_order_audit,
                    }
                )
        nk_reports.append(
            {
                "nk": int(nk),
                "mesh_size": int(points.shape[0]),
                "q_reports": [] if twist_summary_only else q_reports,
            }
        )
    convergence_summary_rows: list[dict[str, Any]] = []
    grouped_items: dict[tuple[str, str, float, float], list[dict[str, Any]]] = {}
    for item in convergence_items:
        key = (
            str(item["q_direction"]),
            f"{float(item['q_norm']):.16g}",
            float(item["temperature_K"]),
            float(item["eta_eV"]),
        )
        grouped_items.setdefault(key, []).append(item)
    for items in grouped_items.values():
        ordered = sorted(items, key=lambda row: (int(row["nk"]), int(row["twist_count"])))
        for level_a, level_b in zip(ordered, ordered[1:], strict=False):
            response_b_norm = float(np.linalg.norm(level_b["total_response"]))
            block_b_norm = float(np.linalg.norm(level_b["current_current_block"]))
            convergence_summary_rows.append(
                {
                    "diagnostic_only": True,
                    "valid_for_casimir_input": False,
                    "level_a": {
                        "nk": int(level_a["nk"]),
                        "twist_count": int(level_a["twist_count"]),
                    },
                    "level_b": {
                        "nk": int(level_b["nk"]),
                        "twist_count": int(level_b["twist_count"]),
                    },
                    "q_direction": str(level_b["q_direction"]),
                    "q_norm": float(level_b["q_norm"]),
                    "temperature_K": float(level_b["temperature_K"]),
                    "eta_eV": float(level_b["eta_eV"]),
                    "response_relative_change_norm": float(
                        np.linalg.norm(level_b["total_response"] - level_a["total_response"])
                        / max(response_b_norm, 1e-300)
                    ),
                    "current_current_block_relative_change_norm": float(
                        np.linalg.norm(level_b["current_current_block"] - level_a["current_current_block"])
                        / max(block_b_norm, 1e-300)
                    ),
                    "ward_residual_change_norm": float(
                        np.linalg.norm(level_b["ward_residual"] - level_a["ward_residual"])
                    ),
                    "translation_error_change_norm": float(
                        np.linalg.norm(level_b["translation_error"] - level_a["translation_error"])
                    ),
                }
            )
    return {
        "audit_name": "normal_finite_q_ward_audit",
        "scope": "diagnostic_only_normal_state_finite_q_ward_residuals",
        "omega_eV": float(config.omega_eV),
        "temperature_K": float(temperature_K),
        "eta_eV": float(eta_eV),
        "nk_values": [int(value) for value in nk_values],
        "twist_counts": [int(value) for value in twist_counts],
        "twist_summary_only": bool(twist_summary_only),
        "q_values": [float(value) for value in q_values],
        "q_directions": list(q_directions),
        "component_labels": list(WARD_COMPONENT_LABELS),
        "response_level_residuals_explain": (
            "bubble/direct/total are normal-state response-level residuals from physical_ward_residuals; "
            "ward_contraction_decomposition stores iomega, qx, qy terms before summing"
        ),
        "operator_level_residuals_explain": (
            "Peierls vertex identity is checked before response assembly and is distinct from response-level residuals"
        ),
        "ward_compatible_quadrature_summary": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "shifted_pair_response_is_raw_equivalent_diagnostic": True,
            "deprecated_note": "Retained for backward compatibility; do not treat shifted-pair response as a Ward-compatible quadrature scheme.",
            "rows": quadrature_summary_rows,
        },
        "finite_mesh_translation_error_summary": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "rows": translation_error_summary_rows,
        },
        "twist_averaged_quadrature_summary": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "twist_offset_rule": "twist_count=1 uses (0,0); twist_count=4 uses fixed 2x2 stratified offsets; 8/16/32 use deterministic Halton bases 2 and 3.",
            "rows": twist_summary_rows,
        },
        "twist_averaged_convergence_summary": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "relative_change_definition": "norm(level_b - level_a) / max(norm(level_b), 1e-300)",
            "rows": convergence_summary_rows,
        },
        "nk_reports": nk_reports,
        "ward_identity_closed": False,
        "valid_for_casimir_input": False,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_seconds = time.perf_counter() - start
    if isinstance(payload.get("runtime_profile_summary"), dict):
        payload["runtime_profile_summary"]["json_write_time_seconds"] = float(write_seconds)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    size_mb = path.stat().st_size / (1024.0 * 1024.0)
    if size_mb > MAX_JSON_SIZE_MB:
        raise RuntimeError(
            f"normal audit JSON is {size_mb:.2f} MB, above {MAX_JSON_SIZE_MB:.1f} MB. "
            "Use --summary-only/--twist-summary-only/--adaptive-summary-only and remove large per-k or matrix fields."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行 normal-state finite-q Ward residual 诊断。")
    parser.add_argument("--omega", type=float, default=0.01)
    parser.add_argument("--q-values", nargs="+", type=float, default=[0.001, 0.002, 0.005, 0.01, 0.02])
    parser.add_argument("--directions", nargs="+", choices=tuple(DIRECTION_VECTORS), default=["x", "y", "diagonal"])
    parser.add_argument("--nk", type=int, default=3, help="Backward-compatible single-nk shortcut.")
    parser.add_argument("--nk-values", nargs="+", type=int)
    parser.add_argument("--twist-counts", nargs="+", type=int, default=[1])
    parser.add_argument("--actual-twist-counts", nargs="+", type=int)
    parser.add_argument("--twist-mode", choices=("halton", "symmetry_paired"), default="symmetry_paired")
    parser.add_argument("--adaptive-mode", choices=("none", "q_covariant"), default="none")
    parser.add_argument("--adaptive-refine-levels", nargs="+", type=int, default=[0])
    parser.add_argument("--fs-window-factor", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--summary-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--twist-summary-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adaptive-summary-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--temperature-K", type=float, default=DEFAULT_TEMPERATURE_K)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    nk_values = tuple(args.nk_values) if args.nk_values is not None else (int(args.nk),)
    if args.summary_only:
        payload = run_compact_summary_audit(
            omega_eV=args.omega,
            q_values=tuple(args.q_values),
            q_directions=tuple(args.directions),
            nk_values=nk_values,
            twist_counts=tuple(args.twist_counts),
            actual_twist_counts=tuple(args.actual_twist_counts) if args.actual_twist_counts is not None else None,
            twist_mode=args.twist_mode,
            adaptive_mode=args.adaptive_mode,
            adaptive_refine_levels=tuple(args.adaptive_refine_levels),
            fs_window_factor=args.fs_window_factor,
            temperature_K=args.temperature_K,
            eta_eV=args.eta,
            workers=max(1, int(args.workers)),
            progress_enabled=not bool(args.no_progress),
        )
    else:
        payload = run_normal_finite_q_ward_audit(
            omega_eV=args.omega,
            q_values=tuple(args.q_values),
            q_directions=tuple(args.directions),
            nk_values=nk_values,
            twist_counts=tuple(args.twist_counts),
            twist_summary_only=bool(args.twist_summary_only),
            temperature_K=args.temperature_K,
            eta_eV=args.eta,
        )
    if args.json_output is not None:
        _write_json(args.json_output, payload)
    print(
        "normal finite-q Ward audit prepared: "
        f"nk_values={payload['nk_values']}, q_values={payload['q_values']}, "
        f"directions={payload['q_directions']}, twist_counts={payload['twist_counts']}, "
        f"summary_only={payload.get('output_format', {}).get('summary_only', False)}, "
        f"valid_for_casimir_input={payload['valid_for_casimir_input']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
