#!/usr/bin/env python3
"""Stage 4.16 full physical Ward response with adaptive quadrature.

Diagnostic-only.  This script applies the Stage 4.15 Fermi-window adaptive
quadrature to the full physical response.  It does not modify the main
response, bubble sign, direct contact, conductivity, reflection, or Casimir
code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327 import KuboConfig, bosonic_matsubara_energy_eV, fermi_function, k_weights, uniform_bz_mesh  # noqa: E402
from lno327.models.lno327_four_orbital.peierls import normal_state_hopping_terms  # noqa: E402
from lno327.collective.ward import physical_ward_residuals  # noqa: E402

from stage4_15_fermi_window_adaptive_quadrature import (  # noqa: E402
    build_adaptive_cells,
    quadrature_points_for_cells,
)

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "ward_identity"
JSON_OUTPUT = OUTPUT_DIR / "stage4_16_full_response_adaptive_ward_diagnostic.json"
MD_OUTPUT = OUTPUT_DIR / "stage4_16_full_response_adaptive_ward_diagnostic.md"

TEMPERATURE_K = 30.0
MATSUBARA_INDEX = 1
ETA_EV = 1e-10
OUTPUT_SI = False
Q_BASE = np.array([0.02, 0.013], dtype=float)
Q_SCALES = (1.0, 0.5, 0.25, 0.125)
COARSE_GRID = 32
MAX_REFINEMENT_LEVELS = (0, 1, 2, 3, 4)
GAUSS_ORDER = 3
FERMI_WINDOW_EV = 0.05
UNIFORM_REFERENCE_MESHES = (32, 64)
DIRECTIONS = ("x", "y")
EPS = 1e-300
INTEGRATION_CHUNK_SIZE = 4096
HOPPING_TERMS = normal_state_hopping_terms()


def to_jsonable(value: Any) -> Any:
    """Convert numpy values to JSON-safe builtins and reject complex scalars."""

    if isinstance(value, complex | np.complexfloating):
        raise TypeError("complex values must be split into real/imag before JSON serialization")
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def _complex_vector_parts(vector: np.ndarray) -> dict[str, list[float]]:
    return {
        "real": [float(item.real) for item in vector],
        "imag": [float(item.imag) for item in vector],
        "abs": [float(abs(item)) for item in vector],
    }


def config_for_temperature() -> KuboConfig:
    omega_eV = bosonic_matsubara_energy_eV(MATSUBARA_INDEX, TEMPERATURE_K)
    return KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=TEMPERATURE_K,
        eta_eV=ETA_EV,
        output_si=OUTPUT_SI,
    )


def _sinc_scalar(x: float) -> float:
    return 1.0 if abs(x) < 1e-12 else float(np.sin(x) / x)


def _batch_hamiltonian(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    matrices = np.zeros((len(kx), 4, 4), dtype=complex)
    for (rx, ry), hopping in HOPPING_TERMS:
        phase = np.exp(1j * (kx * rx + ky * ry))
        matrices += phase[:, None, None] * hopping[None, :, :]
    return matrices


def _batch_vector_vertex(kx: np.ndarray, ky: np.ndarray, q: np.ndarray, direction: str) -> np.ndarray:
    qx, qy = float(q[0]), float(q[1])
    matrices = np.zeros((len(kx), 4, 4), dtype=complex)
    for (rx, ry), hopping in HOPPING_TERMS:
        component = rx if direction == "x" else ry
        if component == 0:
            continue
        phase = np.exp(1j * (kx * rx + ky * ry))
        sinc_factor = _sinc_scalar(0.5 * (qx * rx + qy * ry))
        matrices += 1j * component * sinc_factor * phase[:, None, None] * hopping[None, :, :]
    return matrices


def _batch_contact_vertex(kx: np.ndarray, ky: np.ndarray, q: np.ndarray, direction_i: str, direction_j: str) -> np.ndarray:
    qx, qy = float(q[0]), float(q[1])
    matrices = np.zeros((len(kx), 4, 4), dtype=complex)
    for (rx, ry), hopping in HOPPING_TERMS:
        component_i = rx if direction_i == "x" else ry
        component_j = rx if direction_j == "x" else ry
        if component_i == 0 or component_j == 0:
            continue
        phase = np.exp(1j * (kx * rx + ky * ry))
        sinc_factor = _sinc_scalar(0.5 * (qx * rx + qy * ry))
        matrices += (
            -component_i
            * component_j
            * sinc_factor
            * sinc_factor
            * phase[:, None, None]
            * hopping[None, :, :]
        )
    return matrices


def _thermal_trace_batch(
    energies: np.ndarray,
    states: np.ndarray,
    operator_matrices: np.ndarray,
    config: KuboConfig,
) -> np.ndarray:
    occupations = fermi_function(energies, config.fermi_level_eV, config.temperature_eV)
    operator_band = np.einsum("...am,...ab,...bn->...mn", states.conjugate(), operator_matrices, states)
    diagonal = np.diagonal(operator_band, axis1=-2, axis2=-1)
    return np.sum(occupations * diagonal, axis=-1)


def _finite_q_bubble_batch(
    energies_minus: np.ndarray,
    states_minus: np.ndarray,
    energies_plus: np.ndarray,
    states_plus: np.ndarray,
    observable_vertices: tuple[np.ndarray, np.ndarray, np.ndarray],
    source_vertices: tuple[np.ndarray, np.ndarray, np.ndarray],
    config: KuboConfig,
) -> np.ndarray:
    occupations_minus = fermi_function(energies_minus, config.fermi_level_eV, config.temperature_eV)
    occupations_plus = fermi_function(energies_plus, config.fermi_level_eV, config.temperature_eV)
    observable_matrices = tuple(
        np.einsum("...am,...ab,...bn->...mn", states_minus.conjugate(), vertex, states_plus)
        for vertex in observable_vertices
    )
    source_matrices = tuple(
        np.einsum("...am,...ab,...bn->...mn", states_minus.conjugate(), vertex, states_plus)
        for vertex in source_vertices
    )
    denominator = 1j * config.omega_eV + energies_minus[:, :, None] - energies_plus[:, None, :]
    factor = (occupations_minus[:, :, None] - occupations_plus[:, None, :]) / denominator
    response = np.zeros((len(energies_minus), 3, 3), dtype=complex)
    for mu, observable_matrix in enumerate(observable_matrices):
        for nu, source_matrix in enumerate(source_matrices):
            response[:, mu, nu] = np.sum(factor * observable_matrix * np.conjugate(source_matrix), axis=(1, 2))
    return response


def integrate_physical_components_on_points(
    points: np.ndarray,
    weights: np.ndarray,
    q: np.ndarray,
    config: KuboConfig,
) -> dict[str, np.ndarray]:
    """Integrate full physical response components on identical points/weights."""

    qx, qy = float(q[0]), float(q[1])
    bubble = np.zeros((3, 3), dtype=complex)
    direct = np.zeros((3, 3), dtype=complex)
    identity = np.eye(4, dtype=complex)
    for start in range(0, len(points), INTEGRATION_CHUNK_SIZE):
        stop = min(start + INTEGRATION_CHUNK_SIZE, len(points))
        chunk = points[start:stop]
        chunk_weights = weights[start:stop]
        kx = chunk[:, 0]
        ky = chunk[:, 1]
        h_minus = _batch_hamiltonian(kx - 0.5 * qx, ky - 0.5 * qy)
        h_plus = _batch_hamiltonian(kx + 0.5 * qx, ky + 0.5 * qy)
        h_mid = _batch_hamiltonian(kx, ky)
        e_minus, u_minus = np.linalg.eigh(h_minus)
        e_plus, u_plus = np.linalg.eigh(h_plus)
        e_mid, u_mid = np.linalg.eigh(h_mid)
        rho = np.broadcast_to(identity, h_minus.shape)
        vector_x = _batch_vector_vertex(kx, ky, q, "x")
        vector_y = _batch_vector_vertex(kx, ky, q, "y")
        observable_vertices = (rho, -vector_x, -vector_y)
        source_vertices = (rho, vector_x, vector_y)
        bubble_values = _finite_q_bubble_batch(
            e_minus,
            u_minus,
            e_plus,
            u_plus,
            observable_vertices,
            source_vertices,
            config,
        )
        bubble += np.einsum("p,pij->ij", chunk_weights, bubble_values)
        for i, direction_i in enumerate(DIRECTIONS):
            for j, direction_j in enumerate(DIRECTIONS):
                contact = _batch_contact_vertex(kx, ky, q, direction_i, direction_j)
                expect_mij = _thermal_trace_batch(e_mid, u_mid, contact, config)
                direct[1 + i, 1 + j] += np.sum(chunk_weights * (-expect_mij))
    return {"bubble": bubble, "direct": direct, "total": bubble + direct}


def project_spatial_components(q: np.ndarray, spatial: np.ndarray) -> tuple[complex, complex]:
    qnorm = float(np.linalg.norm(q))
    if qnorm <= 0.0:
        raise ValueError("q norm must be positive")
    qhat = q / qnorm
    that = np.array([-qhat[1], qhat[0]], dtype=float)
    longitudinal = qhat[0] * spatial[0] + qhat[1] * spatial[1]
    transverse = that[0] * spatial[0] + that[1] * spatial[1]
    return complex(longitudinal), complex(transverse)


def response_result_row(
    *,
    label: str,
    q_scale: float,
    q: np.ndarray,
    components: dict[str, np.ndarray],
    config: KuboConfig,
    num_cells_total: int,
    num_cells_refined: int,
    num_quadrature_points: int,
    same_points_weights_for_bubble_and_direct: bool,
    refinement_level: int | None = None,
) -> dict[str, Any]:
    left, right = physical_ward_residuals(components["total"], config.omega_eV, q)
    left_long, left_trans = project_spatial_components(q, left[1:])
    right_long, right_trans = project_spatial_components(q, right[1:])
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    row: dict[str, Any] = {
        "label": label,
        "q_scale": float(q_scale),
        "q_model": [float(q[0]), float(q[1])],
        "q_norm": float(np.linalg.norm(q)),
        "left_norm": left_norm,
        "right_norm": right_norm,
        "max_norm": max(left_norm, right_norm),
        "left_density_source_abs": float(abs(left[0])),
        "left_spatial_source_norm": float(np.linalg.norm(left[1:])),
        "right_density_observable_abs": float(abs(right[0])),
        "right_spatial_observable_norm": float(np.linalg.norm(right[1:])),
        "left_longitudinal_abs": float(abs(left_long)),
        "left_transverse_abs": float(abs(left_trans)),
        "right_longitudinal_abs": float(abs(right_long)),
        "right_transverse_abs": float(abs(right_trans)),
        "left_residual": _complex_vector_parts(left),
        "right_residual": _complex_vector_parts(right),
        "num_cells_total": int(num_cells_total),
        "num_cells_refined": int(num_cells_refined),
        "num_quadrature_points": int(num_quadrature_points),
        "same_points_weights_for_bubble_and_direct": bool(same_points_weights_for_bubble_and_direct),
    }
    if refinement_level is not None:
        row["refinement_level"] = int(refinement_level)
    return row


def adaptive_response_rows(
    *,
    q_scale: float,
    q: np.ndarray,
    config: KuboConfig,
    coarse_grid: int,
    max_refinement_levels: tuple[int, ...] | list[int],
    gauss_order: int,
    fermi_window_eV: float,
) -> list[dict[str, Any]]:
    rows = []
    for refinement_level in max_refinement_levels:
        cells, refined_count, _flagged_base = build_adaptive_cells(
            q,
            coarse_grid=int(coarse_grid),
            refinement_level=int(refinement_level),
            fermi_window_eV=float(fermi_window_eV),
            fermi_level_eV=config.fermi_level_eV,
        )
        points, weights = quadrature_points_for_cells(cells, int(gauss_order))
        components = integrate_physical_components_on_points(points, weights, q, config)
        rows.append(
            response_result_row(
                label="adaptive_fermi_window_full_response",
                q_scale=q_scale,
                q=q,
                components=components,
                config=config,
                num_cells_total=len(cells),
                num_cells_refined=refined_count,
                num_quadrature_points=len(points),
                same_points_weights_for_bubble_and_direct=True,
                refinement_level=int(refinement_level),
            )
        )
    return rows


def uniform_reference_rows(q_scales: tuple[float, ...] | list[float], config: KuboConfig) -> list[dict[str, Any]]:
    rows = []
    for mesh_size in UNIFORM_REFERENCE_MESHES:
        mesh = uniform_bz_mesh(int(mesh_size))
        weights = k_weights(mesh)
        for q_scale in q_scales:
            q = float(q_scale) * Q_BASE
            components = integrate_physical_components_on_points(mesh, weights, q, config)
            rows.append(
                response_result_row(
                    label=f"uniform_mesh_{mesh_size}",
                    q_scale=float(q_scale),
                    q=q,
                    components=components,
                    config=config,
                    num_cells_total=mesh_size * mesh_size,
                    num_cells_refined=0,
                    num_quadrature_points=len(mesh),
                    same_points_weights_for_bubble_and_direct=True,
                )
            )
    return rows


def classify_adaptive_improvement(adaptive_results: list[dict[str, Any]], uniform_results: list[dict[str, Any]]) -> str:
    uniform64 = [row for row in uniform_results if row["label"] == "uniform_mesh_64"]
    if not adaptive_results or not uniform64:
        return "ADAPTIVE_FULL_RESPONSE_INCONCLUSIVE"
    max_level = max(int(row["refinement_level"]) for row in adaptive_results)
    adaptive_final = [row for row in adaptive_results if int(row["refinement_level"]) == max_level]
    adaptive_max = max(float(row["max_norm"]) for row in adaptive_final)
    uniform64_max = max(float(row["max_norm"]) for row in uniform64)
    if adaptive_max < 0.1 * uniform64_max:
        return "ADAPTIVE_FULL_RESPONSE_IMPROVES_WARD"
    return "ADAPTIVE_FULL_RESPONSE_INCONCLUSIVE"


def classify_refinement_convergence(adaptive_results: list[dict[str, Any]]) -> str:
    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in adaptive_results:
        grouped.setdefault(float(row["q_scale"]), []).append(row)
    if not grouped:
        return "ADAPTIVE_FULL_RESPONSE_INCONCLUSIVE"
    converging = True
    for rows in grouped.values():
        rows_sorted = sorted(rows, key=lambda item: int(item["refinement_level"]))
        values = [float(row["max_norm"]) for row in rows_sorted]
        if values[-1] > values[0]:
            converging = False
            break
        upticks = sum(1 for a, b in zip(values, values[1:], strict=False) if b > 1.05 * a)
        if upticks > 1:
            converging = False
            break
    return "FULL_RESPONSE_REFINEMENT_CONVERGING" if converging else "ADAPTIVE_FULL_RESPONSE_INCONCLUSIVE"


def dominant_remaining_channel(adaptive_results: list[dict[str, Any]]) -> str:
    if not adaptive_results:
        return "unknown"
    max_level = max(int(row["refinement_level"]) for row in adaptive_results)
    final_rows = [row for row in adaptive_results if int(row["refinement_level"]) == max_level]
    totals = {
        "left_density_source": max(float(row["left_density_source_abs"]) for row in final_rows),
        "left_spatial_source": max(float(row["left_spatial_source_norm"]) for row in final_rows),
        "right_density_observable": max(float(row["right_density_observable_abs"]) for row in final_rows),
        "right_spatial_observable": max(float(row["right_spatial_observable_norm"]) for row in final_rows),
        "left_longitudinal": max(float(row["left_longitudinal_abs"]) for row in final_rows),
        "left_transverse": max(float(row["left_transverse_abs"]) for row in final_rows),
        "right_longitudinal": max(float(row["right_longitudinal_abs"]) for row in final_rows),
        "right_transverse": max(float(row["right_transverse_abs"]) for row in final_rows),
    }
    return max(totals, key=totals.get)


def closure_status(adaptive_results: list[dict[str, Any]]) -> str:
    if not adaptive_results:
        return "ADAPTIVE_FULL_RESPONSE_INCONCLUSIVE"
    max_level = max(int(row["refinement_level"]) for row in adaptive_results)
    adaptive_final = [row for row in adaptive_results if int(row["refinement_level"]) == max_level]
    adaptive_max = max(float(row["max_norm"]) for row in adaptive_final)
    if adaptive_max < 1e-6:
        return "NUMERICALLY_CLOSED_WITH_ADAPTIVE_QUADRATURE"
    return "IMPROVED_BUT_NOT_CLOSED"


def likely_issue_and_next_step(improvement_status: str, close_status: str) -> tuple[str, str]:
    if close_status == "NUMERICALLY_CLOSED_WITH_ADAPTIVE_QUADRATURE":
        return (
            "LOW_TEMPERATURE_QUADRATURE_WAS_DOMINANT",
            "Next: document adaptive full Ward closure and add a non-regression test before considering response-cache integration.",
        )
    if improvement_status == "ADAPTIVE_FULL_RESPONSE_IMPROVES_WARD":
        return (
            "QUADRATURE_IMPROVED_REMAINING_ROUTING_OR_DENSITY_CONVENTION",
            "Next: increase adaptive resolution and audit remaining channel decomposition before any conductivity/reflection/Casimir use.",
        )
    return (
        "FULL_RESPONSE_RESIDUAL_NOT_PRIMARILY_QUADRATURE",
        "Next: audit finite-q density vertex embedding, contact expectation routing, and response-level conventions.",
    )


def run_audit(
    *,
    coarse_grid: int = COARSE_GRID,
    max_refinement_levels: tuple[int, ...] | list[int] = MAX_REFINEMENT_LEVELS,
    gauss_order: int = GAUSS_ORDER,
    fermi_window_eV: float = FERMI_WINDOW_EV,
    q_scales: tuple[float, ...] | list[float] = Q_SCALES,
) -> dict[str, Any]:
    config = config_for_temperature()
    adaptive_results = []
    for q_scale in q_scales:
        q = float(q_scale) * Q_BASE
        adaptive_results.extend(
            adaptive_response_rows(
                q_scale=float(q_scale),
                q=q,
                config=config,
                coarse_grid=int(coarse_grid),
                max_refinement_levels=max_refinement_levels,
                gauss_order=int(gauss_order),
                fermi_window_eV=float(fermi_window_eV),
            )
        )
    uniform_results = uniform_reference_rows(q_scales, config)
    improvement = classify_adaptive_improvement(adaptive_results, uniform_results)
    convergence = classify_refinement_convergence(adaptive_results)
    close = closure_status(adaptive_results)
    dominant = dominant_remaining_channel(adaptive_results)
    likely_issue, next_step = likely_issue_and_next_step(improvement, close)
    return {
        "stage": "Stage 4.16",
        "purpose": "Full physical Ward response diagnostic with Fermi-window adaptive quadrature",
        "config": {
            "temperature_K": TEMPERATURE_K,
            "matsubara_index": MATSUBARA_INDEX,
            "omega_eV": float(config.omega_eV),
            "eta_eV": ETA_EV,
            "output_si": OUTPUT_SI,
            "q_base": [float(Q_BASE[0]), float(Q_BASE[1])],
            "q_scales": [float(item) for item in q_scales],
            "coarse_grid": int(coarse_grid),
            "max_refinement_levels": [int(item) for item in max_refinement_levels],
            "gauss_order": int(gauss_order),
            "fermi_window_eV": float(fermi_window_eV),
            "uniform_reference_meshes": [int(item) for item in UNIFORM_REFERENCE_MESHES],
        },
        "adaptive_response_results": adaptive_results,
        "uniform_reference_results": uniform_results,
        "diagnostic_status": {
            "adaptive_improvement_status": improvement,
            "refinement_convergence_status": convergence,
            "closure_status": close,
            "dominant_remaining_channel": dominant,
            "likely_issue": likely_issue,
            "next_step": next_step,
        },
        "boundary": {
            "no_main_response_change": True,
            "no_bubble_sign_change": True,
            "no_direct_contact_change": True,
            "no_source_observable_change": True,
            "no_residual_tuning": True,
            "no_fitted_contact": True,
            "no_E_ET_added": True,
            "no_conductivity_reflection_casimir": True,
        },
    }


def _fmt(value: float) -> str:
    return f"{value:.6e}"


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def render_markdown(data: dict[str, Any]) -> str:
    status = data["diagnostic_status"]
    max_level = max(int(row["refinement_level"]) for row in data["adaptive_response_results"])
    final_rows = [row for row in data["adaptive_response_results"] if int(row["refinement_level"]) == max_level]
    adaptive_table = _table(
        ("q_scale", "level", "max_norm", "left_norm", "right_norm", "quad points"),
        [
            (
                _fmt(float(row["q_scale"])),
                row["refinement_level"],
                _fmt(float(row["max_norm"])),
                _fmt(float(row["left_norm"])),
                _fmt(float(row["right_norm"])),
                row["num_quadrature_points"],
            )
            for row in final_rows
        ],
    )
    uniform_table = _table(
        ("label", "max max_norm"),
        [
            (
                label,
                _fmt(max(float(row["max_norm"]) for row in data["uniform_reference_results"] if row["label"] == label)),
            )
            for label in ("uniform_mesh_32", "uniform_mesh_64")
        ],
    )
    channel_table = _table(
        ("q_scale", "left_long", "left_trans", "right_long", "right_trans"),
        [
            (
                _fmt(float(row["q_scale"])),
                _fmt(float(row["left_longitudinal_abs"])),
                _fmt(float(row["left_transverse_abs"])),
                _fmt(float(row["right_longitudinal_abs"])),
                _fmt(float(row["right_transverse_abs"])),
            )
            for row in final_rows
        ],
    )
    return "\n\n".join(
        [
            "# Stage 4.16 Full response adaptive Ward diagnostic",
            "## Boundary\n\n"
            "- no main response change\n"
            "- no bubble sign change\n"
            "- no direct contact change\n"
            "- no source/observable change\n"
            "- no residual tuning\n"
            "- no fitted contact\n"
            "- no E_ET added\n"
            "- no conductivity / reflection / Casimir",
            "## Formula being tested\n\n"
            "$$\\Pi_{\\mu\\nu}=\\Pi_{\\mu\\nu}^{bubble}+D_{\\mu\\nu},$$\n\n"
            "with $J=(\\rho,-V_x,-V_y)$, $P=(\\rho,V_x,V_y)$, corrected positive bubble prefactor, and $D_{ij}=-\\langle M_{ij}\\rangle$.",
            "## Adaptive quadrature summary\n\nStage 4.13 fixed the bubble sign. Stage 4.15 showed adaptive quadrature improves $C-K$. This stage applies the same point/weight strategy to the full physical response.",
            "## Uniform reference comparison\n\n" + uniform_table,
            "## Ward residual versus refinement\n\n" + adaptive_table,
            "## Longitudinal/transverse decomposition\n\n" + channel_table,
            "## Diagnostic decision\n\n"
            + _table(
                ("quantity", "status"),
                [
                    ("adaptive_improvement_status", status["adaptive_improvement_status"]),
                    ("refinement_convergence_status", status["refinement_convergence_status"]),
                    ("closure_status", status["closure_status"]),
                    ("dominant_remaining_channel", status["dominant_remaining_channel"]),
                    ("likely_issue", status["likely_issue"]),
                ],
            )
            + "\n\nIf not closed, do not revert the Stage 4.13 bubble sign or change direct contact. Full Ward response numerical validation is required before conductivity/reflection/Casimir.",
            "## Next step\n\n" + status["next_step"],
        ]
    ) + "\n"


def main() -> None:
    data = run_audit()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MD_OUTPUT.write_text(render_markdown(data), encoding="utf-8")
    print(f"Wrote {JSON_OUTPUT}")
    print(f"Wrote {MD_OUTPUT}")


if __name__ == "__main__":
    main()
