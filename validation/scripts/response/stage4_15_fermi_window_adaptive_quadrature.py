#!/usr/bin/env python3
"""Stage 4.15 Fermi-window adaptive quadrature for C-K diagnostics.

Diagnostic-only.  This script does not modify the main response, the bubble
sign, direct contact, conductivity, reflection, or Casimir code.
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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327.conductivity import KuboConfig, fermi_function, k_weights, uniform_bz_mesh  # noqa: E402
from lno327.model import normal_state_hamiltonian  # noqa: E402
from lno327.tb_fourier import normal_state_hopping_terms  # noqa: E402

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "ward_identity"
JSON_OUTPUT = OUTPUT_DIR / "stage4_15_fermi_window_adaptive_quadrature.json"
MD_OUTPUT = OUTPUT_DIR / "stage4_15_fermi_window_adaptive_quadrature.md"

TEMPERATURE_K = 30.0
Q_BASE = np.array([0.02, 0.013], dtype=float)
Q_SCALES = (1.0, 0.5, 0.25, 0.125)
DIRECTIONS = ("x", "y")
COARSE_GRID = 32
MAX_REFINEMENT_LEVELS = (0, 1, 2, 3, 4)
GAUSS_ORDER = 3
FERMI_WINDOW_EV = 0.05
TEMPERATURE_SWEEP_K = (30.0, 100.0, 300.0, 1000.0)
UNIFORM_REFERENCE_MESHES = (32, 64)
EPS = 1e-300
INTEGRATION_CHUNK_SIZE = 8192
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


def _complex_parts(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag), "abs": float(abs(value))}


def _rel_error(abs_error: float, *refs: complex | float) -> float:
    return float(abs_error / max(*(abs(ref) for ref in refs), EPS))


def config_for_temperature(temperature_K: float) -> KuboConfig:
    return KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=float(temperature_K),
        eta_eV=1e-10,
        output_si=False,
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
    hamiltonian_matrices: np.ndarray,
    operator_matrices: np.ndarray,
    config: KuboConfig,
) -> np.ndarray:
    energies, states = np.linalg.eigh(hamiltonian_matrices)
    occupations = fermi_function(energies, config.fermi_level_eV, config.temperature_eV)
    operator_band = np.einsum("...am,...ab,...bn->...mn", states.conjugate(), operator_matrices, states)
    diagonal = np.diagonal(operator_band, axis1=-2, axis2=-1)
    return np.sum(occupations * diagonal, axis=-1)


def integrate_ck_on_points(points: np.ndarray, weights: np.ndarray, q: np.ndarray, config: KuboConfig) -> dict[str, tuple[complex, complex]]:
    """Integrate C and K on an identical set of points and weights."""

    totals = {direction: [0.0j, 0.0j] for direction in DIRECTIONS}
    qx, qy = float(q[0]), float(q[1])
    for start in range(0, len(points), INTEGRATION_CHUNK_SIZE):
        stop = min(start + INTEGRATION_CHUNK_SIZE, len(points))
        chunk = points[start:stop]
        chunk_weights = weights[start:stop]
        kx = chunk[:, 0]
        ky = chunk[:, 1]
        h_minus = _batch_hamiltonian(kx - 0.5 * qx, ky - 0.5 * qy)
        h_plus = _batch_hamiltonian(kx + 0.5 * qx, ky + 0.5 * qy)
        h_mid = _batch_hamiltonian(kx, ky)
        for direction in DIRECTIONS:
            vector = _batch_vector_vertex(kx, ky, q, direction)
            c_values = _thermal_trace_batch(h_minus, vector, config) - _thermal_trace_batch(h_plus, vector, config)
            m_xj = _batch_contact_vertex(kx, ky, q, "x", direction)
            m_yj = _batch_contact_vertex(kx, ky, q, "y", direction)
            k_values = _thermal_trace_batch(h_mid, qx * m_xj + qy * m_yj, config)
            totals[direction][0] += np.sum(chunk_weights * c_values)
            totals[direction][1] += np.sum(chunk_weights * k_values)
    return {direction: (complex(values[0]), complex(values[1])) for direction, values in totals.items()}


def _cell_sample_points(x0: float, x1: float, y0: float, y1: float) -> tuple[tuple[float, float], ...]:
    xm = 0.5 * (x0 + x1)
    ym = 0.5 * (y0 + y1)
    return ((x0, y0), (x0, y1), (x1, y0), (x1, y1), (xm, ym))


def _cell_in_fermi_window(
    cell: tuple[float, float, float, float],
    q: np.ndarray,
    fermi_window_eV: float,
    fermi_level_eV: float,
) -> bool:
    x0, x1, y0, y1 = cell
    qx, qy = float(q[0]), float(q[1])
    for kx, ky in _cell_sample_points(x0, x1, y0, y1):
        for sx, sy in ((0.0, 0.0), (0.5 * qx, 0.5 * qy), (-0.5 * qx, -0.5 * qy)):
            energies = np.linalg.eigvalsh(normal_state_hamiltonian(kx + sx, ky + sy))
            if np.any(np.abs(energies - fermi_level_eV) < fermi_window_eV):
                return True
    return False


def build_adaptive_cells(
    q: np.ndarray,
    *,
    coarse_grid: int,
    refinement_level: int,
    fermi_window_eV: float,
    fermi_level_eV: float = 0.0,
) -> tuple[list[tuple[float, float, float, float]], int, int]:
    """Return final cells, number of refined parent cells, and flagged base cells."""

    edges = np.linspace(-np.pi, np.pi, coarse_grid + 1)
    cells = [
        (float(edges[ix]), float(edges[ix + 1]), float(edges[iy]), float(edges[iy + 1]))
        for ix in range(coarse_grid)
        for iy in range(coarse_grid)
    ]
    flagged_base = sum(
        1 for cell in cells if _cell_in_fermi_window(cell, q, fermi_window_eV, fermi_level_eV)
    )
    refined_count = 0
    for _level in range(refinement_level):
        next_cells: list[tuple[float, float, float, float]] = []
        for cell in cells:
            if _cell_in_fermi_window(cell, q, fermi_window_eV, fermi_level_eV):
                x0, x1, y0, y1 = cell
                xm = 0.5 * (x0 + x1)
                ym = 0.5 * (y0 + y1)
                next_cells.extend(
                    [
                        (x0, xm, y0, ym),
                        (x0, xm, ym, y1),
                        (xm, x1, y0, ym),
                        (xm, x1, ym, y1),
                    ]
                )
                refined_count += 1
            else:
                next_cells.append(cell)
        cells = next_cells
    return cells, refined_count, flagged_base


def quadrature_points_for_cells(
    cells: list[tuple[float, float, float, float]],
    gauss_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return quadrature points and normalized BZ-average weights."""

    nodes, node_weights = np.polynomial.legendre.leggauss(gauss_order)
    points = []
    weights = []
    bz_area = (2.0 * np.pi) ** 2
    for x0, x1, y0, y1 in cells:
        x_mid = 0.5 * (x0 + x1)
        y_mid = 0.5 * (y0 + y1)
        x_half = 0.5 * (x1 - x0)
        y_half = 0.5 * (y1 - y0)
        for i, node_x in enumerate(nodes):
            for j, node_y in enumerate(nodes):
                points.append((x_mid + x_half * node_x, y_mid + y_half * node_y))
                weights.append(float(node_weights[i] * node_weights[j] * x_half * y_half / bz_area))
    return np.asarray(points, dtype=float), np.asarray(weights, dtype=float)


def row_from_ck(
    *,
    label: str,
    q_scale: float,
    q: np.ndarray,
    direction: str,
    c_value: complex,
    k_value: complex,
    num_cells_total: int,
    num_cells_refined: int,
    num_quadrature_points: int,
    same_points_weights_for_C_and_K: bool,
    refinement_level: int | None = None,
    temperature_K: float = TEMPERATURE_K,
    coarse_grid: int | None = None,
    gauss_order: int | None = None,
) -> dict[str, Any]:
    c_minus_k = c_value - k_value
    row: dict[str, Any] = {
        "label": label,
        "temperature_K": float(temperature_K),
        "q_scale": float(q_scale),
        "q_model": [float(q[0]), float(q[1])],
        "q_norm": float(np.linalg.norm(q)),
        "direction": direction,
        "C": _complex_parts(c_value),
        "K": _complex_parts(k_value),
        "C_minus_K": _complex_parts(c_minus_k),
        "C_minus_K_rel": _rel_error(abs(c_minus_k), c_value, k_value),
        "num_cells_total": int(num_cells_total),
        "num_cells_refined": int(num_cells_refined),
        "num_quadrature_points": int(num_quadrature_points),
        "same_points_weights_for_C_and_K": bool(same_points_weights_for_C_and_K),
    }
    if refinement_level is not None:
        row["refinement_level"] = int(refinement_level)
    if coarse_grid is not None:
        row["coarse_grid"] = int(coarse_grid)
    if gauss_order is not None:
        row["gauss_order"] = int(gauss_order)
    return row


def adaptive_rows_for_q(
    *,
    q_scale: float,
    q: np.ndarray,
    config: KuboConfig,
    coarse_grid: int,
    max_refinement_levels: tuple[int, ...] | list[int],
    gauss_order: int,
    fermi_window_eV: float,
    temperature_K: float,
) -> list[dict[str, Any]]:
    rows = []
    for refinement_level in max_refinement_levels:
        cells, refined_count, _flagged_base = build_adaptive_cells(
            q,
            coarse_grid=coarse_grid,
            refinement_level=int(refinement_level),
            fermi_window_eV=fermi_window_eV,
            fermi_level_eV=config.fermi_level_eV,
        )
        points, weights = quadrature_points_for_cells(cells, gauss_order)
        integrated = integrate_ck_on_points(points, weights, q, config)
        for direction in DIRECTIONS:
            c_value, k_value = integrated[direction]
            rows.append(
                row_from_ck(
                    label="adaptive_fermi_window",
                    q_scale=q_scale,
                    q=q,
                    direction=direction,
                    c_value=c_value,
                    k_value=k_value,
                    num_cells_total=len(cells),
                    num_cells_refined=refined_count,
                    num_quadrature_points=len(points),
                    same_points_weights_for_C_and_K=True,
                    refinement_level=int(refinement_level),
                    temperature_K=temperature_K,
                    coarse_grid=coarse_grid,
                    gauss_order=gauss_order,
                )
            )
    return rows


def uniform_reference_rows(q_scales: tuple[float, ...] | list[float], temperature_K: float = TEMPERATURE_K) -> list[dict[str, Any]]:
    config = config_for_temperature(temperature_K)
    rows = []
    for mesh_size in UNIFORM_REFERENCE_MESHES:
        mesh = uniform_bz_mesh(mesh_size)
        weights = k_weights(mesh)
        for q_scale in q_scales:
            q = float(q_scale) * Q_BASE
            integrated = integrate_ck_on_points(mesh, weights, q, config)
            for direction in DIRECTIONS:
                c_value, k_value = integrated[direction]
                rows.append(
                    row_from_ck(
                        label=f"uniform_mesh_{mesh_size}",
                        q_scale=float(q_scale),
                        q=q,
                        direction=direction,
                        c_value=c_value,
                        k_value=k_value,
                        num_cells_total=mesh_size * mesh_size,
                        num_cells_refined=0,
                        num_quadrature_points=len(mesh),
                        same_points_weights_for_C_and_K=True,
                        temperature_K=temperature_K,
                    )
                )
    return rows


def temperature_sweep_rows(
    *,
    coarse_grid: int,
    max_refinement_level: int,
    gauss_order: int,
    fermi_window_eV: float,
    temperature_sweep_K: tuple[float, ...] | list[float],
) -> list[dict[str, Any]]:
    rows = []
    q = Q_BASE
    for temperature_K in temperature_sweep_K:
        config = config_for_temperature(float(temperature_K))
        rows.extend(
            adaptive_rows_for_q(
                q_scale=1.0,
                q=q,
                config=config,
                coarse_grid=coarse_grid,
                max_refinement_levels=(max_refinement_level,),
                gauss_order=gauss_order,
                fermi_window_eV=fermi_window_eV,
                temperature_K=float(temperature_K),
            )
        )
    for row in rows:
        row["label"] = "temperature_sweep_adaptive"
    return rows


def classify_adaptive_improvement(adaptive_results: list[dict[str, Any]], uniform_results: list[dict[str, Any]]) -> str:
    uniform64 = [
        row for row in uniform_results if row["label"] == "uniform_mesh_64" and float(row["temperature_K"]) == TEMPERATURE_K
    ]
    if not adaptive_results or not uniform64:
        return "ADAPTIVE_QUADRATURE_INCONCLUSIVE"
    max_level = max(int(row["refinement_level"]) for row in adaptive_results)
    adaptive_max = max(
        float(row["C_minus_K_rel"])
        for row in adaptive_results
        if int(row["refinement_level"]) == max_level and float(row["temperature_K"]) == TEMPERATURE_K
    )
    uniform64_max = max(float(row["C_minus_K_rel"]) for row in uniform64)
    if adaptive_max < 0.1 * uniform64_max:
        return "ADAPTIVE_QUADRATURE_IMPROVES_CK"
    return "ADAPTIVE_QUADRATURE_INCONCLUSIVE"


def classify_refinement_convergence(adaptive_results: list[dict[str, Any]]) -> str:
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for row in adaptive_results:
        if float(row["temperature_K"]) != TEMPERATURE_K:
            continue
        grouped.setdefault((float(row["q_scale"]), str(row["direction"])), []).append(row)
    if not grouped:
        return "ADAPTIVE_QUADRATURE_INCONCLUSIVE"
    converging = True
    for rows in grouped.values():
        rows_sorted = sorted(rows, key=lambda item: int(item["refinement_level"]))
        rels = [float(row["C_minus_K_rel"]) for row in rows_sorted]
        if rels[-1] > rels[0]:
            converging = False
            break
        upticks = sum(1 for a, b in zip(rels, rels[1:], strict=False) if b > 1.05 * a)
        if upticks > 1:
            converging = False
            break
    return "REFINEMENT_CONVERGING" if converging else "ADAPTIVE_QUADRATURE_INCONCLUSIVE"


def classify_temperature_sanity(temperature_results: list[dict[str, Any]]) -> str:
    high_rows = [row for row in temperature_results if abs(float(row["temperature_K"]) - 1000.0) < 1e-9]
    if high_rows and max(float(row["C_minus_K_rel"]) for row in high_rows) < 1e-2:
        return "TEMPERATURE_SANITY_CONFIRMED"
    return "ADAPTIVE_QUADRATURE_INCONCLUSIVE"


def likely_issue_and_next_step(adaptive_status: str, refinement_status: str, temperature_status: str) -> tuple[str, str]:
    if adaptive_status == "ADAPTIVE_QUADRATURE_IMPROVES_CK":
        return (
            "FERMI_WINDOW_ADAPTIVE_QUADRATURE_CONFIRMED",
            "Next: test the same adaptive quadrature strategy on the full Ward response diagnostic before any conductivity/reflection/Casimir use.",
        )
    if temperature_status == "TEMPERATURE_SANITY_CONFIRMED" and refinement_status == "REFINEMENT_CONVERGING":
        return (
            "FERMI_SURFACE_QUADRATURE_PARTIALLY_CONFIRMED",
            "Next: increase adaptive resolution or improve the Fermi-window cell selector before changing response-level terms.",
        )
    return (
        "ADAPTIVE_QUADRATURE_INCONCLUSIVE",
        "Next: audit finite-q density vertex embedding or contact expectation routing before adding new Ward terms.",
    )


def run_audit(
    *,
    coarse_grid: int = COARSE_GRID,
    max_refinement_levels: tuple[int, ...] | list[int] = MAX_REFINEMENT_LEVELS,
    gauss_order: int = GAUSS_ORDER,
    fermi_window_eV: float = FERMI_WINDOW_EV,
    q_scales: tuple[float, ...] | list[float] = Q_SCALES,
    temperature_sweep_K: tuple[float, ...] | list[float] = TEMPERATURE_SWEEP_K,
) -> dict[str, Any]:
    config = config_for_temperature(TEMPERATURE_K)
    adaptive_results = []
    for q_scale in q_scales:
        q = float(q_scale) * Q_BASE
        adaptive_results.extend(
            adaptive_rows_for_q(
                q_scale=float(q_scale),
                q=q,
                config=config,
                coarse_grid=int(coarse_grid),
                max_refinement_levels=max_refinement_levels,
                gauss_order=int(gauss_order),
                fermi_window_eV=float(fermi_window_eV),
                temperature_K=TEMPERATURE_K,
            )
        )
    uniform_results = uniform_reference_rows(q_scales, TEMPERATURE_K)
    temperature_results = temperature_sweep_rows(
        coarse_grid=int(coarse_grid),
        max_refinement_level=max(int(level) for level in max_refinement_levels),
        gauss_order=int(gauss_order),
        fermi_window_eV=float(fermi_window_eV),
        temperature_sweep_K=temperature_sweep_K,
    )
    adaptive_status = classify_adaptive_improvement(adaptive_results, uniform_results)
    refinement_status = classify_refinement_convergence(adaptive_results)
    temperature_status = classify_temperature_sanity(temperature_results)
    likely_issue, next_step = likely_issue_and_next_step(adaptive_status, refinement_status, temperature_status)
    return {
        "stage": "Stage 4.15",
        "purpose": "Fermi-window adaptive quadrature for C-K Ward diagnostic",
        "config": {
            "temperature_K": TEMPERATURE_K,
            "q_base": [float(Q_BASE[0]), float(Q_BASE[1])],
            "q_scales": [float(item) for item in q_scales],
            "directions": list(DIRECTIONS),
            "coarse_grid": int(coarse_grid),
            "max_refinement_levels": [int(item) for item in max_refinement_levels],
            "gauss_order": int(gauss_order),
            "fermi_window_eV": float(fermi_window_eV),
            "temperature_sweep_K": [float(item) for item in temperature_sweep_K],
        },
        "adaptive_results": adaptive_results,
        "uniform_reference_results": uniform_results,
        "temperature_sweep_results": temperature_results,
        "diagnostic_status": {
            "adaptive_improvement_status": adaptive_status,
            "refinement_convergence_status": refinement_status,
            "temperature_sanity_status": temperature_status,
            "likely_issue": likely_issue,
            "next_step": next_step,
        },
        "boundary": {
            "no_main_response_change": True,
            "no_bubble_sign_change": True,
            "no_direct_contact_change": True,
            "no_residual_tuning": True,
            "no_fitted_contact": True,
            "no_E_ET_added": True,
            "no_conductivity_reflection_casimir": True,
            "does_not_claim_ward_closure": True,
        },
    }


def _fmt(value: float) -> str:
    return f"{value:.6e}"


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _max_by_label(rows: list[dict[str, Any]], label: str) -> float:
    values = [float(row["C_minus_K_rel"]) for row in rows if row["label"] == label]
    return max(values) if values else float("nan")


def render_markdown(data: dict[str, Any]) -> str:
    status = data["diagnostic_status"]
    max_level = max(int(row["refinement_level"]) for row in data["adaptive_results"])
    adaptive_final = [row for row in data["adaptive_results"] if int(row["refinement_level"]) == max_level]
    adaptive_table = _table(
        ("q_scale", "direction", "level", "|C-K| rel", "cells", "quad points"),
        [
            (
                _fmt(float(row["q_scale"])),
                row["direction"],
                row["refinement_level"],
                _fmt(float(row["C_minus_K_rel"])),
                row["num_cells_total"],
                row["num_quadrature_points"],
            )
            for row in adaptive_final
        ],
    )
    uniform_table = _table(
        ("label", "max |C-K| rel"),
        [
            ("uniform_mesh_32", _fmt(_max_by_label(data["uniform_reference_results"], "uniform_mesh_32"))),
            ("uniform_mesh_64", _fmt(_max_by_label(data["uniform_reference_results"], "uniform_mesh_64"))),
            ("adaptive_final", _fmt(max(float(row["C_minus_K_rel"]) for row in adaptive_final))),
        ],
    )
    temp_table = _table(
        ("temperature_K", "max |C-K| rel"),
        [
            (
                _fmt(float(temp)),
                _fmt(
                    max(
                        float(row["C_minus_K_rel"])
                        for row in data["temperature_sweep_results"]
                        if abs(float(row["temperature_K"]) - float(temp)) < 1e-9
                    )
                ),
            )
            for temp in data["config"]["temperature_sweep_K"]
        ],
    )
    return "\n\n".join(
        [
            "# Stage 4.15 Fermi-window adaptive quadrature",
            "## Boundary\n\n"
            "- no main response change\n"
            "- no bubble sign change\n"
            "- no direct contact change\n"
            "- no residual tuning\n"
            "- no fitted contact\n"
            "- no E_ET added\n"
            "- no conductivity / reflection / Casimir\n"
            "- no Ward closure claim",
            "## Purpose\n\n"
            "Stage 4.13 fixed the bubble sign. Stage 4.14 pointed to low-temperature "
            "Fermi-surface quadrature. This stage checks whether Fermi-window adaptive "
            "quadrature improves $C_j-K_j$ using the same quadrature points and weights for both quantities.",
            "## Adaptive final-level results\n\n" + adaptive_table,
            "## Uniform reference comparison\n\n" + uniform_table,
            "## Temperature sanity\n\n" + temp_table,
            "## Diagnostic decision\n\n"
            + _table(
                ("quantity", "status"),
                [
                    ("adaptive_improvement_status", status["adaptive_improvement_status"]),
                    ("refinement_convergence_status", status["refinement_convergence_status"]),
                    ("temperature_sanity_status", status["temperature_sanity_status"]),
                    ("likely_issue", status["likely_issue"]),
                ],
            ),
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
