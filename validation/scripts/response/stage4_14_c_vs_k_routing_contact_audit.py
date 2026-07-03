#!/usr/bin/env python3
"""Stage 4.14 C_j versus K_j routing/contact audit.

Diagnostic-only.  This script does not modify the main response, the Stage
4.13 bubble sign fix, the direct contact definition, conductivity, reflection,
or Casimir code.
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327.conductivity import KuboConfig, bosonic_matsubara_energy_eV, fermi_function, k_weights, uniform_bz_mesh  # noqa: E402
from lno327.models.lno327_four_orbital.normal import normal_state_hamiltonian  # noqa: E402
from lno327.tb_fourier import (  # noqa: E402
    normal_state_hamiltonian_from_hoppings,
    peierls_hamiltonian_contact_vertex,
    peierls_hamiltonian_vector_vertex,
)

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "ward_identity"
JSON_OUTPUT = OUTPUT_DIR / "stage4_14_c_vs_k_routing_contact_audit.json"
MD_OUTPUT = OUTPUT_DIR / "stage4_14_c_vs_k_routing_contact_audit.md"

TEMPERATURE_K = 30.0
MATSUBARA_INDEX = 1
ETA_EV = 1e-10
OUTPUT_SI = False
Q_BASE = np.array([0.02, 0.013], dtype=float)
Q_SCALES = (1.0, 0.5, 0.25, 0.125)
MESH_SIZES = (8, 12, 16, 24, 32, 48, 64)
TEMPERATURE_SWEEP_K = (30.0, 100.0, 300.0, 1000.0)
DIRECTIONS = ("x", "y")
RANDOM_SEED = 12345
RANDOM_NUM = 32
BASELINE_MESH_SIZE = 16
TEMPERATURE_SWEEP_MESH_SIZE = 32
EPS = 1e-300


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


def thermal_trace_expectation(
    hamiltonian_matrix: np.ndarray,
    operator_matrix: np.ndarray,
    config: KuboConfig,
) -> complex:
    """Return Tr[f(H) O] using the eigenbasis of H."""

    energies, states = np.linalg.eigh(hamiltonian_matrix)
    occupations = fermi_function(energies, config.fermi_level_eV, config.temperature_eV)
    operator_band = states.conjugate().T @ operator_matrix @ states
    return complex(np.sum(occupations * np.diag(operator_band)))


def hamiltonian_representation_check(mesh_size: int = BASELINE_MESH_SIZE) -> dict[str, Any]:
    """Check H0_trig(k) against the hopping/Fourier reconstruction."""

    mesh = uniform_bz_mesh(mesh_size)
    rng = np.random.default_rng(RANDOM_SEED)
    random_points = rng.uniform(-np.pi, np.pi, size=(RANDOM_NUM, 2))
    points = np.vstack([mesh, random_points])
    max_abs = 0.0
    max_rel = 0.0
    for kx, ky in points:
        h_model = normal_state_hamiltonian(float(kx), float(ky))
        h_hopping = normal_state_hamiltonian_from_hoppings(float(kx), float(ky))
        abs_error = float(np.linalg.norm(h_model - h_hopping))
        rel_error = abs_error / max(float(np.linalg.norm(h_model)), EPS)
        max_abs = max(max_abs, abs_error)
        max_rel = max(max_rel, rel_error)
    status = "H_REPRESENTATION_MATCH" if max_abs < 1e-12 else "H_REPRESENTATION_MISMATCH"
    return {
        "mesh_size": int(mesh_size),
        "random_num": RANDOM_NUM,
        "random_seed": RANDOM_SEED,
        "num_points": int(len(points)),
        "max_abs_H_model_minus_H_hopping": max_abs,
        "max_rel_H_model_minus_H_hopping": max_rel,
        "H_representation_status": status,
    }


def second_order_identity_check(mesh_size: int = BASELINE_MESH_SIZE, q: np.ndarray = Q_BASE) -> dict[str, Any]:
    """Check q_i M_ij(k,q) = V_j(k+q/2,q)-V_j(k-q/2,q)."""

    mesh = uniform_bz_mesh(mesh_size)
    qx, qy = float(q[0]), float(q[1])
    max_abs = 0.0
    max_rel = 0.0
    for kx_value, ky_value in mesh:
        kx = float(kx_value)
        ky = float(ky_value)
        for direction_j in DIRECTIONS:
            m_xj = peierls_hamiltonian_contact_vertex(kx, ky, qx, qy, "x", direction_j)
            m_yj = peierls_hamiltonian_contact_vertex(kx, ky, qx, qy, "y", direction_j)
            lhs = qx * m_xj + qy * m_yj
            v_plus = peierls_hamiltonian_vector_vertex(kx + 0.5 * qx, ky + 0.5 * qy, qx, qy, direction_j)
            v_minus = peierls_hamiltonian_vector_vertex(kx - 0.5 * qx, ky - 0.5 * qy, qx, qy, direction_j)
            rhs = v_plus - v_minus
            abs_error = float(np.linalg.norm(lhs - rhs))
            rel_error = abs_error / max(float(np.linalg.norm(rhs)), EPS)
            max_abs = max(max_abs, abs_error)
            max_rel = max(max_rel, rel_error)
    status = "SECOND_ORDER_IDENTITY_MATCH" if max_rel < 1e-10 else "SECOND_ORDER_IDENTITY_MISMATCH"
    return {
        "mesh_size": int(mesh_size),
        "q_model": [qx, qy],
        "max_abs_second_order_identity_error": max_abs,
        "max_rel_second_order_identity_error": max_rel,
        "second_order_identity_status": status,
    }


def C_commutator(
    mesh: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    q: np.ndarray,
    direction_j: str,
) -> complex:
    """Return C_j=sum_k Tr[(f(H_-)-f(H_+)) V_j(k,q)]."""

    qx, qy = float(q[0]), float(q[1])
    total = 0.0j
    for weight, (kx_value, ky_value) in zip(weights, mesh, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        h_minus = normal_state_hamiltonian(kx - 0.5 * qx, ky - 0.5 * qy)
        h_plus = normal_state_hamiltonian(kx + 0.5 * qx, ky + 0.5 * qy)
        vertex = peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, direction_j)
        total += float(weight) * (
            thermal_trace_expectation(h_minus, vertex, config)
            - thermal_trace_expectation(h_plus, vertex, config)
        )
    return complex(total)


def K_midpoint_contact(
    mesh: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    q: np.ndarray,
    direction_j: str,
) -> complex:
    """Return K_j^mid=sum_k Tr[f(H(k)) q_i M_ij(k,q)]."""

    qx, qy = float(q[0]), float(q[1])
    total = 0.0j
    for weight, (kx_value, ky_value) in zip(weights, mesh, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        h0 = normal_state_hamiltonian(kx, ky)
        m_xj = peierls_hamiltonian_contact_vertex(kx, ky, qx, qy, "x", direction_j)
        m_yj = peierls_hamiltonian_contact_vertex(kx, ky, qx, qy, "y", direction_j)
        total += float(weight) * thermal_trace_expectation(h0, qx * m_xj + qy * m_yj, config)
    return complex(total)


def K_deltaV_midpoint(
    mesh: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    q: np.ndarray,
    direction_j: str,
) -> complex:
    """Return K_j from the midpoint expectation of V_j(k+q/2)-V_j(k-q/2)."""

    qx, qy = float(q[0]), float(q[1])
    total = 0.0j
    for weight, (kx_value, ky_value) in zip(weights, mesh, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        h0 = normal_state_hamiltonian(kx, ky)
        v_plus = peierls_hamiltonian_vector_vertex(kx + 0.5 * qx, ky + 0.5 * qy, qx, qy, direction_j)
        v_minus = peierls_hamiltonian_vector_vertex(kx - 0.5 * qx, ky - 0.5 * qy, qx, qy, direction_j)
        total += float(weight) * thermal_trace_expectation(h0, v_plus - v_minus, config)
    return complex(total)


def CK_row(
    mesh_size: int,
    q_scale: float,
    direction_j: str,
    mesh: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    q: np.ndarray,
    *,
    label: str,
) -> dict[str, Any]:
    c_value = C_commutator(mesh, weights, config, q, direction_j)
    k_value = K_midpoint_contact(mesh, weights, config, q, direction_j)
    k_delta_v = K_deltaV_midpoint(mesh, weights, config, q, direction_j)
    c_minus_k = c_value - k_value
    k_contact_minus_delta_v = k_value - k_delta_v
    mesh_shift_error = c_value - k_delta_v
    return {
        "label": label,
        "mesh_size": int(mesh_size),
        "temperature_K": float(config.temperature_eV / 8.617333262145e-5),
        "q_scale": float(q_scale),
        "q_model": [float(q[0]), float(q[1])],
        "q_norm": float(np.linalg.norm(q)),
        "direction_j": direction_j,
        "C": _complex_parts(c_value),
        "K_midpoint_contact": _complex_parts(k_value),
        "K_deltaV_midpoint": _complex_parts(k_delta_v),
        "C_minus_K": _complex_parts(c_minus_k),
        "K_contact_minus_K_deltaV": _complex_parts(k_contact_minus_delta_v),
        "mesh_shift_error": _complex_parts(mesh_shift_error),
        "C_minus_K_rel": _rel_error(abs(c_minus_k), c_value, k_value),
        "K_contact_minus_K_deltaV_rel": _rel_error(abs(k_contact_minus_delta_v), k_value, k_delta_v),
        "mesh_shift_error_rel": _rel_error(abs(mesh_shift_error), c_value, k_delta_v),
    }


def config_for_temperature(temperature_K: float = TEMPERATURE_K) -> KuboConfig:
    omega_eV = bosonic_matsubara_energy_eV(MATSUBARA_INDEX, float(temperature_K))
    return KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=float(temperature_K),
        eta_eV=ETA_EV,
        output_si=OUTPUT_SI,
    )


def mesh_convergence_rows(mesh_sizes: tuple[int, ...] | list[int], q_scales: tuple[float, ...] | list[float]) -> tuple[list[dict[str, Any]], str]:
    config = config_for_temperature(TEMPERATURE_K)
    rows = []
    for mesh_size in mesh_sizes:
        mesh = uniform_bz_mesh(int(mesh_size))
        weights = k_weights(mesh)
        for q_scale in q_scales:
            q = float(q_scale) * Q_BASE
            for direction_j in DIRECTIONS:
                rows.append(CK_row(int(mesh_size), float(q_scale), direction_j, mesh, weights, config, q, label="q_base"))

    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["q_scale"]), str(row["direction_j"]))].append(row)
    results = []
    statuses = []
    for (q_scale, direction_j), group_rows in sorted(grouped.items()):
        sorted_rows = sorted(group_rows, key=lambda item: int(item["mesh_size"]))
        meshes = np.array([int(row["mesh_size"]) for row in sorted_rows], dtype=float)
        abs_errors = np.array([float(row["C_minus_K"]["abs"]) for row in sorted_rows], dtype=float)
        rel_errors = np.array([float(row["C_minus_K_rel"]) for row in sorted_rows], dtype=float)
        slope = float(np.polyfit(np.log(meshes), np.log(np.maximum(abs_errors, EPS)), 1)[0]) if len(meshes) > 1 else 0.0
        finest_rel = float(rel_errors[-1])
        if finest_rel < 1e-8:
            status = "NUMERICALLY_CONVERGED"
        elif slope < -0.75:
            status = "CONVERGING_WITH_MESH"
        else:
            status = "NOT_CONVERGING_OR_INCONCLUSIVE"
        statuses.append(status)
        results.append(
            {
                "q_scale": float(q_scale),
                "direction_j": direction_j,
                "mesh_sizes": [int(item) for item in meshes],
                "finest_mesh_size": int(meshes[-1]),
                "finest_C_minus_K_abs": float(abs_errors[-1]),
                "finest_C_minus_K_rel": finest_rel,
                "slope_log_error_vs_log_mesh": slope,
                "status": status,
                "rows": sorted_rows,
            }
        )
    if results and all(status == "NUMERICALLY_CONVERGED" for status in statuses):
        global_status = "NUMERICALLY_CONVERGED"
    elif results and all(status in {"NUMERICALLY_CONVERGED", "CONVERGING_WITH_MESH"} for status in statuses):
        global_status = "CONVERGING_WITH_MESH"
    else:
        global_status = "NOT_CONVERGING_OR_INCONCLUSIVE"
    return results, global_status


def commensurate_q_results(mesh_sizes: tuple[int, ...] | list[int]) -> tuple[list[dict[str, Any]], str]:
    config = config_for_temperature(TEMPERATURE_K)
    rows = []
    for mesh_size in mesh_sizes:
        mesh = uniform_bz_mesh(int(mesh_size))
        weights = k_weights(mesh)
        delta_k = 2.0 * np.pi / float(mesh_size)
        q_cases = {
            "q_comm_x": np.array([2.0 * delta_k, 0.0], dtype=float),
            "q_comm_y": np.array([0.0, 2.0 * delta_k], dtype=float),
        }
        for label, q in q_cases.items():
            for direction_j in DIRECTIONS:
                rows.append(CK_row(int(mesh_size), 1.0, direction_j, mesh, weights, config, q, label=label))
    close = bool(rows) and all(float(row["C_minus_K_rel"]) < 1e-8 for row in rows)
    status = "COMMENSURATE_SHIFT_CLOSE" if close else "COMMENSURATE_SHIFT_NOT_CLOSE"
    return rows, status


def temperature_sweep_results(temperature_sweep_K: tuple[float, ...] | list[float]) -> tuple[list[dict[str, Any]], str]:
    rows = []
    mesh = uniform_bz_mesh(TEMPERATURE_SWEEP_MESH_SIZE)
    weights = k_weights(mesh)
    for temperature_K in temperature_sweep_K:
        config = config_for_temperature(float(temperature_K))
        for direction_j in DIRECTIONS:
            rows.append(
                CK_row(
                    TEMPERATURE_SWEEP_MESH_SIZE,
                    1.0,
                    direction_j,
                    mesh,
                    weights,
                    config,
                    Q_BASE,
                    label="temperature_sweep_q_base",
                )
            )
    by_temp: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        by_temp[float(row["temperature_K"])].append(float(row["C_minus_K_rel"]))
    low = max(by_temp[min(by_temp)]) if by_temp else np.inf
    high = max(by_temp[max(by_temp)]) if by_temp else np.inf
    status = "TEMPERATURE_IMPROVES_CK_CONVERGENCE" if high < 0.1 * low else "TEMPERATURE_DOES_NOT_RESOLVE_CK"
    return rows, status


def baseline_rows(q_scales: tuple[float, ...] | list[float]) -> list[dict[str, Any]]:
    config = config_for_temperature(TEMPERATURE_K)
    mesh = uniform_bz_mesh(BASELINE_MESH_SIZE)
    weights = k_weights(mesh)
    rows = []
    for q_scale in q_scales:
        q = float(q_scale) * Q_BASE
        for direction_j in DIRECTIONS:
            rows.append(CK_row(BASELINE_MESH_SIZE, float(q_scale), direction_j, mesh, weights, config, q, label="baseline_q_base"))
    return rows


def likely_issue_and_next_step(
    h_status: str,
    second_order_status: str,
    q_base_status: str,
    commensurate_status: str,
    temperature_status: str,
) -> tuple[str, str]:
    if h_status == "H_REPRESENTATION_MISMATCH":
        return (
            "HAMILTONIAN_HOPPING_REPRESENTATION_MISMATCH",
            "Fix H0(k) and hopping reconstruction consistency before further Ward audits.",
        )
    if second_order_status == "SECOND_ORDER_IDENTITY_MISMATCH":
        return (
            "PEIERLS_SECOND_ORDER_IDENTITY_MISMATCH",
            "Fix V/M Peierls vertex consistency before changing response-level terms.",
        )
    if commensurate_status == "COMMENSURATE_SHIFT_CLOSE" and q_base_status != "NUMERICALLY_CONVERGED":
        return (
            "NONCOMMENSURATE_MESH_SHIFT_QUADRATURE",
            "Use shift-compatible quadrature, denser adaptive integration, or commensurate finite-q meshes before interpreting C-K as a physics-level residual.",
        )
    if temperature_status == "TEMPERATURE_IMPROVES_CK_CONVERGENCE":
        return (
            "LOW_TEMPERATURE_FERMI_SURFACE_QUADRATURE",
            "Improve Fermi-surface quadrature or use higher mesh before introducing new Ward terms.",
        )
    return (
        "CONTACT_EXPECTATION_OR_DENSITY_Q_CONVENTION",
        "Next: analytically audit finite-q density vertex embedding and contact expectation routing. Do not revert the Stage 4.13 bubble sign fix.",
    )


def run_audit(
    mesh_sizes: tuple[int, ...] | list[int] = MESH_SIZES,
    q_scales: tuple[float, ...] | list[float] = Q_SCALES,
    temperature_sweep_K: tuple[float, ...] | list[float] = TEMPERATURE_SWEEP_K,
) -> dict[str, Any]:
    h_check = hamiltonian_representation_check(BASELINE_MESH_SIZE)
    second_check = second_order_identity_check(BASELINE_MESH_SIZE, Q_BASE)
    baseline = baseline_rows(q_scales)
    mesh_results, q_base_status = mesh_convergence_rows(mesh_sizes, q_scales)
    comm_rows, comm_status = commensurate_q_results(mesh_sizes)
    temp_rows, temp_status = temperature_sweep_results(temperature_sweep_K)
    likely_issue, next_step = likely_issue_and_next_step(
        str(h_check["H_representation_status"]),
        str(second_check["second_order_identity_status"]),
        q_base_status,
        comm_status,
        temp_status,
    )
    omega_eV = bosonic_matsubara_energy_eV(MATSUBARA_INDEX, TEMPERATURE_K)
    return {
        "stage": "Stage 4.14",
        "purpose": "C_j versus K_j routing and contact expectation audit",
        "config": {
            "temperature_K": TEMPERATURE_K,
            "matsubara_index": MATSUBARA_INDEX,
            "omega_eV": float(omega_eV),
            "eta_eV": ETA_EV,
            "q_base": [float(Q_BASE[0]), float(Q_BASE[1])],
            "q_scales": [float(item) for item in q_scales],
            "mesh_sizes": [int(item) for item in mesh_sizes],
            "temperature_sweep_K": [float(item) for item in temperature_sweep_K],
            "mesh_64_retained": bool(max(mesh_sizes) >= 64),
        },
        "hamiltonian_representation_check": h_check,
        "second_order_identity_check": second_check,
        "baseline_CK_results": baseline,
        "mesh_convergence_results": mesh_results,
        "commensurate_q_results": comm_rows,
        "temperature_sweep_results": temp_rows,
        "diagnostic_status": {
            "H_representation_status": h_check["H_representation_status"],
            "second_order_identity_status": second_check["second_order_identity_status"],
            "q_base_CK_convergence_status": q_base_status,
            "commensurate_q_status": comm_status,
            "temperature_sweep_status": temp_status,
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


def render_markdown(data: dict[str, Any]) -> str:
    h_check = data["hamiltonian_representation_check"]
    second = data["second_order_identity_check"]
    baseline_table = _table(
        ("q_scale", "direction", "|C-K| rel", "|K-K_deltaV| rel", "mesh_shift rel"),
        [
            (
                _fmt(float(row["q_scale"])),
                row["direction_j"],
                _fmt(float(row["C_minus_K_rel"])),
                _fmt(float(row["K_contact_minus_K_deltaV_rel"])),
                _fmt(float(row["mesh_shift_error_rel"])),
            )
            for row in data["baseline_CK_results"]
        ],
    )
    mesh_table = _table(
        ("q_scale", "direction", "finest mesh", "finest rel", "slope", "status"),
        [
            (
                _fmt(float(row["q_scale"])),
                row["direction_j"],
                row["finest_mesh_size"],
                _fmt(float(row["finest_C_minus_K_rel"])),
                _fmt(float(row["slope_log_error_vs_log_mesh"])),
                row["status"],
            )
            for row in data["mesh_convergence_results"]
        ],
    )
    comm_table = _table(
        ("mesh", "q_label", "direction", "|C-K| rel"),
        [
            (
                row["mesh_size"],
                row["label"],
                row["direction_j"],
                _fmt(float(row["C_minus_K_rel"])),
            )
            for row in data["commensurate_q_results"]
        ],
    )
    temp_table = _table(
        ("temperature_K", "direction", "|C-K| rel"),
        [
            (
                _fmt(float(row["temperature_K"])),
                row["direction_j"],
                _fmt(float(row["C_minus_K_rel"])),
            )
            for row in data["temperature_sweep_results"]
        ],
    )
    status = data["diagnostic_status"]
    return "\n\n".join(
        [
            "# Stage 4.14 C_j versus K_j routing/contact audit",
            "## Boundary\n\n"
            "- no main response change\n"
            "- no bubble sign change\n"
            "- no direct contact change\n"
            "- no source/observable change\n"
            "- no residual tuning\n"
            "- no fitted contact\n"
            "- no E_ET added\n"
            "- no conductivity / reflection / Casimir\n"
            "- no Ward closure claim",
            "## Analytic identity being tested\n\n"
            "$$C_j=\\sum_k\\operatorname{Tr}[(f(H_-)-f(H_+))V_j(k,q)].$$\n\n"
            "$$K_j=\\sum_k\\operatorname{Tr}[f(H(k))q_iM_{ij}(k,q)].$$\n\n"
            "In the continuous BZ integral, the expected identity is $C_j=K_j$.",
            "## Hamiltonian representation consistency\n\n"
            + _table(
                ("quantity", "value"),
                [
                    ("status", h_check["H_representation_status"]),
                    ("max_abs_H_model_minus_H_hopping", _fmt(float(h_check["max_abs_H_model_minus_H_hopping"]))),
                    ("max_rel_H_model_minus_H_hopping", _fmt(float(h_check["max_rel_H_model_minus_H_hopping"]))),
                ],
            ),
            "## Second-order Peierls identity\n\n"
            + _table(
                ("quantity", "value"),
                [
                    ("status", second["second_order_identity_status"]),
                    ("max_abs_second_order_identity_error", _fmt(float(second["max_abs_second_order_identity_error"]))),
                    ("max_rel_second_order_identity_error", _fmt(float(second["max_rel_second_order_identity_error"]))),
                ],
            ),
            "## Baseline C-K results\n\n" + baseline_table,
            "## Mesh convergence\n\n" + mesh_table,
            "## Commensurate q shift test\n\n" + comm_table,
            "## Temperature sweep\n\n" + temp_table,
            "## Diagnostic decision\n\n"
            + _table(
                ("quantity", "status"),
                [
                    ("H_representation_status", status["H_representation_status"]),
                    ("second_order_identity_status", status["second_order_identity_status"]),
                    ("q_base_CK_convergence_status", status["q_base_CK_convergence_status"]),
                    ("commensurate_q_status", status["commensurate_q_status"]),
                    ("temperature_sweep_status", status["temperature_sweep_status"]),
                    ("likely_issue", status["likely_issue"]),
                ],
            )
            + "\n\nThe remaining C-K mismatch should not be addressed by reverting the Stage 4.13 bubble sign fix or changing direct-contact signs.",
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
