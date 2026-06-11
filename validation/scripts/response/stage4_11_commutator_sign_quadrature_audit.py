#!/usr/bin/env python3
"""Stage 4.11 commutator sign and quadrature convergence audit.

This script is diagnostic-only.  It does not change the bubble formula, the
main response path, contact signs, conductivity, reflection, or Casimir code.
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
from lno327.model import normal_state_hamiltonian  # noqa: E402
from lno327.tb_fourier import peierls_hamiltonian_contact_vertex, peierls_hamiltonian_vector_vertex  # noqa: E402
from lno327.ward_response import (  # noqa: E402
    normal_physical_density_current_response_components_imag_axis,
    physical_ward_residuals,
)

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "ward_identity"
JSON_OUTPUT = OUTPUT_DIR / "stage4_11_commutator_sign_quadrature_audit.json"
MD_OUTPUT = OUTPUT_DIR / "stage4_11_commutator_sign_quadrature_audit.md"

TEMPERATURE_K = 30.0
MATSUBARA_INDEX = 1
ETA_EV = 1e-10
Q_BASE = np.array([0.02, 0.013], dtype=float)
Q_SCALES = (1.0, 0.5, 0.25, 0.125)
MESH_SIZES = (8, 12, 16, 24, 32)
EPS = 1e-300

BUBBLE_CANDIDATE_KEYS = (
    ("PLUS_C_PLUS", "err_bubble_plus_C_plus_rel"),
    ("MINUS_C_PLUS", "err_bubble_minus_C_plus_rel"),
    ("PLUS_C_MINUS", "err_bubble_plus_C_minus_rel"),
    ("MINUS_C_MINUS", "err_bubble_minus_C_minus_rel"),
)


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


def _commutator_candidate(
    mesh: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    q: np.ndarray,
    direction_j: str,
    *,
    vertex_q_sign: float,
) -> complex:
    qx, qy = float(q[0]), float(q[1])
    total = 0.0j
    for weight, (kx_value, ky_value) in zip(weights, mesh, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        h_minus = normal_state_hamiltonian(kx - 0.5 * qx, ky - 0.5 * qy)
        h_plus = normal_state_hamiltonian(kx + 0.5 * qx, ky + 0.5 * qy)
        vertex = peierls_hamiltonian_vector_vertex(
            kx,
            ky,
            vertex_q_sign * qx,
            vertex_q_sign * qy,
            direction_j,
        )
        expect_minus = thermal_trace_expectation(h_minus, vertex, config)
        expect_plus = thermal_trace_expectation(h_plus, vertex, config)
        total += float(weight) * (expect_minus - expect_plus)
    return complex(total)


def commutator_candidate_c_plus_q(
    mesh: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    q: np.ndarray,
    direction_j: str,
) -> complex:
    """Return C_j^(+q)."""

    return _commutator_candidate(mesh, weights, config, q, direction_j, vertex_q_sign=1.0)


def commutator_candidate_c_minus_q(
    mesh: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    q: np.ndarray,
    direction_j: str,
) -> complex:
    """Return C_j^(-q), keeping H_pm fixed."""

    return _commutator_candidate(mesh, weights, config, q, direction_j, vertex_q_sign=-1.0)


def direct_contact_contraction_k(
    mesh: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    q: np.ndarray,
    direction_j: str,
) -> complex:
    """Return K_j(q)=q_i <M_ij>."""

    qx, qy = float(q[0]), float(q[1])
    total = 0.0j
    for weight, (kx_value, ky_value) in zip(weights, mesh, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        h0 = normal_state_hamiltonian(kx, ky)
        m_xj = peierls_hamiltonian_contact_vertex(kx, ky, qx, qy, "x", direction_j)
        m_yj = peierls_hamiltonian_contact_vertex(kx, ky, qx, qy, "y", direction_j)
        operator = qx * m_xj + qy * m_yj
        total += float(weight) * thermal_trace_expectation(h0, operator, config)
    return complex(total)


def _rel_error(abs_error: float, *refs: complex | float) -> float:
    return float(abs_error / max(*(abs(ref) for ref in refs), EPS))


def classify_bubble_sign(row: dict[str, Any]) -> tuple[str, float, str]:
    """Return best bubble sign candidate, relative error, and status."""

    candidate, key = min(BUBBLE_CANDIDATE_KEYS, key=lambda item: float(row[item[1]]))
    rel_error = float(row[key])
    return candidate, rel_error, ("MATCH" if rel_error < 1e-8 else "MISMATCH")


def classify_direct_sign(err_direct_minus_k_rel: float, err_direct_plus_k_rel: float) -> str:
    if err_direct_minus_k_rel < 1e-10:
        return "MATCH_R_DIRECT_EQUALS_MINUS_K"
    if err_direct_plus_k_rel < 1e-10:
        return "MATCH_R_DIRECT_EQUALS_PLUS_K"
    return "MISMATCH"


def classify_convergence(finest_rel_error: float, mesh_slope: float) -> str:
    if finest_rel_error < 1e-8:
        return "NUMERICALLY_CONVERGED"
    if mesh_slope < -0.75:
        return "CONVERGING_WITH_MESH"
    return "NOT_CONVERGING_OR_INCONCLUSIVE"


def classify_bubble_sign_global(rows: list[dict[str, Any]], finest_mesh_size: int) -> str:
    candidates = {str(row["best_bubble_sign_candidate"]) for row in rows}
    finest_rows = [row for row in rows if int(row["mesh_size"]) == finest_mesh_size]
    if len(candidates) == 1 and finest_rows and all(float(row["best_bubble_sign_rel_error"]) < 1e-8 for row in finest_rows):
        return f"CONSISTENT_MATCH_{next(iter(candidates))}"
    return "UNRESOLVED_OR_INCONSISTENT"


def classify_direct_sign_global(rows: list[dict[str, Any]]) -> str:
    if rows and all(str(row["direct_sign_status"]) == "MATCH_R_DIRECT_EQUALS_MINUS_K" for row in rows):
        return "MATCH_R_DIRECT_EQUALS_MINUS_K"
    return "MISMATCH_OR_INCONSISTENT"


def _complex_parts(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag), "abs": float(abs(value))}


def sign_audit_row(
    mesh_size: int,
    q_scale: float,
    direction_j: str,
    mesh: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    q: np.ndarray,
) -> dict[str, Any]:
    components = normal_physical_density_current_response_components_imag_axis(mesh, config, q, weights)
    left_bubble, _right_bubble = physical_ward_residuals(components["bubble"], config.omega_eV, q)
    left_direct, _right_direct = physical_ward_residuals(components["direct"], config.omega_eV, q)
    left_total, _right_total = physical_ward_residuals(components["total"], config.omega_eV, q)
    j_index = 0 if direction_j == "x" else 1
    r_bubble = complex(left_bubble[1 + j_index])
    r_direct = complex(left_direct[1 + j_index])
    r_total = complex(left_total[1 + j_index])
    c_plus = commutator_candidate_c_plus_q(mesh, weights, config, q, direction_j)
    c_minus = commutator_candidate_c_minus_q(mesh, weights, config, q, direction_j)
    k_value = direct_contact_contraction_k(mesh, weights, config, q, direction_j)

    err_bubble_plus_c_plus = abs(r_bubble - c_plus)
    err_bubble_minus_c_plus = abs(r_bubble + c_plus)
    err_bubble_plus_c_minus = abs(r_bubble - c_minus)
    err_bubble_minus_c_minus = abs(r_bubble + c_minus)
    err_direct_minus_k = abs(r_direct + k_value)
    err_direct_plus_k = abs(r_direct - k_value)
    err_cplus_minus_k = abs(c_plus - k_value)
    err_cminus_minus_k = abs(c_minus - k_value)
    cplus_minus_k = c_plus - k_value
    err_total_if_cplus_minus_k = abs(r_total - cplus_minus_k)

    row: dict[str, Any] = {
        "mesh_size": int(mesh_size),
        "q_scale": float(q_scale),
        "q_model": [float(q[0]), float(q[1])],
        "q_norm": float(np.linalg.norm(q)),
        "direction_j": direction_j,
        "R_bubble": _complex_parts(r_bubble),
        "R_direct": _complex_parts(r_direct),
        "R_total": _complex_parts(r_total),
        "C_plus": _complex_parts(c_plus),
        "C_minus": _complex_parts(c_minus),
        "K": _complex_parts(k_value),
        "err_bubble_plus_C_plus_abs": float(err_bubble_plus_c_plus),
        "err_bubble_plus_C_plus_rel": _rel_error(err_bubble_plus_c_plus, r_bubble, c_plus),
        "err_bubble_minus_C_plus_abs": float(err_bubble_minus_c_plus),
        "err_bubble_minus_C_plus_rel": _rel_error(err_bubble_minus_c_plus, r_bubble, c_plus),
        "err_bubble_plus_C_minus_abs": float(err_bubble_plus_c_minus),
        "err_bubble_plus_C_minus_rel": _rel_error(err_bubble_plus_c_minus, r_bubble, c_minus),
        "err_bubble_minus_C_minus_abs": float(err_bubble_minus_c_minus),
        "err_bubble_minus_C_minus_rel": _rel_error(err_bubble_minus_c_minus, r_bubble, c_minus),
        "err_direct_minus_K_abs": float(err_direct_minus_k),
        "err_direct_minus_K_rel": _rel_error(err_direct_minus_k, r_direct, k_value),
        "err_direct_plus_K_abs": float(err_direct_plus_k),
        "err_direct_plus_K_rel": _rel_error(err_direct_plus_k, r_direct, k_value),
        "err_Cplus_minus_K_abs": float(err_cplus_minus_k),
        "err_Cplus_minus_K_rel": _rel_error(err_cplus_minus_k, c_plus, k_value),
        "err_Cminus_minus_K_abs": float(err_cminus_minus_k),
        "err_Cminus_minus_K_rel": _rel_error(err_cminus_minus_k, c_minus, k_value),
        "err_total_if_Cplus_minus_K_abs": float(err_total_if_cplus_minus_k),
        "err_total_if_Cplus_minus_K_rel": _rel_error(err_total_if_cplus_minus_k, r_total, cplus_minus_k),
    }
    candidate, rel_error, bubble_status = classify_bubble_sign(row)
    row["best_bubble_sign_candidate"] = candidate
    row["best_bubble_sign_rel_error"] = rel_error
    row["bubble_sign_status"] = bubble_status
    row["direct_sign_status"] = classify_direct_sign(
        float(row["err_direct_minus_K_rel"]),
        float(row["err_direct_plus_K_rel"]),
    )
    return row


def _mesh_slope(mesh_sizes: list[int], values: list[float]) -> float:
    if len(mesh_sizes) < 2:
        return 0.0
    return float(np.polyfit(np.log(np.array(mesh_sizes, dtype=float)), np.log(np.maximum(values, EPS)), 1)[0])


def mesh_convergence_rows(sign_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str, str]:
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sign_rows:
        grouped[(float(row["q_scale"]), str(row["direction_j"]))].append(row)
    results = []
    for (q_scale, direction_j), rows in sorted(grouped.items()):
        rows_sorted = sorted(rows, key=lambda item: int(item["mesh_size"]))
        meshes = [int(row["mesh_size"]) for row in rows_sorted]
        cplus_rel = [float(row["err_Cplus_minus_K_rel"]) for row in rows_sorted]
        cminus_rel = [float(row["err_Cminus_minus_K_rel"]) for row in rows_sorted]
        cplus_abs = [float(row["err_Cplus_minus_K_abs"]) for row in rows_sorted]
        cminus_abs = [float(row["err_Cminus_minus_K_abs"]) for row in rows_sorted]
        r_total_abs = [float(row["R_total"]["abs"]) for row in rows_sorted]
        cplus_slope = _mesh_slope(meshes, cplus_abs)
        cminus_slope = _mesh_slope(meshes, cminus_abs)
        rtotal_slope = _mesh_slope(meshes, r_total_abs)
        results.append(
            {
                "q_scale": q_scale,
                "direction_j": direction_j,
                "mesh_sizes": meshes,
                "finest_mesh_size": meshes[-1],
                "finest_err_Cplus_minus_K_rel": cplus_rel[-1],
                "finest_err_Cminus_minus_K_rel": cminus_rel[-1],
                "finest_abs_Cplus_minus_K": cplus_abs[-1],
                "finest_abs_Cminus_minus_K": cminus_abs[-1],
                "finest_abs_R_total": r_total_abs[-1],
                "mesh_slope_Cplus_minus_K": cplus_slope,
                "mesh_slope_Cminus_minus_K": cminus_slope,
                "mesh_slope_R_total": rtotal_slope,
                "Cplus_K_convergence_status": classify_convergence(cplus_rel[-1], cplus_slope),
                "Cminus_K_convergence_status": classify_convergence(cminus_rel[-1], cminus_slope),
            }
        )
    cplus_global = (
        "CONVERGING_OR_CONVERGED"
        if results
        and all(
            row["Cplus_K_convergence_status"] in {"NUMERICALLY_CONVERGED", "CONVERGING_WITH_MESH"}
            for row in results
        )
        else "NOT_CONVERGING_OR_INCONCLUSIVE"
    )
    cminus_global = (
        "CONVERGING_OR_CONVERGED"
        if results
        and all(
            row["Cminus_K_convergence_status"] in {"NUMERICALLY_CONVERGED", "CONVERGING_WITH_MESH"}
            for row in results
        )
        else "NOT_CONVERGING_OR_INCONCLUSIVE"
    )
    return results, cplus_global, cminus_global


def likely_issue_and_next_step(
    direct_sign_global_status: str,
    bubble_sign_global_status: str,
    quadrature_global_status: str,
) -> tuple[str, str]:
    if direct_sign_global_status != "MATCH_R_DIRECT_EQUALS_MINUS_K":
        likely_issue = "DIRECT_CONTACT_SIGN_OR_IMPLEMENTATION"
        next_step = "Next: audit physical direct contact implementation against R_direct = -q_i<M_ij>."
    elif bubble_sign_global_status.startswith("CONSISTENT_MATCH_MINUS"):
        likely_issue = "BUBBLE_WARD_CONTRACTION_SIGN_CONVENTION"
        next_step = "Next: audit Matsubara Fourier convention and left Ward contraction sign before modifying any vertex or contact term."
    elif bubble_sign_global_status == "UNRESOLVED_OR_INCONSISTENT":
        likely_issue = "BUBBLE_OR_DENSITY_Q_ROUTING"
        next_step = "Next: audit density operator q convention and source reverse matrix element routing."
    elif bubble_sign_global_status.startswith("CONSISTENT_MATCH_PLUS"):
        likely_issue = "C_MINUS_K_ROUTING_OR_CONTACT_EXPECTATION"
        next_step = "Next: audit C_j versus K_j routing, density q-convention, and contact thermal expectation. Do not revert the Stage 4.13 bubble prefactor sign fix."
    elif quadrature_global_status == "CONVERGING_OR_CONVERGED":
        likely_issue = "FINITE_MESH_QUADRATURE"
        next_step = "Next: rerun Ward residual regression with denser BZ meshes and q values commensurate with mesh spacing before adding any E_ET term."
    else:
        likely_issue = "DENSITY_SOURCE_CONVENTION_OR_MATRIX_ELEMENT_ROUTING"
        next_step = "Next: audit scalar source convention, density normalization, and finite-q matrix-element routing."
    return likely_issue, next_step


def run_audit(
    mesh_sizes: tuple[int, ...] | list[int] = MESH_SIZES,
    q_scales: tuple[float, ...] | list[float] = Q_SCALES,
) -> dict[str, Any]:
    omega_eV = bosonic_matsubara_energy_eV(MATSUBARA_INDEX, TEMPERATURE_K)
    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=TEMPERATURE_K,
        eta_eV=ETA_EV,
        output_si=False,
    )
    sign_rows = []
    for mesh_size in mesh_sizes:
        mesh = uniform_bz_mesh(int(mesh_size))
        weights = k_weights(mesh)
        for q_scale in q_scales:
            q = float(q_scale) * Q_BASE
            for direction_j in ("x", "y"):
                sign_rows.append(sign_audit_row(int(mesh_size), float(q_scale), direction_j, mesh, weights, config, q))
    convergence_results, cplus_global, cminus_global = mesh_convergence_rows(sign_rows)
    finest_mesh_size = max(int(item) for item in mesh_sizes)
    bubble_global = classify_bubble_sign_global(sign_rows, finest_mesh_size)
    direct_global = classify_direct_sign_global(sign_rows)
    quadrature_global = cplus_global
    likely_issue, next_step = likely_issue_and_next_step(direct_global, bubble_global, quadrature_global)
    return {
        "stage": "Stage 4.11",
        "purpose": "Commutator sign and quadrature convergence audit",
        "config": {
            "temperature_K": TEMPERATURE_K,
            "matsubara_index": MATSUBARA_INDEX,
            "omega_eV": float(omega_eV),
            "eta_eV": ETA_EV,
            "q_base": [float(Q_BASE[0]), float(Q_BASE[1])],
            "q_scales": [float(item) for item in q_scales],
            "mesh_sizes": [int(item) for item in mesh_sizes],
        },
        "sign_audit": {"results": sign_rows},
        "mesh_convergence": {
            "results": convergence_results,
            "global_Cplus_K_convergence_status": cplus_global,
            "global_Cminus_K_convergence_status": cminus_global,
        },
        "diagnostic_status": {
            "bubble_sign_global_status": bubble_global,
            "direct_sign_global_status": direct_global,
            "quadrature_global_status": quadrature_global,
            "likely_issue": likely_issue,
            "next_step": next_step,
        },
        "boundary": {
            "no_residual_tuning": True,
            "no_bubble_formula_change": True,
            "no_main_response_change": True,
            "no_contact_sign_scan_for_formula_choice": True,
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
    sign_rows = data["sign_audit"]["results"]
    finest = max(int(item) for item in data["config"]["mesh_sizes"])
    finest_rows = [row for row in sign_rows if int(row["mesh_size"]) == finest]
    bubble_table = _table(
        (
            "q_scale",
            "direction",
            "best_candidate",
            "best_rel",
            "+C+ rel",
            "-C+ rel",
            "+C- rel",
            "-C- rel",
        ),
        [
            (
                _fmt(float(row["q_scale"])),
                row["direction_j"],
                row["best_bubble_sign_candidate"],
                _fmt(float(row["best_bubble_sign_rel_error"])),
                _fmt(float(row["err_bubble_plus_C_plus_rel"])),
                _fmt(float(row["err_bubble_minus_C_plus_rel"])),
                _fmt(float(row["err_bubble_plus_C_minus_rel"])),
                _fmt(float(row["err_bubble_minus_C_minus_rel"])),
            )
            for row in finest_rows
        ],
    )
    direct_table = _table(
        ("q_scale", "direction", "direct_status", "R_direct=-K rel", "R_direct=+K rel"),
        [
            (
                _fmt(float(row["q_scale"])),
                row["direction_j"],
                row["direct_sign_status"],
                _fmt(float(row["err_direct_minus_K_rel"])),
                _fmt(float(row["err_direct_plus_K_rel"])),
            )
            for row in finest_rows
        ],
    )
    ck_table = _table(
        ("q_scale", "direction", "C+ vs K rel", "C- vs K rel", "R_total abs"),
        [
            (
                _fmt(float(row["q_scale"])),
                row["direction_j"],
                _fmt(float(row["err_Cplus_minus_K_rel"])),
                _fmt(float(row["err_Cminus_minus_K_rel"])),
                _fmt(float(row["R_total"]["abs"])),
            )
            for row in finest_rows
        ],
    )
    mesh_table = _table(
        (
            "q_scale",
            "direction",
            "C+ status",
            "C+ slope",
            "C- status",
            "C- slope",
            "R_total slope",
        ),
        [
            (
                _fmt(float(row["q_scale"])),
                row["direction_j"],
                row["Cplus_K_convergence_status"],
                _fmt(float(row["mesh_slope_Cplus_minus_K"])),
                row["Cminus_K_convergence_status"],
                _fmt(float(row["mesh_slope_Cminus_minus_K"])),
                _fmt(float(row["mesh_slope_R_total"])),
            )
            for row in data["mesh_convergence"]["results"]
        ],
    )
    status = data["diagnostic_status"]
    return "\n\n".join(
        [
            "# Stage 4.11 Commutator sign and quadrature convergence audit",
            "## Boundary\n\n- no residual tuning\n- no bubble formula change\n- no main response path change\n- no fitted contact\n- no conductivity / reflection / Casimir\n- no Ward closure claim",
            "## Stage 4.13 note\n\nAfter Stage 4.13 the main response path uses the corrected positive fermion-loop bubble prefactor. Therefore a main bubble match to $+C_j$ is the expected post-fix sign bookkeeping, not a residual-tuning choice.",
            "## Fixed formulas\n\n"
            "$C_j^{(+q)}=\\sum_k\\operatorname{Tr}[(f(H_-)-f(H_+))V_j(k,q)]$.\n\n"
            "$C_j^{(-q)}=\\sum_k\\operatorname{Tr}[(f(H_-)-f(H_+))V_j(k,-q)]$.\n\n"
            "$K_j=q_i\\langle M_{ij}\\rangle$.\n\n"
            "$R^{direct}_{L,j}$ should satisfy $R^{direct}_{L,j}=-K_j$.",
            "## Bubble sign audit table\n\n" + bubble_table,
            "## Direct contact sign audit table\n\n" + direct_table,
            "## C_j vs K_j comparison\n\n" + ck_table,
            "## Mesh convergence table\n\n" + mesh_table,
            "## Diagnostic decision\n\n"
            + _table(
                ("quantity", "status"),
                [
                    ("bubble_sign_global_status", status["bubble_sign_global_status"]),
                    ("direct_sign_global_status", status["direct_sign_global_status"]),
                    ("quadrature_global_status", status["quadrature_global_status"]),
                    ("global_Cplus_K_convergence_status", data["mesh_convergence"]["global_Cplus_K_convergence_status"]),
                    ("global_Cminus_K_convergence_status", data["mesh_convergence"]["global_Cminus_K_convergence_status"]),
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
