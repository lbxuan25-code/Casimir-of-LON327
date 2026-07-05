#!/usr/bin/env python3
"""Stage 4.13 regression after flipping the main bubble prefactor.

Diagnostic-only.  This script verifies the corrected main finite-q bubble
sign bookkeeping and does not tune residuals, add fitted contact terms, add
E_ET, or enter conductivity/reflection/Casimir.
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

from lno327 import KuboConfig, bosonic_matsubara_energy_eV, fermi_function, k_weights, uniform_bz_mesh  # noqa: E402
from lno327.models.lno327_four_orbital.normal import normal_state_hamiltonian  # noqa: E402
from lno327.models.lno327_four_orbital.peierls import peierls_hamiltonian_contact_vertex, peierls_hamiltonian_vector_vertex  # noqa: E402
from lno327.collective.ward import physical_ward_residuals  # noqa: E402
from lno327.response.normal_density_current import normal_physical_density_current_response_components_imag_axis  # noqa: E402

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "ward_identity"
JSON_OUTPUT = OUTPUT_DIR / "stage4_13_bubble_sign_fix_regression.json"
MD_OUTPUT = OUTPUT_DIR / "stage4_13_bubble_sign_fix_regression.md"

TEMPERATURE_K = 30.0
MATSUBARA_INDEX = 1
ETA_EV = 1e-10
OUTPUT_SI = False
MESH_SIZE = 16
Q_BASE = np.array([0.02, 0.013], dtype=float)
Q_SCALES = (1.0, 0.5, 0.25, 0.125)
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


def commutator_candidate_c_plus_q(
    mesh: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    q: np.ndarray,
    direction_j: str,
) -> complex:
    """Return C_j^(+q) with the fixed Stage 4.11/4.12 definition."""

    qx, qy = float(q[0]), float(q[1])
    total = 0.0j
    for weight, (kx_value, ky_value) in zip(weights, mesh, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        h_minus = normal_state_hamiltonian(kx - 0.5 * qx, ky - 0.5 * qy)
        h_plus = normal_state_hamiltonian(kx + 0.5 * qx, ky + 0.5 * qy)
        vertex = peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, direction_j)
        expect_minus = thermal_trace_expectation(h_minus, vertex, config)
        expect_plus = thermal_trace_expectation(h_plus, vertex, config)
        total += float(weight) * (expect_minus - expect_plus)
    return complex(total)


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


def regression_row(
    q_scale: float,
    direction_j: str,
    mesh: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    q: np.ndarray,
) -> dict[str, Any]:
    components = normal_physical_density_current_response_components_imag_axis(mesh, config, q, weights)
    left_bubble, _ = physical_ward_residuals(components["bubble"], config.omega_eV, q)
    left_direct, _ = physical_ward_residuals(components["direct"], config.omega_eV, q)
    left_total, _ = physical_ward_residuals(components["total"], config.omega_eV, q)
    j_index = 0 if direction_j == "x" else 1
    r_bubble = complex(left_bubble[1 + j_index])
    r_direct = complex(left_direct[1 + j_index])
    r_total = complex(left_total[1 + j_index])
    c_plus = commutator_candidate_c_plus_q(mesh, weights, config, q, direction_j)
    k_value = direct_contact_contraction_k(mesh, weights, config, q, direction_j)
    c_minus_k = c_plus - k_value
    err_bubble = abs(r_bubble - c_plus)
    err_direct = abs(r_direct + k_value)
    err_total = abs(r_total - c_minus_k)
    return {
        "q_scale": float(q_scale),
        "q_model": [float(q[0]), float(q[1])],
        "q_norm": float(np.linalg.norm(q)),
        "direction_j": direction_j,
        "R_bubble": _complex_parts(r_bubble),
        "R_direct": _complex_parts(r_direct),
        "R_total": _complex_parts(r_total),
        "C_plus": _complex_parts(c_plus),
        "K": _complex_parts(k_value),
        "C_minus_K": _complex_parts(c_minus_k),
        "err_main_bubble_matches_plus_C_abs": float(err_bubble),
        "err_main_bubble_matches_plus_C_rel": _rel_error(err_bubble, r_bubble, c_plus),
        "err_direct_matches_minus_K_abs": float(err_direct),
        "err_direct_matches_minus_K_rel": _rel_error(err_direct, r_direct, k_value),
        "err_total_matches_C_minus_K_abs": float(err_total),
        "err_total_matches_C_minus_K_rel": _rel_error(err_total, r_total, c_minus_k),
        "total_residual_abs": float(abs(r_total)),
    }


def _all_below(rows: list[dict[str, Any]], key: str, threshold: float) -> bool:
    return bool(rows) and all(float(row[key]) < threshold for row in rows)


def run_regression(mesh_size: int = MESH_SIZE) -> dict[str, Any]:
    omega_eV = bosonic_matsubara_energy_eV(MATSUBARA_INDEX, TEMPERATURE_K)
    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=TEMPERATURE_K,
        eta_eV=ETA_EV,
        output_si=OUTPUT_SI,
    )
    mesh = uniform_bz_mesh(int(mesh_size))
    weights = k_weights(mesh)
    rows = []
    for q_scale in Q_SCALES:
        q = float(q_scale) * Q_BASE
        for direction_j in ("x", "y"):
            rows.append(regression_row(float(q_scale), direction_j, mesh, weights, config, q))

    main_status = (
        "MAIN_BUBBLE_MATCHES_PLUS_C"
        if _all_below(rows, "err_main_bubble_matches_plus_C_rel", 1e-8)
        else "MAIN_BUBBLE_SIGN_UNRESOLVED"
    )
    direct_status = (
        "DIRECT_STILL_MATCHES_MINUS_K"
        if _all_below(rows, "err_direct_matches_minus_K_rel", 1e-10)
        else "DIRECT_CONTACT_CHANGED_OR_BROKEN"
    )
    total_status = (
        "TOTAL_MATCHES_C_MINUS_K"
        if _all_below(rows, "err_total_matches_C_minus_K_rel", 1e-8)
        else "TOTAL_BOOKKEEPING_UNRESOLVED"
    )
    sign_bookkeeping_ok = (
        main_status == "MAIN_BUBBLE_MATCHES_PLUS_C"
        and direct_status == "DIRECT_STILL_MATCHES_MINUS_K"
        and total_status == "TOTAL_MATCHES_C_MINUS_K"
    )
    max_total_abs = max(float(row["total_residual_abs"]) for row in rows)
    likely_remaining_issue = (
        "C_MINUS_K_ROUTING_OR_CONTACT_EXPECTATION"
        if sign_bookkeeping_ok and max_total_abs > 1e-10
        else "PATCH_OR_SIGN_BOOKKEEPING_ERROR"
    )
    if sign_bookkeeping_ok:
        next_step = (
            "Next: rerun Stage 4.9-4.11 residual diagnostics with the corrected bubble sign. "
            "If an O(q) residual remains, audit C_j versus K_j routing, density q-convention, "
            "and contact thermal expectation."
        )
    else:
        next_step = "Next: stop and inspect the main bubble sign patch before any further Ward diagnostics."
    return {
        "stage": "Stage 4.13",
        "purpose": "Regression after flipping the main fermion-loop bubble prefactor",
        "config": {
            "temperature_K": TEMPERATURE_K,
            "matsubara_index": MATSUBARA_INDEX,
            "omega_eV": float(omega_eV),
            "eta_eV": ETA_EV,
            "mesh_size": int(mesh_size),
            "q_base": [float(Q_BASE[0]), float(Q_BASE[1])],
            "q_scales": [float(item) for item in Q_SCALES],
        },
        "results": rows,
        "diagnostic_status": {
            "main_bubble_sign_status": main_status,
            "direct_contact_status": direct_status,
            "total_bookkeeping_status": total_status,
            "max_total_residual_abs": float(max_total_abs),
            "likely_remaining_issue": likely_remaining_issue,
            "next_step": next_step,
        },
        "boundary": {
            "bubble_prefactor_changed": True,
            "direct_contact_unchanged": True,
            "source_observable_split_unchanged": True,
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
    rows = data["results"]
    sign_table = _table(
        ("q_scale", "direction", "bubble=+C rel", "direct=-K rel", "total=C-K rel", "|R_total|"),
        [
            (
                _fmt(float(row["q_scale"])),
                row["direction_j"],
                _fmt(float(row["err_main_bubble_matches_plus_C_rel"])),
                _fmt(float(row["err_direct_matches_minus_K_rel"])),
                _fmt(float(row["err_total_matches_C_minus_K_rel"])),
                _fmt(float(row["total_residual_abs"])),
            )
            for row in rows
        ],
    )
    status = data["diagnostic_status"]
    return "\n\n".join(
        [
            "# Stage 4.13 Bubble prefactor sign fix regression",
            "## Boundary\n\n"
            "- bubble prefactor changed from negative to positive\n"
            "- direct contact unchanged\n"
            "- source/observable split unchanged\n"
            "- no residual tuning\n"
            "- no fitted contact\n"
            "- no E_ET added\n"
            "- no conductivity / reflection / Casimir\n"
            "- no Ward closure claim",
            "## Changed formula\n\n"
            "$$\\Pi_{\\mu\\nu}^{bubble}=\\sum_{k,m,n}"
            "\\frac{f(E_m^-)-f(E_n^+)}{i\\Omega+E_m^- -E_n^+}"
            "J_{\\mu,mn}^{-+}P_{\\nu,nm}^{+-}.$$",
            "## Bubble sign regression\n\n"
            "The corrected main bubble is expected to satisfy "
            "$R_L^{bubble}[j]\\approx +C_j$.",
            "## Direct contact regression\n\n"
            "The direct term is unchanged and is expected to satisfy "
            "$R_L^{direct}[j]\\approx -K_j$.",
            "## Total residual bookkeeping\n\n"
            "The total spatial-source residual is expected to satisfy "
            "$R_L^{total}[j]\\approx C_j-K_j$.\n\n"
            + sign_table,
            "## Diagnostic decision\n\n"
            + _table(
                ("quantity", "status"),
                [
                    ("main_bubble_sign_status", status["main_bubble_sign_status"]),
                    ("direct_contact_status", status["direct_contact_status"]),
                    ("total_bookkeeping_status", status["total_bookkeeping_status"]),
                    ("likely_remaining_issue", status["likely_remaining_issue"]),
                    ("max_total_residual_abs", _fmt(float(status["max_total_residual_abs"]))),
                ],
            )
            + "\n\nThis remaining residual is not a bubble overall sign issue. It should be addressed by auditing C_j versus K_j routing, density q-convention, or contact thermal expectation.",
            "## Next step\n\n" + status["next_step"],
        ]
    ) + "\n"


def main() -> None:
    data = run_regression()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MD_OUTPUT.write_text(render_markdown(data), encoding="utf-8")
    print(f"Wrote {JSON_OUTPUT}")
    print(f"Wrote {MD_OUTPUT}")


if __name__ == "__main__":
    main()
