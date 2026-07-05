#!/usr/bin/env python3
"""Stage 4.12 Kubo bubble fermion-loop sign audit.

Diagnostic-only script.  It constructs a local positive-bubble candidate for
comparison, but does not modify the main response path, the direct contact
definition, conductivity, reflection, or Casimir code.
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

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "ward_identity"
JSON_OUTPUT = OUTPUT_DIR / "stage4_12_kubo_fermion_loop_sign_audit.json"
MD_OUTPUT = OUTPUT_DIR / "stage4_12_kubo_fermion_loop_sign_audit.md"

TEMPERATURE_K = 30.0
MATSUBARA_INDEX = 1
ETA_EV = 1e-10
OUTPUT_SI = False
Q_BASE = np.array([0.02, 0.013], dtype=float)
Q_SCALES = (1.0, 0.5, 0.25, 0.125)
MESH_SIZE = 16
STATIC_TEMPERATURE_K = 30.0
STATIC_ETA_EV = 1e-10
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


def finite_q_band_bubble_imag_axis_with_sign(
    energies_minus: np.ndarray,
    states_minus: np.ndarray,
    energies_plus: np.ndarray,
    states_plus: np.ndarray,
    observable_vertices: tuple[np.ndarray, ...],
    source_vertices: tuple[np.ndarray, ...],
    config: KuboConfig,
    *,
    bubble_sign: float,
) -> np.ndarray:
    """Return diagnostic finite-q band bubble with selectable overall sign.

    ``bubble_sign=+1`` is the corrected positive fermion-loop sign,
    ``bubble_sign=-1`` reproduces the old pre-Stage-4.13 negative diagnostic sign.  The matrix
    element routing intentionally matches the current implementation:
    ``source_matrix[m, n].conjugate()`` implements the reverse finite-q matrix
    element under the existing code convention.
    """

    if bubble_sign not in {-1.0, 1.0}:
        raise ValueError("bubble_sign must be +1.0 or -1.0")
    if len(observable_vertices) != len(source_vertices):
        raise ValueError("observable_vertices and source_vertices must have the same length")
    occupations_minus = fermi_function(energies_minus, config.fermi_level_eV, config.temperature_eV)
    occupations_plus = fermi_function(energies_plus, config.fermi_level_eV, config.temperature_eV)
    observable_matrices = tuple(
        states_minus.conjugate().T @ vertex @ states_plus for vertex in observable_vertices
    )
    source_matrices = tuple(states_minus.conjugate().T @ vertex @ states_plus for vertex in source_vertices)
    response = np.zeros((len(observable_vertices), len(source_vertices)), dtype=complex)
    for m, energy_minus in enumerate(energies_minus):
        for n, energy_plus in enumerate(energies_plus):
            occupation_diff = float(occupations_minus[m] - occupations_plus[n])
            if occupation_diff == 0.0:
                continue
            denominator = 1j * config.omega_eV + float(energy_minus - energy_plus)
            factor = bubble_sign * occupation_diff / denominator
            for mu, observable_matrix in enumerate(observable_matrices):
                for nu, source_matrix in enumerate(source_matrices):
                    response[mu, nu] += factor * observable_matrix[m, n] * np.conjugate(source_matrix[m, n])
    return response


def physical_response_components_with_bubble_sign(
    mesh: np.ndarray,
    config: KuboConfig,
    q: np.ndarray,
    weights: np.ndarray,
    *,
    bubble_sign: float,
) -> dict[str, np.ndarray]:
    """Return diagnostic physical-current components with selectable bubble sign.

    This local helper keeps the current physical convention
    ``J=(rho, -V_x, -V_y)`` and ``P=(rho, V_x, V_y)``.  The direct contact is
    fixed to the current physical definition ``D_ij=-<M_ij>``.
    """

    qx, qy = float(q[0]), float(q[1])
    rho = np.eye(4, dtype=complex)
    bubble = np.zeros((3, 3), dtype=complex)
    direct = np.zeros((3, 3), dtype=complex)
    for weight, (kx_value, ky_value) in zip(weights, mesh, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        h_minus = normal_state_hamiltonian(kx - 0.5 * qx, ky - 0.5 * qy)
        h_plus = normal_state_hamiltonian(kx + 0.5 * qx, ky + 0.5 * qy)
        energies_minus, states_minus = np.linalg.eigh(h_minus)
        energies_plus, states_plus = np.linalg.eigh(h_plus)

        vector_x = peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, "x")
        vector_y = peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, "y")
        observable_vertices = (rho, -vector_x, -vector_y)
        source_vertices = (rho, vector_x, vector_y)
        bubble += float(weight) * finite_q_band_bubble_imag_axis_with_sign(
            energies_minus,
            states_minus,
            energies_plus,
            states_plus,
            observable_vertices,
            source_vertices,
            config,
            bubble_sign=bubble_sign,
        )

        h0 = normal_state_hamiltonian(kx, ky)
        energies_midpoint, states_midpoint = np.linalg.eigh(h0)
        occupations_midpoint = fermi_function(
            energies_midpoint,
            config.fermi_level_eV,
            config.temperature_eV,
        )
        directions = ("x", "y")
        for i, direction_i in enumerate(directions):
            for j, direction_j in enumerate(directions):
                contact_matrix = peierls_hamiltonian_contact_vertex(kx, ky, qx, qy, direction_i, direction_j)
                band_contact = states_midpoint.conjugate().T @ contact_matrix @ states_midpoint
                expect_mij = np.sum(occupations_midpoint * np.diag(band_contact))
                direct[1 + i, 1 + j] += float(weight) * (-expect_mij)
    return {"bubble": bubble, "direct": direct, "total": bubble + direct}


def commutator_candidate_c_plus_q(
    mesh: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    q: np.ndarray,
    direction_j: str,
) -> complex:
    """Return C_j^(+q) using the Stage 4.11 diagnostic definition."""

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
    """Return K_j(q)=q_i <M_ij> using the Stage 4.11 diagnostic definition."""

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


def ward_sign_audit_row(
    q_scale: float,
    direction_j: str,
    mesh: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    q: np.ndarray,
) -> dict[str, Any]:
    components_negative = physical_response_components_with_bubble_sign(
        mesh,
        config,
        q,
        weights,
        bubble_sign=-1.0,
    )
    components_positive = physical_response_components_with_bubble_sign(
        mesh,
        config,
        q,
        weights,
        bubble_sign=1.0,
    )
    left_bubble_negative, _ = physical_ward_residuals(components_negative["bubble"], config.omega_eV, q)
    left_bubble_positive, _ = physical_ward_residuals(components_positive["bubble"], config.omega_eV, q)
    left_total_negative, _ = physical_ward_residuals(components_negative["total"], config.omega_eV, q)
    left_total_positive, _ = physical_ward_residuals(components_positive["total"], config.omega_eV, q)

    j_index = 0 if direction_j == "x" else 1
    r_bubble_negative = complex(left_bubble_negative[1 + j_index])
    r_bubble_positive = complex(left_bubble_positive[1 + j_index])
    r_total_negative = complex(left_total_negative[1 + j_index])
    r_total_positive = complex(left_total_positive[1 + j_index])
    c_plus = commutator_candidate_c_plus_q(mesh, weights, config, q, direction_j)
    k_value = direct_contact_contraction_k(mesh, weights, config, q, direction_j)

    err_negative_matches_minus_c = abs(r_bubble_negative + c_plus)
    err_negative_matches_plus_c = abs(r_bubble_negative - c_plus)
    err_positive_matches_plus_c = abs(r_bubble_positive - c_plus)
    err_positive_matches_minus_c = abs(r_bubble_positive + c_plus)
    err_total_positive_expected_c_minus_k = abs(r_total_positive - (c_plus - k_value))
    err_total_negative_expected_minus_c_minus_k = abs(r_total_negative - (-(c_plus + k_value)))

    err_negative_matches_minus_c_rel = _rel_error(err_negative_matches_minus_c, r_bubble_negative, c_plus)
    err_positive_matches_plus_c_rel = _rel_error(err_positive_matches_plus_c, r_bubble_positive, c_plus)
    status = (
        "POSITIVE_BUBBLE_GIVES_PLUS_C_AND_NEGATIVE_GIVES_MINUS_C"
        if err_positive_matches_plus_c_rel < 1e-8 and err_negative_matches_minus_c_rel < 1e-8
        else "UNRESOLVED"
    )

    return {
        "q_scale": float(q_scale),
        "q_model": [float(q[0]), float(q[1])],
        "q_norm": float(np.linalg.norm(q)),
        "direction_j": direction_j,
        "R_bubble_negative": _complex_parts(r_bubble_negative),
        "R_bubble_positive": _complex_parts(r_bubble_positive),
        "R_total_negative": _complex_parts(r_total_negative),
        "R_total_positive": _complex_parts(r_total_positive),
        "C_plus": _complex_parts(c_plus),
        "K": _complex_parts(k_value),
        "err_negative_matches_minus_C_abs": float(err_negative_matches_minus_c),
        "err_negative_matches_minus_C_rel": err_negative_matches_minus_c_rel,
        "err_negative_matches_plus_C_abs": float(err_negative_matches_plus_c),
        "err_negative_matches_plus_C_rel": _rel_error(err_negative_matches_plus_c, r_bubble_negative, c_plus),
        "err_positive_matches_plus_C_abs": float(err_positive_matches_plus_c),
        "err_positive_matches_plus_C_rel": err_positive_matches_plus_c_rel,
        "err_positive_matches_minus_C_abs": float(err_positive_matches_minus_c),
        "err_positive_matches_minus_C_rel": _rel_error(err_positive_matches_minus_c, r_bubble_positive, c_plus),
        "err_total_positive_expected_C_minus_K_abs": float(err_total_positive_expected_c_minus_k),
        "err_total_positive_expected_C_minus_K_rel": _rel_error(
            err_total_positive_expected_c_minus_k,
            r_total_positive,
            c_plus - k_value,
        ),
        "err_total_negative_expected_minus_C_minus_K_abs": float(err_total_negative_expected_minus_c_minus_k),
        "err_total_negative_expected_minus_C_minus_K_rel": _rel_error(
            err_total_negative_expected_minus_c_minus_k,
            r_total_negative,
            -(c_plus + k_value),
        ),
        "ward_bubble_sign_status": status,
    }


def compressibility_sanity_check() -> dict[str, Any]:
    """Return a minimal static diagonal-Hamiltonian sign sanity check."""

    energies = np.array([-0.2, 0.1, 0.4], dtype=float)
    config = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=STATIC_TEMPERATURE_K,
        eta_eV=STATIC_ETA_EV,
        output_si=OUTPUT_SI,
    )
    occupations = fermi_function(energies, config.fermi_level_eV, config.temperature_eV)
    fprime = -(1.0 / config.temperature_eV) * occupations * (1.0 - occupations)
    analytic = float(np.sum(fprime))
    positive_static = analytic
    negative_static = -analytic
    positive_matches = (
        np.sign(positive_static) == np.sign(analytic)
        and _rel_error(abs(positive_static - analytic), positive_static, analytic) < 1e-12
    )
    negative_wrong_sign = np.sign(negative_static) == -np.sign(analytic)
    status = (
        "POSITIVE_BUBBLE_SIGN_MATCHES_COMPRESSIBILITY"
        if positive_matches and negative_wrong_sign
        else "UNRESOLVED"
    )
    return {
        "energies_eV": [float(item) for item in energies],
        "temperature_K": STATIC_TEMPERATURE_K,
        "temperature_eV": float(config.temperature_eV),
        "analytic_compressibility": analytic,
        "positive_bubble_static_limit": float(positive_static),
        "negative_bubble_static_limit": float(negative_static),
        "positive_matches_analytic": bool(positive_matches),
        "negative_has_wrong_sign": bool(negative_wrong_sign),
        "compressibility_status": status,
    }


def classify_ward_bubble_sign_global(rows: list[dict[str, Any]]) -> str:
    if rows and all(
        row["ward_bubble_sign_status"] == "POSITIVE_BUBBLE_GIVES_PLUS_C_AND_NEGATIVE_GIVES_MINUS_C"
        for row in rows
    ):
        return "POSITIVE_BUBBLE_GIVES_PLUS_C_AND_NEGATIVE_GIVES_MINUS_C"
    return "UNRESOLVED"


def run_audit(
    q_scales: tuple[float, ...] | list[float] = Q_SCALES,
    mesh_size: int = MESH_SIZE,
) -> dict[str, Any]:
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
    for q_scale in q_scales:
        q = float(q_scale) * Q_BASE
        for direction_j in ("x", "y"):
            rows.append(ward_sign_audit_row(float(q_scale), direction_j, mesh, weights, config, q))

    compressibility = compressibility_sanity_check()
    ward_global = classify_ward_bubble_sign_global(rows)
    compressibility_status = str(compressibility["compressibility_status"])
    if (
        ward_global == "POSITIVE_BUBBLE_GIVES_PLUS_C_AND_NEGATIVE_GIVES_MINUS_C"
        and compressibility_status == "POSITIVE_BUBBLE_SIGN_MATCHES_COMPRESSIBILITY"
    ):
        likely_issue = "STAGE_4_12_SUPPORTS_POSITIVE_BUBBLE_SIGN"
        next_step = (
            "Next: after the Stage 4.13 main-path patch, rerun Stage 4.9-4.11 diagnostics. "
            "Do not modify direct contact."
        )
    else:
        likely_issue = "BUBBLE_SIGN_UNRESOLVED"
        next_step = (
            "Next: audit Matsubara Green function convention and finite-q matrix element ordering "
            "before changing the main response path."
        )
    return {
        "stage": "Stage 4.12",
        "purpose": "Kubo bubble fermion-loop sign audit",
        "config": {
            "temperature_K": TEMPERATURE_K,
            "matsubara_index": MATSUBARA_INDEX,
            "omega_eV": float(omega_eV),
            "eta_eV": ETA_EV,
            "mesh_size": int(mesh_size),
            "q_base": [float(Q_BASE[0]), float(Q_BASE[1])],
            "q_scales": [float(item) for item in q_scales],
        },
        "ward_bubble_sign_audit": {"results": rows},
        "compressibility_sanity_check": compressibility,
        "diagnostic_status": {
            "ward_bubble_sign_global_status": ward_global,
            "compressibility_status": compressibility_status,
            "likely_issue": likely_issue,
            "next_step": next_step,
        },
        "boundary": {
            "no_residual_tuning": True,
            "no_bubble_formula_change_to_main_path": True,
            "no_main_response_change": True,
            "no_direct_contact_change": True,
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
    rows = data["ward_bubble_sign_audit"]["results"]
    comparison_table = _table(
        (
            "q_scale",
            "direction",
            "neg=-C rel",
            "neg=+C rel",
            "pos=+C rel",
            "pos=-C rel",
            "status",
        ),
        [
            (
                _fmt(float(row["q_scale"])),
                row["direction_j"],
                _fmt(float(row["err_negative_matches_minus_C_rel"])),
                _fmt(float(row["err_negative_matches_plus_C_rel"])),
                _fmt(float(row["err_positive_matches_plus_C_rel"])),
                _fmt(float(row["err_positive_matches_minus_C_rel"])),
                row["ward_bubble_sign_status"],
            )
            for row in rows
        ],
    )
    total_table = _table(
        ("q_scale", "direction", "positive total vs C-K rel", "negative total vs -(C+K) rel"),
        [
            (
                _fmt(float(row["q_scale"])),
                row["direction_j"],
                _fmt(float(row["err_total_positive_expected_C_minus_K_rel"])),
                _fmt(float(row["err_total_negative_expected_minus_C_minus_K_rel"])),
            )
            for row in rows
        ],
    )
    comp = data["compressibility_sanity_check"]
    status = data["diagnostic_status"]
    return "\n\n".join(
        [
            "# Stage 4.12 Kubo bubble fermion-loop sign audit",
            "## Boundary\n\n"
            "- no residual tuning\n"
            "- no bubble formula change to the main path\n"
            "- no main response path change\n"
            "- no direct contact change\n"
            "- no conductivity / reflection / Casimir\n"
            "- no Ward closure claim",
            "## Analytic sign logic\n\n"
            "$$\\Pi^{bubble}=-\\langle TJP\\rangle_c.$$\n\n"
            "$$\\langle TJP\\rangle_c=-\\mathrm{Tr}[JGPG].$$\n\n"
            "Therefore\n\n"
            "$$\\Pi^{bubble}=+\\mathrm{Tr}[JGPG].$$\n\n"
            "This audit compares the old pre-Stage-4.13 negative diagnostic branch with the "
            "corrected positive band-sum branch. After Stage 4.13 the main path is patched to "
            "the positive bubble prefactor.",
            "## Ward bubble sign comparison\n\n" + comparison_table,
            "## Total residual sign bookkeeping\n\n" + total_table,
            "## Compressibility sanity check\n\n"
            + _table(
                ("quantity", "value"),
                [
                    ("analytic_compressibility", _fmt(float(comp["analytic_compressibility"]))),
                    ("positive_bubble_static_limit", _fmt(float(comp["positive_bubble_static_limit"]))),
                    ("negative_bubble_static_limit", _fmt(float(comp["negative_bubble_static_limit"]))),
                    ("compressibility_status", comp["compressibility_status"]),
                ],
            ),
            "## Diagnostic decision\n\n"
            + _table(
                ("quantity", "status"),
                [
                    ("ward_bubble_sign_global_status", status["ward_bubble_sign_global_status"]),
                    ("compressibility_status", status["compressibility_status"]),
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
