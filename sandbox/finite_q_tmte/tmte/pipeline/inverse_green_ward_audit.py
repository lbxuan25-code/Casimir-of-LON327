"""Diagnostic-only inverse-Green Ward audit for finite-q BdG vertices."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from lno327.bdg.hamiltonian import bdg_hamiltonian_from_model_pairing

from ..adapters.collective_adapter import collective_vertices
from ..adapters.model_adapter import build_model_scan_inputs
from ..adapters.primitive_vertices_adapter import primitive_observable_vertices, primitive_source_vertices
from ..io.writers import write_json
from ..theory.conventions import finite_q_conventions
from ..theory.frequency import frequency_payload, matsubara_xi_eV
from ..theory.vertices import longitudinal_transverse_vertices
from .collective_schur_factors import collective_order_from_ansatz
from .vertex_convention_audit import _candidate_coefficients, _norm, _safe_ratio, linear_combination_report, matrix_report

SCHEMA_VERSION = "finite_q_tmte_inverse_green_ward_audit_v1"


def inverse_green_matrix(z_eV: complex, h_bdg: np.ndarray) -> np.ndarray:
    """Return G^{-1}(z,k)=z I-H_BdG(k) for diagnostic matrix identities."""

    h = np.asarray(h_bdg, dtype=complex)
    return complex(z_eV) * np.eye(h.shape[0], dtype=complex) - h


def inverse_green_pair(
    *,
    h_minus: np.ndarray,
    h_plus: np.ndarray,
    xi_eV: float,
    fermionic_energy_eV: float,
    frequency_convention: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build G^{-1}_- and G^{-1}_+ under a diagnostic transfer-frequency convention."""

    nu = float(fermionic_energy_eV)
    xi = float(xi_eV)
    if frequency_convention == "matsubara_i_transfer":
        z_minus = 1j * nu
        z_plus = 1j * (nu + xi)
    elif frequency_convention == "matsubara_minus_i_transfer":
        z_minus = 1j * nu
        z_plus = 1j * (nu - xi)
    elif frequency_convention == "real_transfer_debug":
        z_minus = 1j * nu
        z_plus = 1j * nu + xi
    else:
        raise ValueError(f"unknown frequency convention {frequency_convention!r}")
    return inverse_green_matrix(z_minus, h_minus), inverse_green_matrix(z_plus, h_plus), {
        "frequency_convention": frequency_convention,
        "z_minus_eV": complex(z_minus),
        "z_plus_eV": complex(z_plus),
        "transfer_z_plus_minus_z_minus_eV": complex(z_plus - z_minus),
        "valid_for_casimir_input": False,
    }


def inverse_green_reference_matrices(g_minus_inv: np.ndarray, g_plus_inv: np.ndarray, gamma0: np.ndarray) -> dict[str, np.ndarray]:
    """Return diagnostic references including Nambu tau3-sandwiched Ward forms."""

    gm = np.asarray(g_minus_inv, dtype=complex)
    gp = np.asarray(g_plus_inv, dtype=complex)
    tau = np.asarray(gamma0, dtype=complex)
    return {
        "plain_delta_Ginv_plus_minus_minus": gp - gm,
        "nambu_tau_left_Gplus_minus_Gminus_tau_right": tau @ gp - gm @ tau,
        "nambu_Gplus_tau_right_minus_tau_left_Gminus": gp @ tau - tau @ gm,
        "left_tau_plain_delta": tau @ (gp - gm),
        "plain_delta_right_tau": (gp - gm) @ tau,
        "valid_for_casimir_input": False,
    }


def reference_payload(references: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    return [matrix_report(name, matrix) for name, matrix in references.items() if name != "valid_for_casimir_input"]


def ward_combo_matrix(gamma0: np.ndarray, gamma_l: np.ndarray, gamma_phase: np.ndarray, coeffs: dict[str, Any]) -> np.ndarray:
    return complex(coeffs["a0"]) * gamma0 + complex(coeffs["l"]) * gamma_l + complex(coeffs["phase"]) * gamma_phase


def combo_vs_references(
    *,
    side: str,
    gamma0: np.ndarray,
    gamma_l: np.ndarray,
    gamma_phase: np.ndarray,
    references: dict[str, np.ndarray],
    xi_eV: float,
    q_norm: float,
    delta0_eV: float,
) -> list[dict[str, Any]]:
    rows = []
    usable_refs = {name: matrix for name, matrix in references.items() if name != "valid_for_casimir_input"}
    for coeffs in _candidate_coefficients(xi_eV, q_norm, delta0_eV):
        combo = ward_combo_matrix(gamma0, gamma_l, gamma_phase, coeffs)
        comparisons = [
            {
                "reference": ref_name,
                "combo_minus_reference": linear_combination_report("combo_minus_reference", combo, ref_matrix),
                "combo_plus_reference": linear_combination_report("combo_plus_reference", combo, -ref_matrix),
                "valid_for_casimir_input": False,
            }
            for ref_name, ref_matrix in usable_refs.items()
        ]
        rows.append(
            {
                "side": side,
                "candidate": coeffs["name"],
                "coefficients": {"A0": complex(coeffs["a0"]), "L": complex(coeffs["l"]), "phase_eta2": complex(coeffs["phase"])},
                "combo_report": matrix_report("ward_combo", combo),
                "comparisons": comparisons,
                "accepted_convention": False,
                "valid_for_casimir_input": False,
            }
        )
    return rows


def run_inverse_green_ward_audit(
    *,
    model_name: str,
    pairing_name: str,
    matsubara_index: int,
    temperature_K: float,
    q_value: float,
    kx: float,
    ky: float,
    fermionic_energy_eV: float = 0.0,
    nk_for_model: int = 5,
    delta0_eV: float | None = None,
    eta_eV: float = 1e-8,
    current_vertex: str = "peierls",
    frequency_conventions: tuple[str, ...] = ("matsubara_i_transfer", "matsubara_minus_i_transfer", "real_transfer_debug"),
) -> dict[str, Any]:
    """Run diagnostic inverse-Green matrix Ward checks at one representative k point."""

    xi_eV = matsubara_xi_eV(matsubara_index, temperature_K)
    inputs = build_model_scan_inputs(
        model_name=model_name,
        pairing_name=pairing_name,
        xi_eV=xi_eV,
        nk=nk_for_model,
        delta0_eV=delta0_eV,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
    )
    q = np.asarray([float(q_value), 0.0], dtype=float)
    qx, qy = float(q[0]), float(q[1])
    conventions = finite_q_conventions(q, xi_eV)
    kx0 = float(kx)
    ky0 = float(ky)
    pairing_params = inputs.pairing_params
    delta_minus = inputs.ansatz.mean_pairing(kx0 - 0.5 * qx, ky0 - 0.5 * qy, pairing_params)
    delta_plus = inputs.ansatz.mean_pairing(kx0 + 0.5 * qx, ky0 + 0.5 * qy, pairing_params)
    h_minus = bdg_hamiltonian_from_model_pairing(inputs.spec, kx0 - 0.5 * qx, ky0 - 0.5 * qy, delta_minus)
    h_plus = bdg_hamiltonian_from_model_pairing(inputs.spec, kx0 + 0.5 * qx, ky0 + 0.5 * qy, delta_plus)

    src0, srcx, srcy = primitive_source_vertices(inputs.spec, kx0, ky0, qx, qy, current_vertex=current_vertex)
    obs0, obsx, obsy = primitive_observable_vertices(inputs.spec, kx0, ky0, qx, qy, current_vertex=current_vertex)
    src_l, _ = longitudinal_transverse_vertices(srcx, srcy, conventions)
    obs_l, _ = longitudinal_transverse_vertices(obsx, obsy, conventions)
    coll = collective_vertices(inputs.ansatz, kx0, ky0, qx, qy, pairing_params)
    collective_order, raw_names = collective_order_from_ansatz(inputs.ansatz, len(coll))
    phase_index = collective_order.index("phase_eta2") if "phase_eta2" in collective_order else len(coll) - 1
    phase = np.asarray(coll[phase_index], dtype=complex)
    delta0 = float(getattr(pairing_params, "delta0_eV", 0.0))

    frequency_results = []
    for convention_name in frequency_conventions:
        g_minus, g_plus, freq_meta = inverse_green_pair(
            h_minus=h_minus,
            h_plus=h_plus,
            xi_eV=xi_eV,
            fermionic_energy_eV=fermionic_energy_eV,
            frequency_convention=convention_name,
        )
        source_refs = inverse_green_reference_matrices(g_minus, g_plus, src0)
        observable_refs = inverse_green_reference_matrices(g_minus, g_plus, obs0)
        frequency_results.append(
            {
                "frequency": freq_meta,
                "Ginv_minus_report": matrix_report("Ginv_minus", g_minus),
                "Ginv_plus_report": matrix_report("Ginv_plus", g_plus),
                "references": {
                    "source": reference_payload(source_refs),
                    "observable": reference_payload(observable_refs),
                    "valid_for_casimir_input": False,
                },
                "candidate_comparisons": {
                    "source": combo_vs_references(side="source", gamma0=src0, gamma_l=src_l, gamma_phase=phase, references=source_refs, xi_eV=xi_eV, q_norm=conventions.gL, delta0_eV=delta0),
                    "observable": combo_vs_references(side="observable", gamma0=obs0, gamma_l=obs_l, gamma_phase=phase, references=observable_refs, xi_eV=xi_eV, q_norm=conventions.gL, delta0_eV=delta0),
                    "valid_for_casimir_input": False,
                },
                "valid_for_casimir_input": False,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "diagnostic_only_not_a_fix": True,
            "accepted_convention": False,
            "requires_analytic_derivation_before_convention_change": True,
            "valid_for_casimir_input": False,
            "reason": "inverse_green_ward_audit_not_production_convention",
        },
        "model": {"name": model_name, "pairing": pairing_name, "valid_for_casimir_input": False},
        "point": {
            "kx": kx0,
            "ky": ky0,
            "q_model": [qx, qy],
            "xi_eV": float(xi_eV),
            "fermionic_energy_eV": float(fermionic_energy_eV),
            "matsubara_index": int(matsubara_index),
            "temperature_K": float(temperature_K),
            "delta0_eV": delta0,
            "current_vertex": current_vertex,
            "valid_for_casimir_input": False,
        },
        "frequency": frequency_payload(matsubara_index, temperature_K),
        "collective_order": list(collective_order),
        "raw_ansatz_channel_names": list(raw_names) if raw_names is not None else None,
        "matrix_norms": {
            "H_minus_norm": _norm(h_minus),
            "H_plus_norm": _norm(h_plus),
            "H_plus_minus_H_minus_norm": _norm(np.asarray(h_plus, dtype=complex) - np.asarray(h_minus, dtype=complex)),
            "source_A0_norm": _norm(src0),
            "source_L_norm": _norm(src_l),
            "phase_eta2_norm": _norm(phase),
            "valid_for_casimir_input": False,
        },
        "nambu_reference_note": "Diagnostic references include plain delta G^{-1} and tau3-sandwiched forms such as tau3 G^{-1}_+ - G^{-1}_- tau3. No reference is accepted without analytic derivation.",
        "frequency_convention_results": frequency_results,
        "valid_for_casimir_input": False,
    }


def run_and_write_inverse_green_ward_audit(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_inverse_green_ward_audit(**kwargs)
    write_json(Path(output_dir) / "inverse_green_ward_audit.json", payload)
    return payload
