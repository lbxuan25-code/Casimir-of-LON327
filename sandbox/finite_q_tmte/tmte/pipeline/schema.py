"""JSON schema assembly for finite-q TM/TE sandbox outputs."""

from __future__ import annotations

from typing import Any

from ..theory.conventions import BASIS_NORMALIZATION, SOURCE_ORDER_DIAGNOSTIC, SOURCE_ORDER_PHYSICAL

SCHEMA_VERSION = "finite_q_tmte_sandbox_v1"


def status_payload() -> dict[str, Any]:
    return {
        "diagnostic_run_completed": True,
        "valid_for_casimir_input": False,
        "reason": "tmte_response_normalization_not_yet_matched_to_casimir_kernel",
    }


def basis_payload(conventions: Any) -> dict[str, Any]:
    return {
        "source_order_diagnostic": list(SOURCE_ORDER_DIAGNOSTIC),
        "source_order_physical": list(SOURCE_ORDER_PHYSICAL),
        "normalization": BASIS_NORMALIZATION,
        "qhat": conventions.qhat,
        "that": conventions.that,
        "gauge_coefficients": {"g0": conventions.g0, "gL": conventions.gL},
    }


def scan_payload(
    *,
    model_name: str,
    pairing_name: str,
    frequency: dict[str, Any],
    nk: int,
    first_result: dict[str, Any],
    results: list[dict[str, Any]],
    shifted_mesh_average: dict[str, Any],
) -> dict[str, Any]:
    first_summary = {
        "q_model": first_result["q_model"],
        "q_norm": first_result["q_norm"],
        "basis": first_result["basis"],
        "diagnostics": first_result["diagnostics"],
        "schur": first_result["schur"],
        "shifted_mesh_average": first_result["shifted_mesh_average"],
        "valid_for_casimir_input": False,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status_payload(),
        "model": {"name": model_name, "pairing": pairing_name, "valid_for_casimir_input": False},
        "frequency": frequency,
        "scan_parameters": {
            "xi_eV": float(frequency["xi_eV"]),
            "nk": int(nk),
            "q_count": len({tuple(result["q_model"]) for result in results}),
            "result_count": len(results),
            "source_order_diagnostic": list(SOURCE_ORDER_DIAGNOSTIC),
            "source_order_physical": list(SOURCE_ORDER_PHYSICAL),
            "basis_normalization": BASIS_NORMALIZATION,
            "shifted_mesh_average": shifted_mesh_average,
            "valid_for_casimir_input": False,
        },
        "results": results,
        "first_result_summary": first_summary,
        "valid_for_casimir_input": False,
    }
