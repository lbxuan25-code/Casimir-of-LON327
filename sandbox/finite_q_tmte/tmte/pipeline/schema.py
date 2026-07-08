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
    xi: float,
    nk: int,
    first_result: dict[str, Any],
    results: list[dict[str, Any]],
    shifted_mesh_average: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status_payload(),
        "model": {"name": model_name, "pairing": pairing_name, "nk": int(nk), "valid_for_casimir_input": False},
        "xi": float(xi),
        "basis": first_result["basis"],
        "shifted_mesh_average": shifted_mesh_average,
        "effective_response": first_result["effective_response"],
        "diagnostics": first_result["diagnostics"],
        "results": results,
        "valid_for_casimir_input": False,
    }

