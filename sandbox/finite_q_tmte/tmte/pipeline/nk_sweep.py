"""nk-sweep aggregation for finite-q TM/TE sandbox diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..io.complex_json import complex_matrix_from_json
from ..io.writers import write_json
from .scan_runner import run_scan

SCHEMA_VERSION = "finite_q_tmte_nk_sweep_v1"
RATIO_EPS = 1e-30
SOURCE_INDICES = {"G": 0, "TM": 1, "TE": 2}


def validate_nk_values(nk_values: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in nk_values)
    if not values:
        raise ValueError("nk_values must not be empty")
    if any(value <= 0 for value in values):
        raise ValueError("nk_values must be positive integers")
    return values


def _matrix_from_payload(value: Any) -> np.ndarray:
    if isinstance(value, dict) and "shape" in value and "data" in value:
        return complex_matrix_from_json(value)
    return np.asarray(value, dtype=complex)


def selected_matrix_elements(k_gtmte_eff: Any) -> dict[str, complex]:
    """Extract compact diagnostic elements from K_GTMTE_eff."""

    matrix = _matrix_from_payload(k_gtmte_eff)
    if matrix.shape != (3, 3):
        raise ValueError("K_GTMTE_eff must have shape (3, 3)")
    g = SOURCE_INDICES["G"]
    tm = SOURCE_INDICES["TM"]
    te = SOURCE_INDICES["TE"]
    return {
        "K_GG": complex(matrix[g, g]),
        "K_GTM": complex(matrix[g, tm]),
        "K_TMG": complex(matrix[tm, g]),
        "K_TMTM": complex(matrix[tm, tm]),
        "K_TETE": complex(matrix[te, te]),
        "K_TMTE": complex(matrix[tm, te]),
        "K_TETM": complex(matrix[te, tm]),
    }


def diagnostic_ratios(diagnostics: dict[str, Any], elements: dict[str, complex], *, eps: float = RATIO_EPS) -> dict[str, float | bool]:
    """Return diagnostic-only gauge ratios."""

    denominator_eps = float(eps)
    gauge_row_norm = float(diagnostics["gauge_row_norm"])
    gauge_gg_norm = float(diagnostics["gauge_gg_norm"])
    physical_matrix_norm = float(diagnostics["physical_matrix_norm"])
    tm_abs = float(abs(elements["K_TMTM"]))
    return {
        "gauge_over_physical": gauge_row_norm / max(physical_matrix_norm, denominator_eps),
        "gauge_over_tm_abs": gauge_row_norm / max(tm_abs, denominator_eps),
        "gauge_gg_over_tm_abs": gauge_gg_norm / max(tm_abs, denominator_eps),
        "ratio_eps": denominator_eps,
        "valid_for_casimir_input": False,
    }


def nk_result_summary(*, nk: int, scan_payload: dict[str, Any], eps: float = RATIO_EPS) -> dict[str, Any]:
    """Summarize one single-q scan payload for nk-sweep comparison."""

    results = list(scan_payload.get("results", []))
    if len(results) != 1:
        raise ValueError("nk_sweep v1 expects each single-nk scan to contain exactly one q result")
    result = results[0]
    elements = selected_matrix_elements(result["effective_response"]["K_GTMTE_eff"])
    return {
        "nk": int(nk),
        "q_model": result["q_model"],
        "q_norm": result["q_norm"],
        "diagnostics": result["diagnostics"],
        "schur": result["schur"],
        "shifted_mesh_average": result["shifted_mesh_average"],
        "selected_matrix_elements": elements,
        "ratios": diagnostic_ratios(result["diagnostics"], elements, eps=eps),
        "valid_for_casimir_input": False,
    }


def nk_sweep_payload(
    *,
    model_name: str,
    pairing_name: str,
    frequency: dict[str, Any],
    sweep_parameters: dict[str, Any],
    nk_results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "valid_for_casimir_input": False,
            "reason": "nk_sweep_diagnostic_not_casimir_input",
        },
        "model": {"name": model_name, "pairing": pairing_name, "valid_for_casimir_input": False},
        "frequency": frequency,
        "sweep_parameters": {**sweep_parameters, "valid_for_casimir_input": False},
        "nk_results": list(nk_results),
        "valid_for_casimir_input": False,
    }


def run_nk_sweep(
    *,
    model_name: str,
    pairing_name: str,
    matsubara_index: int,
    temperature_K: float,
    q_values: Sequence[float],
    nk_values: Sequence[int],
    delta0_eV: float | None = None,
    eta_eV: float = 1e-8,
    shift_fractions: Sequence[float] = (0.0,),
    ratio_eps: float = RATIO_EPS,
) -> dict[str, Any]:
    """Run single-q scans for each nk and aggregate compact diagnostics."""

    q_values_tuple = tuple(float(value) for value in q_values)
    if len(q_values_tuple) != 1:
        raise ValueError("nk_sweep v1 supports exactly one q value")
    nks = validate_nk_values(nk_values)
    summaries: list[dict[str, Any]] = []
    frequency: dict[str, Any] | None = None
    for nk in nks:
        scan = run_scan(
            model_name=model_name,
            pairing_name=pairing_name,
            matsubara_index=matsubara_index,
            temperature_K=temperature_K,
            q_values=q_values_tuple,
            nk=nk,
            delta0_eV=delta0_eV,
            eta_eV=eta_eV,
            shift_fractions=tuple(shift_fractions),
        )
        frequency = scan["frequency"]
        summaries.append(nk_result_summary(nk=nk, scan_payload=scan, eps=ratio_eps))
    return nk_sweep_payload(
        model_name=model_name,
        pairing_name=pairing_name,
        frequency=frequency or {},
        sweep_parameters={
            "nk_values": list(nks),
            "q_values": [float(value) for value in q_values_tuple],
            "q_direction": [1.0, 0.0],
            "shift_fractions": [float(value) for value in shift_fractions],
            "ratio_eps": float(ratio_eps),
            "single_q_summary": True,
        },
        nk_results=summaries,
    )


def run_and_write_nk_sweep(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_nk_sweep(**kwargs)
    write_json(Path(output_dir) / "nk_sweep.json", payload)
    return payload
