from __future__ import annotations

import json

import numpy as np

from sandbox.finite_q_tmte.tmte.io.complex_json import to_jsonable
from sandbox.finite_q_tmte.tmte.pipeline.schema import SCHEMA_VERSION, basis_payload, scan_payload
from sandbox.finite_q_tmte.tmte.theory.conventions import finite_q_conventions
from sandbox.finite_q_tmte.tmte.theory.frequency import frequency_payload


def _fake_result(qx: float) -> dict[str, object]:
    conventions = finite_q_conventions(np.asarray([0.2, 0.0]), xi_eV=0.01)
    return {
        "q_model": np.asarray([qx, 0.0], dtype=float),
        "q_norm": float(abs(qx)),
        "basis": basis_payload(conventions),
        "bare_blocks": {"K_SS": np.eye(3, dtype=complex)},
        "effective_response": {"K_GTMTE_eff": np.eye(3, dtype=complex), "K_TMTE_eff": np.eye(2, dtype=complex)},
        "diagnostics": {"gauge_row_norm": 0.0},
        "schur": {
            "solve_method": "solve",
            "etaeta_condition_number": 1.0,
            "condition_threshold": 1e12,
            "numerically_suspect": False,
            "valid_for_casimir_input": False,
        },
        "shifted_mesh_average": {"average_order": "average_blocks_then_schur"},
        "valid_for_casimir_input": False,
    }


def test_schema_fields_and_complex_serialization():
    first = _fake_result(0.02)
    payload = scan_payload(
        model_name="symmetry_bdg_2band",
        pairing_name="dwave",
        frequency=frequency_payload(1, 10.0),
        nk=1,
        first_result=first,
        results=[first],
        shifted_mesh_average={"average_order": "average_blocks_then_schur"},
    )
    encoded = to_jsonable(payload)
    assert encoded["schema_version"] == SCHEMA_VERSION
    assert encoded["status"]["valid_for_casimir_input"] is False
    assert "effective_response" not in encoded
    assert "diagnostics" not in encoded
    assert encoded["scan_parameters"]["basis_normalization"] == "unnormalized_gauge_orthogonal_tm_te"
    assert encoded["frequency"]["source"] == "matsubara_index"
    assert encoded["frequency"]["matsubara_index"] == 1
    assert encoded["frequency"]["temperature_K"] == 10.0
    assert "xi_eV" in encoded["frequency"]
    assert "xi" not in encoded
    assert "xi" not in encoded["scan_parameters"]
    assert encoded["results"][0]["effective_response"]["K_TMTE_eff"]["shape"] == [2, 2]
    assert "first_result_summary" in encoded
    assert "effective_response" not in encoded["first_result_summary"]
    json.dumps(encoded)


def test_multi_result_schema_keeps_results_without_global_response():
    first = _fake_result(0.02)
    second = _fake_result(0.03)
    payload = scan_payload(
        model_name="symmetry_bdg_2band",
        pairing_name="dwave",
        frequency=frequency_payload(1, 10.0),
        nk=1,
        first_result=first,
        results=[first, second],
        shifted_mesh_average={"average_order": "average_blocks_then_schur"},
    )
    assert payload["scan_parameters"]["q_count"] == 2
    assert payload["scan_parameters"]["result_count"] == 2
    assert len(payload["results"]) == 2
    assert payload["first_result_summary"]["q_norm"] == first["q_norm"]
    assert "effective_response" not in payload
    assert "diagnostics" not in payload
