from __future__ import annotations

import json

import numpy as np

from sandbox.finite_q_tmte.tmte.io.complex_json import to_jsonable
from sandbox.finite_q_tmte.tmte.pipeline.schema import SCHEMA_VERSION, basis_payload, scan_payload
from sandbox.finite_q_tmte.tmte.theory.conventions import finite_q_conventions


def test_schema_fields_and_complex_serialization():
    conventions = finite_q_conventions(np.asarray([0.2, 0.0]), xi=0.01)
    first = {
        "basis": basis_payload(conventions),
        "effective_response": {"K_GTMTE_eff": np.eye(3, dtype=complex), "K_TMTE_eff": np.eye(2, dtype=complex)},
        "diagnostics": {"gauge_row_norm": 0.0},
    }
    payload = scan_payload(
        model_name="symmetry_bdg_2band",
        pairing_name="dwave",
        xi=0.01,
        nk=1,
        first_result=first,
        results=[first],
        shifted_mesh_average={"average_order": "average_blocks_then_schur"},
    )
    encoded = to_jsonable(payload)
    assert encoded["schema_version"] == SCHEMA_VERSION
    assert encoded["status"]["valid_for_casimir_input"] is False
    assert encoded["basis"]["normalization"] == "unnormalized_gauge_orthogonal_tm_te"
    assert encoded["effective_response"]["K_TMTE_eff"]["shape"] == [2, 2]
    json.dumps(encoded)

