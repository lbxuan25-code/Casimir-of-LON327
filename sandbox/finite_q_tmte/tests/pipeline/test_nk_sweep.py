from __future__ import annotations

import numpy as np

from sandbox.finite_q_tmte.tmte.pipeline.nk_sweep import (
    SCHEMA_VERSION,
    diagnostic_ratios,
    nk_result_summary,
    nk_sweep_payload,
    selected_matrix_elements,
)
from sandbox.finite_q_tmte.tmte.theory.frequency import frequency_payload


def _fake_scan_payload(nk: int) -> dict[str, object]:
    matrix = np.asarray(
        [
            [1.0 + 0.0j, 2.0 + 1.0j, 3.0 + 0.0j],
            [4.0 - 1.0j, 5.0 + 0.0j, 6.0 + 2.0j],
            [7.0 + 0.0j, 8.0 - 2.0j, 9.0 + 0.0j],
        ],
        dtype=complex,
    )
    return {
        "frequency": frequency_payload(1, 10.0),
        "results": [
            {
                "q_model": np.asarray([0.02, 0.0], dtype=float),
                "q_norm": 0.02,
                "effective_response": {"K_GTMTE_eff": matrix},
                "diagnostics": {
                    "gauge_row_norm": 10.0 + nk,
                    "gauge_col_norm": 11.0 + nk,
                    "gauge_gg_norm": 2.5,
                    "physical_matrix_norm": 20.0,
                    "etaeta_condition_number": 30.0,
                },
                "schur": {"solve_method": "solve", "numerically_suspect": False},
                "shifted_mesh_average": {"enabled": False},
                "valid_for_casimir_input": False,
            }
        ],
    }


def test_selected_matrix_elements_and_ratios_for_fake_matrix():
    matrix = _fake_scan_payload(5)["results"][0]["effective_response"]["K_GTMTE_eff"]  # type: ignore[index]
    elements = selected_matrix_elements(matrix)
    assert elements["K_GG"] == 1.0 + 0.0j
    assert elements["K_GTM"] == 2.0 + 1.0j
    assert elements["K_TMG"] == 4.0 - 1.0j
    assert elements["K_TMTM"] == 5.0 + 0.0j
    assert elements["K_TETE"] == 9.0 + 0.0j
    assert elements["K_TMTE"] == 6.0 + 2.0j
    assert elements["K_TETM"] == 8.0 - 2.0j
    ratios = diagnostic_ratios(
        {"gauge_row_norm": 10.0, "gauge_gg_norm": 2.5, "physical_matrix_norm": 20.0},
        elements,
        eps=1e-30,
    )
    assert ratios["gauge_over_physical"] == 0.5
    assert ratios["gauge_over_tm_abs"] == 2.0
    assert ratios["gauge_gg_over_tm_abs"] == 0.5
    assert ratios["valid_for_casimir_input"] is False


def test_nk_sweep_schema_keeps_all_nk_values_without_top_level_response():
    nk_results = [
        nk_result_summary(nk=5, scan_payload=_fake_scan_payload(5)),
        nk_result_summary(nk=7, scan_payload=_fake_scan_payload(7)),
    ]
    payload = nk_sweep_payload(
        model_name="symmetry_bdg_2band",
        pairing_name="dwave",
        frequency=frequency_payload(1, 10.0),
        sweep_parameters={"nk_values": [5, 7], "ratio_eps": 1e-30},
        nk_results=nk_results,
    )
    assert payload["schema_version"] == SCHEMA_VERSION
    assert "effective_response" not in payload
    assert [row["nk"] for row in payload["nk_results"]] == [5, 7]
    assert payload["valid_for_casimir_input"] is False
    assert all(row["valid_for_casimir_input"] is False for row in payload["nk_results"])

