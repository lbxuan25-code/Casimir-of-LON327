from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sandbox.finite_q_tmte.tmte.pipeline.normal_equal_time_robustness_scan import (
    SCHEMA_VERSION,
    aggregate_rows,
    summarize_equal_time_payload,
)


def _load_debug_script():
    path = Path("sandbox/finite_q_tmte/scripts/debug_normal_equal_time_robustness_scan.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_normal_equal_time_robustness_scan_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _complex(real: float, imag: float = 0.0) -> dict[str, float]:
    return {"real": real, "imag": imag, "abs": abs(complex(real, imag))}


def fake_payload() -> dict[str, object]:
    return {
        "debug_parameters": {"q_value": 0.02, "nk": 13},
        "frequency": {"matsubara_index": 1, "temperature_K": 10.0},
        "vertex_identity": {"max_abs_error_over_shifted_meshes": 1e-14},
        "ward_decomposition": {
            "left": {
                "total": {"norm": 1e-3},
                "contact_required_over_current": {"alpha": 0.75 + 0.0j},
            },
            "right": {
                "total": {"norm": 1e-3},
                "contact_required_over_current": {"alpha": 0.75 + 0.0j},
            },
        },
        "candidate_equal_time_vectors_ranked": [
            {
                "name": "minus_translation_forward",
                "difference_over_target_norm": 1e-13,
                "fit_to_target": {"alpha": 1.0 + 0.0j, "residual_over_target_norm": 2e-13},
            }
        ],
    }


def test_schema_version():
    assert SCHEMA_VERSION == "finite_q_tmte_normal_equal_time_robustness_scan_v1"


def test_summarize_equal_time_payload_passes_expected_top_candidate():
    row = summarize_equal_time_payload(fake_payload(), diff_tol=1e-10, fit_res_tol=1e-10, alpha_tol=1e-10)
    assert row["passed_translation_identity"] is True
    assert row["top_candidate"] == "minus_translation_forward"
    assert row["q_value"] == pytest.approx(0.02)
    assert row["contact_alpha_left"]["real"] == pytest.approx(0.75)


def test_summarize_equal_time_payload_fails_unexpected_candidate():
    payload = fake_payload()
    payload["candidate_equal_time_vectors_ranked"][0]["name"] = "qM_mid"
    row = summarize_equal_time_payload(payload, diff_tol=1e-10, fit_res_tol=1e-10, alpha_tol=1e-10)
    assert row["passed_translation_identity"] is False


def test_aggregate_rows_counts_passes():
    rows = [
        {"passed_translation_identity": True, "top_difference_over_missing": 1e-13, "top_fit_residual_over_missing": 2e-13, "top_fit_alpha": _complex(1.0), "top_candidate": "minus_translation_forward"},
        {"passed_translation_identity": False, "top_difference_over_missing": 1e-2, "top_fit_residual_over_missing": 1e-2, "top_fit_alpha": _complex(0.9), "top_candidate": "qM_mid"},
    ]
    agg = aggregate_rows(rows)
    assert agg["num_rows"] == 2
    assert agg["num_passed"] == 1
    assert agg["all_passed"] is False
    assert agg["top_candidate_counts"]["qM_mid"] == 1


def test_cli_rejects_nonpositive_nk(tmp_path):
    module = _load_debug_script()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
                "--matsubara-indices",
                "1",
                "--q-values",
                "0.02",
                "--nk-values",
                "0",
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_cli_rejects_nonpositive_q(tmp_path):
    module = _load_debug_script()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
                "--matsubara-indices",
                "1",
                "--q-values",
                "0.0",
                "--nk-values",
                "5",
                "--output-dir",
                str(tmp_path),
            ]
        )
