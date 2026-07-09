from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sandbox.finite_q_tmte.tmte.pipeline.primitive_extended_translation_collective_robustness_scan import (
    SCHEMA_VERSION,
    aggregate_rows,
    summarize_payload,
)


def _load_debug_script():
    path = Path("sandbox/finite_q_tmte/scripts/debug_primitive_extended_translation_collective_robustness_scan.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_primitive_extended_translation_collective_robustness_scan_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fit_row(name: str = "minus_translation_plus_qM", diff: float = 1e-12, alpha: complex = 1.0 + 0j, fit_res: float = 2e-12) -> dict[str, object]:
    return {
        "name": name,
        "difference_over_target_norm": diff,
        "fit_to_target": {"alpha": alpha, "residual_over_target_norm": fit_res},
    }


def fake_payload() -> dict[str, object]:
    return {
        "model": {"pairing": "dwave", "delta0_eV": 0.1},
        "frequency": {"matsubara_index": 1, "temperature_K": 10.0},
        "debug_parameters": {"q_value": 0.02, "nk": 13},
        "ward_decomposition": {
            "left": {
                "em_total": {"norm": 1.0},
                "mixed_collective": {"norm": 0.8},
                "extended_total": {"norm": 0.2},
                "em_to_extended_reduction": 0.2,
            },
            "right": {
                "em_total": {"norm": 1.0},
                "mixed_collective": {"norm": 0.8},
                "extended_total": {"norm": 0.2},
                "em_to_extended_reduction": 0.2,
            },
        },
        "left_translation_candidates_ranked": [_fit_row()],
        "right_translation_candidates_ranked": [_fit_row()],
    }


def test_schema_version():
    assert SCHEMA_VERSION == "finite_q_tmte_primitive_extended_translation_collective_robustness_scan_v1"


def test_summarize_payload_passes_both_sides():
    row = summarize_payload(fake_payload(), diff_tol=1e-9, fit_res_tol=1e-9, alpha_tol=1e-9)
    assert row["passed_translation_identity"] is True
    assert row["left"]["top_candidate"] == "minus_translation_plus_qM"
    assert row["right"]["top_candidate"] == "minus_translation_plus_qM"
    assert row["left_extended_over_em"] == pytest.approx(0.2)


def test_summarize_payload_fails_wrong_top_candidate():
    payload = fake_payload()
    payload["left_translation_candidates_ranked"] = [_fit_row(name="translation_forward")]
    row = summarize_payload(payload, diff_tol=1e-9, fit_res_tol=1e-9, alpha_tol=1e-9)
    assert row["passed_translation_identity"] is False


def test_aggregate_rows_counts_pairing_passes():
    row1 = summarize_payload(fake_payload(), diff_tol=1e-9, fit_res_tol=1e-9, alpha_tol=1e-9)
    payload2 = fake_payload()
    payload2["model"]["pairing"] = "spm"
    payload2["left_translation_candidates_ranked"] = [_fit_row(name="translation_forward")]
    row2 = summarize_payload(payload2, diff_tol=1e-9, fit_res_tol=1e-9, alpha_tol=1e-9)
    agg = aggregate_rows([row1, row2])
    assert agg["num_rows"] == 2
    assert agg["num_passed"] == 1
    assert agg["all_passed"] is False
    assert agg["pairing_pass_counts"]["dwave"] == 1
    assert agg["pairing_pass_counts"]["spm"] == 0


def test_cli_rejects_nonpositive_q(tmp_path):
    module = _load_debug_script()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
                "--pairings",
                "dwave",
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


def test_cli_rejects_nonpositive_nk(tmp_path):
    module = _load_debug_script()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
                "--pairings",
                "dwave",
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
