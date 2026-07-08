from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.pipeline.primitive_response_ward_audit import (
    SCHEMA_VERSION,
    evaluate_primitive_ward_candidate,
    primitive_schur_effective,
    primitive_ward_candidate_vectors,
)


def _load_debug_script():
    path = Path("sandbox/finite_q_tmte/scripts/debug_primitive_response_ward_audit.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_primitive_response_ward_audit_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_schema_version_is_primitive_response_audit():
    assert SCHEMA_VERSION == "finite_q_tmte_primitive_response_ward_audit_v1"


def test_primitive_schur_effective_matches_manual_schur():
    primitive = {
        "k_ss": np.asarray([[1.0, 0.2], [0.3, 2.0]], dtype=complex),
        "k_seta": np.asarray([[0.1], [0.2]], dtype=complex),
        "k_etas": np.asarray([[0.4, 0.5]], dtype=complex),
        "k_etaeta": np.asarray([[2.0]], dtype=complex),
    }
    effective, correction, schur = primitive_schur_effective(primitive)
    expected_correction = primitive["k_seta"] @ (np.linalg.solve(primitive["k_etaeta"], primitive["k_etas"]))
    np.testing.assert_allclose(correction, expected_correction)
    np.testing.assert_allclose(effective, primitive["k_ss"] - expected_correction)
    assert schur["solve_method"] == "solve"
    assert schur["valid_for_casimir_input"] is False


def test_matrix_inferred_candidate_vectors_are_asymmetric():
    rows = primitive_ward_candidate_vectors(0.01, 0.2, 0.1)
    by_name = {row["candidate"]: row for row in rows}
    cand = by_name["matrix_inferred_matsubara_i_asymmetric"]
    np.testing.assert_allclose(cand["left_u"], [0.01j, 0.2, 0.0])
    np.testing.assert_allclose(cand["right_u"], [-0.01j, 0.2, 0.0])
    np.testing.assert_allclose(cand["left_w"], [0.0, -0.2j])
    np.testing.assert_allclose(cand["right_w"], [0.0, -0.2j])


def test_evaluate_primitive_ward_candidate_closes_constructed_blocks():
    u_left = np.asarray([1.0, 0.0, 0.0], dtype=complex)
    w_left = np.asarray([0.0], dtype=complex)
    u_right = np.asarray([1.0, 0.0, 0.0], dtype=complex)
    w_right = np.asarray([0.0], dtype=complex)
    primitive = {
        "k_ss": np.asarray([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=complex),
        "k_seta": np.asarray([[0.0], [0.1], [0.2]], dtype=complex),
        "k_etas": np.asarray([[0.0, 0.3, 0.4]], dtype=complex),
        "k_etaeta": np.asarray([[1.0]], dtype=complex),
    }
    effective, _, _ = primitive_schur_effective(primitive)
    candidate = {
        "candidate": "constructed",
        "description": "constructed zero row and column",
        "left_u": u_left,
        "left_w": w_left,
        "right_u": u_right,
        "right_w": w_right,
    }
    result = evaluate_primitive_ward_candidate(
        primitive=primitive,
        effective=effective,
        candidate=candidate,
        primitive_norm=1.0,
        effective_norm=1.0,
        collective_order=("phase_eta2",),
    )
    assert result["norms"]["left_total_extended_norm"] < 1e-14
    assert result["norms"]["right_total_extended_norm"] < 1e-14
    assert result["norms"]["left_effective_norm"] < 1e-14
    assert result["norms"]["right_effective_norm"] < 1e-14
    assert result["accepted_convention"] is False


def test_primitive_response_cli_rejects_nonpositive_nk(tmp_path):
    module = _load_debug_script()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
                "--pairing",
                "dwave",
                "--matsubara-index",
                "1",
                "--q",
                "0.02",
                "--nk",
                "0",
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_primitive_response_cli_rejects_negative_matsubara_index(tmp_path):
    module = _load_debug_script()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
                "--pairing",
                "dwave",
                "--matsubara-index",
                "-1",
                "--q",
                "0.02",
                "--nk",
                "5",
                "--output-dir",
                str(tmp_path),
            ]
        )
