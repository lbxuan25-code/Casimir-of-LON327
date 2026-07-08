from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.pipeline.primitive_response_ward_decomposition import (
    DEFAULT_CANDIDATES,
    SCHEMA_VERSION,
    block_norms,
    decompose_candidate,
    sum_terms,
    term_payload,
)


def _load_debug_script():
    path = Path("sandbox/finite_q_tmte/scripts/debug_primitive_response_ward_decomposition.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_primitive_response_ward_decomposition_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_schema_version_and_default_candidate():
    assert SCHEMA_VERSION == "finite_q_tmte_primitive_response_ward_decomposition_v1"
    assert DEFAULT_CANDIDATES == ("matrix_inferred_matsubara_i_asymmetric",)


def test_sum_terms_reconstructs_complex_vector():
    labels = ("A0", "L", "T")
    first = term_payload("first", np.asarray([1.0 + 1.0j, 0.0, 2.0], dtype=complex), labels, 1.0)
    second = term_payload("second", np.asarray([-1.0, 3.0j, 0.5], dtype=complex), labels, 1.0)
    total = sum_terms([first, second])
    np.testing.assert_allclose(total, [1.0j, 3.0j, 2.5])


def test_block_norms_reports_expected_keys():
    primitive = {
        "k_ss_bubble": np.eye(3, dtype=complex),
        "k_ss_contact": 2.0 * np.eye(3, dtype=complex),
        "k_ss": 3.0 * np.eye(3, dtype=complex),
        "k_seta": np.ones((3, 1), dtype=complex),
        "k_etas": np.ones((1, 3), dtype=complex),
        "k_etaeta_bubble": np.eye(1, dtype=complex),
        "k_etaeta_counterterm": 2.0 * np.eye(1, dtype=complex),
        "k_etaeta": 3.0 * np.eye(1, dtype=complex),
    }
    norms = block_norms(primitive, np.eye(3, dtype=complex), 0.25 * np.eye(3, dtype=complex))
    assert norms["K_SS_bubble_norm"] > 0.0
    assert norms["K_etaeta_counterterm_norm"] == 2.0
    assert norms["valid_for_casimir_input"] is False


def test_decompose_candidate_terms_sum_to_totals():
    primitive = {
        "k_ss_bubble": np.diag([1.0, 2.0, 3.0]).astype(complex),
        "k_ss_contact": np.diag([0.1, 0.2, 0.3]).astype(complex),
        "k_ss": np.diag([1.1, 2.2, 3.3]).astype(complex),
        "k_seta": np.asarray([[0.5], [0.1], [0.2]], dtype=complex),
        "k_etas": np.asarray([[0.3, 0.4, 0.5]], dtype=complex),
        "k_etaeta_bubble": np.asarray([[2.0]], dtype=complex),
        "k_etaeta_counterterm": np.asarray([[0.25]], dtype=complex),
        "k_etaeta": np.asarray([[2.25]], dtype=complex),
    }
    effective = primitive["k_ss"].copy()
    schur_correction = np.zeros((3, 3), dtype=complex)
    candidate = {
        "candidate": "constructed",
        "description": "constructed decomposition",
        "left_u": np.asarray([1.0, 0.0, 0.0], dtype=complex),
        "right_u": np.asarray([0.0, 1.0, 0.0], dtype=complex),
        "left_w": np.asarray([2.0], dtype=complex),
        "right_w": np.asarray([-1.0], dtype=complex),
    }
    result = decompose_candidate(
        primitive=primitive,
        effective=effective,
        schur_correction=schur_correction,
        candidate=candidate,
        collective_order=("phase_eta2",),
        block_reference_norm=1.0,
        effective_reference_norm=1.0,
    )
    left_em_total = result["left_em_decomposition"]["total"]
    np.testing.assert_allclose(
        [item["value"] for item in left_em_total["values"]],
        np.asarray([1.0, 0.0, 0.0]) + np.asarray([0.1, 0.0, 0.0]) + 2.0 * primitive["k_etas"][0],
    )
    assert result["accepted_convention"] is False
    assert result["valid_for_casimir_input"] is False


def test_decomposition_cli_rejects_nonpositive_nk(tmp_path):
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


def test_decomposition_cli_rejects_negative_matsubara_index(tmp_path):
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
