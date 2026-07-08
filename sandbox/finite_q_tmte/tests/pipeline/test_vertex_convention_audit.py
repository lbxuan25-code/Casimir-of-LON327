from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.pipeline.vertex_convention_audit import (
    SCHEMA_VERSION,
    linear_combination_report,
    matrix_report,
    relation_report,
)


def _load_debug_script():
    path = Path("sandbox/finite_q_tmte/scripts/debug_vertex_convention_audit.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_vertex_convention_audit_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_matrix_report_identifies_hermitian_matrix():
    matrix = np.asarray([[1.0, 2.0 - 1.0j], [2.0 + 1.0j, 3.0]], dtype=complex)
    report = matrix_report("H", matrix)
    assert report["name"] == "H"
    assert report["hermitian_residual_over_norm"] < 1e-14
    assert report["antihermitian_residual_over_norm"] > 1.0


def test_matrix_report_identifies_antihermitian_matrix():
    hermitian = np.asarray([[1.0, 2.0 - 1.0j], [2.0 + 1.0j, 3.0]], dtype=complex)
    antihermitian = 1j * hermitian
    report = matrix_report("iH", antihermitian)
    assert report["antihermitian_residual_over_norm"] < 1e-14
    assert report["i_times_matrix_hermitian_residual_over_norm"] < 1e-14


def test_relation_report_detects_opposite_dagger_sign():
    right = np.asarray([[1.0, 2.0j], [-2.0j, 3.0]], dtype=complex)
    left = -right.conj().T
    report = relation_report("L", left, right)
    assert report["left_plus_right_dagger_over_norm"] < 1e-14
    assert report["left_minus_right_dagger_over_norm"] > 1.0


def test_linear_combination_report_reference_residual():
    matrix = np.eye(2, dtype=complex)
    reference = np.eye(2, dtype=complex)
    report = linear_combination_report("same", matrix, reference)
    assert report["residual_to_reference_norm"] < 1e-14
    assert report["valid_for_casimir_input"] is False


def test_schema_version_is_vertex_convention_audit():
    assert SCHEMA_VERSION == "finite_q_tmte_vertex_convention_audit_v1"


def test_vertex_audit_cli_rejects_nonpositive_nk_for_model(tmp_path):
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
                "--nk-for-model",
                "0",
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_vertex_audit_cli_rejects_negative_matsubara_index(tmp_path):
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
                "--output-dir",
                str(tmp_path),
            ]
        )
