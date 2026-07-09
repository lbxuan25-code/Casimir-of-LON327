from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.pipeline.primitive_em_translation_rhs_audit import (
    DEFAULT_CANDIDATE,
    SCHEMA_VERSION,
    _fit,
    _match,
)


def _load_debug_script():
    path = Path("sandbox/finite_q_tmte/scripts/debug_primitive_em_translation_rhs_audit.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_primitive_em_translation_rhs_audit_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_schema_and_default_candidate():
    assert SCHEMA_VERSION == "finite_q_tmte_primitive_em_translation_rhs_audit_v1"
    assert DEFAULT_CANDIDATE == "matrix_inferred_matsubara_i_asymmetric"


def test_fit_recovers_parallel_alpha():
    candidate = np.asarray([1.0, -2.0j, 0.5], dtype=complex)
    alpha = -0.4 + 0.2j
    report = _fit(alpha * candidate, candidate)
    np.testing.assert_allclose(report["alpha"], alpha)
    assert report["residual_norm"] < 1e-12


def test_match_exact_vector():
    target = np.asarray([0.0, 1.0, 0.0], dtype=complex)
    report = _match("exact", target, target)
    assert report["difference_norm"] == pytest.approx(0.0)
    assert report["difference_over_target_norm"] == pytest.approx(0.0)


def test_cli_rejects_nonpositive_q(tmp_path):
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
                "0.0",
                "--nk",
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
