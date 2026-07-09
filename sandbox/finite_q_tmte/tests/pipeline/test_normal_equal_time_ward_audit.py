from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.pipeline.normal_equal_time_ward_audit import (
    SCHEMA_VERSION,
    _fit,
    _match,
)


def _load_debug_script():
    path = Path("sandbox/finite_q_tmte/scripts/debug_normal_equal_time_ward_audit.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_normal_equal_time_ward_audit_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_schema_version():
    assert SCHEMA_VERSION == "finite_q_tmte_normal_equal_time_ward_audit_v1"


def test_fit_recovers_parallel_alpha():
    candidate = np.asarray([1.0, 2.0j, -0.5], dtype=complex)
    alpha = 0.25 - 0.1j
    report = _fit(alpha * candidate, candidate)
    np.testing.assert_allclose(report["alpha"], alpha)
    assert report["residual_norm"] < 1e-12


def test_match_reports_direct_difference():
    target = np.asarray([1.0, 0.0, 0.0], dtype=complex)
    candidate = np.asarray([1.0, 0.0, 0.0], dtype=complex)
    report = _match("candidate", candidate, target)
    assert report["difference_norm"] == pytest.approx(0.0)
    assert report["difference_over_target_norm"] == pytest.approx(0.0)


def test_normal_equal_time_cli_rejects_nonpositive_nk(tmp_path):
    module = _load_debug_script()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
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


def test_normal_equal_time_cli_rejects_negative_matsubara_index(tmp_path):
    module = _load_debug_script()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
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
