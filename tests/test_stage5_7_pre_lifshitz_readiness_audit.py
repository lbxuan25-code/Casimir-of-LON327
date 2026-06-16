from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from lno327.lifshitz_readiness import (
    lab_q_to_crystal_q,
    round_trip_factor,
    scalar_sheet_te_tm_reflection,
    trace_log_integrand,
    trace_log_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "src" / "lno327" / "lifshitz_readiness.py"
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage5_7_pre_lifshitz_readiness_audit.py"
DOC = ROOT / "docs" / "notes" / "stage5_7_pre_lifshitz_readiness_audit_zh.md"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage5_7_audit", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _synthetic_stage5_6(path: Path, *, status: str = "STAGE5_6_TE_TM_ADAPTER_PASSED") -> None:
    module = _load_module()
    row = {
        "q_case": "synthetic",
        "matsubara_n": 1,
        "q_scale": 1.0,
        "kappa_m_inv": 2.0e7,
        "reflection_TE_TM": np.array([[-0.02, 0.001], [0.002, 0.3]], dtype=complex),
    }
    data = {
        "stage": "Stage 5.6",
        "diagnostic_status": {"stage5_6_status": status},
        "converted_results": [row],
    }
    path.write_text(json.dumps(module.to_jsonable(data)), encoding="utf-8")


def test_round_trip_factor():
    assert round_trip_factor(2.0e7, 100.0e-9) == pytest.approx(np.exp(-4.0))


def test_trace_log_matrix_order():
    r1 = np.array([[0.2, 0.4], [0.0, -0.1]], dtype=complex)
    r2 = np.array([[0.3, 0.0], [0.5, 0.1]], dtype=complex)
    u2 = round_trip_factor(2.0e7, 100.0e-9)
    expected = np.eye(2, dtype=complex) - u2 * (r1 @ r2)
    reverse = np.eye(2, dtype=complex) - u2 * (r2 @ r1)
    actual = trace_log_matrix(r1, r2, 2.0e7, 100.0e-9)
    np.testing.assert_allclose(actual, expected)
    assert not np.allclose(actual, reverse)


def test_zero_reflection_integrand_zero():
    zero = np.zeros((2, 2), dtype=complex)
    np.testing.assert_allclose(trace_log_matrix(zero, zero, 2.0e7, 100.0e-9), np.eye(2))
    assert trace_log_integrand(zero, zero, 2.0e7, 100.0e-9) == pytest.approx(0.0)


def test_zero_sheet_integrand_zero():
    reflection = scalar_sheet_te_tm_reflection(0.0, eta_L=0.4, eta_T=2.0)
    assert np.allclose(reflection, 0.0)
    assert trace_log_integrand(reflection, reflection, 2.0e7, 100.0e-9) == pytest.approx(0.0)


def test_isotropic_identical_sheets_formula():
    reflection = scalar_sheet_te_tm_reflection(0.25, eta_L=0.4, eta_T=2.0)
    u2 = round_trip_factor(2.0e7, 100.0e-9)
    r_ss = reflection[0, 0]
    r_pp = reflection[1, 1]
    expected = np.log(1.0 - u2 * r_ss**2) + np.log(1.0 - u2 * r_pp**2)
    actual = trace_log_integrand(reflection, reflection, 2.0e7, 100.0e-9)
    assert actual == pytest.approx(expected)


def test_large_separation_limit():
    reflection = np.array([[0.2, 0.03], [0.01, -0.1]], dtype=complex)
    assert abs(trace_log_integrand(reflection, reflection, 2.0e7, 1.0)) < 1e-12


def test_isotropic_angle_independence():
    reflection = scalar_sheet_te_tm_reflection(0.25, eta_L=0.4, eta_T=2.0)
    values = []
    for theta in (0.0, np.pi / 5.0, np.pi / 2.0):
        _ = lab_q_to_crystal_q(np.array([1.0, 0.0]), theta)
        values.append(trace_log_integrand(reflection, reflection, 2.0e7, 100.0e-9))
    assert max(abs(value - values[0]) for value in values) < 1e-14


def test_rotation_convention_q_lab_to_crystal():
    np.testing.assert_allclose(lab_q_to_crystal_q(np.array([3.0, 0.0]), np.pi / 2.0), np.array([0.0, -3.0]), atol=1e-14)


def test_reject_failed_stage5_6_by_default(tmp_path):
    module = _load_module()
    input_json = tmp_path / "bad_stage5_6.json"
    _synthetic_stage5_6(input_json, status="FAILED")
    with pytest.raises(ValueError, match="STAGE5_6"):
        module.run_audit(input_json, separation_nm=100.0, max_representative_rows=6, require_stage5_6_passed=True)


def test_script_with_synthetic_stage5_6_input(tmp_path):
    input_json = tmp_path / "stage5_6.json"
    output_json = tmp_path / "stage5_7.json"
    output_md = tmp_path / "stage5_7.md"
    _synthetic_stage5_6(input_json)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-json",
            str(input_json),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
    )
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["diagnostic_status"]["stage5_7_status"] == "STAGE5_7_PRE_LIFSHITZ_READINESS_PASSED"
    assert data["representative_real_stage5_6_integrand_checks"]
    assert output_md.exists()


def test_no_response_or_casimir_imports():
    imported_modules: list[str] = []
    for path in (HELPER, SCRIPT):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module)
    banned = ("response", "casimir", "energy", "force", "torque")
    assert not any(any(word in module.lower() for word in banned) for module in imported_modules)


def test_no_g_symbol():
    text = SCRIPT.read_text(encoding="utf-8") + "\n" + DOC.read_text(encoding="utf-8")
    assert " g " not in text
    assert '"g"' not in text
