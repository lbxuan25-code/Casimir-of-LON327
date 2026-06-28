from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from lno327.casimir_integrand import (
    casimir_integrand_single_point,
    rotate_2x2_te_tm_toy_matrix,
    toy_anisotropic_symmetric_reflection,
    toy_isotropic_reflection,
    toy_zero_reflection,
)
from lno327.lifshitz_readiness import round_trip_factor, trace_log_matrix

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "src" / "lno327" / "casimir_integrand.py"
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage5_8_casimir_integrand_prototype.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage5_8_prototype", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _synthetic_stage5_6(path: Path, *, status: str = "STAGE5_6_TE_TM_ADAPTER_PASSED") -> None:
    module = _load_module()
    data = {
        "stage": "Stage 5.6",
        "diagnostic_status": {"stage5_6_status": status},
        "converted_results": [
            {
                "q_case": "synthetic",
                "matsubara_n": 1,
                "q_scale": 1.0,
                "kappa_m_inv": 2.0e7,
                "reflection_TE_TM": np.array([[-0.02, 0.001], [0.002, 0.3]], dtype=complex),
            }
        ],
    }
    path.write_text(json.dumps(module.to_jsonable(data)), encoding="utf-8")


def test_zero_reflection_integrand_zero():
    zero = toy_zero_reflection()
    assert casimir_integrand_single_point(zero, zero, 2.0e7, 100.0e-9)["logdet_integrand"] == pytest.approx(0.0)


def test_one_zero_plate_integrand_zero():
    zero = toy_zero_reflection()
    nonzero = toy_isotropic_reflection(-0.05, 0.25)
    assert casimir_integrand_single_point(zero, nonzero, 2.0e7, 100.0e-9)["logdet_integrand"] == pytest.approx(0.0)


def test_large_separation_limit():
    reflection = toy_anisotropic_symmetric_reflection(-0.2, 0.3, 0.04)
    assert abs(casimir_integrand_single_point(reflection, reflection, 2.0e7, 1.0)["logdet_integrand"]) < 1e-12


def test_small_separation_larger_magnitude_toy():
    reflection = toy_isotropic_reflection(-0.02, 0.03)
    near = casimir_integrand_single_point(reflection, reflection, 2.0e7, 20.0e-9)["logdet_integrand"]
    far = casimir_integrand_single_point(reflection, reflection, 2.0e7, 200.0e-9)["logdet_integrand"]
    assert abs(near) > abs(far)
    assert near.real < far.real < 0.0


def test_isotropic_identical_sheets_formula():
    reflection = toy_isotropic_reflection(-0.07, 0.21)
    u2 = round_trip_factor(2.0e7, 100.0e-9)
    expected = np.log(1.0 - u2 * reflection[0, 0] ** 2) + np.log(1.0 - u2 * reflection[1, 1] ** 2)
    actual = casimir_integrand_single_point(reflection, reflection, 2.0e7, 100.0e-9)["logdet_integrand"]
    assert actual == pytest.approx(expected)


def test_isotropic_angle_independence():
    reflection = toy_isotropic_reflection(0.12, 0.12)
    values = [
        casimir_integrand_single_point(reflection, rotate_2x2_te_tm_toy_matrix(reflection, theta), 2.0e7, 100.0e-9)[
            "logdet_integrand"
        ]
        for theta in (0.0, np.pi / 8.0, np.pi / 2.0)
    ]
    assert max(abs(value - values[0]) for value in values) < 1e-14


def test_anisotropic_toy_pi_periodicity():
    reflection = toy_anisotropic_symmetric_reflection(-0.12, 0.28, 0.05)
    theta = np.pi / 5.0
    value = casimir_integrand_single_point(reflection, rotate_2x2_te_tm_toy_matrix(reflection, theta), 2.0e7, 100.0e-9)[
        "logdet_integrand"
    ]
    shifted = casimir_integrand_single_point(
        reflection,
        rotate_2x2_te_tm_toy_matrix(reflection, theta + np.pi),
        2.0e7,
        100.0e-9,
    )["logdet_integrand"]
    zero = casimir_integrand_single_point(reflection, rotate_2x2_te_tm_toy_matrix(reflection, 0.0), 2.0e7, 100.0e-9)[
        "logdet_integrand"
    ]
    pi = casimir_integrand_single_point(reflection, rotate_2x2_te_tm_toy_matrix(reflection, np.pi), 2.0e7, 100.0e-9)[
        "logdet_integrand"
    ]
    assert abs(value - shifted) < 1e-14
    assert abs(zero - pi) < 1e-14


def test_matrix_order_direct_entries():
    r1 = np.array([[0.2, 0.4], [0.0, -0.1]], dtype=complex)
    r2 = np.array([[0.3, 0.0], [0.5, 0.1]], dtype=complex)
    u2 = round_trip_factor(2.0e7, 100.0e-9)
    expected = np.eye(2, dtype=complex) - u2 * (r1 @ r2)
    reverse = np.eye(2, dtype=complex) - u2 * (r2 @ r1)
    actual = trace_log_matrix(r1, r2, 2.0e7, 100.0e-9)
    np.testing.assert_allclose(actual, expected)
    assert not np.allclose(actual, reverse)


def test_integrand_complex_output_allowed():
    r1 = np.array([[0.1 + 0.02j, 0.03], [0.0, -0.2]], dtype=complex)
    r2 = np.array([[0.2, 0.01j], [0.04, 0.1]], dtype=complex)
    value = casimir_integrand_single_point(r1, r2, 2.0e7, 100.0e-9)["logdet_integrand"]
    assert isinstance(value, complex)
    passive = casimir_integrand_single_point(toy_isotropic_reflection(-0.07, 0.21), toy_isotropic_reflection(-0.07, 0.21), 2.0e7, 100.0e-9)[
        "logdet_integrand"
    ]
    assert abs(passive.imag) < 1e-14


def test_script_with_synthetic_stage5_6_input(tmp_path):
    input_json = tmp_path / "stage5_6.json"
    output_json = tmp_path / "stage5_8.json"
    output_md = tmp_path / "stage5_8.md"
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
    assert data["diagnostic_status"]["stage5_8_status"] == "STAGE5_8_CASIMIR_INTEGRAND_PROTOTYPE_PASSED"
    assert data["representative_stage5_6_checks"]
    assert output_md.exists()


def test_reject_failed_stage5_6_by_default(tmp_path):
    module = _load_module()
    input_json = tmp_path / "bad_stage5_6.json"
    _synthetic_stage5_6(input_json, status="FAILED")
    with pytest.raises(ValueError, match="STAGE5_6"):
        module.run_prototype(input_json, separation_nm=100.0, max_representative_rows=6, require_stage5_6_passed=True)


def test_no_response_or_casimir_imports():
    imported_modules: list[str] = []
    for path in (HELPER, SCRIPT):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module)
    banned = ("response", "energy", "force", "torque")
    assert not any(any(word in module.lower() for word in banned) for module in imported_modules)


def test_no_g_symbol():
    text = SCRIPT.read_text(encoding="utf-8")
    assert " g " not in text
    assert '"g"' not in text
