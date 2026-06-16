from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from lno327.constants import C0, EV_TO_J, HBAR
from lno327.reflection_input import (
    model_q_to_si_wavevector,
    omega_eV_to_xi_si,
    rotate_sigma_tilde_xy_to_lt,
    tangential_electric_reflection_matrix_LT,
    vacuum_admittance_LT,
    vacuum_kappa,
    xy_to_lt_rotation,
)

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "src" / "lno327" / "reflection_input.py"
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage5_5b_reflection_input_tensor_formatter.py"
DOC = ROOT / "docs" / "notes" / "stage5_5b_reflection_input_tensor_formatter_zh.md"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage5_5b_formatter", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _synthetic_stage5_4b(path: Path, *, status: str = "STAGE5_4B_CONDUCTIVITY_CONVERSION_PASSED") -> None:
    data = {
        "stage": "Stage 5.4b",
        "diagnostic_status": {"stage5_4b_status": status},
        "converted_results": [
            {
                "q_case": "qx",
                "matsubara_index": 1,
                "q_scale": 1.0,
                "omega_eV": 0.02,
                "q_model": [0.02, 0.0],
                "sigma_tilde_xx": {"real": 0.1, "imag": 0.0, "abs": 0.1},
                "sigma_tilde_xy": {"real": 0.01, "imag": 0.0, "abs": 0.01},
                "sigma_tilde_yx": {"real": 0.01, "imag": 0.0, "abs": 0.01},
                "sigma_tilde_yy": {"real": 0.2, "imag": 0.0, "abs": 0.2},
            }
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_model_q_to_si_wavevector():
    qx, qy, q = model_q_to_si_wavevector(0.02, 0.01, 2.0e-10, 4.0e-10)
    assert qx == pytest.approx(1.0e8)
    assert qy == pytest.approx(2.5e7)
    assert q == pytest.approx(np.hypot(qx, qy))


def test_omega_eV_to_xi_si():
    assert omega_eV_to_xi_si(0.02) == pytest.approx(0.02 * EV_TO_J / HBAR)


def test_vacuum_kappa():
    q = 2.0e7
    xi = 3.0e13
    assert vacuum_kappa(q, xi) == pytest.approx(np.sqrt(q**2 + (xi / C0) ** 2))


def test_xy_to_lt_rotation_qx():
    np.testing.assert_allclose(xy_to_lt_rotation(1.0, 0.0), np.eye(2))


def test_xy_to_lt_rotation_qy():
    np.testing.assert_allclose(xy_to_lt_rotation(0.0, 1.0), np.array([[0.0, 1.0], [-1.0, 0.0]]))


def test_rotate_tensor_qy():
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=complex)
    np.testing.assert_allclose(rotate_sigma_tilde_xy_to_lt(tensor, 0.0, 1.0), np.array([[4.0, -3.0], [-2.0, 1.0]], dtype=complex))


def test_isotropic_scalar_no_mixing():
    sigma = 0.2 * np.eye(2, dtype=complex)
    sigma_lt = rotate_sigma_tilde_xy_to_lt(sigma, 1.0, 1.0)
    xi, kappa = 3.0e13, 2.0e6
    reflection = tangential_electric_reflection_matrix_LT(sigma_lt, xi, kappa)
    y0 = vacuum_admittance_LT(xi, kappa)
    expected = np.diag([-0.2 / (2.0 * y0[0, 0] + 0.2), -0.2 / (2.0 * y0[1, 1] + 0.2)])
    np.testing.assert_allclose(sigma_lt, sigma)
    np.testing.assert_allclose(reflection, expected)
    assert abs(reflection[0, 1]) < 1e-14


def test_diagonal_LT_no_mixing():
    reflection = tangential_electric_reflection_matrix_LT(np.diag([0.1, 0.2]).astype(complex), 3.0e13, 2.0e6)
    assert abs(reflection[0, 1]) < 1e-14
    assert abs(reflection[1, 0]) < 1e-14


def test_offdiag_LT_retains_mixing():
    reflection = tangential_electric_reflection_matrix_LT(np.array([[0.1, 0.03], [0.03, 0.2]], dtype=complex), 3.0e13, 2.0e6)
    assert abs(reflection[0, 1]) > 0.0 or abs(reflection[1, 0]) > 0.0


def test_weak_sheet_limit():
    sigma = 1e-6 * np.array([[1.0, 0.2], [0.2, 2.0]], dtype=complex)
    xi, kappa = 3.0e13, 2.0e6
    reflection = tangential_electric_reflection_matrix_LT(sigma, xi, kappa)
    expected = -0.5 * np.linalg.solve(vacuum_admittance_LT(xi, kappa), sigma)
    assert np.linalg.norm(reflection - expected) / np.linalg.norm(expected) < 1e-5


def test_script_reads_stage5_4b_or_synthetic(tmp_path):
    input_json = tmp_path / "stage5_4b.json"
    output_json = tmp_path / "stage5_5b.json"
    output_md = tmp_path / "stage5_5b.md"
    _synthetic_stage5_4b(input_json)
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
    assert data["diagnostic_status"]["stage5_5b_status"] == "STAGE5_5B_REFLECTION_INPUT_FORMATTER_PASSED"
    assert data["reflection_input_results"][0]["Q_m_inv"] > 0.0
    assert output_md.exists()


def test_reject_non_passed_stage5_4b_by_default(tmp_path):
    module = _load_module()
    input_json = tmp_path / "bad_stage5_4b.json"
    _synthetic_stage5_4b(input_json, status="FAILED")
    with pytest.raises(ValueError, match="STAGE5_4B"):
        module.run_formatter(input_json, allow_q_zero=False, require_stage5_4b_passed=True)


def test_no_reflection_casimir_imports():
    imported_modules: list[str] = []
    for path in (HELPER, SCRIPT):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module)
    assert not any("casimir" in module.lower() for module in imported_modules)


def test_no_g_symbol():
    text = SCRIPT.read_text(encoding="utf-8") + "\n" + DOC.read_text(encoding="utf-8")
    assert " g " not in text
    assert '"g"' not in text
