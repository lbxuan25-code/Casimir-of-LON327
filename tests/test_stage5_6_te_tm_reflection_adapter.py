from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from lno327.material_structure import LNO327_THIN_FILM_SLAO_IN_PLANE
from lno327.reflection_input import (
    sigma_tilde_xy_to_te_tm_reflection_matrix,
    tangential_electric_LT_to_TE_TM,
    tangential_electric_reflection_matrix_LT,
)

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "src" / "lno327" / "reflection_input.py"
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage5_6_te_tm_reflection_adapter.py"
DOC = ROOT / "docs" / "notes" / "stage5_6_te_tm_reflection_adapter_zh.md"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage5_6_adapter", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _direct(sigma: np.ndarray | float):
    ax = LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m
    ay = LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_y_m
    if np.isscalar(sigma):
        sigma = float(sigma) * np.eye(2, dtype=complex)
    return sigma_tilde_xy_to_te_tm_reflection_matrix(np.asarray(sigma, dtype=complex), 0.02, 0.0, 0.02, ax, ay)


def _synthetic_stage5_5b(path: Path, *, status: str = "STAGE5_5B_REFLECTION_INPUT_FORMATTER_PASSED") -> None:
    module = _load_module()
    direct = _direct(np.array([[0.1, 0.01], [0.01, 0.2]], dtype=complex))
    row = {
        "q_case": "qx",
        "matsubara_n": 1,
        "q_scale": 1.0,
        "omega_eV": direct["omega_eV"],
        "q_model_x": direct["q_model_x"],
        "q_model_y": direct["q_model_y"],
        "q_model_source": "synthetic",
        "sigma_tilde_xy_matrix": direct["sigma_tilde_xy_matrix"],
        "sigma_tilde_LT_matrix": direct["sigma_tilde_LT_matrix"],
        "reflection_tangential_E_LT": direct["reflection_tangential_E_LT"],
    }
    data = {
        "stage": "Stage 5.5b",
        "diagnostic_status": {"stage5_5b_status": status},
        "reflection_input_results": [row],
    }
    path.write_text(json.dumps(module.to_jsonable(data)), encoding="utf-8")


def test_adapter_index_formula():
    sample = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=complex)
    expected = np.array([[4.0, 3.0], [-2.0, -1.0]], dtype=complex)
    np.testing.assert_allclose(tangential_electric_LT_to_TE_TM(sample), expected)


def test_zero_sheet_reflection_zero():
    direct = _direct(np.zeros((2, 2), dtype=complex))
    np.testing.assert_allclose(direct["reflection_tangential_E_LT"], 0.0)
    np.testing.assert_allclose(direct["reflection_TE_TM"], 0.0)


def test_isotropic_scalar_no_mixing():
    direct = _direct(0.2)
    assert abs(direct["reflection_TE_TM"][0, 1]) < 1e-14
    assert abs(direct["reflection_TE_TM"][1, 0]) < 1e-14


def test_scalar_limit_formulas():
    sigma = 0.2
    direct = _direct(sigma)
    y0 = direct["vacuum_admittance_Y0_LT"]
    r_ss = -sigma / (2.0 * y0[1, 1] + sigma)
    r_pp = sigma / (2.0 * y0[0, 0] + sigma)
    assert direct["reflection_TE_TM"][0, 0] == pytest.approx(r_ss)
    assert direct["reflection_TE_TM"][1, 1] == pytest.approx(r_pp)


def test_strong_sheet_limit():
    direct = _direct(1.0e9)
    assert direct["reflection_TE_TM"][0, 0].real == pytest.approx(-1.0, abs=1e-5)
    assert direct["reflection_TE_TM"][1, 1].real == pytest.approx(1.0, abs=1e-5)


def test_weak_sheet_limit():
    sigma = 1.0e-9
    direct = _direct(sigma)
    y0 = direct["vacuum_admittance_Y0_LT"]
    expected = np.array([[-sigma / (2.0 * y0[1, 1]), 0.0], [0.0, sigma / (2.0 * y0[0, 0])]], dtype=complex)
    assert np.linalg.norm(direct["reflection_TE_TM"] - expected) / np.linalg.norm(expected) < 1e-6


def test_symmetric_offdiag_mixing_retained():
    module = _load_module()
    sigma_lt = np.array([[0.1, 0.03], [0.03, 0.2]], dtype=complex)
    reflection = tangential_electric_reflection_matrix_LT(sigma_lt, 3.0e13, 2.0e6)
    te_tm = tangential_electric_LT_to_TE_TM(reflection)
    assert abs(te_tm[0, 1]) > 0.0 or abs(te_tm[1, 0]) > 0.0
    assert module.classify_offdiag_marker(sigma_lt) == "symmetric_finite_q_mixing"


def test_hall_like_marker_diagnostic():
    module = _load_module()
    sigma_lt = np.array([[0.1, 0.02], [-0.02, 0.1]], dtype=complex)
    assert module.classify_offdiag_marker(sigma_lt) == "antisymmetric_marker"


def test_script_with_synthetic_stage5_5b_input(tmp_path):
    input_json = tmp_path / "stage5_5b.json"
    output_json = tmp_path / "stage5_6.json"
    output_md = tmp_path / "stage5_6.md"
    _synthetic_stage5_5b(input_json)
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
    assert data["diagnostic_status"]["stage5_6_status"] == "STAGE5_6_TE_TM_ADAPTER_PASSED"
    assert data["converted_results"][0]["reflection_TE_TM"]
    assert output_md.exists()


def test_reject_failed_stage5_5b_by_default(tmp_path):
    module = _load_module()
    input_json = tmp_path / "bad_stage5_5b.json"
    _synthetic_stage5_5b(input_json, status="FAILED")
    with pytest.raises(ValueError, match="STAGE5_5B"):
        module.run_adapter(input_json, allow_q_zero=False, require_stage5_5b_passed=True)


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
