from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from lno327.response_conventions import e2_over_hbar_siemens, four_pi_alpha
from lno327.material_structure import LNO327_THIN_FILM_SLAO_IN_PLANE

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage5_4b_convert_model_conductivity_to_si_sheet.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage5_4b_convert", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _synthetic_input(path: Path, *, status: str = "MONITOR") -> None:
    data = {
        "stage": "Stage synthetic",
        "diagnostic_status": {"conductivity_sanity_status": "CONDUCTIVITY_SANITY_MONITOR_OFFDIAG"},
        "scan_results": [
            {
                "matsubara_index": 1,
                "q_case": "q_diag_pos",
                "q_scale": 1.0,
                "status": status,
                "sigma_xx_model": {"real": 1.0, "imag": 0.0, "abs": 1.0},
                "sigma_xy_model": {"real": 2.0, "imag": 0.0, "abs": 2.0},
                "sigma_yx_model": {"real": 3.0, "imag": 0.0, "abs": 3.0},
                "sigma_yy_model": {"real": 4.0, "imag": 0.0, "abs": 4.0},
                "relative_offdiag_norm": float(np.sqrt(13.0) / np.sqrt(17.0)),
            }
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_parse_complex_component():
    module = _load_module()
    assert module.parse_complex_component({"real": 1.0, "imag": 2.0, "abs": 5.0}) == 1.0 + 2.0j


def test_convert_single_synthetic_row_square_lattice():
    module = _load_module()
    row = {
        "sigma_xx_model": 1.0 + 0.0j,
        "sigma_xy_model": 2.0 + 0.0j,
        "sigma_yx_model": 3.0 + 0.0j,
        "sigma_yy_model": 4.0 + 0.0j,
    }
    convention = module.SheetConductivityUnitConvention(
        LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m,
        LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_y_m,
        LNO327_THIN_FILM_SLAO_IN_PLANE.unit_cell_area_m2,
    )
    converted = module.convert_row(row, convention)
    np.testing.assert_allclose(converted["sigma_SI_sheet_matrix"], e2_over_hbar_siemens() * np.array([[1, 2], [3, 4]], dtype=complex))
    np.testing.assert_allclose(converted["sigma_tilde_matrix"], four_pi_alpha() * np.array([[1, 2], [3, 4]], dtype=complex), rtol=1e-10)


def test_relative_structure_preserved():
    module = _load_module()
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=complex)
    assert module.relative_offdiag_norm(matrix) == pytest.approx(module.relative_offdiag_norm(four_pi_alpha() * matrix))


def test_stage5_4b_script_with_synthetic_input(tmp_path):
    input_json = tmp_path / "input.json"
    output_json = tmp_path / "output.json"
    output_md = tmp_path / "output.md"
    _synthetic_input(input_json, status="MONITOR")
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
    assert data["diagnostic_status"]["stage5_4b_status"] == "STAGE5_4B_CONDUCTIVITY_CONVERSION_PASSED"
    assert "sigma_tilde_xx" in data["converted_results"][0]
    assert output_md.exists()


def test_reject_fail_input_by_default(tmp_path):
    module = _load_module()
    input_json = tmp_path / "input_fail.json"
    _synthetic_input(input_json, status="FAIL")
    convention = module.SheetConductivityUnitConvention(3.754e-10, 3.754e-10, (3.754e-10) ** 2)
    with pytest.raises(ValueError, match="FAIL"):
        module.run_conversion(input_json, convention)


def test_allow_monitor_input(tmp_path):
    module = _load_module()
    input_json = tmp_path / "input_monitor.json"
    _synthetic_input(input_json, status="MONITOR")
    convention = module.SheetConductivityUnitConvention(3.754e-10, 3.754e-10, (3.754e-10) ** 2)
    data = module.run_conversion(input_json, convention)
    assert data["input"]["allow_monitor_input"] is True
    assert data["summary"]["num_cases"] == 1


def test_no_reflection_or_casimir_imports():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)
    assert not any("reflection" in module.lower() for module in imported_modules)
    assert not any("casimir" in module.lower() for module in imported_modules)


def test_no_heavy_response_run():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "run_case" not in text
    assert "integrate_physical_components" not in text
    assert "normal_physical_density_current_response" not in text
