from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from lno327.response_conventions import (
    SheetConductivityUnitConvention,
    conductivity_unit_conversion_metadata,
    e2_over_hbar_siemens,
    four_pi_alpha,
    model_to_dimensionless_sheet_conductivity,
    model_to_si_sheet_conductivity,
    sheet_geometry_factor_tensor,
    z0_e2_over_hbar,
)
from lno327.material_structure import LNO327_THIN_FILM_SLAO_IN_PLANE

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "src" / "lno327" / "response_conventions.py"
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage5_4a_conductivity_unit_conversion.py"


def test_square_lattice_model_to_si():
    a = 3.85e-10
    convention = SheetConductivityUnitConvention(a, a)
    sigma_model = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=complex)
    np.testing.assert_allclose(model_to_si_sheet_conductivity(sigma_model, convention), e2_over_hbar_siemens() * sigma_model)


def test_rectangular_geometry_tensor():
    a = 3.85e-10
    convention = SheetConductivityUnitConvention(2.0 * a, a, unit_cell_area_m2=2.0 * a * a)
    np.testing.assert_allclose(sheet_geometry_factor_tensor(convention), np.array([[2.0, 1.0], [1.0, 0.5]]))


def test_model_to_dimensionless_square():
    a = 3.85e-10
    convention = SheetConductivityUnitConvention(a, a)
    sigma_model = np.array([[1.0, -0.2], [0.3, 4.0]], dtype=complex)
    np.testing.assert_allclose(
        model_to_dimensionless_sheet_conductivity(sigma_model, convention),
        four_pi_alpha() * sigma_model,
        rtol=1e-10,
        atol=1e-15,
    )


def test_z0_e2_over_hbar_equals_four_pi_alpha():
    assert abs(z0_e2_over_hbar() - four_pi_alpha()) / four_pi_alpha() < 1e-10


def test_invalid_shape_raises():
    convention = SheetConductivityUnitConvention(3.85e-10, 3.85e-10)
    with pytest.raises(ValueError, match=r"shape \(2, 2\)"):
        model_to_si_sheet_conductivity(np.ones((3, 3)), convention)


def test_invalid_lattice_constants_raise():
    with pytest.raises(ValueError, match="lattice_a_x_m"):
        SheetConductivityUnitConvention(0.0, 3.85e-10)
    with pytest.raises(ValueError, match="lattice_a_y_m"):
        SheetConductivityUnitConvention(3.85e-10, -1.0)
    with pytest.raises(ValueError, match="unit_cell_area_m2"):
        SheetConductivityUnitConvention(3.85e-10, 3.85e-10, unit_cell_area_m2=0.0)


def test_complex_dtype_preserved():
    convention = SheetConductivityUnitConvention(3.85e-10, 3.85e-10)
    sigma_model = np.array([[1.0 + 2.0j, 0.0], [0.0, 3.0 - 4.0j]], dtype=complex)
    assert np.iscomplexobj(model_to_si_sheet_conductivity(sigma_model, convention))


def test_metadata_fields():
    convention = SheetConductivityUnitConvention(3.85e-10, 3.85e-10)
    metadata = conductivity_unit_conversion_metadata(convention)
    assert "formula_model_to_si" in metadata
    assert metadata["normalization"] == "bilayer-normalized 2D sheet conductivity"
    assert metadata["dimensionless_symbol"] == "sigma_tilde_ij = Z0 * sigma_SI_sheet_ij"
    assert metadata["bulk_3d_conductivity"] is False
    assert metadata["single_layer_conductivity"] is False


def test_validation_script_outputs(tmp_path):
    output_json = tmp_path / "stage5_4a.json"
    output_md = tmp_path / "stage5_4a.md"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
    )
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert output_json.exists()
    assert output_md.exists()
    assert data["diagnostic_status"]["stage5_4a_status"] == "STAGE5_4A_CONDUCTIVITY_UNIT_CONVERSION_PASSED"


def test_default_lattice_is_thin_film_config(tmp_path):
    output_json = tmp_path / "stage5_4a.json"
    output_md = tmp_path / "stage5_4a.md"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output-json", str(output_json), "--output-md", str(output_md)],
        check=True,
    )
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["geometry"]["lattice_a_x_m"] == 3.754e-10
    assert data["geometry"]["lattice_a_y_m"] == 3.754e-10
    assert data["geometry"]["is_placeholder"] is False


def test_material_structure_config():
    assert LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m == 3.754e-10
    assert LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_y_m == 3.754e-10
    assert LNO327_THIN_FILM_SLAO_IN_PLANE.is_placeholder is False


def test_no_reflection_or_casimir_imports():
    imported_modules: list[str] = []
    for path in (HELPER, SCRIPT):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module)
    assert not any("reflection" in module.lower() for module in imported_modules)
    assert not any("casimir" in module.lower() for module in imported_modules)
