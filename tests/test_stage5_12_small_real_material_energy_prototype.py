from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from lno327.material_energy_prototype import (
    integrate_small_real_material_energy_prototype,
    jsonable_complex_matrix_to_numpy,
    load_stage5_11_reflection_grid,
    prototype_polar_weights,
)

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "src" / "lno327" / "material_energy_prototype.py"


def _matrix_json(value: np.ndarray):
    return [[{"re": float(x.real), "im": float(x.imag), "abs": float(abs(x))} for x in row] for row in value]


def _synthetic_grid(path: Path, *, status: str = "STAGE5_11_REAL_MATERIAL_REFLECTION_GRID_PROTOTYPE_PASSED", zero: bool = False) -> None:
    rows = []
    for n in (1, 2):
        for q_nm in (0.05, 0.10):
            for phi in (0.0, 90.0):
                reflection = np.zeros((2, 2), dtype=complex) if zero else np.diag([-0.02, 0.3]).astype(complex)
                rows.append(
                    {
                        "status": "PASS",
                        "n": n,
                        "temperature_K": 10.0,
                        "Q_nm_inv": q_nm,
                        "Q_m_inv": q_nm * 1e9,
                        "phi_deg": phi,
                        "phi_rad": np.deg2rad(phi),
                        "kappa_m_inv": q_nm * 1e9 + n * 1e5,
                        "reflection_TE_TM": _matrix_json(reflection),
                    }
                )
    path.write_text(
        json.dumps({"stage": "Stage 5.11", "diagnostic_status": {"stage5_11_status": status}, "point_results": rows}),
        encoding="utf-8",
    )


def test_complex_deserialization():
    matrix = jsonable_complex_matrix_to_numpy([[{"re": 1.0, "im": 2.0, "abs": 3.0}]])
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == 1.0 + 2.0j


def test_polar_weights_finite_and_nonnegative():
    weights = prototype_polar_weights(np.array([1.0, 2.0]), np.array([0.0, np.pi / 2.0]))["weights"]
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0.0)


def test_input_status_check(tmp_path):
    path = tmp_path / "stage5_11.json"
    _synthetic_grid(path)
    data = load_stage5_11_reflection_grid(path)
    assert len(data["point_results"]) == 8


def test_reject_failed_stage5_11_input(tmp_path):
    path = tmp_path / "bad_stage5_11.json"
    _synthetic_grid(path, status="FAILED")
    with pytest.raises(ValueError, match="STAGE5_11"):
        load_stage5_11_reflection_grid(path)


def test_finite_prototype_energy_using_synthetic_grid(tmp_path):
    path = tmp_path / "stage5_11.json"
    _synthetic_grid(path)
    data = load_stage5_11_reflection_grid(path)
    result = integrate_small_real_material_energy_prototype(data, separation_m=100e-9)
    assert np.isfinite(result["F_proto_over_area_J_m2"].real)
    assert np.isfinite(result["F_proto_over_area_J_m2"].imag)
    assert result["num_points_used"] == 8


def test_zero_reflection_gives_zero_prototype_energy(tmp_path):
    path = tmp_path / "stage5_11_zero.json"
    _synthetic_grid(path, zero=True)
    data = load_stage5_11_reflection_grid(path)
    result = integrate_small_real_material_energy_prototype(data, separation_m=100e-9)
    assert result["F_proto_over_area_J_m2"] == pytest.approx(0.0)


def test_separation_trend_for_simple_synthetic_grid(tmp_path):
    path = tmp_path / "stage5_11.json"
    _synthetic_grid(path)
    data = load_stage5_11_reflection_grid(path)
    values = [integrate_small_real_material_energy_prototype(data, separation_m=d)["F_proto_over_area_J_m2"] for d in (50e-9, 100e-9, 200e-9)]
    assert abs(values[0]) > abs(values[1]) > abs(values[2])


def test_warnings_present(tmp_path):
    path = tmp_path / "stage5_11.json"
    _synthetic_grid(path)
    data = load_stage5_11_reflection_grid(path)
    result = integrate_small_real_material_energy_prototype(data, separation_m=100e-9)
    for key in (
        "n0_excluded",
        "zero_mode_not_included",
        "matsubara_grid_incomplete",
        "angular_grid_sparse",
        "Q_grid_sparse",
        "not_production_quadrature",
        "not_physical_prediction",
    ):
        assert key in result["warnings"]


def test_no_force_torque_production_claim():
    text = HELPER.read_text(encoding="utf-8")
    assert "force" in text
    assert "torque" in text
    assert "production energy" in text


def test_no_response_rerun_import_from_heavy_response_scripts():
    tree = ast.parse(HELPER.read_text(encoding="utf-8"))
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)
    assert not any("stage4_" in module or "ward_response" in module or "conductivity" == module for module in imported_modules)
