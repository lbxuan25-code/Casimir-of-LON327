from __future__ import annotations

import importlib.util
import inspect
import ast
from pathlib import Path

import numpy as np
import pytest

from lno327.ward_response import physical_ward_residuals, physical_ward_residuals_legacy

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage4_18_corrected_full_response_ward_validation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage4_18_corrected_ward", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def small_validation():
    module = _load_module()
    return module.run_validation(coarse_grid=8, max_refinement_level=1, gauss_order=2, q_scales=[1.0])


def test_corrected_right_residual_uses_minus_q_contraction():
    response = np.array(
        [
            [1.0 + 0.1j, 2.0 - 0.2j, -0.5 + 0.3j],
            [0.7 + 0.4j, -1.0 + 0.2j, 0.8 - 0.1j],
            [0.2 - 0.6j, 0.5 + 0.5j, 1.3 - 0.7j],
        ],
        dtype=complex,
    )
    omega_eV = 0.03
    q = np.array([0.02, 0.013], dtype=float)
    left, right = physical_ward_residuals(response, omega_eV, q)
    expected_left = 1j * omega_eV * response[0, :] + q[0] * response[1, :] + q[1] * response[2, :]
    expected_right = 1j * omega_eV * response[:, 0] - q[0] * response[:, 1] - q[1] * response[:, 2]
    np.testing.assert_allclose(left, expected_left)
    np.testing.assert_allclose(right, expected_right)


def test_legacy_right_residual_uses_plus_q_contraction():
    response = np.array(
        [
            [1.0 + 0.1j, 2.0 - 0.2j, -0.5 + 0.3j],
            [0.7 + 0.4j, -1.0 + 0.2j, 0.8 - 0.1j],
            [0.2 - 0.6j, 0.5 + 0.5j, 1.3 - 0.7j],
        ],
        dtype=complex,
    )
    omega_eV = 0.03
    q = np.array([0.02, 0.013], dtype=float)
    left, right = physical_ward_residuals_legacy(response, omega_eV, q)
    expected_left = 1j * omega_eV * response[0, :] + q[0] * response[1, :] + q[1] * response[2, :]
    expected_right = 1j * omega_eV * response[:, 0] + q[0] * response[:, 1] + q[1] * response[:, 2]
    np.testing.assert_allclose(left, expected_left)
    np.testing.assert_allclose(right, expected_right)


def test_fast_run_outputs_required_top_level_fields(small_validation):
    assert small_validation["stage"] == "Stage 4.18"
    assert "corrected_validation_results" in small_validation
    assert "legacy_comparison_results" in small_validation
    assert "diagnostic_status" in small_validation
    assert "boundary" in small_validation


def test_fast_run_contains_corrected_and_legacy_residual_fields(small_validation):
    row = small_validation["corrected_validation_results"][0]
    for key in (
        "left_norm",
        "right_norm",
        "max_corrected_norm",
        "legacy_right_norm",
        "left_density_source_abs",
        "left_spatial_source_norm",
        "right_density_observable_abs",
        "right_spatial_observable_norm",
        "left_longitudinal_abs",
        "left_transverse_abs",
        "right_longitudinal_abs",
        "right_transverse_abs",
        "num_cells_total",
        "num_cells_refined",
        "num_quadrature_points",
    ):
        assert key in row
    assert small_validation["legacy_comparison_results"]


def test_boundary_fields_are_all_true(small_validation):
    for value in small_validation["boundary"].values():
        assert value is True


def test_docstring_contains_source_side_identity():
    doc = inspect.getdoc(physical_ward_residuals)
    assert doc is not None
    assert "G_+^{-1}-G_-^{-1}=iΩρ-q_iV_i" in doc


def test_script_does_not_import_downstream_casimir_reflection_modules():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)
    assert not any("reflection" in module for module in imported_modules)
    assert not any("casimir" in module.lower() for module in imported_modules)
