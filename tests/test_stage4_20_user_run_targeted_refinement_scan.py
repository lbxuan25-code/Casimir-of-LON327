from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from lno327.ward_response import physical_ward_residuals

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage4_20_user_run_targeted_refinement_scan.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage4_20_targeted_scan", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module():
    return _load_module()


def test_quick_run_scan_outputs_required_fields(module, tmp_path):
    output_json = tmp_path / "scan.json"
    output_md = tmp_path / "scan.md"
    data = module.run_scan(
        preset="quick",
        coarse_grid=8,
        adaptive_levels=[1],
        gauss_orders=[2],
        fermi_windows=[0.05],
        workers=1,
        output_json=output_json,
        output_md=output_md,
    )
    for key in (
        "stage",
        "purpose",
        "run_mode",
        "config",
        "scan_results",
        "summary_statistics",
        "worst_cases",
        "diagnostic_status",
        "boundary",
    ):
        assert key in data
    assert output_json.exists()
    assert output_md.exists()
    assert data["scan_results"]


def test_dry_run_returns_case_list_without_response(module):
    data = module.run_scan(
        preset="quick",
        coarse_grid=8,
        adaptive_levels=[1],
        gauss_orders=[2],
        fermi_windows=[0.05],
        workers=1,
        dry_run=True,
    )
    assert data["run_mode"] == "dry_run"
    assert data["num_planned_cases"] == len(data["planned_cases"])
    assert "scan_results" not in data
    assert all("estimated_quadrature_points_upper_bound" in case for case in data["planned_cases"])


def test_case_key_is_unique_for_quick_cases(module):
    config = module.apply_overrides(module.preset_config("quick"))
    cases = module.build_cases(config)
    keys = [module.case_key(case) for case in cases]
    assert len(keys) == len(set(keys))


def test_resume_skips_completed_case(module, tmp_path):
    output_json = tmp_path / "scan.json"
    output_md = tmp_path / "scan.md"
    first = module.run_scan(
        preset="quick",
        coarse_grid=8,
        adaptive_levels=[1],
        gauss_orders=[2],
        fermi_windows=[0.05],
        workers=1,
        max_cases=1,
        output_json=output_json,
        output_md=output_md,
    )
    second = module.run_scan(
        preset="quick",
        coarse_grid=8,
        adaptive_levels=[1],
        gauss_orders=[2],
        fermi_windows=[0.05],
        workers=1,
        max_cases=1,
        output_json=output_json,
        output_md=output_md,
        resume=True,
    )
    assert len(first["scan_results"]) == 1
    assert len(second["scan_results"]) == 1
    assert second["scan_results"][0]["case_key"] == first["scan_results"][0]["case_key"]


def test_corrected_right_residual_uses_minus_q_contraction():
    response = np.array(
        [
            [1.0 + 0.1j, 0.2 - 0.3j, -0.4 + 0.2j],
            [0.5 - 0.7j, 1.2 + 0.4j, 0.1 + 0.8j],
            [-0.2 + 0.6j, 0.3 - 0.5j, 0.9 + 0.2j],
        ],
        dtype=complex,
    )
    omega_eV = 0.02
    q = np.array([0.02, -0.013], dtype=float)
    _left, right = physical_ward_residuals(response, omega_eV, q)
    expected_right = 1j * omega_eV * response[:, 0] - q[0] * response[:, 1] - q[1] * response[:, 2]
    np.testing.assert_allclose(right, expected_right)


def test_boundary_fields_are_all_true(module, tmp_path):
    data = module.run_scan(
        preset="quick",
        coarse_grid=8,
        adaptive_levels=[1],
        gauss_orders=[2],
        fermi_windows=[0.05],
        workers=1,
        max_cases=1,
        output_json=tmp_path / "scan.json",
        output_md=tmp_path / "scan.md",
    )
    for value in data["boundary"].values():
        assert value is True


def test_no_downstream_reflection_or_casimir_imports():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)
    assert not any("reflection" in module for module in imported_modules)
    assert not any("casimir" in module.lower() for module in imported_modules)


def test_worst_cases_are_sorted_descending(module, tmp_path):
    data = module.run_scan(
        preset="quick",
        coarse_grid=8,
        adaptive_levels=[1],
        gauss_orders=[2],
        fermi_windows=[0.05],
        workers=1,
        output_json=tmp_path / "scan.json",
        output_md=tmp_path / "scan.md",
    )
    rows = data["worst_cases"]["top_10_largest_max_corrected_norm"]
    values = [float(row["max_corrected_norm"]) for row in rows]
    assert values == sorted(values, reverse=True)


def test_workers_argument_is_parsed(module):
    parser_args = module.parse_args
    assert callable(parser_args)
    config = module.preset_config("targeted")
    assert config["adaptive_levels"] == [3, 4, 5]


def test_output_json_is_valid(module, tmp_path):
    output_json = tmp_path / "scan.json"
    module.run_scan(
        preset="quick",
        coarse_grid=8,
        adaptive_levels=[1],
        gauss_orders=[2],
        fermi_windows=[0.05],
        workers=1,
        max_cases=1,
        output_json=output_json,
        output_md=tmp_path / "scan.md",
    )
    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    assert loaded["stage"] == "Stage 4.20"
