from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys

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


def _fake_row(module, *, fermi_window: float, max_norm: float, q_scale: float = 1.0) -> dict:
    row = {
        "temperature_K": 30.0,
        "matsubara_index": 1,
        "omega_eV": 0.016,
        "q_case": "q_diag_pos",
        "q_scale": q_scale,
        "q_model": [0.02 * q_scale, 0.013 * q_scale],
        "adaptive_level": 1,
        "gauss_order": 2,
        "fermi_window_eV": fermi_window,
        "coarse_grid": 8,
        "left_norm": max_norm,
        "right_norm": max_norm,
        "max_corrected_norm": max_norm,
        "left_density_source_abs": 0.0,
        "left_spatial_source_norm": max_norm,
        "right_density_observable_abs": 0.0,
        "right_spatial_observable_norm": max_norm,
        "left_longitudinal_abs": max_norm,
        "left_transverse_abs": 0.0,
        "right_longitudinal_abs": max_norm,
        "right_transverse_abs": 0.0,
        "num_cells_total": 1,
        "num_cells_refined": 0,
        "num_quadrature_points": 4,
        "runtime_seconds": 0.0,
        "status": "CLOSED" if max_norm < 1e-6 else "NOT_CLOSED",
    }
    row["case_key"] = module.case_key(row)
    return row


def test_resume_filters_out_inactive_existing_cases(module, tmp_path):
    output_json = tmp_path / "scan.json"
    output_md = tmp_path / "scan.md"
    old_bad = _fake_row(module, fermi_window=0.03, max_norm=1e-2)
    active_good = _fake_row(module, fermi_window=0.05, max_norm=2e-7)
    output_json.write_text(json.dumps({"scan_results": [old_bad, active_good]}), encoding="utf-8")

    data = module.run_scan(
        preset="quick",
        coarse_grid=8,
        adaptive_levels=[1],
        gauss_orders=[2],
        fermi_windows=[0.05],
        q_scales=[1.0],
        workers=1,
        output_json=output_json,
        output_md=output_md,
        checkpoint_jsonl=tmp_path / "scan.jsonl",
        resume=True,
    )

    assert data["summary_statistics"]["num_completed_cases"] == 1
    assert data["summary_statistics"]["max_corrected_norm_global"] == active_good["max_corrected_norm"]
    assert data["summary_statistics"]["num_not_closed"] == 0
    assert all(row["fermi_window_eV"] != 0.03 for row in data["scan_results"])
    assert all(row["fermi_window_eV"] != 0.03 for row in data["worst_cases"]["top_10_largest_max_corrected_norm"])
    assert data["filtering"]["ignored_existing_case_count"] == 1
    assert data["filtering"]["loaded_existing_active_case_count"] == 1


def test_filter_existing_to_active_grid_no_compute(module, tmp_path, monkeypatch):
    output_json = tmp_path / "scan.json"
    output_md = tmp_path / "scan.md"
    output_json.write_text(
        json.dumps(
            {
                "scan_results": [
                    _fake_row(module, fermi_window=0.03, max_norm=1e-2),
                    _fake_row(module, fermi_window=0.05, max_norm=2e-7),
                ]
            }
        ),
        encoding="utf-8",
    )

    def fail_worker(_case):
        raise AssertionError("filter-only mode must not compute response")

    monkeypatch.setattr(module, "_worker", fail_worker)
    data = module.run_scan(
        preset="quick",
        coarse_grid=8,
        adaptive_levels=[1],
        gauss_orders=[2],
        fermi_windows=[0.05],
        q_scales=[1.0],
        workers=1,
        output_json=output_json,
        output_md=output_md,
        checkpoint_jsonl=tmp_path / "scan.jsonl",
        filter_existing_to_active_grid_only=True,
    )
    assert data["run_mode"] == "filter_existing_to_active_grid"
    assert len(data["scan_results"]) == 1
    assert data["scan_results"][0]["fermi_window_eV"] == 0.05


def test_fresh_ignores_existing_results(module, tmp_path):
    output_json = tmp_path / "scan.json"
    output_md = tmp_path / "scan.md"
    output_json.write_text(
        json.dumps({"scan_results": [_fake_row(module, fermi_window=0.05, max_norm=1e-2)]}),
        encoding="utf-8",
    )
    data = module.run_scan(
        preset="quick",
        coarse_grid=8,
        adaptive_levels=[1],
        gauss_orders=[2],
        fermi_windows=[0.05],
        q_scales=[1.0],
        workers=1,
        output_json=output_json,
        output_md=output_md,
        checkpoint_jsonl=tmp_path / "scan.jsonl",
        fresh=True,
    )
    assert data["filtering"]["loaded_existing_case_count"] == 0
    assert data["filtering"]["loaded_existing_active_case_count"] == 0
    assert data["filtering"]["fresh_mode"] is True


def test_no_003_clean_summary(module, tmp_path):
    output_json = tmp_path / "scan.json"
    output_md = tmp_path / "scan.md"
    output_json.write_text(
        json.dumps(
            {
                "scan_results": [
                    _fake_row(module, fermi_window=0.03, max_norm=1e-2),
                    _fake_row(module, fermi_window=0.05, max_norm=2e-7),
                ]
            }
        ),
        encoding="utf-8",
    )
    data = module.run_scan(
        preset="quick",
        coarse_grid=8,
        adaptive_levels=[1],
        gauss_orders=[2],
        fermi_windows=[0.05],
        q_scales=[1.0],
        workers=1,
        output_json=output_json,
        output_md=output_md,
        checkpoint_jsonl=tmp_path / "scan.jsonl",
        filter_existing_to_active_grid_only=True,
    )
    assert all(float(row["fermi_window_eV"]) != 0.03 for row in data["scan_results"])
    assert all(
        float(row["fermi_window_eV"]) != 0.03
        for row in data["worst_cases"]["top_10_largest_max_corrected_norm"]
    )


def test_mutually_exclusive_fresh_and_resume(module, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["stage4_20", "--fresh", "--resume"],
    )
    with pytest.raises(SystemExit):
        module.parse_args()
