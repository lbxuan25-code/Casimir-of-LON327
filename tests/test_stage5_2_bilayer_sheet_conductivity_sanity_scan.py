from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage5_2_bilayer_sheet_conductivity_sanity_scan.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage5_2_bilayer_sheet_conductivity", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_script_imports():
    module = _load_module()
    assert module.JSON_OUTPUT.name == "stage5_2_bilayer_sheet_conductivity_sanity_scan.json"


def test_quick_mode_runs_and_outputs_json_md(tmp_path):
    output_json = tmp_path / "stage5_2.json"
    output_md = tmp_path / "stage5_2.md"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--quick",
            "--workers",
            "2",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
    )
    assert output_json.exists()
    assert output_md.exists()
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["stage"] == "Stage 5.2"
    assert data["scan_results"]


def test_json_top_level_fields_and_boundaries(tmp_path):
    output_json = tmp_path / "stage5_2_dry.json"
    output_md = tmp_path / "stage5_2_dry.md"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--quick",
            "--workers",
            "2",
            "--dry-run",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
    )
    data = json.loads(output_json.read_text(encoding="utf-8"))
    for key in (
        "stage",
        "purpose",
        "boundary",
        "conductivity_convention",
        "config",
        "scan_results",
        "summary_statistics",
        "diagnostic_status",
    ):
        assert key in data
    assert all(data["boundary"].values())


def test_conductivity_convention_matches_stage5_1b():
    module = _load_module()
    args = module.parse_args([])
    args.quick = True
    config = module.build_scan_config(args)
    data = module.run_scan(config, max_cases=0)
    assert data["conductivity_convention"]["formula"] == "sigma_model_ij(iOmega) = - response[1:3,1:3] / omega_eV"
    assert data["conductivity_convention"]["normalization"] == "bilayer-normalized 2D sheet conductivity"
    assert data["conductivity_convention"]["si_scaling_applied"] is False
    assert data["conductivity_convention"]["bulk_3d_conductivity"] is False
    assert data["conductivity_convention"]["single_layer_conductivity"] is False


def test_zero_or_negative_omega_not_silent():
    module = _load_module()
    status, reasons = module.case_status_from_metrics(
        finite_values=True,
        omega_eV=0.0,
        sigma_diag_min_real=1.0,
        relative_offdiag_norm=0.0,
        ward_max_norm=0.0,
    )
    assert status == "FAIL"
    assert "NONPOSITIVE_OMEGA" in reasons


def test_workers_argument_rejects_nonpositive():
    module = _load_module()
    with pytest.raises(SystemExit):
        module.parse_args(["--workers", "0"])
    with pytest.raises(SystemExit):
        module.parse_args(["--workers", "-2"])


def test_parallel_and_serial_order_match_dry_or_mocked():
    module = _load_module()
    config = {
        "temperature_K": 30.0,
        "matsubara_indices": [1, 2, 4],
        "q_cases": ["q_diag_pos"],
        "adaptive_levels": [1],
        "gauss_orders": [2],
        "fermi_windows_eV": [0.05],
        "coarse_grid": 8,
    }
    cases = module.planned_cases(config)

    def fake_run_case(case, *, eta_eV):
        return {
            **case,
            "omega_eV": float(case["matsubara_index"]),
            "sigma_xx_model": 1.0 + 0.0j,
            "sigma_xy_model": 0.0 + 0.0j,
            "sigma_yx_model": 0.0 + 0.0j,
            "sigma_yy_model": 1.0 + 0.0j,
            "sigma_trace_real": 2.0,
            "sigma_diag_min_real": 1.0,
            "sigma_diag_positive": True,
            "offdiag_norm": 0.0,
            "diag_norm": 2**0.5,
            "relative_offdiag_norm": 0.0,
            "xy_plus_yx_abs": 0.0,
            "xy_minus_yx_abs": 0.0,
            "xx_minus_yy_abs": 0.0,
            "relative_xx_yy_anisotropy": 0.0,
            "ward_left_norm": 0.0,
            "ward_right_norm": 0.0,
            "ward_max_norm": 0.0,
            "num_quadrature_points": 0,
            "runtime_seconds": 0.0,
            "status": "PASS",
            "status_reasons": [],
        }

    serial = module.run_cases_parallel(cases, eta_eV=1e-10, workers=1, worker=fake_run_case)
    parallel = module.run_cases_parallel(
        cases,
        eta_eV=1e-10,
        workers=2,
        worker=fake_run_case,
        executor_factory=ThreadPoolExecutor,
    )
    assert [row["matsubara_index"] for row in serial] == [1, 2, 4]
    assert [row["matsubara_index"] for row in parallel] == [1, 2, 4]
    assert [row["case_index"] for row in serial] == [0, 1, 2]
    assert [row["case_index"] for row in parallel] == [0, 1, 2]

    status, reasons = module.case_status_from_metrics(
        finite_values=True,
        omega_eV=-0.1,
        sigma_diag_min_real=1.0,
        relative_offdiag_norm=0.0,
        ward_max_norm=0.0,
    )
    assert status == "FAIL"
    assert "NONPOSITIVE_OMEGA" in reasons


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            dict(
                finite_values=True,
                omega_eV=0.02,
                sigma_diag_min_real=1.0,
                relative_offdiag_norm=1e-4,
                ward_max_norm=1e-8,
            ),
            "PASS",
        ),
        (
            dict(
                finite_values=True,
                omega_eV=0.02,
                sigma_diag_min_real=-1e-3,
                relative_offdiag_norm=1e-4,
                ward_max_norm=1e-8,
            ),
            "FAIL",
        ),
        (
            dict(
                finite_values=True,
                omega_eV=0.02,
                sigma_diag_min_real=1.0,
                relative_offdiag_norm=2e-3,
                ward_max_norm=1e-8,
            ),
            "MONITOR",
        ),
        (
            dict(
                finite_values=True,
                omega_eV=0.02,
                sigma_diag_min_real=1.0,
                relative_offdiag_norm=1e-4,
                ward_max_norm=1e-5,
            ),
            "FAIL",
        ),
    ],
)
def test_case_status_logic(kwargs, expected):
    module = _load_module()
    status, _reasons = module.case_status_from_metrics(**kwargs)
    assert status == expected


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


def test_dry_run_does_not_execute_response(tmp_path):
    output_json = tmp_path / "stage5_2_dry.json"
    output_md = tmp_path / "stage5_2_dry.md"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--quick",
            "--workers",
            "2",
            "--dry-run",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
    )
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["config"]["dry_run"] is True
    assert data["config"]["planned_num_cases"] > 0
    assert data["scan_results"] == []
