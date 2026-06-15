from __future__ import annotations

import ast
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
