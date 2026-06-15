from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage5_3b_bilayer_sheet_conductivity_offdiag_convergence_audit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage5_3b_offdiag_convergence", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _synthetic_row(**updates):
    row = {
        "matsubara_index": 1,
        "q_case": "q_diag_pos",
        "base_q_case": "q_diag_pos",
        "q_scale": 1.0,
        "adaptive_level": 4,
        "gauss_order": 5,
        "fermi_window_eV": 0.05,
        "sigma_xx_model": 10.0 + 0.0j,
        "sigma_yy_model": 11.0 + 0.0j,
        "relative_offdiag_norm": 0.12,
        "relative_LT_offdiag_norm": 0.04,
        "symmetric_offdiag_abs": 1.0,
        "antisymmetric_offdiag_abs": 1e-6,
        "relative_antisymmetric_to_symmetric": 1e-6,
        "sigma_diag_min_real": 10.0,
        "ward_max_norm": 1e-9,
    }
    row.update(updates)
    return row


def test_imports():
    module = _load_module()
    assert module.JSON_OUTPUT.name == "stage5_3b_bilayer_sheet_conductivity_offdiag_convergence_audit.json"


def test_targeted_default_case_count():
    module = _load_module()
    config = module.build_scan_config(module.parse_args([]))
    assert len(module.planned_cases(config)) == 24


def test_convergence_comparison_synthetic_pass():
    module = _load_module()
    baseline = _synthetic_row()
    comparison = _synthetic_row(adaptive_level=5, relative_offdiag_norm=0.125, relative_LT_offdiag_norm=0.041)
    result = module.compare_to_baseline(baseline, comparison)
    assert result["comparison_status"] == "PASS"


def test_convergence_comparison_synthetic_fail():
    module = _load_module()
    baseline = _synthetic_row()
    comparison = _synthetic_row(adaptive_level=5, relative_offdiag_norm=0.2, relative_LT_offdiag_norm=0.12)
    result = module.compare_to_baseline(baseline, comparison)
    assert result["comparison_status"] == "FAIL"


def test_q_sign_symmetry_pass():
    module = _load_module()
    pos = {
        "matsubara_index": 1,
        "q_scale": 1.0,
        "adaptive_level": 4,
        "gauss_order": 5,
        "fermi_window_eV": 0.05,
        "sigma_xx_model": 3.0 + 0.0j,
        "sigma_yy_model": 4.0 + 0.0j,
        "sigma_xy_model": 0.5 + 0.0j,
        "sigma_yx_model": 0.5 + 0.0j,
    }
    neg = {**pos, "sigma_xy_model": -0.5 + 0.0j, "sigma_yx_model": -0.5 + 0.0j}
    assert module.q_sign_symmetry_pair(pos, neg)["q_sign_symmetry_status"] == "PASS"


def test_q_scaling_decrease():
    module = _load_module()
    rows = [
        _synthetic_row(q_scale=1.0, relative_offdiag_norm=0.12, relative_LT_offdiag_norm=0.04),
        _synthetic_row(q_scale=0.5, relative_offdiag_norm=0.04, relative_LT_offdiag_norm=0.02),
    ]
    summary = module.q_scaling_summary(rows)
    assert summary["all_xy_decrease"] is True
    assert summary["pairs"][0]["xy_offdiag_decreases_with_q"] is True


def test_symmetric_dominates():
    module = _load_module()
    sigma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    assert module.offdiag_decomposition(sigma)["relative_antisymmetric_to_symmetric"] < 1e-12


def test_hall_like_not_misclassified():
    module = _load_module()
    sigma = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=complex)
    row = _synthetic_row(**module.offdiag_decomposition(sigma))
    summary = module.symmetric_antisymmetric_summary([row])
    assert summary["status"] != "SYMMETRIC_OFFDIAG_DOMINATES"


def test_quick_run_outputs(tmp_path):
    output_json = tmp_path / "stage5_3b.json"
    output_md = tmp_path / "stage5_3b.md"
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
    data = json.loads(output_json.read_text(encoding="utf-8"))
    for key in (
        "stage",
        "boundary",
        "conductivity_convention",
        "config",
        "scan_results",
        "convergence_comparisons",
        "q_sign_symmetry_summary",
        "q_scaling_summary",
        "symmetric_antisymmetric_summary",
        "global_summary",
        "diagnostic_status",
    ):
        assert key in data
    assert output_md.exists()


def test_dry_run_no_heavy_response(tmp_path):
    output_json = tmp_path / "stage5_3b_dry.json"
    output_md = tmp_path / "stage5_3b_dry.md"
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


def test_convention_matches_stage5_1b():
    module = _load_module()
    data = module.run_audit(module.build_scan_config(module.parse_args(["--quick", "--dry-run"])))
    assert data["conductivity_convention"]["formula"] == "sigma_model_ij(iOmega) = - response[1:3,1:3] / omega_eV"
    assert data["conductivity_convention"]["normalization"] == "bilayer-normalized 2D sheet conductivity"
