from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage5_3_bilayer_sheet_conductivity_symmetry_convergence_audit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage5_3_bilayer_sheet_conductivity", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_lt_rotation_diagonalizes_synthetic_geometric_tensor():
    module = _load_module()
    q = np.array([1.0, 1.0])
    sigma_lt = np.array([[3.0, 0.0], [0.0, 1.0]], dtype=complex)
    rotation = module.xy_to_lt_rotation(q)
    sigma_xy = rotation.T @ sigma_lt @ rotation
    assert abs(sigma_xy[0, 1]) > 0.5
    recovered = module.xy_to_lt(sigma_xy, q)
    np.testing.assert_allclose(recovered, sigma_lt, atol=1e-12)


def test_symmetric_antisymmetric_decomposition():
    module = _load_module()
    symmetric = module.offdiag_decomposition(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex))
    antisymmetric = module.offdiag_decomposition(np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=complex))
    assert symmetric["symmetric_offdiag_abs"] > symmetric["antisymmetric_offdiag_abs"]
    assert antisymmetric["antisymmetric_offdiag_abs"] > antisymmetric["symmetric_offdiag_abs"]


def test_q_sign_symmetry_synthetic():
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
    result = module.q_sign_symmetry_pair(pos, neg)
    assert result["q_sign_symmetry_status"] == "PASS"


def test_case_status_ward_fail():
    module = _load_module()
    rows = [{"ward_max_norm": 1e-4, "sigma_diag_min_real": 1.0}]
    summaries = {
        "lt_projection_summary": {},
        "q_sign_symmetry_summary": {},
        "axial_vs_diagonal_summary": {},
        "q_scaling_summary": {},
        "convergence_summary": {},
    }
    status = module.diagnostic_status(rows, summaries)
    assert status["conductivity_symmetry_audit_status"] == "CONDUCTIVITY_SYMMETRY_AUDIT_FAILED_WARD"


def test_quick_mode_runs(tmp_path):
    output_json = tmp_path / "stage5_3.json"
    output_md = tmp_path / "stage5_3.md"
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
    for key in (
        "stage",
        "boundary",
        "conductivity_convention",
        "scan_results",
        "lt_projection_summary",
        "q_sign_symmetry_summary",
        "axial_vs_diagonal_summary",
        "q_scaling_summary",
        "convergence_summary",
        "diagnostic_status",
    ):
        assert key in data
    assert data["stage"] == "Stage 5.3"


def test_dry_run_no_response(tmp_path):
    output_json = tmp_path / "stage5_3_dry.json"
    output_md = tmp_path / "stage5_3_dry.md"
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
    args = module.parse_args(["--quick", "--dry-run"])
    data = module.run_audit(module.build_scan_config(args))
    assert data["conductivity_convention"]["formula"] == "sigma_model_ij(iOmega) = - response[1:3,1:3] / omega_eV"
    assert data["conductivity_convention"]["normalization"] == "bilayer-normalized 2D sheet conductivity"
