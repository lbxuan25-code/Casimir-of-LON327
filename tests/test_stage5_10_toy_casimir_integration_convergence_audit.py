from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from lno327.casimir_toy_integration import (
    convergence_scan_toy,
    integrate_toy_free_energy_density,
    xi_c_from_omega_eV,
)

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "src" / "lno327" / "casimir_toy_integration.py"
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage5_10_toy_casimir_integration_convergence_audit.py"
DOC = ROOT / "docs" / "notes" / "stage5_10_toy_casimir_integration_convergence_audit_zh.md"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage5_10_audit", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _params():
    return {"Qc_m_inv": 0.2e9, "xi_c_si": xi_c_from_omega_eV(0.05)}


def _base(**overrides):
    kwargs = {
        "temperature_K": 10.0,
        "n_max": 4,
        "Q_max_m_inv": 0.5e9,
        "n_Q": 8,
        "n_phi": 8,
        "separation_m": 100.0e-9,
        "theta_rad": 0.0,
        **_params(),
    }
    kwargs.update(overrides)
    return kwargs


def _synthetic_stage5_9(path: Path, *, status: str = "STAGE5_9_CASIMIR_GRID_SCAFFOLD_PASSED") -> None:
    data = {
        "stage": "Stage 5.9",
        "diagnostic_status": {"stage5_9_status": status},
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_zero_reflector_full_integration_zero():
    result = integrate_toy_free_energy_density(model="zero", **_base())
    assert result["free_energy_density_J_m2"] == pytest.approx(0.0)
    assert result["imag_part_J_m2"] == pytest.approx(0.0)


def test_isotropic_toy_angle_independence():
    values = [
        integrate_toy_free_energy_density(model="isotropic_identical", **_base(theta_rad=theta))["free_energy_density_J_m2"]
        for theta in (0.0, np.pi / 6.0, np.pi / 4.0, np.pi / 2.0, np.pi)
    ]
    assert max(abs(value - values[0]) for value in values) < 1e-30


def test_anisotropic_toy_pi_periodicity():
    zero = integrate_toy_free_energy_density(model="anisotropic_relative_rotation", **_base(theta_rad=0.0))["free_energy_density_J_m2"]
    pi = integrate_toy_free_energy_density(model="anisotropic_relative_rotation", **_base(theta_rad=np.pi))["free_energy_density_J_m2"]
    quarter = integrate_toy_free_energy_density(model="anisotropic_relative_rotation", **_base(theta_rad=np.pi / 4.0))[
        "free_energy_density_J_m2"
    ]
    assert abs(zero - pi) < 1e-30
    assert abs(quarter - zero) > 0.0


def test_distance_dependence_magnitude():
    values = [
        integrate_toy_free_energy_density(model="isotropic_identical", **_base(separation_m=d))["free_energy_density_J_m2"]
        for d in (50.0e-9, 100.0e-9, 200.0e-9)
    ]
    assert values[0] < values[1] < values[2] < 0.0
    assert abs(values[0]) > abs(values[1]) > abs(values[2])


def test_no_nan_inf_in_convergence_scans():
    scan = convergence_scan_toy(
        temperature_K=10.0,
        n_max_values=[2, 4, 8],
        Q_max_values_m_inv=[0.25e9, 0.5e9, 0.75e9],
        n_Q_values=[8, 12, 16],
        n_phi_values=[8, 12, 16],
        separation_m=100.0e-9,
        theta_rad=0.0,
        model="isotropic_identical",
        **_params(),
    )
    for section in scan.values():
        assert np.all(np.isfinite(section["free_energy_density_J_m2"]))
        assert section["status"] in {"PASS", "MONITOR"}


def test_matsubara_nmax_scan_runs():
    scan = convergence_scan_toy(
        temperature_K=10.0,
        n_max_values=[2, 4, 8],
        Q_max_values_m_inv=[0.5e9],
        n_Q_values=[8],
        n_phi_values=[8],
        separation_m=100.0e-9,
        theta_rad=0.0,
        model="isotropic_identical",
        **_params(),
    )
    assert len(scan["n_max"]["free_energy_density_J_m2"]) == 3


def test_Qmax_scan_runs():
    scan = convergence_scan_toy(
        temperature_K=10.0,
        n_max_values=[4],
        Q_max_values_m_inv=[0.25e9, 0.5e9, 0.75e9],
        n_Q_values=[8],
        n_phi_values=[8],
        separation_m=100.0e-9,
        theta_rad=0.0,
        model="isotropic_identical",
        **_params(),
    )
    assert len(scan["Q_max"]["free_energy_density_J_m2"]) == 3


def test_nQ_scan_runs():
    scan = convergence_scan_toy(
        temperature_K=10.0,
        n_max_values=[4],
        Q_max_values_m_inv=[0.5e9],
        n_Q_values=[8, 12, 16],
        n_phi_values=[8],
        separation_m=100.0e-9,
        theta_rad=0.0,
        model="isotropic_identical",
        **_params(),
    )
    assert len(scan["n_Q"]["free_energy_density_J_m2"]) == 3


def test_nphi_scan_isotropic_stability():
    values = [
        integrate_toy_free_energy_density(model="isotropic_identical", **_base(n_phi=n_phi))["free_energy_density_J_m2"]
        for n_phi in (8, 12, 16)
    ]
    assert max(abs(value - values[0]) for value in values) < 1e-28


def test_imaginary_part_sanity():
    result = integrate_toy_free_energy_density(model="isotropic_identical", **_base())
    assert abs(result["imag_part_J_m2"]) < 1e-30


def test_reject_failed_stage5_9_by_default(tmp_path):
    module = _load_module()
    input_json = tmp_path / "bad_stage5_9.json"
    _synthetic_stage5_9(input_json, status="FAILED")
    with pytest.raises(ValueError, match="STAGE5_9"):
        module.run_audit(input_json, temperature_K=10.0, separation_nm=100.0, require_stage5_9_passed=True)


def test_script_with_synthetic_stage5_9_input(tmp_path):
    input_json = tmp_path / "stage5_9.json"
    output_json = tmp_path / "stage5_10.json"
    output_md = tmp_path / "stage5_10.md"
    _synthetic_stage5_9(input_json)
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
            "--temperature-K",
            "10",
            "--separation-nm",
            "100",
        ],
        check=True,
    )
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["diagnostic_status"]["stage5_10_status"] in {
        "STAGE5_10_TOY_CASIMIR_INTEGRATION_CONVERGENCE_AUDIT_PASSED",
        "STAGE5_10_TOY_CASIMIR_INTEGRATION_CONVERGENCE_AUDIT_MONITOR",
    }
    assert output_md.exists()


def test_no_response_or_real_material_imports():
    imported_modules: list[str] = []
    for path in (HELPER, SCRIPT):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module)
    banned = ("response", "material", "production")
    assert not any(any(word in module.lower() for word in banned) for module in imported_modules)


def test_no_energy_force_torque_physical_claim(tmp_path):
    module = _load_module()
    input_json = tmp_path / "stage5_9.json"
    _synthetic_stage5_9(input_json)
    data = module.run_audit(input_json, temperature_K=10.0, separation_nm=100.0, require_stage5_9_passed=True)
    assert data["scope"]["toy_model_only"]
    assert data["scope"]["not_material_prediction"]
    assert data["scope"]["no_real_LNO327_energy"]
    assert data["scope"]["no_force"]
    assert data["scope"]["no_torque"]


def test_no_g_symbol():
    text = SCRIPT.read_text(encoding="utf-8") + "\n" + DOC.read_text(encoding="utf-8")
    assert " g " not in text
    assert '"g"' not in text
