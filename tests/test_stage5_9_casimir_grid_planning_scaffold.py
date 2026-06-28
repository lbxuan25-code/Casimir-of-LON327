from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from lno327.casimir_grid import (
    matsubara_prime_weights,
    matsubara_xi_grid,
    omega_eV_to_xi_si,
    polar_measure_weights,
    q_polar_grid,
    round_trip_factor_from_xi_Q_d,
    xi_si_to_omega_eV,
)

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "src" / "lno327" / "casimir_grid.py"
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage5_9_casimir_grid_planning_scaffold.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage5_9_scaffold", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _synthetic_stage5_8(path: Path, *, status: str = "STAGE5_8_CASIMIR_INTEGRAND_PROTOTYPE_PASSED") -> None:
    data = {
        "stage": "Stage 5.8",
        "diagnostic_status": {"stage5_8_status": status},
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_matsubara_grid_length_and_zero():
    xi = matsubara_xi_grid(10.0, 8)
    assert len(xi) == 9
    assert xi[0] == 0.0


def test_matsubara_prime_weights():
    np.testing.assert_allclose(matsubara_prime_weights(4), np.array([0.5, 1.0, 1.0, 1.0, 1.0]))


def test_matsubara_linear_in_n():
    xi = matsubara_xi_grid(10.0, 8)
    np.testing.assert_allclose(np.diff(xi), np.diff(xi)[0])


def test_xi_omega_eV_round_trip():
    xi = matsubara_xi_grid(10.0, 8)
    np.testing.assert_allclose(omega_eV_to_xi_si(xi_si_to_omega_eV(xi)), xi, rtol=1e-14, atol=1e-6)


def test_phi_grid_no_duplicate_endpoint():
    grid = q_polar_grid(5.0e8, 8, 12)
    phi = grid["phi_rad"]
    assert phi[0] == 0.0
    assert phi[-1] < 2.0 * np.pi


def test_q_grid_shapes():
    grid = q_polar_grid(5.0e8, 8, 12)
    assert grid["Qx_m_inv"].shape == (8, 12)
    assert grid["Qy_m_inv"].shape == (8, 12)
    assert grid["Q_m_inv"].shape == (8,)
    assert grid["phi_rad"].shape == (12,)


def test_polar_measure_nonnegative():
    grid = q_polar_grid(5.0e8, 8, 12)
    weights = polar_measure_weights(grid["Q_m_inv"], grid["phi_rad"])
    assert np.all(weights >= 0.0)


def test_round_trip_factor_range():
    xi = matsubara_xi_grid(10.0, 8)
    q = q_polar_grid(5.0e8, 8, 12)["Q_m_inv"]
    factors = [round_trip_factor_from_xi_Q_d(x, qq, 100.0e-9) for x in xi for qq in q]
    assert all(0.0 < factor <= 1.0 for factor in factors)


def test_q_zero_warning_present(tmp_path):
    module = _load_module()
    input_json = tmp_path / "stage5_8.json"
    _synthetic_stage5_8(input_json)
    data = module.run_scaffold(
        input_json,
        temperature_K=10.0,
        n_max=8,
        q_max_nm_inv=0.5,
        n_q=8,
        n_phi=12,
        separation_nm=100.0,
        require_stage5_8_passed=True,
    )
    assert module.Q0_WARNING in data["warnings"]


def test_response_grid_insufficiency_warning_present(tmp_path):
    module = _load_module()
    input_json = tmp_path / "stage5_8.json"
    _synthetic_stage5_8(input_json)
    data = module.run_scaffold(
        input_json,
        temperature_K=10.0,
        n_max=8,
        q_max_nm_inv=0.5,
        n_q=8,
        n_phi=12,
        separation_nm=100.0,
        require_stage5_8_passed=True,
    )
    assert module.RESPONSE_GRID_WARNING in data["warnings"]


def test_reject_failed_stage5_8_by_default(tmp_path):
    module = _load_module()
    input_json = tmp_path / "bad_stage5_8.json"
    _synthetic_stage5_8(input_json, status="FAILED")
    with pytest.raises(ValueError, match="STAGE5_8"):
        module.run_scaffold(
            input_json,
            temperature_K=10.0,
            n_max=8,
            q_max_nm_inv=0.5,
            n_q=8,
            n_phi=12,
            separation_nm=100.0,
            require_stage5_8_passed=True,
        )


def test_script_with_synthetic_stage5_8_input(tmp_path):
    input_json = tmp_path / "stage5_8.json"
    output_json = tmp_path / "stage5_9.json"
    output_md = tmp_path / "stage5_9.md"
    _synthetic_stage5_8(input_json)
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
            "--n-max",
            "8",
            "--q-max-nm-inv",
            "0.5",
            "--n-q",
            "8",
            "--n-phi",
            "12",
            "--separation-nm",
            "100",
        ],
        check=True,
    )
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["diagnostic_status"]["stage5_9_status"] == "STAGE5_9_CASIMIR_GRID_SCAFFOLD_PASSED"
    assert output_md.exists()


def test_no_energy_force_torque_output(tmp_path):
    module = _load_module()
    input_json = tmp_path / "stage5_8.json"
    _synthetic_stage5_8(input_json)
    data = module.run_scaffold(
        input_json,
        temperature_K=10.0,
        n_max=8,
        q_max_nm_inv=0.5,
        n_q=8,
        n_phi=12,
        separation_nm=100.0,
        require_stage5_8_passed=True,
    )
    text = json.dumps(module.to_jsonable(data))
    assert "casimir_energy" not in text
    assert "casimir_force" not in text
    assert "casimir_torque" not in text
    assert data["boundary"]["no_energy_output"]
    assert data["boundary"]["no_force_output"]
    assert data["boundary"]["no_torque_output"]


def test_no_response_or_production_imports():
    imported_modules: list[str] = []
    for path in (HELPER, SCRIPT):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module)
    banned = ("response", "production", "energy", "force", "torque")
    assert not any(any(word in module.lower() for word in banned) for module in imported_modules)


def test_no_g_symbol():
    text = SCRIPT.read_text(encoding="utf-8")
    assert " g " not in text
    assert '"g"' not in text
