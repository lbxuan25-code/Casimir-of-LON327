from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from lno327.material_production_grid import (
    PRODUCTION_GRID_LEVELS,
    build_production_grid,
    classify_energy_convergence,
    integrate_grid_energy_from_rows,
    interior_q_nodes_nm_inv,
    production_grid_plan_from_stage5_13,
    relative_change,
    validate_stage5_13_input,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage5_14_real_material_production_grid_convergence.py"


def _synthetic_stage5_13(
    path: Path,
    *,
    status: str = "STAGE5_13_ZERO_MODE_GRID_CONVERGENCE_AUDIT_PASSED",
) -> None:
    path.write_text(
        json.dumps(
            {
                "stage": "Stage 5.13",
                "diagnostic_status": {"stage5_13_status": status},
                "grid_convergence_plan": {
                    **PRODUCTION_GRID_LEVELS,
                    "Q0_policy": "exclude endpoint Q=0 and use interior quadrature nodes",
                    "n0_policy": "use extrapolated xi->0+ reflection matrix; do not divide by omega=0",
                },
            }
        ),
        encoding="utf-8",
    )


def _synthetic_rows_for_grid(level: str) -> list[dict[str, object]]:
    grid = build_production_grid(level, Q_max_nm_inv=0.2, temperature_K=10.0)
    rows = []
    for n in grid.n_positive:
        for q in grid.Q_nm_inv:
            for phi in grid.phi_deg:
                rows.append(
                    {
                        "status": "PASS",
                        "n": n,
                        "Q_nm_inv": float(q),
                        "phi_deg": float(phi),
                        "integrand_identical_sheet": {
                            "logdet": -1.0e-6 / (1.0 + n + float(q)),
                            "status": "PASS",
                        },
                    }
                )
    return rows


def test_production_grid_levels_match_requirement():
    assert PRODUCTION_GRID_LEVELS["coarse"] == {"n_max": 8, "n_Q": 16, "n_phi": 8}
    assert PRODUCTION_GRID_LEVELS["medium"] == {"n_max": 16, "n_Q": 24, "n_phi": 12}
    assert PRODUCTION_GRID_LEVELS["fine"] == {"n_max": 32, "n_Q": 32, "n_phi": 16}


def test_interior_q_nodes_exclude_q0():
    nodes = interior_q_nodes_nm_inv(0.5, 16)
    assert len(nodes) == 16
    assert nodes[0] > 0.0
    assert nodes[-1] < 0.5
    with pytest.raises(ValueError):
        interior_q_nodes_nm_inv(0.0, 16)


def test_stage5_13_input_status_and_plan_validation(tmp_path):
    input_json = tmp_path / "stage5_13.json"
    _synthetic_stage5_13(input_json)
    data = json.loads(input_json.read_text(encoding="utf-8"))
    assert validate_stage5_13_input(data) == "STAGE5_13_ZERO_MODE_GRID_CONVERGENCE_AUDIT_PASSED"
    plan = production_grid_plan_from_stage5_13(data)
    assert plan["fine"]["n_phi"] == 16

    bad = {**data, "diagnostic_status": {"stage5_13_status": "FAILED"}}
    with pytest.raises(ValueError):
        validate_stage5_13_input(bad)


def test_energy_convergence_classification():
    assert relative_change(100.0, 98.0) == pytest.approx(0.02)
    assert classify_energy_convergence(0.049) == "PASS"
    assert classify_energy_convergence(0.10) == "MONITOR"
    assert classify_energy_convergence(0.20) == "FAIL"
    assert classify_energy_convergence(float("nan")) == "FAIL"


def test_synthetic_grid_energy_includes_n0_extrapolated_proxy():
    grid = build_production_grid("coarse", Q_max_nm_inv=0.2, temperature_K=10.0)
    result = integrate_grid_energy_from_rows(grid, _synthetic_rows_for_grid("coarse"))
    assert result["status"] == "PASS"
    assert result["num_missing_points"] == 0
    assert result["num_response_points_expected"] == 8 * 16 * 8
    assert result["num_energy_points_including_n0"] == 9 * 16 * 8
    assert result["real_J_m2"] < 0.0
    assert "xi->0+" in result["n0_policy"]


def test_script_dry_run_outputs_structure_without_response(tmp_path):
    input_json = tmp_path / "stage5_13.json"
    output_json = tmp_path / "stage5_14.json"
    output_md = tmp_path / "stage5_14.md"
    _synthetic_stage5_13(input_json)
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
            "--dry-run-grid-only",
            "--workers",
            "2",
            "--resume",
            "--skip-existing",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
        check=True,
    )
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["diagnostic_status"]["stage5_14_status"] == "STAGE5_14_DRY_RUN_GRID_ONLY_PASSED"
    assert data["grid_runs"]["coarse"]["num_response_points_expected"] == 8 * 16 * 8
    assert data["grid_runs"]["fine"]["num_energy_points_including_n0"] == 33 * 32 * 16
    assert data["energy_convergence"]["status"] == "DRY_RUN"
    assert data["cache_summary"]["resume"] is True
    assert data["cache_summary"]["skip_existing"] is True
    assert output_md.exists()


def test_script_rejects_failed_stage5_13(tmp_path):
    input_json = tmp_path / "bad_stage5_13.json"
    output_json = tmp_path / "stage5_14.json"
    output_md = tmp_path / "stage5_14.md"
    _synthetic_stage5_13(input_json, status="FAILED")
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-json",
            str(input_json),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--dry-run-grid-only",
        ],
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert "Stage 5.13" in proc.stderr or "Stage 5.13" in proc.stdout


def test_no_g_symbol():
    text = SCRIPT.read_text(encoding="utf-8")
    assert " g " not in text
    assert '"g"' not in text
