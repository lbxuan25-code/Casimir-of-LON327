from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from lno327.material_grid_convergence import (
    default_small_q_points,
    default_zero_mode_points,
    grid_convergence_plan,
    q_nm_phi_to_si_model,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage5_13_zero_mode_grid_convergence_audit.py"


def _synthetic_stage5_12(path: Path, *, status: str = "STAGE5_12_SMALL_REAL_MATERIAL_ENERGY_PROTOTYPE_PASSED") -> None:
    path.write_text(json.dumps({"stage": "Stage 5.12", "diagnostic_status": {"stage5_12_status": status}}), encoding="utf-8")


def test_default_point_counts():
    assert len(default_small_q_points()) == 48
    assert len(default_zero_mode_points()) == 36


def test_smoke_point_counts():
    assert len(default_small_q_points(smoke=True)) == 8
    assert len(default_zero_mode_points(smoke=True)) == 8


def test_q_conversion_excludes_q0_and_converts_units():
    converted = q_nm_phi_to_si_model(0.01, 60.0, 2.0e-10, 4.0e-10)
    assert converted["Q_m_inv"] == pytest.approx(1.0e7)
    assert converted["q_model_x"] == pytest.approx(converted["Qx_m_inv"] * 2.0e-10)
    assert converted["q_model_y"] == pytest.approx(converted["Qy_m_inv"] * 4.0e-10)
    with pytest.raises(ValueError):
        q_nm_phi_to_si_model(0.0, 0.0, 2.0e-10, 4.0e-10)


def test_grid_plan_contains_three_levels():
    plan = grid_convergence_plan()
    assert set(["coarse", "medium", "fine"]).issubset(plan)
    assert plan["coarse"]["n_max"] == 8
    assert "Q0_policy" in plan
    assert "n0_policy" in plan


def test_script_rejects_failed_stage5_12(tmp_path):
    input_json = tmp_path / "bad_stage5_12.json"
    output_json = tmp_path / "stage5_13.json"
    output_md = tmp_path / "stage5_13.md"
    _synthetic_stage5_12(input_json, status="FAILED")
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
            "--smoke",
        ],
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert "Stage 5.12" in proc.stderr or "Stage 5.12" in proc.stdout


def test_script_dry_run_smoke_outputs_structure(tmp_path):
    input_json = tmp_path / "stage5_12.json"
    output_json = tmp_path / "stage5_13.json"
    output_md = tmp_path / "stage5_13.md"
    _synthetic_stage5_12(input_json)
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
            "--smoke",
            "--workers",
            "2",
        ],
        check=True,
    )
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["checks"]["input_status"] == "PASS"
    assert data["small_Q_audit"]["status"] == "PASS"
    assert data["zero_mode_audit"]["status"] == "PASS"
    assert data["grid_convergence_plan"]["fine"]["n_phi"] == 16
    assert output_md.exists()


def test_no_g_symbol():
    text = SCRIPT.read_text(encoding="utf-8")
    assert " g " not in text
    assert '"g"' not in text
