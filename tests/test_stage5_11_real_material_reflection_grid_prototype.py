from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from lno327.material_reflection_grid import (
    MaterialReflectionGridPoint,
    complex_matrix_to_jsonable,
    default_stage5_11_points,
    grid_point_to_si_and_model_q,
    material_reflection_grid_prototype_metadata,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage5_11_real_material_reflection_grid_prototype.py"


def _synthetic_stage5_10(path: Path, *, status: str = "STAGE5_10_TOY_CASIMIR_INTEGRATION_CONVERGENCE_AUDIT_PASSED") -> None:
    data = {
        "stage": "Stage 5.10",
        "diagnostic_status": {"stage5_10_status": status},
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_default_points_count():
    assert len(default_stage5_11_points()) == 36


def test_smoke_points_count():
    assert len(default_stage5_11_points(smoke=True)) == 8


def test_grid_point_conversion_units():
    point = MaterialReflectionGridPoint(n=1, Q_nm_inv=0.10, phi_deg=60.0, temperature_K=10.0)
    converted = grid_point_to_si_and_model_q(point, 2.0e-10, 4.0e-10)
    assert converted["Q_m_inv"] == pytest.approx(1.0e8)
    assert converted["Qx_m_inv"] == pytest.approx(5.0e7)
    assert converted["Qy_m_inv"] == pytest.approx(np.sin(np.pi / 3.0) * 1.0e8)
    assert converted["q_model_x"] == pytest.approx(converted["Qx_m_inv"] * 2.0e-10)
    assert converted["q_model_y"] == pytest.approx(converted["Qy_m_inv"] * 4.0e-10)


def test_n0_excluded():
    assert all(point.n > 0 for point in default_stage5_11_points())


def test_Q0_excluded():
    assert all(point.Q_nm_inv > 0.0 for point in default_stage5_11_points())


def test_complex_matrix_serialization():
    serialized = complex_matrix_to_jsonable(np.array([[1.0 + 2.0j, 3.0 - 4.0j]]))
    assert serialized == [[{"re": 1.0, "im": 2.0}, {"re": 3.0, "im": -4.0}]]


def test_metadata_boundaries():
    metadata = material_reflection_grid_prototype_metadata()
    assert metadata["no_energy_output"]
    assert metadata["no_force_output"]
    assert metadata["no_torque_output"]
    assert metadata["not_production"]


def test_script_rejects_failed_stage5_10(tmp_path):
    input_json = tmp_path / "bad_stage5_10.json"
    output_json = tmp_path / "stage5_11.json"
    output_md = tmp_path / "stage5_11.md"
    _synthetic_stage5_10(input_json, status="FAILED")
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
    assert "Stage 5.10" in proc.stderr or "Stage 5.10" in proc.stdout


def test_script_with_synthetic_stage5_10_input_smoke_dry_run(tmp_path):
    input_json = tmp_path / "stage5_10.json"
    output_json = tmp_path / "stage5_11.json"
    output_md = tmp_path / "stage5_11.md"
    _synthetic_stage5_10(input_json)
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
            "--smoke",
            "--dry-run-grid-only",
        ],
        check=True,
    )
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["diagnostic_status"]["stage5_11_status"] == "STAGE5_11_DRY_RUN_GRID_ONLY_PASSED"
    assert data["prototype_grid"]["num_requested_points"] == 8
    assert output_md.exists()


def test_dry_run_workers_preserve_point_order(tmp_path):
    input_json = tmp_path / "stage5_10.json"
    _synthetic_stage5_10(input_json)

    outputs = []
    for workers in (1, 2):
        output_json = tmp_path / f"stage5_11_workers_{workers}.json"
        output_md = tmp_path / f"stage5_11_workers_{workers}.md"
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
                "--smoke",
                "--dry-run-grid-only",
                "--workers",
                str(workers),
            ],
            check=True,
        )
        outputs.append(json.loads(output_json.read_text(encoding="utf-8")))

    ids_1 = [row["point_id"] for row in outputs[0]["point_results"]]
    ids_2 = [row["point_id"] for row in outputs[1]["point_results"]]
    assert outputs[0]["prototype_grid"]["num_requested_points"] == outputs[1]["prototype_grid"]["num_requested_points"] == 8
    assert ids_1 == ids_2


def test_no_g_symbol():
    text = SCRIPT.read_text(encoding="utf-8")
    assert " g " not in text
    assert '"g"' not in text


def test_no_production_energy_claim(tmp_path):
    input_json = tmp_path / "stage5_10.json"
    output_json = tmp_path / "stage5_11.json"
    output_md = tmp_path / "stage5_11.md"
    _synthetic_stage5_10(input_json)
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
        ],
        check=True,
    )
    text = output_json.read_text(encoding="utf-8")
    assert "real_LNO327_energy" not in text
    assert "no_energy_output" in text
    assert "no_force_output" in text
    assert "no_torque_output" in text
