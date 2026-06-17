from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from lno327.conductivity_units import SheetConductivityUnitConvention
from lno327.material_reflection_grid import default_stage5_11_points, grid_point_to_si_and_model_q
from lno327.material_response_cache import (
    atomic_write_json,
    cache_filename_for_point_id,
    cache_path_for_point,
    is_reusable_cache,
    load_reusable_point_cache,
    write_point_cache,
)
from lno327.material_structure import LNO327_THIN_FILM_SLAO_IN_PLANE

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage5_11_real_material_reflection_grid_prototype.py"


def _response_config() -> dict[str, object]:
    return {
        "adaptive_level": 4,
        "gauss_order": 5,
        "fermi_window_eV": 0.05,
        "coarse_grid": 32,
        "eta_eV": 1e-10,
    }


def _lattice_convention() -> dict[str, object]:
    return {
        "a_x_m": LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m,
        "a_y_m": LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_y_m,
        "source": LNO327_THIN_FILM_SLAO_IN_PLANE.source_note,
        "is_placeholder": LNO327_THIN_FILM_SLAO_IN_PLANE.is_placeholder,
    }


def _convention() -> SheetConductivityUnitConvention:
    return SheetConductivityUnitConvention(
        lattice_a_x_m=LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m,
        lattice_a_y_m=LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_y_m,
        unit_cell_area_m2=LNO327_THIN_FILM_SLAO_IN_PLANE.unit_cell_area_m2,
    )


def _point_id(converted: dict[str, object]) -> str:
    return f"n{converted['n']}_Q{converted['Q_nm_inv']:.3f}_phi{converted['phi_deg']:.1f}"


def _synthetic_row(point, *, status: str = "PASS") -> dict[str, object]:
    convention = _convention()
    converted = grid_point_to_si_and_model_q(point, convention.lattice_a_x_m, convention.lattice_a_y_m)
    matrix = np.array([[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]])
    return {
        "point_id": _point_id(converted),
        **converted,
        "ward_residual": {"left_max": 0.0, "right_max": 0.0, "total_max": 0.0, "status": "PASS"},
        "response_matrix": matrix,
        "sigma_model_xy": matrix,
        "sigma_tilde_xy": matrix,
        "sigma_tilde_LT": matrix,
        "reflection_tangential_E_LT": matrix,
        "reflection_TE_TM": matrix,
        "conductivity_sanity": {
            "status": "PASS",
            "max_abs_sigma_tilde": 0.0,
            "sigma_tilde_xx_real": 0.0,
            "sigma_tilde_yy_real": 0.0,
            "offdiag_norm_ratio": 0.0,
        },
        "reflection_sanity": {"status": "PASS", "max_abs_R_TE_TM": 0.0},
        "integrand_identical_sheet": {
            "separation_m": 100.0e-9,
            "round_trip_factor": 0.0,
            "logdet": 0.0 + 0.0j,
            "status": "PASS",
        },
        "num_quadrature_points": 0,
        "refined_cell_count": 0,
        "runtime_seconds": 0.0,
        "status": status,
    }


def _synthetic_stage5_10(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "stage": "Stage 5.10",
                "diagnostic_status": {"stage5_10_status": "STAGE5_10_TOY_CASIMIR_INTEGRATION_CONVERGENCE_AUDIT_PASSED"},
            }
        ),
        encoding="utf-8",
    )


def test_cache_filename_is_stable_and_readable():
    assert cache_filename_for_point_id("n1_Q0.050_phi0.0") == "n1_Q0.050_phi0.0.json"
    assert cache_filename_for_point_id("n1/Q0.050 phi0.0") == "n1_Q0.050_phi0.0.json"


def test_same_config_cache_is_reusable(tmp_path):
    row = _synthetic_row(default_stage5_11_points(smoke=True)[0])
    path = write_point_cache(
        tmp_path,
        row,
        response_config=_response_config(),
        lattice_convention=_lattice_convention(),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert is_reusable_cache(
        payload,
        point_id=row["point_id"],
        response_config=_response_config(),
        lattice_convention=_lattice_convention(),
    )
    restored = load_reusable_point_cache(
        tmp_path,
        point_id=row["point_id"],
        response_config=_response_config(),
        lattice_convention=_lattice_convention(),
    )
    assert restored is not None
    assert np.asarray(restored["sigma_tilde_xy"], dtype=complex).shape == (2, 2)


def test_different_config_cache_is_invalid(tmp_path):
    row = _synthetic_row(default_stage5_11_points(smoke=True)[0])
    write_point_cache(
        tmp_path,
        row,
        response_config=_response_config(),
        lattice_convention=_lattice_convention(),
    )
    changed_config = {**_response_config(), "gauss_order": 7}
    assert (
        load_reusable_point_cache(
            tmp_path,
            point_id=row["point_id"],
            response_config=changed_config,
            lattice_convention=_lattice_convention(),
        )
        is None
    )


def test_fail_cache_is_not_reused(tmp_path):
    row = _synthetic_row(default_stage5_11_points(smoke=True)[0], status="FAIL")
    write_point_cache(
        tmp_path,
        row,
        response_config=_response_config(),
        lattice_convention=_lattice_convention(),
    )
    assert (
        load_reusable_point_cache(
            tmp_path,
            point_id=row["point_id"],
            response_config=_response_config(),
            lattice_convention=_lattice_convention(),
        )
        is None
    )


def test_atomic_write_json_produces_complete_json(tmp_path):
    path = tmp_path / "cache.json"
    atomic_write_json(path, {"point_id": "n1_Q0.050_phi0.0", "value": 1.0 + 2.0j})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["point_id"] == "n1_Q0.050_phi0.0"
    assert payload["value"]["re"] == 1.0
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_skip_existing_preserves_point_result_order(tmp_path):
    input_json = tmp_path / "stage5_10.json"
    output_json = tmp_path / "stage5_11.json"
    output_md = tmp_path / "stage5_11.md"
    cache_dir = tmp_path / "cache"
    _synthetic_stage5_10(input_json)

    expected_ids = []
    for point in default_stage5_11_points(smoke=True):
        row = _synthetic_row(point)
        expected_ids.append(row["point_id"])
        write_point_cache(
            cache_dir,
            row,
            response_config=_response_config(),
            lattice_convention=_lattice_convention(),
        )
        assert cache_path_for_point(cache_dir, row["point_id"]).exists()

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
            "--smoke",
            "--workers",
            "2",
            "--resume",
            "--skip-existing",
            "--cache-dir",
            str(cache_dir),
        ],
        check=True,
    )
    data = json.loads(output_json.read_text(encoding="utf-8"))
    actual_ids = [row["point_id"] for row in data["point_results"]]
    assert actual_ids == expected_ids
    assert all(row["cache"]["source"] == "hit" for row in data["point_results"])
