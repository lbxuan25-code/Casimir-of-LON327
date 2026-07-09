from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sandbox.finite_q_tmte.tmte.pipeline.minimal_casimir_shift_scan import (
    SCHEMA_VERSION,
    _row_from_phi_row,
    run_and_write_minimal_casimir_shift_scan,
    run_minimal_casimir_shift_scan,
)


def _load_cli():
    path = Path("sandbox/finite_q_tmte/scripts/debug_minimal_casimir_shift_scan.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_minimal_casimir_shift_scan_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_schema_version():
    assert SCHEMA_VERSION == "finite_q_tmte_minimal_casimir_shift_scan_v1"


def test_row_large_r_norm_flag():
    row = _row_from_phi_row(
        shift=0.2,
        r_norm_warning_threshold=2.0,
        row={
            "phi_mod_deg": 15.0,
            "logdet_abs": 0.1,
            "delta_logdet_abs_from_phi0": 0.01,
            "Rdiff": 3.0,
            "R1_norm": 1.5,
            "R2_norm": 2.5,
            "p1_Keff_norm": 0.7,
            "p2_Keff_norm": 0.8,
            "finite_R1": True,
            "finite_R2": True,
            "finite_logdet": True,
            "kappa_match": True,
        },
    )
    assert row["max_R_norm"] == pytest.approx(2.5)
    assert row["large_R_norm"] is True


def test_shift_scan_rows_summary_and_guard():
    payload = run_minimal_casimir_shift_scan(
        model_name="symmetry_bdg_2band",
        pairing_name="dwave",
        matsubara_index=1,
        temperature_K=10.0,
        q_magnitude=0.02,
        phi_values_deg=[0.0, 45.0],
        plate2_theta_deg=45.0,
        nk=1,
        separation_nm=20.0,
        shift_values=[0.0, 0.5],
        include_rhs_aware_validation=False,
        r_norm_warning_threshold=2.0,
    )
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["status"]["shift_scan_only"] is True
    assert payload["status"]["r_norm_guard_included"] is True
    assert len(payload["rows"]) == 4
    assert payload["summary"]["num_rows"] == 4
    assert payload["summary"]["all_finite_logdet"] is True
    assert payload["summary"]["all_kappa_match"] is True
    assert len(payload["summary_by_shift"]) == 2
    assert payload["valid_for_casimir_input"] is False


def test_shift_scan_rejects_duplicate_shifts():
    with pytest.raises(ValueError, match="unique"):
        run_minimal_casimir_shift_scan(
            model_name="symmetry_bdg_2band",
            pairing_name="dwave",
            matsubara_index=1,
            temperature_K=10.0,
            q_magnitude=0.02,
            phi_values_deg=[0.0, 45.0],
            plate2_theta_deg=45.0,
            nk=1,
            separation_nm=20.0,
            shift_values=[0.0, 0.0],
            include_rhs_aware_validation=False,
        )


def test_run_and_write_shift_scan_outputs_json_and_csv(tmp_path):
    payload = run_and_write_minimal_casimir_shift_scan(
        tmp_path,
        model_name="symmetry_bdg_2band",
        pairing_name="dwave",
        matsubara_index=1,
        temperature_K=10.0,
        q_magnitude=0.02,
        phi_values_deg=[0.0, 45.0],
        plate2_theta_deg=45.0,
        nk=1,
        separation_nm=20.0,
        shift_values=[0.0, 0.5],
        include_rhs_aware_validation=False,
    )
    assert (tmp_path / "minimal_casimir_shift_scan.json").exists()
    csv_path = tmp_path / "minimal_casimir_shift_scan.csv"
    assert csv_path.exists()
    assert "max_R_norm" in csv_path.read_text()
    assert payload["summary"]["num_rows"] == 4


def test_cli_accepts_shift_scan_args(tmp_path):
    module = _load_cli()
    parser = module.build_parser()
    args = parser.parse_args([
        "--matsubara-index", "1", "--q", "0.02", "--phi-values", "0", "45", "--plate2-theta-deg", "45", "--nk", "13", "--separation-nm", "20", "--shift-values", "0.0", "0.5", "--output-dir", str(tmp_path)
    ])
    assert args.q == 0.02
    assert args.phi_values == [0.0, 45.0]
    assert args.shift_values == [0.0, 0.5]
