from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sandbox.finite_q_tmte.tmte.pipeline.minimal_casimir_phi_scan import (
    SCHEMA_VERSION,
    run_and_write_minimal_casimir_phi_scan,
    run_minimal_casimir_phi_scan,
)


def _load_cli():
    path = Path("sandbox/finite_q_tmte/scripts/debug_minimal_casimir_phi_scan.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_minimal_casimir_phi_scan_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_schema_version():
    assert SCHEMA_VERSION == "finite_q_tmte_minimal_casimir_phi_scan_v1"


def test_phi_scan_rows_summary_and_periodic_diagnostics():
    payload = run_minimal_casimir_phi_scan(
        model_name="symmetry_bdg_2band",
        pairing_name="dwave",
        matsubara_index=1,
        temperature_K=10.0,
        q_magnitude=0.02,
        phi_values_deg=[0.0, 90.0, 180.0, 270.0],
        plate2_theta_deg=45.0,
        nk=1,
        separation_nm=20.0,
        include_rhs_aware_validation=False,
    )
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["status"]["phi_scan_only"] is True
    assert len(payload["rows"]) == 4
    assert payload["summary"]["num_rows"] == 4
    assert payload["summary"]["all_finite_logdet"] is True
    assert payload["summary"]["all_kappa_match"] is True
    assert payload["summary"]["periodic_phi_integral_logdet_abs_diagnostic"] is not None
    assert payload["summary"]["periodic_phi_average_logdet_abs_diagnostic"] is not None
    assert payload["rows"][0]["d_logdet_abs_dphi_rad_diagnostic"] is not None
    assert payload["valid_for_casimir_input"] is False


def test_phi_scan_single_row_has_no_derivative_or_integral():
    payload = run_minimal_casimir_phi_scan(
        model_name="symmetry_bdg_2band",
        pairing_name="dwave",
        matsubara_index=1,
        temperature_K=10.0,
        q_magnitude=0.02,
        phi_values_deg=[30.0],
        plate2_theta_deg=45.0,
        nk=1,
        separation_nm=20.0,
        include_rhs_aware_validation=False,
    )
    assert payload["rows"][0]["d_logdet_abs_dphi_rad_diagnostic"] is None
    assert payload["summary"]["periodic_phi_integral_logdet_abs_diagnostic"] is None


def test_phi_scan_rejects_periodic_duplicates():
    with pytest.raises(ValueError, match="unique modulo 360"):
        run_minimal_casimir_phi_scan(
            model_name="symmetry_bdg_2band",
            pairing_name="dwave",
            matsubara_index=1,
            temperature_K=10.0,
            q_magnitude=0.02,
            phi_values_deg=[0.0, 360.0],
            plate2_theta_deg=45.0,
            nk=1,
            separation_nm=20.0,
            include_rhs_aware_validation=False,
        )


def test_run_and_write_phi_scan_outputs_json_and_csv(tmp_path):
    payload = run_and_write_minimal_casimir_phi_scan(
        tmp_path,
        model_name="symmetry_bdg_2band",
        pairing_name="dwave",
        matsubara_index=1,
        temperature_K=10.0,
        q_magnitude=0.02,
        phi_values_deg=[0.0, 180.0],
        plate2_theta_deg=45.0,
        nk=1,
        separation_nm=20.0,
        include_rhs_aware_validation=False,
    )
    assert (tmp_path / "minimal_casimir_phi_scan.json").exists()
    csv_path = tmp_path / "minimal_casimir_phi_scan.csv"
    assert csv_path.exists()
    assert "phi_deg" in csv_path.read_text()
    assert payload["summary"]["num_rows"] == 2


def test_cli_accepts_phi_values(tmp_path):
    module = _load_cli()
    parser = module.build_parser()
    args = parser.parse_args([
        "--matsubara-index", "1", "--q", "0.02", "--phi-values", "0", "90", "180", "270", "--plate2-theta-deg", "45", "--nk", "13", "--separation-nm", "20", "--output-dir", str(tmp_path)
    ])
    assert args.q == 0.02
    assert args.phi_values == [0.0, 90.0, 180.0, 270.0]
