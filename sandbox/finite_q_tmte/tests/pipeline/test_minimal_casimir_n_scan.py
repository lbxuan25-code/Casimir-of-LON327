from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sandbox.finite_q_tmte.tmte.pipeline.minimal_casimir_n_scan import (
    K_B_EV_PER_K,
    SCHEMA_VERSION,
    matsubara_xi_eV,
    run_and_write_minimal_casimir_n_scan,
    run_minimal_casimir_n_scan,
)


def _load_cli():
    path = Path("sandbox/finite_q_tmte/scripts/debug_minimal_casimir_n_scan.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_minimal_casimir_n_scan_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_matsubara_xi_formula():
    assert matsubara_xi_eV(1, 10.0) == pytest.approx(2.0 * 3.141592653589793 * K_B_EV_PER_K * 10.0)
    with pytest.raises(ValueError, match="n=0"):
        matsubara_xi_eV(0, 10.0)


def test_n_scan_rows_summary_and_partial_sum():
    payload = run_minimal_casimir_n_scan(
        model_name="symmetry_bdg_2band",
        pairing_name="dwave",
        matsubara_indices=[1, 2],
        temperature_K=10.0,
        q_values=[0.01, 0.02],
        phi_values_deg=[0.0, 180.0],
        plate2_theta_deg=45.0,
        nk=1,
        separation_nm=20.0,
        include_rhs_aware_validation=False,
    )
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["status"]["n_scan_only"] is True
    assert payload["status"]["no_n0_policy"] is True
    assert len(payload["rows"]) == 2
    assert payload["rows"][0]["matsubara_index"] == 1
    assert payload["rows"][1]["matsubara_index"] == 2
    assert payload["rows"][0]["ratio_abs_to_previous_n"] is None
    assert payload["rows"][1]["ratio_abs_to_previous_n"] is not None
    assert payload["summary"]["num_rows"] == 2
    assert payload["summary"]["all_finite_logdet"] is True
    assert payload["summary"]["all_kappa_match"] is True
    assert payload["summary"]["n0_policy_included"] is False
    assert payload["valid_for_casimir_input"] is False


def test_n_scan_rejects_n0_and_duplicate_indices():
    with pytest.raises(ValueError, match="n=0"):
        run_minimal_casimir_n_scan(
            model_name="symmetry_bdg_2band",
            pairing_name="dwave",
            matsubara_indices=[0, 1],
            temperature_K=10.0,
            q_values=[0.01, 0.02],
            phi_values_deg=[0.0, 180.0],
            plate2_theta_deg=45.0,
            nk=1,
            separation_nm=20.0,
            include_rhs_aware_validation=False,
        )
    with pytest.raises(ValueError, match="unique"):
        run_minimal_casimir_n_scan(
            model_name="symmetry_bdg_2band",
            pairing_name="dwave",
            matsubara_indices=[1, 1],
            temperature_K=10.0,
            q_values=[0.01, 0.02],
            phi_values_deg=[0.0, 180.0],
            plate2_theta_deg=45.0,
            nk=1,
            separation_nm=20.0,
            include_rhs_aware_validation=False,
        )


def test_run_and_write_n_scan_outputs_json_and_csv(tmp_path):
    payload = run_and_write_minimal_casimir_n_scan(
        tmp_path,
        model_name="symmetry_bdg_2band",
        pairing_name="dwave",
        matsubara_indices=[1, 2],
        temperature_K=10.0,
        q_values=[0.01, 0.02],
        phi_values_deg=[0.0, 180.0],
        plate2_theta_deg=45.0,
        nk=1,
        separation_nm=20.0,
        include_rhs_aware_validation=False,
    )
    assert (tmp_path / "minimal_casimir_n_scan.json").exists()
    csv_path = tmp_path / "minimal_casimir_n_scan.csv"
    assert csv_path.exists()
    assert "xi_eV" in csv_path.read_text()
    assert payload["summary"]["num_rows"] == 2


def test_cli_accepts_n_scan_args(tmp_path):
    module = _load_cli()
    parser = module.build_parser()
    args = parser.parse_args([
        "--matsubara-indices", "1", "2", "3", "--q-values", "0.01", "0.02", "--phi-values", "0", "180", "--plate2-theta-deg", "45", "--nk", "13", "--separation-nm", "20", "--output-dir", str(tmp_path)
    ])
    assert args.matsubara_indices == [1, 2, 3]
    assert args.shift_fractions == [0.0]
