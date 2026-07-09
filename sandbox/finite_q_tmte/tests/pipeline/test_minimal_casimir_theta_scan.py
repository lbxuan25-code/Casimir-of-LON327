from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.pipeline.minimal_casimir_qvec_path import q_model_vector_from_polar
from sandbox.finite_q_tmte.tmte.pipeline.minimal_casimir_theta_scan import (
    SCHEMA_VERSION,
    run_and_write_minimal_casimir_theta_scan,
    run_minimal_casimir_theta_scan,
)


def _load_cli():
    path = Path("sandbox/finite_q_tmte/scripts/debug_minimal_casimir_theta_scan.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_minimal_casimir_theta_scan_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_schema_version():
    assert SCHEMA_VERSION == "finite_q_tmte_minimal_casimir_theta_scan_v1"


def test_theta_scan_rows_and_summary():
    payload = run_minimal_casimir_theta_scan(
        model_name="symmetry_bdg_2band",
        pairing_name="dwave",
        matsubara_index=1,
        temperature_K=10.0,
        q_lab_vector=q_model_vector_from_polar(0.02, 30.0),
        theta_values_deg=[0.0, 45.0, 90.0],
        nk=1,
        separation_nm=20.0,
        include_rhs_aware_validation=False,
    )
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["status"]["theta_scan_only"] is True
    assert len(payload["rows"]) == 3
    assert payload["summary"]["num_rows"] == 3
    assert payload["summary"]["all_finite_logdet"] is True
    assert payload["summary"]["all_kappa_match"] is True
    assert payload["rows"][0]["Rdiff"] < 1e-12
    assert payload["rows"][1]["d_logdet_abs_dtheta_rad_diagnostic"] is not None
    assert payload["valid_for_casimir_input"] is False


def test_theta_scan_single_row_has_no_derivative():
    payload = run_minimal_casimir_theta_scan(
        model_name="symmetry_bdg_2band",
        pairing_name="dwave",
        matsubara_index=1,
        temperature_K=10.0,
        q_lab_vector=[0.02, 0.0],
        theta_values_deg=[0.0],
        nk=1,
        separation_nm=20.0,
        include_rhs_aware_validation=False,
    )
    assert payload["rows"][0]["d_logdet_abs_dtheta_rad_diagnostic"] is None


def test_theta_scan_rejects_duplicate_angles():
    with pytest.raises(ValueError, match="unique"):
        run_minimal_casimir_theta_scan(
            model_name="symmetry_bdg_2band",
            pairing_name="dwave",
            matsubara_index=1,
            temperature_K=10.0,
            q_lab_vector=[0.02, 0.0],
            theta_values_deg=[0.0, 0.0],
            nk=1,
            separation_nm=20.0,
            include_rhs_aware_validation=False,
        )


def test_run_and_write_theta_scan_outputs_json_and_csv(tmp_path):
    payload = run_and_write_minimal_casimir_theta_scan(
        tmp_path,
        model_name="symmetry_bdg_2band",
        pairing_name="dwave",
        matsubara_index=1,
        temperature_K=10.0,
        q_lab_vector=[0.02, 0.0],
        theta_values_deg=[0.0, 90.0],
        nk=1,
        separation_nm=20.0,
        include_rhs_aware_validation=False,
    )
    assert (tmp_path / "minimal_casimir_theta_scan.json").exists()
    csv_path = tmp_path / "minimal_casimir_theta_scan.csv"
    assert csv_path.exists()
    assert "theta_deg" in csv_path.read_text()
    assert payload["summary"]["num_rows"] == 2


def test_cli_accepts_theta_values_and_q_input(tmp_path):
    module = _load_cli()
    parser = module.build_parser()
    args = parser.parse_args([
        "--matsubara-index", "1", "--q", "0.02", "--phi-deg", "30", "--theta-values", "0", "45", "90", "--nk", "13", "--separation-nm", "20", "--output-dir", str(tmp_path)
    ])
    q = module.q_vector_from_args(args, parser)
    assert np.allclose(np.linalg.norm(q), 0.02)

    with pytest.raises(SystemExit):
        args = parser.parse_args([
            "--matsubara-index", "1", "--q", "0.02", "--qx", "0.02", "--qy", "0.0", "--theta-values", "0", "45", "--nk", "13", "--separation-nm", "20", "--output-dir", str(tmp_path)
        ])
        module.q_vector_from_args(args, parser)
