from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.pipeline.minimal_casimir_path import (
    SCHEMA_VERSION,
    response_to_minimal_casimir_point,
    run_minimal_casimir_path,
)


def _load_cli():
    path = Path("sandbox/finite_q_tmte/scripts/debug_minimal_casimir_path.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_minimal_casimir_path_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_schema_version():
    assert SCHEMA_VERSION == "finite_q_tmte_minimal_casimir_path_v1"


def test_zero_spatial_response_gives_zero_reflection_and_zero_logdet():
    payload = response_to_minimal_casimir_point(
        np.zeros((3, 3), dtype=complex),
        omega_eV=0.005,
        q_value=0.02,
        separation_nm=20.0,
    )
    assert payload["response"]["spatial_response_norm"] == 0.0
    assert payload["conductivity"]["sigma_model_norm"] == 0.0
    assert payload["reflection"]["R_TE_TM_norm"] == 0.0
    assert abs(payload["trace_log"]["logdet_integrand"]) < 1e-14
    assert payload["trace_log"]["logdet_is_finite"] is True
    assert payload["valid_for_casimir_input"] is False


def test_scalar_synthetic_response_has_diagonal_te_tm_reflection():
    omega = 0.005
    sigma_model_scalar = 0.1
    response = np.zeros((3, 3), dtype=complex)
    response[1:3, 1:3] = -omega * sigma_model_scalar * np.eye(2, dtype=complex)
    payload = response_to_minimal_casimir_point(
        response,
        omega_eV=omega,
        q_value=0.02,
        separation_nm=20.0,
    )
    reflection = np.asarray(payload["reflection"]["R1_TE_TM"], dtype=complex)
    assert np.isfinite(reflection.real).all()
    assert np.isfinite(reflection.imag).all()
    assert abs(reflection[0, 1]) < 1e-14
    assert abs(reflection[1, 0]) < 1e-14
    assert payload["trace_log"]["logdet_is_finite"] is True


def test_response_to_minimal_casimir_point_rejects_bad_shape():
    with pytest.raises(ValueError, match="shape"):
        response_to_minimal_casimir_point(np.zeros((2, 2)), omega_eV=0.005, q_value=0.02, separation_nm=20.0)


def test_response_to_minimal_casimir_point_rejects_q0():
    with pytest.raises(ValueError, match="q_value"):
        response_to_minimal_casimir_point(np.zeros((3, 3)), omega_eV=0.005, q_value=0.0, separation_nm=20.0)


def test_minimal_path_rejects_n0_before_heavy_compute():
    with pytest.raises(ValueError, match="n>=1"):
        run_minimal_casimir_path(
            model_name="symmetry_bdg_2band",
            pairing_name="dwave",
            matsubara_index=0,
            temperature_K=10.0,
            q_value=0.02,
            nk=3,
            separation_nm=20.0,
            include_rhs_aware_validation=False,
        )


def test_minimal_path_rejects_theta_nonzero_before_heavy_compute():
    with pytest.raises(ValueError, match="theta_deg=0"):
        run_minimal_casimir_path(
            model_name="symmetry_bdg_2band",
            pairing_name="dwave",
            matsubara_index=1,
            temperature_K=10.0,
            q_value=0.02,
            nk=3,
            separation_nm=20.0,
            theta_deg=5.0,
            include_rhs_aware_validation=False,
        )


def test_cli_rejects_n0_q0_and_theta_nonzero(tmp_path):
    module = _load_cli()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--matsubara-index", "0", "--q", "0.02", "--nk", "13", "--separation-nm", "20", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--matsubara-index", "1", "--q", "0", "--nk", "13", "--separation-nm", "20", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--matsubara-index", "1", "--q", "0.02", "--nk", "13", "--separation-nm", "20", "--theta-deg", "1", "--output-dir", str(tmp_path)])
