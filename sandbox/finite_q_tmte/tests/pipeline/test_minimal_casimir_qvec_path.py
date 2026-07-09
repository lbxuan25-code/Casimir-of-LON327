from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.pipeline.minimal_casimir_qvec_path import (
    SCHEMA_VERSION,
    q_model_vector_from_polar,
    response_to_minimal_casimir_qvec_point,
    run_minimal_casimir_qvec_path,
)


def _load_cli():
    path = Path("sandbox/finite_q_tmte/scripts/debug_minimal_casimir_qvec_path.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_minimal_casimir_qvec_path_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_schema_version():
    assert SCHEMA_VERSION == "finite_q_tmte_minimal_casimir_qvec_path_v1"


def test_q_model_vector_from_polar():
    q = q_model_vector_from_polar(0.02, 30.0)
    assert np.allclose(np.linalg.norm(q), 0.02)
    assert q[0] > 0.0
    assert q[1] > 0.0


def test_zero_response_qvec_gives_zero_reflection_and_zero_logdet():
    q = q_model_vector_from_polar(0.02, 30.0)
    payload = response_to_minimal_casimir_qvec_point(
        np.zeros((3, 3), dtype=complex),
        omega_eV=0.005,
        q_model_vector=q,
        separation_nm=20.0,
    )
    assert payload["response"]["spatial_response_norm"] == 0.0
    assert payload["conductivity"]["sigma_model_norm"] == 0.0
    assert payload["reflection"]["R_TE_TM_norm"] == 0.0
    assert abs(payload["trace_log"]["logdet_integrand"]) < 1e-14
    assert payload["trace_log"]["logdet_is_finite"] is True
    assert payload["reflection"]["LT_native_reflection_path"] is True
    assert payload["valid_for_casimir_input"] is False


def test_scalar_synthetic_LT_response_has_diagonal_te_tm_reflection_for_angled_q():
    omega = 0.005
    sigma_model_scalar = 0.1
    response = np.zeros((3, 3), dtype=complex)
    response[1:3, 1:3] = -omega * sigma_model_scalar * np.eye(2, dtype=complex)
    payload = response_to_minimal_casimir_qvec_point(
        response,
        omega_eV=omega,
        q_model_vector=q_model_vector_from_polar(0.02, 45.0),
        separation_nm=20.0,
    )
    reflection = np.asarray(payload["reflection"]["R1_TE_TM"], dtype=complex)
    assert np.isfinite(reflection.real).all()
    assert np.isfinite(reflection.imag).all()
    assert abs(reflection[0, 1]) < 1e-14
    assert abs(reflection[1, 0]) < 1e-14
    assert payload["trace_log"]["logdet_is_finite"] is True


def test_qvec_path_rejects_n0_and_theta_before_heavy_compute():
    with pytest.raises(ValueError, match="n>=1"):
        run_minimal_casimir_qvec_path(
            model_name="symmetry_bdg_2band",
            pairing_name="dwave",
            matsubara_index=0,
            temperature_K=10.0,
            q_model_vector=[0.02, 0.0],
            nk=3,
            separation_nm=20.0,
            include_rhs_aware_validation=False,
        )
    with pytest.raises(ValueError, match="theta_deg=0"):
        run_minimal_casimir_qvec_path(
            model_name="symmetry_bdg_2band",
            pairing_name="dwave",
            matsubara_index=1,
            temperature_K=10.0,
            q_model_vector=[0.02, 0.0],
            nk=3,
            separation_nm=20.0,
            theta_deg=5.0,
            include_rhs_aware_validation=False,
        )


def test_qvec_nonzero_qy_marks_rhs_guard_not_run():
    payload = run_minimal_casimir_qvec_path(
        model_name="symmetry_bdg_2band",
        pairing_name="dwave",
        matsubara_index=1,
        temperature_K=10.0,
        q_model_vector=[0.02, 0.01],
        nk=1,
        separation_nm=20.0,
        include_rhs_aware_validation=True,
    )
    assert payload["status"]["arbitrary_q_vec_supported"] is True
    assert payload["rhs_aware_validation"]["status"]["diagnostic_run_completed"] is False
    assert payload["valid_for_casimir_input"] is False


def test_cli_accepts_polar_and_explicit_qvec_and_rejects_bad_mix(tmp_path):
    module = _load_cli()
    parser = module.build_parser()
    args = parser.parse_args(["--matsubara-index", "1", "--q", "0.02", "--phi-deg", "30", "--nk", "13", "--separation-nm", "20", "--output-dir", str(tmp_path)])
    q = module.q_vector_from_args(args, parser)
    assert np.allclose(np.linalg.norm(q), 0.02)

    args = parser.parse_args(["--matsubara-index", "1", "--qx", "0.02", "--qy", "0.01", "--nk", "13", "--separation-nm", "20", "--output-dir", str(tmp_path)])
    q = module.q_vector_from_args(args, parser)
    assert np.allclose(q, [0.02, 0.01])

    with pytest.raises(SystemExit):
        parser.parse_args(["--matsubara-index", "1", "--nk", "13", "--separation-nm", "20", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        args = parser.parse_args(["--matsubara-index", "1", "--q", "0.02", "--qx", "0.02", "--qy", "0.0", "--nk", "13", "--separation-nm", "20", "--output-dir", str(tmp_path)])
        module.q_vector_from_args(args, parser)
