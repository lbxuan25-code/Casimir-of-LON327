from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.pipeline.minimal_casimir_qvec_path import q_model_vector_from_polar
from sandbox.finite_q_tmte.tmte.pipeline.minimal_casimir_theta_path import (
    SCHEMA_VERSION,
    crystal_q_from_lab_q,
    rotation_matrix_deg,
    run_minimal_casimir_theta_path,
)


def _load_cli():
    path = Path("sandbox/finite_q_tmte/scripts/debug_minimal_casimir_theta_path.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_minimal_casimir_theta_path_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_schema_version():
    assert SCHEMA_VERSION == "finite_q_tmte_minimal_casimir_theta_path_v1"


def test_rotation_convention():
    q_lab = np.asarray([1.0, 0.0])
    assert np.allclose(rotation_matrix_deg(90.0) @ q_lab, [0.0, 1.0])
    assert np.allclose(crystal_q_from_lab_q(q_lab, 90.0), [0.0, -1.0])


def test_theta_zero_gives_identical_plate_reflections():
    payload = run_minimal_casimir_theta_path(
        model_name="symmetry_bdg_2band",
        pairing_name="dwave",
        matsubara_index=1,
        temperature_K=10.0,
        q_lab_vector=[0.02, 0.0],
        plate2_theta_deg=0.0,
        nk=1,
        separation_nm=20.0,
        include_rhs_aware_validation=False,
    )
    assert payload["status"]["theta_diagnostic_supported"] is True
    assert payload["mixed_trace_log"]["R1_minus_R2_norm"] < 1e-12
    assert payload["mixed_trace_log"]["logdet_is_finite"] is True
    assert payload["valid_for_casimir_input"] is False


def test_nonzero_theta_produces_finite_mixed_trace_log():
    payload = run_minimal_casimir_theta_path(
        model_name="symmetry_bdg_2band",
        pairing_name="dwave",
        matsubara_index=1,
        temperature_K=10.0,
        q_lab_vector=q_model_vector_from_polar(0.02, 30.0),
        plate2_theta_deg=45.0,
        nk=1,
        separation_nm=20.0,
        include_rhs_aware_validation=False,
    )
    assert payload["sanity_checks"]["finite_R1"] is True
    assert payload["sanity_checks"]["finite_R2"] is True
    assert payload["sanity_checks"]["finite_logdet"] is True
    assert payload["geometry"]["plate2"]["theta_deg"] == 45.0
    assert payload["valid_for_casimir_input"] is False


def test_theta_path_rejects_n0_and_zero_q():
    with pytest.raises(ValueError, match="n>=1"):
        run_minimal_casimir_theta_path(
            model_name="symmetry_bdg_2band",
            pairing_name="dwave",
            matsubara_index=0,
            temperature_K=10.0,
            q_lab_vector=[0.02, 0.0],
            plate2_theta_deg=45.0,
            nk=1,
            separation_nm=20.0,
            include_rhs_aware_validation=False,
        )
    with pytest.raises(ValueError, match="nonzero"):
        run_minimal_casimir_theta_path(
            model_name="symmetry_bdg_2band",
            pairing_name="dwave",
            matsubara_index=1,
            temperature_K=10.0,
            q_lab_vector=[0.0, 0.0],
            plate2_theta_deg=45.0,
            nk=1,
            separation_nm=20.0,
            include_rhs_aware_validation=False,
        )


def test_cli_accepts_polar_and_explicit_lab_qvec_and_rejects_bad_mix(tmp_path):
    module = _load_cli()
    parser = module.build_parser()
    args = parser.parse_args([
        "--matsubara-index", "1", "--q", "0.02", "--phi-deg", "30", "--plate2-theta-deg", "45", "--nk", "13", "--separation-nm", "20", "--output-dir", str(tmp_path)
    ])
    q = module.q_vector_from_args(args, parser)
    assert np.allclose(np.linalg.norm(q), 0.02)

    args = parser.parse_args([
        "--matsubara-index", "1", "--qx", "0.02", "--qy", "0.01", "--plate2-theta-deg", "45", "--nk", "13", "--separation-nm", "20", "--output-dir", str(tmp_path)
    ])
    q = module.q_vector_from_args(args, parser)
    assert np.allclose(q, [0.02, 0.01])

    with pytest.raises(SystemExit):
        args = parser.parse_args([
            "--matsubara-index", "1", "--q", "0.02", "--qx", "0.02", "--qy", "0.0", "--plate2-theta-deg", "45", "--nk", "13", "--separation-nm", "20", "--output-dir", str(tmp_path)
        ])
        module.q_vector_from_args(args, parser)
