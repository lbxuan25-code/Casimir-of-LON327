from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sandbox.finite_q_tmte.tmte.pipeline.rhs_aware_finite_q_validation import (
    SCHEMA_VERSION,
    summarize_schur_audit,
)


def _load_validation_cli():
    path = Path("sandbox/finite_q_tmte/scripts/debug_rhs_aware_finite_q_validation.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_rhs_aware_finite_q_validation_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fake_side(*, s_res: float, eff_res: float, eta_proj: float = 0.25, r_eff_norm: float = 0.2) -> dict[str, object]:
    return {
        "s_channel_residual": {"norm_over_reference": s_res},
        "effective_residual_over_reference": eff_res,
        "eta_projection_over_rhs_s": eta_proj,
        "eta_channel_total_C_eta": {"norm": 1e-6},
        "effective_rhs_predicted": {"norm": r_eff_norm},
        "effective_direct": {"norm": r_eff_norm},
    }


def _fake_schur_audit(*, s_res: float = 1e-12, eff_res: float = 1e-12, cond: float = 10.0) -> dict[str, object]:
    return {
        "model": {"name": "m", "pairing": "p", "valid_for_casimir_input": False},
        "frequency": {"matsubara_index": 1, "temperature_K": 10.0, "valid_for_casimir_input": False},
        "debug_parameters": {"q_value": 0.02, "nk": 13, "shifted_mesh_average": {"num_shifted_meshes": 1}, "valid_for_casimir_input": False},
        "block_norms": {"K_eff_norm": 2.0, "K_SS_norm": 3.0, "Schur_correction_norm": 0.5},
        "summary": {},
        "schur_solve_metadata": {"etaeta_condition_number": cond, "numerically_suspect": False},
        "ward_decomposition": {
            "left": _fake_side(s_res=s_res, eff_res=eff_res),
            "right": _fake_side(s_res=s_res, eff_res=eff_res),
        },
    }


def test_schema_version():
    assert SCHEMA_VERSION == "finite_q_tmte_rhs_aware_finite_q_validation_v1"


def test_summarize_schur_audit_passes_rhs_aware_residuals():
    payload = summarize_schur_audit(_fake_schur_audit(), residual_tol=1e-9, condition_max=1e12)
    assert payload["status"]["rhs_aware_ward_closed"] is True
    assert payload["status"]["valid_for_casimir_input"] is False
    assert payload["legacy_zero_rhs_check"]["status"] == "invalid_target_at_finite_q"


def test_summarize_schur_audit_fails_large_residual():
    payload = summarize_schur_audit(_fake_schur_audit(eff_res=1e-3), residual_tol=1e-9, condition_max=1e12)
    assert payload["status"]["schur_effective_closed"] is False
    assert payload["status"]["rhs_aware_ward_closed"] is False


def test_summarize_schur_audit_fails_bad_condition():
    payload = summarize_schur_audit(_fake_schur_audit(cond=1e20), residual_tol=1e-9, condition_max=1e12)
    assert payload["status"]["condition_ok"] is False
    assert payload["status"]["rhs_aware_ward_closed"] is False


def test_validation_cli_rejects_nonpositive_q(tmp_path):
    module = _load_validation_cli()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            ["--pairing", "dwave", "--matsubara-index", "1", "--q", "0", "--nk", "13", "--output-dir", str(tmp_path)]
        )


def test_validation_cli_rejects_nonpositive_nk(tmp_path):
    module = _load_validation_cli()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            ["--pairing", "dwave", "--matsubara-index", "1", "--q", "0.02", "--nk", "0", "--output-dir", str(tmp_path)]
        )
