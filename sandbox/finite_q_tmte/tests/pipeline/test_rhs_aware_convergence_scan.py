from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sandbox.finite_q_tmte.tmte.pipeline.rhs_aware_convergence_scan import (
    SCHEMA_VERSION,
    aggregate_rows,
    compute_nk_convergence,
    compute_shift_convergence,
    shift_fractions_for_mode,
)


def _load_scan_cli():
    path = Path("sandbox/finite_q_tmte/scripts/debug_rhs_aware_convergence_scan.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_rhs_aware_convergence_scan_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(*, nk: int, shift: str, k_eff: float, r_eff: float, eta_proj: float, closed: bool = True) -> dict[str, object]:
    return {
        "pairing": "dwave",
        "matsubara_index": 1,
        "q_value": 0.02,
        "nk": nk,
        "shift_mode": shift,
        "rhs_aware_ward_closed": closed,
        "max_s_channel_residual_over_rhs_s": 1e-12,
        "max_effective_residual_over_reference": 2e-12,
        "max_eta_projection_over_rhs_s": eta_proj,
        "max_legacy_zero_rhs_residual_over_k_eff_norm": 1e-3,
        "K_eff_norm": k_eff,
        "K_SS_norm": 1.0,
        "Schur_correction_norm": 0.5,
        "K_etaeta_condition_number": 10.0,
        "left_R_eff_norm": r_eff,
        "right_R_eff_norm": r_eff,
        "left_R_eff_over_K_eff_norm": r_eff / k_eff,
        "right_R_eff_over_K_eff_norm": r_eff / k_eff,
        "left_eta_projection_over_rhs_s": eta_proj,
        "right_eta_projection_over_rhs_s": eta_proj,
        "valid_for_casimir_input": False,
    }


def test_schema_version():
    assert SCHEMA_VERSION == "finite_q_tmte_rhs_aware_convergence_scan_v1"


def test_shift_fractions_for_mode():
    assert shift_fractions_for_mode("noshift") == (0.0,)
    assert shift_fractions_for_mode("shifted2") == (0.0, 0.5)
    assert shift_fractions_for_mode("shifted5") == (0.0, 0.2, 0.4, 0.6, 0.8)
    with pytest.raises(ValueError):
        shift_fractions_for_mode("bad")


def test_compute_nk_convergence_adjacent_pairs():
    rows = [
        _row(nk=9, shift="noshift", k_eff=1.0, r_eff=0.1, eta_proj=0.2),
        _row(nk=13, shift="noshift", k_eff=1.1, r_eff=0.11, eta_proj=0.25),
        _row(nk=17, shift="noshift", k_eff=1.2, r_eff=0.12, eta_proj=0.3),
    ]
    comparisons = compute_nk_convergence(rows)
    assert len(comparisons) == 2
    assert comparisons[0]["nk_from"] == 9
    assert comparisons[0]["nk_to"] == 13
    assert comparisons[1]["nk_from"] == 13
    assert comparisons[1]["nk_to"] == 17


def test_compute_shift_convergence_uses_noshift_reference():
    rows = [
        _row(nk=13, shift="noshift", k_eff=1.0, r_eff=0.1, eta_proj=0.2),
        _row(nk=13, shift="shifted2", k_eff=1.2, r_eff=0.2, eta_proj=0.4),
        _row(nk=13, shift="shifted5", k_eff=1.4, r_eff=0.3, eta_proj=0.8),
    ]
    comparisons = compute_shift_convergence(rows)
    assert len(comparisons) == 2
    assert {row["shift_to"] for row in comparisons} == {"shifted2", "shifted5"}


def test_aggregate_rows_counts_pass_fail():
    rows = [
        _row(nk=9, shift="noshift", k_eff=1.0, r_eff=0.1, eta_proj=0.2, closed=True),
        _row(nk=13, shift="noshift", k_eff=1.1, r_eff=0.2, eta_proj=0.3, closed=False),
    ]
    nk = compute_nk_convergence(rows)
    aggregate = aggregate_rows(rows, nk, [])
    assert aggregate["num_rows"] == 2
    assert aggregate["num_rhs_aware_closed"] == 1
    assert aggregate["all_rhs_aware_closed"] is False
    assert aggregate["num_nk_convergence_pairs"] == 1


def test_convergence_cli_rejects_nonpositive_q(tmp_path):
    module = _load_scan_cli()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            ["--matsubara-indices", "1", "--q-values", "0", "--nk-values", "13", "--output-dir", str(tmp_path)]
        )


def test_convergence_cli_rejects_bad_shift_mode(tmp_path):
    module = _load_scan_cli()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            ["--matsubara-indices", "1", "--q-values", "0.02", "--nk-values", "13", "--shift-modes", "bad", "--output-dir", str(tmp_path)]
        )
