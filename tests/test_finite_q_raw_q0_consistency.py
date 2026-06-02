import subprocess
import sys
from pathlib import Path

import numpy as np

from lno327.finite_q_response import (
    bdg_finite_q_diamagnetic_kernel,
    bdg_finite_q_paramagnetic_kernel,
    bdg_finite_q_sigma_like_response,
    bdg_finite_q_total_kernel,
    compare_raw_q0_bubble_to_local_components,
    finite_q_q0_formula_consistency,
    finite_q_raw_bubble_response,
    _bosonic_matsubara_energy_eV,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "scripts" / "response" / "diagnose_finite_q_raw_q0_consistency.py"
SUMMARY = (
    ROOT
    / "validation"
    / "outputs"
    / "response"
    / "finite_q_raw_q0_consistency"
    / "finite_q_raw_q0_consistency_summary.md"
)


def test_raw_q0_bubble_runs_without_local_hook():
    raw = finite_q_raw_bubble_response(
        "normal",
        matsubara_index=1,
        temperature_K=30.0,
        q_magnitude=0.0,
        q_phi=0.0,
        nk=6,
        delta0=0.04,
        eta=1e-4,
    )
    result = finite_q_q0_formula_consistency(
        "normal",
        matsubara_index=1,
        temperature_K=30.0,
        nk=6,
        delta0=0.04,
        eta=1e-4,
    )

    assert raw.shape == (2, 2)
    assert np.isfinite(raw).all()
    assert np.isfinite(result.error_raw_to_local_sigma)
    assert result.error_hook_to_local_sigma < 1e-10


def test_raw_q0_component_errors_and_best_match_exist():
    result = finite_q_q0_formula_consistency(
        "dwave",
        matsubara_index=1,
        temperature_K=30.0,
        nk=6,
        delta0=0.04,
        eta=1e-4,
        denominator_mode="stable",
        deg_tol=1e-7,
    )

    assert result.best_raw_q0_match_component
    assert np.isfinite(result.best_raw_q0_relative_error)
    assert np.isfinite(result.error_raw_to_local_sigma)
    assert result.gauge_status == "prototype_not_ward_verified"
    assert result.response_layer == "bdg_finite_q_kernel_stack"
    assert result.contains_dia
    assert result.finite_q_dia_status == "q0_fallback_only"
    assert result.ward_status == "not_closed"
    assert not result.valid_for_casimir_input
    assert not result.final_casimir_input
    assert result.not_final_Casimir_conclusion


def test_bdg_kernel_layer_api_and_sigma_definition():
    kwargs = dict(
        kind="spm",
        matsubara_index=1,
        temperature_K=30.0,
        q_magnitude=0.0,
        q_phi=0.0,
        nk=6,
        delta0=0.04,
        eta=1e-4,
        denominator_mode="stable",
        deg_tol=1e-7,
    )
    k_para = bdg_finite_q_paramagnetic_kernel(**kwargs)
    k_dia = bdg_finite_q_diamagnetic_kernel(
        "spm",
        matsubara_index=1,
        temperature_K=30.0,
        q_magnitude=0.0,
        q_phi=0.0,
        nk=6,
        delta0=0.04,
        eta=1e-4,
    )
    k_total = bdg_finite_q_total_kernel(**kwargs)
    sigma = bdg_finite_q_sigma_like_response(**kwargs)
    omega = _bosonic_matsubara_energy_eV(1, 30.0)

    assert k_para.shape == (2, 2)
    assert np.isfinite(k_para).all()
    assert np.allclose(k_total, k_para + k_dia)
    assert np.allclose(sigma, k_total / omega)


def test_sigma_like_response_requires_positive_matsubara():
    with np.testing.assert_raises(ValueError):
        bdg_finite_q_sigma_like_response(
            "dwave",
            matsubara_index=0,
            temperature_K=30.0,
            q_magnitude=0.0,
            q_phi=0.0,
            nk=6,
            delta0=0.04,
            eta=1e-4,
        )


def test_q0_kernel_stack_consistency_is_explicit_not_hidden():
    result = finite_q_q0_formula_consistency(
        "spm",
        matsubara_index=1,
        temperature_K=30.0,
        nk=6,
        delta0=0.04,
        eta=1e-4,
        denominator_mode="stable",
        deg_tol=1e-7,
    )

    assert result.raw_q0_bubble.shape == (2, 2)
    assert np.allclose(result.raw_q0_bubble, result.K_para_q0)
    assert np.allclose(result.K_total_q0, result.K_para_q0 + result.K_dia_q0)
    assert np.isfinite(result.error_K_para_q0_to_local_K_para)
    assert np.isfinite(result.error_K_total_q0_to_local_K_total)
    assert np.isfinite(result.error_Sigma_SC_q0_to_local_K_total_over_omega)
    assert "K_para_q0_consistency_" in result.diagnostic_status
    assert "ward_status=not_closed" in result.diagnostic_status
    assert result.valid_for_casimir_input is False


def test_all_kinds_and_denominator_modes_run():
    rows = compare_raw_q0_bubble_to_local_components(
        kinds=["normal", "spm", "dwave"],
        matsubara_list=[1],
        nk_list=[6],
        denominator_mode_list=["raw", "stable"],
        deg_tol_list=[1e-8],
        temperature_K=30.0,
        delta0=0.04,
        eta=1e-4,
    )

    assert {row.kind for row in rows} == {"normal", "spm", "dwave"}
    assert {row.denominator_mode for row in rows} == {"raw", "stable"}
    assert all(row.gauge_status == "prototype_not_ward_verified" for row in rows)


def test_quick_script_outputs_fields(tmp_path):
    output_prefix = tmp_path / "finite_q_raw_q0_consistency"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output-prefix", str(output_prefix)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    required = {
        "kind",
        "denominator_mode",
        "deg_tol",
        "raw_q0_bubble",
        "K_para_q0",
        "K_dia_q0",
        "K_total_q0",
        "Sigma_SC_q0",
        "local_sigma",
        "hook_q0_response",
        "error_raw_to_local_sigma",
        "error_raw_to_local_K_para",
        "error_raw_to_local_K_total",
        "error_raw_to_local_K_total_over_omega",
        "error_raw_to_normal_kubo_sigma",
        "error_K_para_q0_to_local_K_para",
        "error_K_total_q0_to_local_K_total",
        "error_Sigma_SC_q0_to_local_K_total_over_omega",
        "error_hook_to_local_sigma",
        "best_raw_q0_match_component",
        "best_raw_q0_relative_error",
        "formula_layer_diagnosis",
        "response_layer",
        "contains_dia",
        "finite_q_dia_status",
        "ward_status",
        "valid_for_casimir_input",
        "gauge_status",
        "final_casimir_input",
        "not_final_Casimir_conclusion",
    }
    with np.load(output_prefix.with_suffix(".npz"), allow_pickle=True) as data:
        assert required.issubset(data.files)
        assert set(data["kind"]) == {"normal", "spm", "dwave"}
        assert set(data["denominator_mode"]) == {"raw", "stable"}
        assert np.all(np.isfinite(data["error_raw_to_local_sigma"]))
        assert np.all(np.isfinite(data["best_raw_q0_relative_error"]))
        assert np.nanmax(data["error_hook_to_local_sigma"]) < 1e-10
        assert set(data["gauge_status"]) == {"prototype_not_ward_verified"}
        assert set(data["ward_status"]) == {"not_closed"}
        assert not np.any(data["valid_for_casimir_input"])
        assert not np.any(data["final_casimir_input"])
        assert np.all(data["not_final_Casimir_conclusion"])
    assert SUMMARY.exists()
