from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from lno327.collective.ward import contact_ward_rhs, physical_ward_residuals
from validation.lib.finite_q_schur_ward_localization import run_bdg_schur_ward_algebra_localization
from validation.lib import finite_q_ward_triage as triage


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "validation" / "scripts" / "bdg_finite_q" / "run_finite_q_ward_report.py"


def _load_report_module():
    spec = importlib.util.spec_from_file_location("run_finite_q_ward_report_for_triage_test", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _component(real: float, imag: float = 0.0, component: str = "density"):
    return {"component": component, "real": real, "imag": imag}


def _vector_row(pairing: str, response: str, q: float, left, right=None):
    right = left if right is None else right
    labels = ("density", "current_x", "current_y")
    left_vector = tuple(_component(float(value), component=labels[index]) for index, value in enumerate(left))
    right_vector = tuple(_component(float(value), component=labels[index]) for index, value in enumerate(right))
    max_norm = max(float(np.linalg.norm(np.asarray(left, dtype=complex))), float(np.linalg.norm(np.asarray(right, dtype=complex))))
    return SimpleNamespace(
        pairing_name=pairing,
        response_name=response,
        q_model=(q, 0.0),
        q_norm=q,
        left_ward_residual_norm=max_norm,
        right_ward_residual_norm=max_norm,
        residual_component_labels=labels,
        left_ward_residual_vector=left_vector,
        right_ward_residual_vector=right_vector,
        max_ward_residual_norm=max_norm,
        residual_ratio_to_bare=None,
        collective_matrix_condition_number=None,
        inverse_method="not_used",
        pinv_diagnostic_used=False,
    )


def _criterion_rows(pairings=("spm", "dwave"), q_values=(0.01,), closure_delta=1e-8):
    rows = []
    for pairing in pairings:
        for q in q_values:
            direct = np.array([0.0, 0.3, 0.0])
            rows.extend(
                [
                    _vector_row(pairing, "bare_bubble", q, [1e-8, 0.0, 0.0]),
                    _vector_row(pairing, "direct", q, direct),
                    _vector_row(pairing, "bare_total", q, direct + np.array([2e-8, 0.0, 0.0])),
                    _vector_row(pairing, "minus_schur", q, [3e-8, 0.0, 0.0]),
                    _vector_row(pairing, "amplitude_phase_schur", q, [closure_delta, 0.0, 0.0]),
                ]
            )
    return rows


def test_normal_finite_q_ward_triage_reports_fields(monkeypatch):
    matrix = np.eye(3, dtype=complex)

    monkeypatch.setattr(
        triage,
        "get_finite_q_validation_model",
        lambda model_name: SimpleNamespace(name=model_name, spec=object()),
    )
    monkeypatch.setattr(triage, "uniform_bz_mesh", lambda nk: np.zeros((nk, 2), dtype=float))
    monkeypatch.setattr(triage, "k_weights", lambda points: np.ones(points.shape[0]) / points.shape[0])
    monkeypatch.setattr(
        triage.KuboConfig,
        "from_kelvin",
        staticmethod(lambda **kwargs: SimpleNamespace(omega_eV=kwargs["omega_eV"])),
    )
    monkeypatch.setattr(
        triage,
        "normal_physical_density_current_response_components_imag_axis_from_model",
        lambda *args, **kwargs: {
            "bare_bubble": matrix,
            "direct": -0.5 * matrix,
            "total": 0.5 * matrix,
        },
    )

    payload = triage.run_normal_finite_q_ward_triage(nk=2)

    assert payload["available"] is True
    assert payload["valid_for_casimir_input"] is False
    assert payload["workspace_evaluation"] is True
    assert set(payload["residual_components"]) == {"bare_bubble", "direct", "total"}
    assert "max_norm" in payload["residual_components"]["total"]
    assert "vector_norm" in payload["residual_components"]["total"]
    assert payload["component_labels"] == ["density", "current_x", "current_y"]
    assert payload["suspected_layer"] in {
        "normal_closed",
        "normal_contact_or_vertex",
        "normal_response_assembly",
        "unknown",
    }


def test_operator_identity_triage_reports_unavailable_subchecks(monkeypatch):
    monkeypatch.setattr(
        triage,
        "get_finite_q_validation_model",
        lambda model_name: SimpleNamespace(name=model_name, spec=object()),
    )
    monkeypatch.setattr(triage, "uniform_bz_mesh", lambda nk: np.zeros((nk, 2), dtype=float))

    payload = triage.run_operator_identity_triage(nk=2)

    assert payload["available"] is True
    assert payload["valid_for_casimir_input"] is False
    assert "normal_vertex" in payload
    assert "bdg_vertex" in payload
    assert "pairing_sector" in payload
    assert payload["normal_vertex"]["available"] is False
    assert payload["bdg_vertex"]["available"] is False
    assert set(payload["pairing_sector"]["by_pairing"]) == {"spm", "dwave"}
    assert payload["suspected_layer"] in {
        "normal_peierls_vertex",
        "bdg_vertex_construction",
        "pairing_gauge_vertex",
        "response_assembly_or_collective",
        "unknown",
    }


def test_contact_cancellation_triage_reports_geometry(monkeypatch):
    rows = (
        SimpleNamespace(
            pairing_name="spm",
            response_name="bare_bubble",
            left_ward_residual_vector=({"real": 1.0, "imag": 0.0}, {"real": 0.0, "imag": 0.0}, {"real": 0.0, "imag": 0.0}),
            right_ward_residual_vector=({"real": 1.0, "imag": 0.0}, {"real": 0.0, "imag": 0.0}, {"real": 0.0, "imag": 0.0}),
        ),
        SimpleNamespace(
            pairing_name="spm",
            response_name="direct",
            left_ward_residual_vector=({"real": -0.8, "imag": 0.0}, {"real": 0.0, "imag": 0.0}, {"real": 0.0, "imag": 0.0}),
            right_ward_residual_vector=({"real": -0.8, "imag": 0.0}, {"real": 0.0, "imag": 0.0}, {"real": 0.0, "imag": 0.0}),
        ),
        SimpleNamespace(
            pairing_name="spm",
            response_name="bare_total",
            left_ward_residual_vector=({"real": 0.2, "imag": 0.0}, {"real": 0.0, "imag": 0.0}, {"real": 0.0, "imag": 0.0}),
            right_ward_residual_vector=({"real": 0.2, "imag": 0.0}, {"real": 0.0, "imag": 0.0}, {"real": 0.0, "imag": 0.0}),
        ),
    )

    class Model:
        name = "symmetry_bdg_2band"

        @staticmethod
        def build_pairing_params(delta0_eV):
            return SimpleNamespace(delta0_eV=delta0_eV)

    monkeypatch.setattr(triage, "get_finite_q_validation_model", lambda model_name: Model())
    monkeypatch.setitem(
        sys.modules,
        "validation.scripts.bdg_finite_q.finite_q_ward_scan",
        SimpleNamespace(run_finite_q_ward_scan=lambda *args, **kwargs: SimpleNamespace(rows=rows)),
    )

    payload = triage.run_contact_cancellation_triage(pairings=("spm",), nk=2)
    spm = payload["by_pairing"]["spm"]

    assert payload["available"] is True
    assert payload["valid_for_casimir_input"] is False
    assert "cosine_bubble_vs_minus_direct" in spm
    assert "cancellation_fraction" in spm
    assert spm["interpretation"] in {
        "bubble_and_direct_cancel_well",
        "contact_magnitude_mismatch",
        "contact_direction_mismatch",
        "bubble_dominant",
        "direct_dominant",
        "unknown",
    }


def test_normal_contact_direct_audit_reports_fields(monkeypatch):
    bubble = np.eye(3, dtype=complex)
    direct = np.zeros((3, 3), dtype=complex)
    direct[1, 1] = 2.0
    direct[2, 2] = 3.0

    monkeypatch.setattr(
        triage,
        "get_finite_q_validation_model",
        lambda model_name: SimpleNamespace(name=model_name, spec=object()),
    )
    monkeypatch.setattr(triage, "uniform_bz_mesh", lambda nk: np.zeros((nk, 2), dtype=float))
    monkeypatch.setattr(triage, "k_weights", lambda points: np.ones(points.shape[0]) / points.shape[0])
    monkeypatch.setattr(
        triage.KuboConfig,
        "from_kelvin",
        staticmethod(lambda **kwargs: SimpleNamespace(omega_eV=kwargs["omega_eV"])),
    )
    monkeypatch.setattr(
        triage,
        "normal_physical_density_current_response_components_imag_axis_from_model",
        lambda *args, **kwargs: {
            "bubble": bubble,
            "direct": direct,
            "total": bubble + direct,
        },
    )

    payload = triage.run_normal_contact_direct_audit(nk=2)

    assert payload["available"] is True
    assert payload["valid_for_casimir_input"] is False
    assert payload["matrix_shape"] == [3, 3]
    assert payload["direct_block_interpretation"] == "current_current_only"
    assert "direct_nonzero_pattern" in payload
    assert "residual_component_audit" in payload
    assert "direct_sign_candidates" in payload
    assert "q_scaling" in payload
    assert "summary" in payload
    assert payload["summary"]["valid_for_casimir_input"] is False


def test_normal_contact_direct_audit_detects_current_current_block(monkeypatch):
    bubble = np.zeros((3, 3), dtype=complex)
    direct = np.zeros((3, 3), dtype=complex)
    direct[1:3, 1:3] = np.eye(2)

    monkeypatch.setattr(
        triage,
        "get_finite_q_validation_model",
        lambda model_name: SimpleNamespace(name=model_name, spec=object()),
    )
    monkeypatch.setattr(triage, "uniform_bz_mesh", lambda nk: np.zeros((nk, 2), dtype=float))
    monkeypatch.setattr(triage, "k_weights", lambda points: np.ones(points.shape[0]) / points.shape[0])
    monkeypatch.setattr(
        triage.KuboConfig,
        "from_kelvin",
        staticmethod(lambda **kwargs: SimpleNamespace(omega_eV=kwargs["omega_eV"])),
    )
    monkeypatch.setattr(
        triage,
        "normal_physical_density_current_response_components_imag_axis_from_model",
        lambda *args, **kwargs: {
            "bubble": bubble,
            "direct": direct,
            "total": bubble + direct,
        },
    )

    payload = triage.run_normal_contact_direct_audit(nk=2)

    assert payload["direct_block_interpretation"] == "current_current_only"
    assert payload["direct_nonzero_pattern"]["density_current_norm"] == 0.0
    assert payload["direct_nonzero_pattern"]["current_density_norm"] == 0.0


def test_normal_contact_direct_audit_flags_sign_or_magnitude_candidate(monkeypatch):
    bubble = np.eye(3, dtype=complex)
    direct = np.eye(3, dtype=complex)

    monkeypatch.setattr(
        triage,
        "get_finite_q_validation_model",
        lambda model_name: SimpleNamespace(name=model_name, spec=object()),
    )
    monkeypatch.setattr(triage, "uniform_bz_mesh", lambda nk: np.zeros((nk, 2), dtype=float))
    monkeypatch.setattr(triage, "k_weights", lambda points: np.ones(points.shape[0]) / points.shape[0])
    monkeypatch.setattr(
        triage.KuboConfig,
        "from_kelvin",
        staticmethod(lambda **kwargs: SimpleNamespace(omega_eV=kwargs["omega_eV"])),
    )
    monkeypatch.setattr(
        triage,
        "normal_physical_density_current_response_components_imag_axis_from_model",
        lambda *args, **kwargs: {
            "bubble": bubble,
            "direct": direct,
            "total": bubble + direct,
        },
    )

    payload = triage.run_normal_contact_direct_audit(nk=2)

    assert payload["summary"]["suspected_issue"] in {
        "direct_sign_suspicious",
        "direct_magnitude_suspicious",
        "direct_has_density_mixing",
    }


def test_contact_aware_helper_algebra_for_current_contact():
    contact = np.zeros((3, 3), dtype=complex)
    contact[1, 1] = 0.706
    q = np.array([0.01, 0.0])

    left, right = physical_ward_residuals(contact, 0.01, q)
    rhs_left, rhs_right = contact_ward_rhs(contact, q)

    assert left[1] == pytest.approx(0.00706)
    assert right[1] == pytest.approx(-0.00706)
    np.testing.assert_allclose(left, rhs_left)
    np.testing.assert_allclose(right, rhs_right)


def test_schur_ward_rectangular_contraction_reproduces_existing_validator_for_dwave():
    payload = run_bdg_schur_ward_algebra_localization(pairing_name="dwave")
    reproduction = payload["validator_reproduction"]

    assert reproduction["matches_existing_validator"] is True
    assert reproduction["left_difference_norm"] <= 1e-14
    assert reproduction["right_difference_norm"] <= 1e-14
    assert payload["valid_for_casimir_input"] is False


def test_schur_ward_localization_returns_requested_fields_for_dwave():
    payload = run_bdg_schur_ward_algebra_localization(pairing_name="dwave")

    assert payload["pairing_name"] == "dwave"
    assert payload["matrix_shapes"] == {
        "K_AA": [3, 3],
        "K_Aeta": [3, 2],
        "K_etaA": [2, 3],
        "K_etaeta": [2, 2],
    }
    assert len(payload["candidates"]) == 12
    for candidate in payload["candidates"]:
        assert set(candidate) >= {
            "candidate_name",
            "R",
            "left_aa_norm",
            "left_aeta_norm",
            "right_aa_norm",
            "right_etaa_norm",
            "max_norm",
            "classification",
            "valid_for_casimir_input",
        }
        assert candidate["classification"] in {
            "both_small",
            "first_small_second_large",
            "first_large_second_small",
            "both_large",
        }
        assert candidate["valid_for_casimir_input"] is False


def test_schur_ward_localization_returns_requested_fields_for_spm():
    payload = run_bdg_schur_ward_algebra_localization(pairing_name="spm")

    assert payload["pairing_name"] == "spm"
    assert len(payload["candidates"]) == 12
    assert payload["best_candidate"]["candidate_name"] in {candidate["candidate_name"] for candidate in payload["candidates"]}
    assert payload["validator_reproduction"]["matches_existing_validator"] is True


def test_schur_ward_localization_best_candidate_has_minimal_max_norm():
    payload = run_bdg_schur_ward_algebra_localization(pairing_name="dwave")
    best = payload["best_candidate"]

    assert best["max_norm"] == min(candidate["max_norm"] for candidate in payload["candidates"])


def test_schur_ward_localization_does_not_write_validation_outputs():
    output_dir = ROOT / "validation" / "outputs" / "finite_q_ward"
    before = sorted(path.relative_to(output_dir) for path in output_dir.rglob("*"))

    run_bdg_schur_ward_algebra_localization(pairing_name="dwave")
    run_bdg_schur_ward_algebra_localization(pairing_name="spm")

    after = sorted(path.relative_to(output_dir) for path in output_dir.rglob("*"))
    assert after == before


def test_bdg_ward_criterion_bare_total_uses_minus_direct_vector_algebra():
    rows = _criterion_rows(pairings=("spm",), q_values=(0.01,), closure_delta=1e-8)

    payload = triage.evaluate_finite_q_bdg_ward_criterion(
        finite_q_rows=rows,
        pairings=("spm",),
        q_values=(0.01,),
        q0_precondition_status={"spm": "convention_aware_pass"},
    )

    spm_rows = payload["by_pairing"]["spm"]["rows"]
    bare_total = next(row for row in spm_rows if row["response_name"] == "bare_total")
    direct = np.array([0.0, 0.3, 0.0])
    total = direct + np.array([2e-8, 0.0, 0.0])

    assert payload["evaluated"] is True
    assert bare_total["criterion_type"] == "contact_aware_total"
    assert bare_total["primary_residual_kind"] == "contact_aware_minus_direct"
    assert bare_total["contact_rhs_norm"] == pytest.approx(np.linalg.norm(direct))
    assert bare_total["primary_residual_norm"] == pytest.approx(np.linalg.norm(total - direct))
    assert bare_total["minus_direct_residual_norm"] == pytest.approx(np.linalg.norm(total - direct))
    assert bare_total["passed"] is True
    assert bare_total["valid_for_casimir_input"] is False


def test_bdg_ward_criterion_final_response_uses_homogeneous_primary_residual():
    payload = triage.evaluate_finite_q_bdg_ward_criterion(
        finite_q_rows=_criterion_rows(pairings=("spm",), q_values=(0.01,), closure_delta=1e-8),
        pairings=("spm",),
        q_values=(0.01,),
        q0_precondition_status={"spm": "convention_aware_pass"},
        absolute_tol=1e-6,
        relative_tol=1e-6,
    )

    closure = next(
        row
        for row in payload["by_pairing"]["spm"]["rows"]
        if row["response_name"] == "amplitude_phase_schur"
    )

    assert payload["ward_identity_closed"] is True
    assert closure["criterion_type"] == "collective_corrected_final"
    assert closure["primary_residual_kind"] == "homogeneous"
    assert closure["primary_residual_norm"] == pytest.approx(closure["homogeneous_residual_norm"])
    assert closure["minus_direct_residual_norm"] == pytest.approx(0.3)
    assert closure["passed"] is True
    assert payload["by_pairing"]["spm"]["max_closure_primary_residual_norm"] == pytest.approx(1e-8)


def test_bdg_ward_criterion_refuses_norm_only_rows():
    rows = [
        {
            "pairing_name": "spm",
            "response_name": name,
            "q_model": [0.01, 0.0],
            "max_ward_residual_norm": 0.0,
        }
        for name in ("bare_bubble", "direct", "bare_total", "minus_schur", "amplitude_phase_schur")
    ]

    payload = triage.evaluate_finite_q_bdg_ward_criterion(
        finite_q_rows=rows,
        pairings=("spm",),
        q_values=(0.01,),
        q0_precondition_status={"spm": "convention_aware_pass"},
    )

    spm = payload["by_pairing"]["spm"]
    assert payload["evaluated"] is False
    assert payload["ward_identity_closed"] is False
    assert spm["blocking_reason"] == "missing_residual_vector"
    assert spm["closed"] is False


def test_bdg_ward_criterion_passes_when_all_pairings_close():
    payload = triage.evaluate_finite_q_bdg_ward_criterion(
        finite_q_rows=_criterion_rows(closure_delta=1e-8),
        pairings=("spm", "dwave"),
        q_values=(0.01,),
        q0_precondition_status={"spm": "convention_aware_pass", "dwave": "convention_aware_pass"},
        absolute_tol=1e-6,
        relative_tol=1e-6,
    )

    assert payload["evaluated"] is True
    assert payload["ward_identity_closed"] is True
    assert payload["summary"]["closed_pairings"] == ["spm", "dwave"]
    assert payload["summary"]["failed_pairings"] == []
    assert payload["valid_for_casimir_input"] is False


def test_bdg_ward_criterion_reports_largest_blocker_on_failure():
    rows = _criterion_rows(closure_delta=1e-8)
    rows = [
        _vector_row("dwave", "amplitude_phase_schur", 0.01, [2e-3, 0.0, 0.0])
        if row.pairing_name == "dwave" and row.response_name == "amplitude_phase_schur"
        else row
        for row in rows
    ]

    payload = triage.evaluate_finite_q_bdg_ward_criterion(
        finite_q_rows=rows,
        pairings=("spm", "dwave"),
        q_values=(0.01,),
        q0_precondition_status={"spm": "convention_aware_pass", "dwave": "convention_aware_pass"},
        absolute_tol=1e-6,
        relative_tol=1e-6,
    )

    blocker = payload["summary"]["largest_blocker"]
    assert payload["ward_identity_closed"] is False
    assert payload["by_pairing"]["dwave"]["closed"] is False
    assert blocker["pairing_name"] == "dwave"
    assert blocker["response_name"] == "amplitude_phase_schur"
    assert blocker["primary_residual_norm"] == pytest.approx(2e-3)
    assert payload["by_pairing"]["dwave"]["blocking_reason"] == "closure_primary_residual_above_tolerance"


def test_normal_ward_convention_audit_reports_fields(monkeypatch):
    bubble = np.zeros((3, 3), dtype=complex)
    direct = np.zeros((3, 3), dtype=complex)
    direct[1, 1] = 0.7

    monkeypatch.setattr(
        triage,
        "get_finite_q_validation_model",
        lambda model_name: SimpleNamespace(name=model_name, spec=object()),
    )
    monkeypatch.setattr(triage, "uniform_bz_mesh", lambda nk: np.zeros((nk, 2), dtype=float))
    monkeypatch.setattr(triage, "k_weights", lambda points: np.ones(points.shape[0]) / points.shape[0])
    monkeypatch.setattr(
        triage.KuboConfig,
        "from_kelvin",
        staticmethod(lambda **kwargs: SimpleNamespace(omega_eV=kwargs["omega_eV"])),
    )
    monkeypatch.setattr(
        triage,
        "normal_physical_density_current_response_components_imag_axis_from_model",
        lambda *args, **kwargs: {
            "bubble": bubble,
            "direct": direct,
            "total": bubble + direct,
        },
    )

    payload = triage.run_normal_ward_convention_audit(nk=2)

    assert payload["available"] is True
    assert payload["valid_for_casimir_input"] is False
    assert "homogeneous_residuals" in payload
    assert "contact_aware_candidates" in payload
    assert "contact_rhs" in payload
    assert "consistency_checks" in payload
    assert "summary" in payload
    assert payload["contact_rhs"]["match_error_left"] == pytest.approx(0.0)
    assert payload["contact_rhs"]["match_error_right"] == pytest.approx(0.0)


def test_normal_ward_convention_audit_detects_homogeneous_total_contact_rhs(monkeypatch):
    bubble = np.zeros((3, 3), dtype=complex)
    direct = np.zeros((3, 3), dtype=complex)
    direct[1, 1] = 0.7

    monkeypatch.setattr(
        triage,
        "get_finite_q_validation_model",
        lambda model_name: SimpleNamespace(name=model_name, spec=object()),
    )
    monkeypatch.setattr(triage, "uniform_bz_mesh", lambda nk: np.zeros((nk, 2), dtype=float))
    monkeypatch.setattr(triage, "k_weights", lambda points: np.ones(points.shape[0]) / points.shape[0])
    monkeypatch.setattr(
        triage.KuboConfig,
        "from_kelvin",
        staticmethod(lambda **kwargs: SimpleNamespace(omega_eV=kwargs["omega_eV"])),
    )
    monkeypatch.setattr(
        triage,
        "normal_physical_density_current_response_components_imag_axis_from_model",
        lambda *args, **kwargs: {
            "bubble": bubble,
            "direct": direct,
            "total": bubble + direct,
        },
    )

    payload = triage.run_normal_ward_convention_audit(nk=2)

    assert payload["summary"]["suspected_issue"] == "homogeneous_total_includes_contact_rhs"
    assert payload["consistency_checks"]["total_minus_direct_matches_bubble"] is True


def test_normal_bubble_convergence_audit_reports_fields(monkeypatch):
    bubble = np.eye(3, dtype=complex)

    monkeypatch.setattr(
        triage,
        "get_finite_q_validation_model",
        lambda model_name: SimpleNamespace(name=model_name, spec=object()),
    )
    monkeypatch.setattr(triage, "uniform_bz_mesh", lambda nk: np.zeros((nk, 2), dtype=float))
    monkeypatch.setattr(triage, "k_weights", lambda points: np.ones(points.shape[0]) / points.shape[0])
    monkeypatch.setattr(
        triage.KuboConfig,
        "from_kelvin",
        staticmethod(lambda **kwargs: SimpleNamespace(omega_eV=kwargs["omega_eV"])),
    )
    monkeypatch.setattr(
        triage,
        "normal_physical_density_current_response_components_imag_axis_from_model",
        lambda *args, **kwargs: {
            "bubble": bubble,
            "direct": np.zeros((3, 3), dtype=complex),
            "total": bubble,
        },
    )

    payload = triage.run_normal_bubble_convergence_audit(
        base_nk=2,
        nk_values=(2, 3),
        q_values=(0.005, 0.01),
        omega_values=(0.005, 0.01),
    )

    assert payload["available"] is True
    assert payload["valid_for_casimir_input"] is False
    assert "base_point" in payload
    assert "nk_trend" in payload
    assert "q_trend" in payload
    assert "omega_trend" in payload
    assert "summary" in payload
    assert "run_config" in payload
    assert payload["run_config"]["nk_values"] == [2, 3]
    assert payload["run_config"]["q_values"] == [0.005, 0.01]
    assert payload["run_config"]["omega_values"] == [0.005, 0.01]
    assert payload["run_config"]["computed_in_current_run"] is True
    assert payload["base_point"]["valid_for_casimir_input"] is False


def test_normal_bubble_trend_classifiers():
    assert triage._classify_monotonic_trend([3.0, 2.0, 1.0]) == "decreasing"
    assert triage._classify_monotonic_trend([1.0, 1.01, 0.99]) == "flat"
    assert triage._classify_monotonic_trend([1.0, 3.0, 2.0]) == "nonmonotonic"
    assert triage._classify_monotonic_trend([1.0]) == "inconclusive"


def test_normal_bubble_q_scaling_slope():
    slope = triage._scaling_slope([0.005, 0.01, 0.02], [0.005, 0.01, 0.02])
    assert slope == pytest.approx(1.0)
    assert triage._scaling_slope([0.005], [0.005]) is None
    assert triage._classify_q_residual_trend([0.005, 0.01, 0.02], [0.005, 0.01, 0.02]) == "linear"


def _synthetic_per_k_records(contributions, *, total=None):
    records = []
    for index, value in enumerate(contributions):
        records.append(
            {
                "k_index": [index, 0],
                "k": [float(index), 0.0],
                "k_plus_q": [float(index) + 0.01, 0.0],
                "band_pair": [0, index % 2],
                "contribution": {"real": float(value), "imag": 0.0, "abs": abs(float(value))},
                "energy": {
                    "epsilon_m_k": 0.0005 if index == 0 else 0.2,
                    "epsilon_n_k_plus_q": 0.1,
                    "energy_difference": -0.0995,
                    "abs_energy_difference": 0.0995,
                },
                "occupation": {
                    "f_m_k": 1.0,
                    "f_n_k_plus_q": 0.0 if index == 0 else 0.8,
                    "occupation_difference": 1.0 if index == 0 else 0.2,
                    "abs_occupation_difference": 1.0 if index == 0 else 0.2,
                },
                "denominator": {"real": 0.0 if index == 0 else 0.1, "imag": 0.0001, "abs": 0.0001 if index == 0 else 0.1},
                "vertices": {
                    "density_vertex_abs": 1.0,
                    "current_x_vertex_abs": 10.0 if index == 0 else 0.1,
                    "current_y_vertex_abs": 0.1,
                    "vertex_product_abs": 10.0 if index == 0 else 0.1,
                },
            }
        )
    total_value = float(sum(contributions) if total is None else total)
    return {
        "target_component": "left_current_x",
        "q_model": [0.01, 0.0],
        "omega_eV": 0.01,
        "mesh_size": len(contributions),
        "total_residual_component": {"real": total_value, "imag": 0.0, "abs": abs(total_value)},
        "total_residual_max_norm": abs(total_value),
        "dominant_component": "left_current_x",
        "records": records,
    }


def test_normal_bubble_per_k_outlier_audit_reports_fields(monkeypatch):
    monkeypatch.setattr(
        triage,
        "get_finite_q_validation_model",
        lambda model_name: SimpleNamespace(name=model_name, spec=object()),
    )
    monkeypatch.setattr(triage, "uniform_bz_mesh", lambda nk: np.zeros((nk, 2), dtype=float))
    monkeypatch.setattr(triage, "k_weights", lambda points: np.ones(points.shape[0]) / points.shape[0])
    monkeypatch.setattr(
        triage,
        "normal_physical_bubble_ward_contribution_records_from_model",
        lambda *args, **kwargs: _synthetic_per_k_records([10.0, 1.0, 0.5]),
    )

    payload = triage.run_normal_bubble_per_k_outlier_audit(
        cases=({"case_name": "synthetic", "nk": 3, "shift": (0.0, 0.0)},),
        top_n=2,
    )
    case = payload["cases"][0]

    assert payload["available"] is True
    assert payload["valid_for_casimir_input"] is False
    assert case["top_contributors"]
    assert "concentration" in case
    assert "cross_case_summary" in payload
    assert "summary" in payload
    assert case["sum_matches_total_component"] is True


def test_normal_bubble_per_k_outlier_audit_detects_sum_mismatch(monkeypatch):
    monkeypatch.setattr(
        triage,
        "get_finite_q_validation_model",
        lambda model_name: SimpleNamespace(name=model_name, spec=object()),
    )
    monkeypatch.setattr(triage, "uniform_bz_mesh", lambda nk: np.zeros((nk, 2), dtype=float))
    monkeypatch.setattr(triage, "k_weights", lambda points: np.ones(points.shape[0]) / points.shape[0])
    monkeypatch.setattr(
        triage,
        "normal_physical_bubble_ward_contribution_records_from_model",
        lambda *args, **kwargs: _synthetic_per_k_records([1.0, 1.0], total=5.0),
    )

    payload = triage.run_normal_bubble_per_k_outlier_audit(
        cases=({"case_name": "mismatch", "nk": 2, "shift": (0.0, 0.0)},),
    )

    assert payload["cases"][0]["sum_matches_total_component"] is False
    assert payload["cases"][0]["diagnosis"]["suspected_issue"] == "per_k_sum_mismatch"


def test_normal_bubble_per_k_outlier_classifiers():
    single = triage._bubble_case_diagnosis(
        abs_values=np.array([10.0, 1.0, 0.5]),
        top_records=[],
        sum_matches=True,
    )
    few = triage._bubble_case_diagnosis(
        abs_values=np.array([2.0, 2.0, 2.0, 2.0, 2.0, 0.1, 0.1]),
        top_records=[],
        sum_matches=True,
    )
    broad = triage._bubble_case_diagnosis(
        abs_values=np.ones(30),
        top_records=[],
        sum_matches=True,
    )

    assert single["suspected_issue"] == "single_k_outlier"
    assert few["suspected_issue"] == "few_k_outliers"
    assert broad["suspected_issue"] in {"broad_mesh_aliasing", "outlier_unresolved"}


def test_normal_bubble_per_k_contributor_classification(monkeypatch):
    record = _synthetic_per_k_records([10.0])["records"][0]
    payload = triage._bubble_top_contributor_payload(
        record,
        rank=1,
        target_component="left_current_x",
        denominator_abs=np.array([0.0001, 0.1, 0.2]),
        vertex_abs=np.array([0.1, 0.2, 10.0]),
    )

    assert payload["classification"]["small_denominator"] is True
    assert payload["classification"]["occupation_jump"] is True
    assert payload["classification"]["large_vertex_product"] is True
    assert payload["classification"]["near_fermi"] is True


def test_report_builder_includes_ward_triage_when_enabled(monkeypatch, tmp_path):
    module = _load_report_module()
    scan_report = SimpleNamespace(
        model_name="symmetry_bdg_2band",
        primary_validation_model=True,
        q0_precondition_status={"spm": "convention_aware_pass", "dwave": "convention_aware_pass"},
        diagnostic_run_completed=True,
        ward_identity_closed=False,
        workspace_evaluation=True,
        pairing_names=("spm", "dwave"),
        rows=(),
    )
    q0_reports = (
        SimpleNamespace(
            pairing_name="spm",
            status="convention_aware_pass",
            passed=True,
            comparator_family="local_bdg",
            q0_comparator_available=True,
            best_transformed_match={},
        ),
        SimpleNamespace(
            pairing_name="dwave",
            status="convention_aware_pass",
            passed=True,
            comparator_family="local_bdg",
            q0_comparator_available=True,
            best_transformed_match={},
        ),
    )

    class Model:
        name = "symmetry_bdg_2band"
        primary_validation_model = True

        @staticmethod
        def require_pairing(pairing):
            return None

        @staticmethod
        def build_pairing_params(delta0):
            return SimpleNamespace(delta0_eV=delta0)

    monkeypatch.setattr(module, "get_finite_q_validation_model", lambda model_name: Model())
    monkeypatch.setattr(module, "run_q0_bdg_response_alignment_many", lambda *args, **kwargs: q0_reports)
    monkeypatch.setattr(module, "run_finite_q_ward_scan", lambda *args, **kwargs: scan_report)
    captured_triage_kwargs = {}

    def fake_run_ward_triage(**kwargs):
        captured_triage_kwargs.update(kwargs)
        return {
            "normal_finite_q": {"suspected_layer": "normal_closed", "valid_for_casimir_input": False},
            "operator_identity": {"suspected_layer": "response_assembly_or_collective", "valid_for_casimir_input": False},
            "contact_cancellation": {"by_pairing": {}, "valid_for_casimir_input": False},
            "normal_contact_direct_audit": {
                "summary": {
                    "suspected_issue": "normal_contact_unresolved",
                    "recommended_next_fix": "inspect normal contact",
                    "valid_for_casimir_input": False,
                },
                "valid_for_casimir_input": False,
            },
            "normal_ward_convention_audit": {
                "summary": {
                    "suspected_issue": "homogeneous_total_includes_contact_rhs",
                    "recommended_next_fix": "separate convention checks",
                    "valid_for_casimir_input": False,
                },
                "valid_for_casimir_input": False,
            },
            "normal_bubble_convergence_audit": {
                "summary": {
                    "suspected_issue": "bubble_residual_unresolved",
                    "recommended_next_fix": "inspect bubble",
                    "valid_for_casimir_input": False,
                },
                "valid_for_casimir_input": False,
            },
            "summary": {
                "suspected_layer": "normal_ward_convention",
                "recommended_next_fix": "separate convention checks",
                "valid_for_casimir_input": False,
            },
        }

    monkeypatch.setattr(
        module,
        "_run_ward_triage",
        fake_run_ward_triage,
    )

    report = module.build_report(
        model_name="symmetry_bdg_2band",
        output_dir=tmp_path,
        nk=2,
        omega=0.01,
        delta0=0.1,
        q_values=(0.01,),
        pairings=("spm", "dwave"),
        bubble_audit_nk_values=(5, 7),
        bubble_audit_q_values=(0.003, 0.006),
        bubble_audit_omega_values=(0.004, 0.008),
        bubble_audit_mesh_shifts_enabled=False,
    )
    markdown = module.format_markdown(report)

    assert "ward_triage" in report
    assert captured_triage_kwargs["bubble_audit_nk_values"] == (5, 7)
    assert captured_triage_kwargs["bubble_audit_q_values"] == (0.003, 0.006)
    assert captured_triage_kwargs["bubble_audit_omega_values"] == (0.004, 0.008)
    assert captured_triage_kwargs["bubble_audit_mesh_shifts_enabled"] is False
    assert report["ward_triage_run_config"]["normal_bubble_convergence_audit"]["nk_values"] == [5, 7]
    assert report["ward_triage_run_config"]["normal_bubble_convergence_audit"]["q_values"] == [0.003, 0.006]
    assert report["ward_triage_run_config"]["normal_bubble_convergence_audit"]["omega_values"] == [0.004, 0.008]
    assert report["ward_triage_run_config"]["normal_bubble_convergence_audit"]["computed_in_current_run"] is True
    assert report["ward_triage"]["summary"]["valid_for_casimir_input"] is False
    assert "normal_contact_direct_audit" in report["ward_triage"]
    assert "normal_ward_convention_audit" in report["ward_triage"]
    assert "normal_bubble_convergence_audit" in report["ward_triage"]
    assert "## Ward triage" in markdown
    assert "normal contact/direct audit" in markdown
    assert "normal Ward convention audit" in markdown
    assert "recommended Ward convention fix" in markdown
    assert "normal bubble convergence audit" in markdown
    assert "recommended normal bubble fix" in markdown
    assert "normal bubble audit config" in markdown
    assert "suspected primary layer" in markdown


def test_report_builder_includes_formal_ward_criterion_and_status_linkage(monkeypatch, tmp_path):
    module = _load_report_module()
    scan_report = SimpleNamespace(
        model_name="symmetry_bdg_2band",
        primary_validation_model=True,
        q0_precondition_status={"spm": "convention_aware_pass", "dwave": "convention_aware_pass"},
        diagnostic_run_completed=True,
        ward_identity_closed=False,
        workspace_evaluation=True,
        pairing_names=("spm", "dwave"),
        rows=tuple(_criterion_rows()),
    )
    q0_reports = (
        SimpleNamespace(
            pairing_name="spm",
            status="convention_aware_pass",
            passed=True,
            comparator_family="local_bdg",
            q0_comparator_available=True,
            best_transformed_match={},
        ),
        SimpleNamespace(
            pairing_name="dwave",
            status="convention_aware_pass",
            passed=True,
            comparator_family="local_bdg",
            q0_comparator_available=True,
            best_transformed_match={},
        ),
    )

    class Model:
        name = "symmetry_bdg_2band"
        primary_validation_model = True

        @staticmethod
        def require_pairing(pairing):
            return None

        @staticmethod
        def build_pairing_params(delta0):
            return SimpleNamespace(delta0_eV=delta0)

    monkeypatch.setattr(module, "get_finite_q_validation_model", lambda model_name: Model())
    monkeypatch.setattr(module, "run_q0_bdg_response_alignment_many", lambda *args, **kwargs: q0_reports)
    monkeypatch.setattr(module, "run_finite_q_ward_scan", lambda *args, **kwargs: scan_report)

    report = module.build_report(
        model_name="symmetry_bdg_2band",
        output_dir=tmp_path,
        nk=2,
        omega=0.01,
        delta0=0.1,
        q_values=(0.01,),
        pairings=("spm", "dwave"),
        include_triage=False,
    )
    markdown = module.format_markdown(report)

    assert "ward_criterion" in report
    assert report["ward_criterion"]["criterion_version"] == "contact_aware_v1"
    assert report["ward_criterion"]["ward_identity_closed"] is True
    assert report["finite_q_status"]["ward_identity_closed"] == report["ward_criterion"]["ward_identity_closed"]
    assert report["finite_q_rows"][0]["left_ward_residual_vector"]
    assert "## Ward criterion" in markdown
    assert "full_bdg_ward_closed: True" in markdown
    assert "spm max primary closure residual" in markdown
    assert "dwave max primary closure residual" in markdown
    assert "max contact-aware closure residual" not in markdown
    assert "valid_for_casimir_input: False" in markdown


def test_report_cli_bubble_audit_defaults_and_custom_values():
    module = _load_report_module()

    defaults = module.parse_args([])
    assert defaults.bubble_audit_nk_values == [7, 9, 11]
    assert defaults.bubble_audit_q_values == [0.005, 0.01, 0.02]
    assert defaults.bubble_audit_omega_values == [0.005, 0.01, 0.02]
    assert defaults.disable_bubble_audit_mesh_shifts is False
    assert defaults.ward_criterion == "contact_aware_v1"
    assert defaults.bdg_ward_closure_response == "amplitude_phase_schur"
    assert defaults.bdg_ward_absolute_tol == 1e-6
    assert defaults.bdg_ward_relative_tol == 1e-6

    custom = module.parse_args(
        [
            "--ward-criterion",
            "contact_aware_v1",
            "--bdg-ward-closure-response",
            "minus_schur",
            "--bdg-ward-absolute-tol",
            "1e-7",
            "--bdg-ward-relative-tol",
            "2e-7",
            "--bubble-audit-nk-values",
            "5",
            "7",
            "--bubble-audit-q-values",
            "0.003",
            "0.006",
            "--bubble-audit-omega-values",
            "0.004",
            "0.008",
            "--disable-bubble-audit-mesh-shifts",
        ]
    )
    assert custom.bubble_audit_nk_values == [5, 7]
    assert custom.bubble_audit_q_values == [0.003, 0.006]
    assert custom.bubble_audit_omega_values == [0.004, 0.008]
    assert custom.disable_bubble_audit_mesh_shifts is True
    assert custom.bdg_ward_closure_response == "minus_schur"
    assert custom.bdg_ward_absolute_tol == 1e-7
    assert custom.bdg_ward_relative_tol == 2e-7


def test_report_command_text_records_bubble_audit_provenance():
    module = _load_report_module()
    args = module.parse_args(
        [
            "--bubble-audit-nk-values",
            "5",
            "7",
            "--bubble-audit-q-values",
            "0.003",
            "0.006",
            "--bubble-audit-omega-values",
            "0.004",
            "0.008",
            "--disable-bubble-audit-mesh-shifts",
        ]
    )

    command = module._command_text(args)

    assert "--bubble-audit-nk-values 5 7" in command
    assert "--bubble-audit-q-values 0.003 0.006" in command
    assert "--bubble-audit-omega-values 0.004 0.008" in command
    assert "--disable-bubble-audit-mesh-shifts" in command
    assert "--ward-criterion contact_aware_v1" in command
    assert "--bdg-ward-closure-response amplitude_phase_schur" in command
    assert "--bdg-ward-absolute-tol 1e-06" in command
    assert "--bdg-ward-relative-tol 1e-06" in command


def test_report_bubble_outlier_cli_and_command_text():
    module = _load_report_module()
    args = module.parse_args(["--include-bubble-outlier-audit", "--bubble-outlier-top-n", "5"])

    command = module._command_text(args)

    assert args.include_bubble_outlier_audit is True
    assert args.bubble_outlier_top_n == 5
    assert "--include-bubble-outlier-audit" in command
    assert "--bubble-outlier-top-n 5" in command


def test_report_bubble_outlier_integration_default_and_enabled(monkeypatch, tmp_path):
    module = _load_report_module()
    scan_report = SimpleNamespace(
        model_name="symmetry_bdg_2band",
        primary_validation_model=True,
        q0_precondition_status={"spm": "convention_aware_pass"},
        diagnostic_run_completed=True,
        ward_identity_closed=False,
        workspace_evaluation=True,
        pairing_names=("spm",),
        rows=(),
    )
    q0_reports = (
        SimpleNamespace(
            pairing_name="spm",
            status="convention_aware_pass",
            passed=True,
            comparator_family="local_bdg",
            q0_comparator_available=True,
            best_transformed_match={},
        ),
    )

    class Model:
        name = "symmetry_bdg_2band"
        primary_validation_model = True

        @staticmethod
        def require_pairing(pairing):
            return None

        @staticmethod
        def build_pairing_params(delta0):
            return SimpleNamespace(delta0_eV=delta0)

    def fake_run_ward_triage(**kwargs):
        payload = {
            "summary": {
                "suspected_layer": "normal_ward_convention",
                "recommended_next_fix": "separate convention checks",
                "valid_for_casimir_input": False,
            }
        }
        if kwargs["include_bubble_outlier_audit"]:
            payload["normal_bubble_per_k_outlier_audit"] = {
                "summary": {
                    "suspected_issue": "twist_sensitive_fermi_surface_sampling",
                    "recommended_next_fix": "inspect outliers",
                    "valid_for_casimir_input": False,
                }
            }
        return payload

    monkeypatch.setattr(module, "get_finite_q_validation_model", lambda model_name: Model())
    monkeypatch.setattr(module, "run_q0_bdg_response_alignment_many", lambda *args, **kwargs: q0_reports)
    monkeypatch.setattr(module, "run_finite_q_ward_scan", lambda *args, **kwargs: scan_report)
    monkeypatch.setattr(module, "_run_ward_triage", fake_run_ward_triage)

    default_report = module.build_report(
        model_name="symmetry_bdg_2band",
        output_dir=tmp_path,
        nk=2,
        omega=0.01,
        delta0=0.1,
        q_values=(0.01,),
        pairings=("spm",),
    )
    enabled_report = module.build_report(
        model_name="symmetry_bdg_2band",
        output_dir=tmp_path,
        nk=2,
        omega=0.01,
        delta0=0.1,
        q_values=(0.01,),
        pairings=("spm",),
        include_bubble_outlier_audit=True,
        bubble_outlier_top_n=5,
    )
    markdown = module.format_markdown(enabled_report)

    assert "normal_bubble_per_k_outlier_audit" not in default_report["ward_triage"]
    assert "normal_bubble_per_k_outlier_audit" in enabled_report["ward_triage"]
    assert enabled_report["ward_triage_run_config"]["normal_bubble_per_k_outlier_audit"]["enabled"] is True
    assert enabled_report["ward_triage_run_config"]["normal_bubble_per_k_outlier_audit"]["top_n"] == 5
    assert "normal bubble per-k outlier audit" in markdown
    assert "recommended per-k outlier fix" in markdown


def test_report_writer_still_targets_only_three_files(tmp_path):
    module = _load_report_module()
    report = {
        "pairing_summary": {},
        "finite_q_status": {
            "diagnostic_run_completed": True,
            "ward_identity_closed": False,
        },
        "q0_precondition_status": {},
        "valid_for_casimir_input": False,
        "diagnostic_interpretation": {
            "recommended_next_action": "inspect diagnostics",
            "main_observation": "diagnostic-only",
        },
        "ward_triage": {
            "summary": {
                "suspected_layer": "unknown",
                "recommended_next_fix": "run report",
                "valid_for_casimir_input": False,
            }
        },
    }

    module.write_report(report, tmp_path, "python validation/scripts/bdg_finite_q/run_finite_q_ward_report.py\n")

    assert sorted(path.name for path in tmp_path.iterdir()) == ["command.sh", "report.json", "report.md"]
    assert not [path for path in tmp_path.iterdir() if path.is_dir()]
