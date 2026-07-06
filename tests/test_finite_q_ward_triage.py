from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from lno327.collective.ward import contact_ward_rhs, physical_ward_residuals
from validation.lib import finite_q_ward_criterion as criterion
from validation.lib import finite_q_ward_triage as triage
from validation.lib.finite_q_schur_ward_localization import run_bdg_schur_ward_algebra_localization


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
    return {"component": component, "real": float(real), "imag": float(imag)}


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
            rows.extend(
                [
                    _vector_row(pairing, "bare_bubble", q, [1.0, 0.0, 0.0]),
                    _vector_row(pairing, "direct", q, [2.0, 0.0, 0.0]),
                    _vector_row(pairing, "bare_total", q, [3.0, 0.0, 0.0]),
                    _vector_row(pairing, "minus_schur", q, [4.0, 0.0, 0.0]),
                    _vector_row(pairing, "amplitude_phase_schur", q, [closure_delta, 0.0, 0.0]),
                ]
            )
    return rows


def test_normal_finite_q_ward_triage_reports_fields(monkeypatch):
    matrix = np.eye(3, dtype=complex)
    monkeypatch.setattr(triage, "get_finite_q_validation_model", lambda model_name: SimpleNamespace(name=model_name, spec=object()))
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
        lambda *args, **kwargs: {"bare_bubble": matrix, "direct": -0.5 * matrix, "total": 0.5 * matrix},
    )

    payload = triage.run_normal_finite_q_ward_triage(nk=2)

    assert payload["available"] is True
    assert payload["valid_for_casimir_input"] is False
    assert payload["workspace_evaluation"] is True
    assert set(payload["residual_components"]) == {"bare_bubble", "direct", "total"}
    assert payload["component_labels"] == ["density", "current_x", "current_y"]


def test_operator_identity_triage_reports_unavailable_subchecks(monkeypatch):
    monkeypatch.setattr(triage, "get_finite_q_validation_model", lambda model_name: SimpleNamespace(name=model_name, spec=object()))
    monkeypatch.setattr(triage, "uniform_bz_mesh", lambda nk: np.zeros((nk, 2), dtype=float))

    payload = triage.run_operator_identity_triage(nk=2)

    assert payload["available"] is True
    assert payload["valid_for_casimir_input"] is False
    assert "normal_vertex" in payload
    assert "bdg_vertex" in payload
    assert "pairing_sector" in payload
    assert payload["normal_vertex"]["available"] is False
    assert payload["bdg_vertex"]["available"] is False


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


def test_normal_contact_direct_audit_reports_fields(monkeypatch):
    bubble = np.eye(3, dtype=complex)
    direct = np.zeros((3, 3), dtype=complex)
    direct[1, 1] = 2.0
    direct[2, 2] = 3.0
    monkeypatch.setattr(triage, "get_finite_q_validation_model", lambda model_name: SimpleNamespace(name=model_name, spec=object()))
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
        lambda *args, **kwargs: {"bubble": bubble, "direct": direct, "total": bubble + direct},
    )

    payload = triage.run_normal_contact_direct_audit(nk=2)

    assert payload["available"] is True
    assert payload["valid_for_casimir_input"] is False
    assert payload["direct_block_interpretation"] == "current_current_only"
    assert "summary" in payload


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


def test_schur_ward_localization_uses_full_hessian_identity():
    payload = run_bdg_schur_ward_algebra_localization(pairing_name="dwave")
    identity = payload["analytic_identity"]

    assert payload["validator_reproduction"]["matches_existing_validator"] is True
    assert identity["aa_identity"] == "full_hessian"
    assert identity["aa_kernel_definition"] == "K_AA_full = K_AA_bubble + K_AA_direct"
    assert "contact_aware_left_aa_norm" not in identity
    assert identity["max_full_hessian_norm"] == pytest.approx(
        max(
            identity["full_hessian_left_aa_norm"],
            identity["full_hessian_right_aa_norm"],
            identity["left_aeta_norm"],
            identity["right_etaa_norm"],
        )
    )


def test_schur_ward_analytic_generators_are_fixed_by_delta0():
    payload = run_bdg_schur_ward_algebra_localization(pairing_name="dwave", delta0_eV=0.1)
    identity = payload["analytic_identity"]

    assert identity["analytic_R_left"][1]["imag"] == pytest.approx(0.2)
    assert identity["analytic_R_right"][1]["imag"] == pytest.approx(-0.2)


def test_full_hessian_criterion_final_response_uses_homogeneous_primary_residual():
    payload = criterion.evaluate_finite_q_bdg_ward_criterion(
        finite_q_rows=_criterion_rows(pairings=("spm",), q_values=(0.01,), closure_delta=1e-8),
        pairings=("spm",),
        q_values=(0.01,),
        q0_precondition_status={"spm": "convention_aware_pass"},
        absolute_tol=1e-6,
        relative_tol=1e-6,
    )

    closure = payload["by_pairing"]["spm"]["rows"][0]

    assert payload["criterion_version"] == "full_hessian_v1"
    assert payload["ward_identity_closed"] is True
    assert closure["response_name"] == "amplitude_phase_schur"
    assert closure["criterion_type"] == "full_hessian_collective_schur"
    assert closure["primary_residual_kind"] == "homogeneous_full_hessian_schur"
    assert closure["primary_residual_norm"] == pytest.approx(closure["homogeneous_residual_norm"])
    for forbidden in ("minus_direct_residual_norm", "contact_rhs_norm", "contact_aware_residual_norm"):
        assert forbidden not in closure


def test_full_hessian_criterion_refuses_norm_only_rows():
    rows = [
        {
            "pairing_name": "spm",
            "response_name": "amplitude_phase_schur",
            "q_model": [0.01, 0.0],
            "max_ward_residual_norm": 0.0,
        }
    ]

    payload = criterion.evaluate_finite_q_bdg_ward_criterion(
        finite_q_rows=rows,
        pairings=("spm",),
        q_values=(0.01,),
        q0_precondition_status={"spm": "convention_aware_pass"},
    )

    spm = payload["by_pairing"]["spm"]
    assert payload["evaluated"] is False
    assert payload["ward_identity_closed"] is False
    assert spm["blocking_reason"] == "missing_residual_vector"


def test_full_hessian_criterion_reports_largest_blocker_on_failure():
    rows = _criterion_rows(closure_delta=1e-8)
    rows = [
        _vector_row("dwave", "amplitude_phase_schur", 0.01, [2e-3, 0.0, 0.0])
        if row.pairing_name == "dwave" and row.response_name == "amplitude_phase_schur"
        else row
        for row in rows
    ]

    payload = criterion.evaluate_finite_q_bdg_ward_criterion(
        finite_q_rows=rows,
        pairings=("spm", "dwave"),
        q_values=(0.01,),
        q0_precondition_status={"spm": "convention_aware_pass", "dwave": "convention_aware_pass"},
        absolute_tol=1e-6,
        relative_tol=1e-6,
    )

    blocker = payload["summary"]["largest_blocker"]
    assert payload["ward_identity_closed"] is False
    assert blocker["pairing_name"] == "dwave"
    assert blocker["response_name"] == "amplitude_phase_schur"
    assert blocker["primary_residual_norm"] == pytest.approx(2e-3)


def _patch_report_builder(monkeypatch, module, scan_rows=()):
    scan_report = SimpleNamespace(
        model_name="symmetry_bdg_2band",
        primary_validation_model=True,
        q0_precondition_status={"spm": "convention_aware_pass", "dwave": "convention_aware_pass"},
        diagnostic_run_completed=True,
        ward_identity_closed=False,
        workspace_evaluation=True,
        pairing_names=("spm", "dwave"),
        rows=tuple(scan_rows),
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


def test_report_builder_includes_full_hessian_ward_criterion_and_status_linkage(monkeypatch, tmp_path):
    module = _load_report_module()
    _patch_report_builder(monkeypatch, module, _criterion_rows())

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

    assert report["ward_criterion"]["criterion_version"] == "full_hessian_v1"
    assert report["ward_criterion"]["criterion_formal_name"] == "full_hessian_v1"
    assert report["run_config"]["ward_criterion"] == "full_hessian_v1"
    assert report["finite_q_status"]["ward_identity_closed"] is True
    assert "criterion_version: full_hessian_v1" in markdown
    assert "contact_aware_v1" not in markdown


def test_report_cli_defaults_and_command_text_use_full_hessian():
    module = _load_report_module()

    defaults = module.parse_args([])
    assert defaults.ward_criterion == "full_hessian_v1"
    assert defaults.bubble_audit_nk_values == [7, 9, 11]
    assert defaults.bubble_audit_q_values == [0.005, 0.01, 0.02]
    assert defaults.bubble_audit_omega_values == [0.005, 0.01, 0.02]

    command = module._command_text(defaults)
    assert "--ward-criterion full_hessian_v1" in command
    assert "contact_aware_v1" not in command


def test_report_builder_includes_triage_when_enabled(monkeypatch, tmp_path):
    module = _load_report_module()
    _patch_report_builder(monkeypatch, module)
    captured_triage_kwargs = {}

    def fake_run_ward_triage(**kwargs):
        captured_triage_kwargs.update(kwargs)
        return {
            "normal_finite_q": {"suspected_layer": "normal_closed", "valid_for_casimir_input": False},
            "operator_identity": {"suspected_layer": "response_assembly_or_collective", "valid_for_casimir_input": False},
            "contact_cancellation": {"by_pairing": {}, "valid_for_casimir_input": False},
            "normal_contact_direct_audit": {"summary": {"suspected_issue": "normal_contact_unresolved", "recommended_next_fix": "inspect normal contact", "valid_for_casimir_input": False}},
            "normal_ward_convention_audit": {"summary": {"suspected_issue": "convention_check", "recommended_next_fix": "inspect convention", "valid_for_casimir_input": False}},
            "normal_bubble_convergence_audit": {"summary": {"suspected_issue": "bubble_check", "recommended_next_fix": "inspect bubble", "valid_for_casimir_input": False}},
        }

    monkeypatch.setattr(module, "_run_ward_triage", fake_run_ward_triage)

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

    assert "ward_triage" in report
    assert captured_triage_kwargs["bubble_audit_nk_values"] == (5, 7)
    assert captured_triage_kwargs["bubble_audit_q_values"] == (0.003, 0.006)
    assert captured_triage_kwargs["bubble_audit_omega_values"] == (0.004, 0.008)
    assert captured_triage_kwargs["bubble_audit_mesh_shifts_enabled"] is False
    assert report["ward_triage"]["summary"]["valid_for_casimir_input"] is False


def test_report_bubble_outlier_integration_default_and_enabled(monkeypatch, tmp_path):
    module = _load_report_module()
    _patch_report_builder(monkeypatch, module, _criterion_rows(pairings=("spm",)))

    def fake_run_ward_triage(**kwargs):
        payload = {"summary": {"suspected_layer": "diagnostic", "recommended_next_fix": "inspect", "valid_for_casimir_input": False}}
        if kwargs["include_bubble_outlier_audit"]:
            payload["normal_bubble_per_k_outlier_audit"] = {"summary": {"suspected_issue": "twist_sensitive", "recommended_next_fix": "inspect outliers", "valid_for_casimir_input": False}}
        return payload

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

    assert "normal_bubble_per_k_outlier_audit" not in default_report["ward_triage"]
    assert "normal_bubble_per_k_outlier_audit" in enabled_report["ward_triage"]


def test_report_writer_still_targets_only_three_files(tmp_path):
    module = _load_report_module()
    report = {
        "run_config": {"pairings": []},
        "pairing_summary": {},
        "finite_q_status": {"diagnostic_run_completed": True, "ward_identity_closed": False},
        "q0_precondition_status": {},
        "valid_for_casimir_input": False,
        "diagnostic_interpretation": {"recommended_next_action": "inspect diagnostics", "main_observation": "diagnostic-only"},
        "ward_criterion": {
            "criterion_version": "full_hessian_v1",
            "criterion_formal_name": "full_hessian_v1",
            "closure_response_name": "amplitude_phase_schur",
            "ward_identity_closed": False,
            "valid_for_casimir_input": False,
            "summary": {"largest_blocker": None, "recommended_next_fix": "inspect", "valid_for_casimir_input": False},
        },
        "ward_triage": {"summary": {"suspected_layer": "unknown", "recommended_next_fix": "run report", "valid_for_casimir_input": False}},
    }

    module.write_report(report, tmp_path, "python validation/scripts/bdg_finite_q/run_finite_q_ward_report.py\n")

    assert sorted(path.name for path in tmp_path.iterdir()) == ["command.sh", "report.json", "report.md"]
