from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

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
    monkeypatch.setattr(
        module,
        "_run_ward_triage",
        lambda **kwargs: {
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
            "summary": {
                "suspected_layer": "response_assembly_or_collective",
                "recommended_next_fix": "inspect response assembly",
                "valid_for_casimir_input": False,
            },
        },
    )

    report = module.build_report(
        model_name="symmetry_bdg_2band",
        output_dir=tmp_path,
        nk=2,
        omega=0.01,
        delta0=0.1,
        q_values=(0.01,),
        pairings=("spm", "dwave"),
    )
    markdown = module.format_markdown(report)

    assert "ward_triage" in report
    assert report["ward_triage"]["summary"]["valid_for_casimir_input"] is False
    assert "normal_contact_direct_audit" in report["ward_triage"]
    assert "## Ward triage" in markdown
    assert "normal contact/direct audit" in markdown
    assert "suspected primary layer" in markdown


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
