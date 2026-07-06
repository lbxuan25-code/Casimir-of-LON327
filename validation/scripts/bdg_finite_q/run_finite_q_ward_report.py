#!/usr/bin/env python3
"""Unified finite-q Ward diagnostic report writer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from finite_q_ward_scan import run_finite_q_ward_scan  # noqa: E402
from q0_bdg_response_alignment import run_q0_bdg_response_alignment_many  # noqa: E402
from validation.lib.finite_q_validation_models import (  # noqa: E402
    available_finite_q_validation_models,
    get_finite_q_validation_model,
)
from validation.lib.finite_q_ward_triage import (  # noqa: E402
    evaluate_finite_q_bdg_ward_criterion,
    run_contact_cancellation_triage,
    run_normal_bubble_convergence_audit,
    run_normal_bubble_per_k_outlier_audit,
    run_normal_contact_direct_audit,
    run_normal_finite_q_ward_triage,
    run_normal_ward_convention_audit,
    run_operator_identity_triage,
    summarize_ward_triage,
)


DEFAULT_OUTPUT_DIR = ROOT / "validation" / "outputs" / "finite_q_ward"
DEFAULT_Q_VALUES = (0.0025, 0.005, 0.01, 0.02)
DEFAULT_PAIRINGS = ("spm", "dwave")


def _round_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _compact_q0_summary(q0_reports: tuple[Any, ...]) -> dict[str, Any]:
    return {
        report.pairing_name: {
            "status": report.status,
            "passed": bool(report.passed),
            "comparator_family": report.comparator_family,
            "q0_comparator_available": bool(report.q0_comparator_available),
            "best_transformed_match": dict(report.best_transformed_match),
            "valid_for_casimir_input": False,
        }
        for report in q0_reports
    }


def _compact_rows(scan_report: Any) -> list[dict[str, Any]]:
    return [
        {
            "pairing_name": row.pairing_name,
            "response_name": row.response_name,
            "q_model": [float(row.q_model[0]), float(row.q_model[1])],
            "q_norm": float(row.q_norm),
            "left_ward_residual_norm": float(row.left_ward_residual_norm),
            "right_ward_residual_norm": float(row.right_ward_residual_norm),
            "residual_component_labels": list(row.residual_component_labels),
            "left_ward_residual_vector": list(row.left_ward_residual_vector),
            "right_ward_residual_vector": list(row.right_ward_residual_vector),
            "max_ward_residual_norm": float(row.max_ward_residual_norm),
            "residual_ratio_to_bare": _round_float(row.residual_ratio_to_bare),
            "collective_matrix_condition_number": _round_float(row.collective_matrix_condition_number),
            "inverse_method": row.inverse_method,
            "pinv_diagnostic_used": bool(row.pinv_diagnostic_used),
            "valid_for_casimir_input": False,
        }
        for row in scan_report.rows
    ]


def _pairing_summary(scan_report: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for pairing_name in scan_report.pairing_names:
        rows = [row for row in scan_report.rows if row.pairing_name == pairing_name]
        closure_rows = [row for row in rows if row.response_name in {"bare_total", "minus_schur", "amplitude_phase_schur"}]
        max_residual = max((row.max_ward_residual_norm for row in rows), default=None)
        max_closure_residual = max((row.max_ward_residual_norm for row in closure_rows), default=None)
        summary[pairing_name] = {
            "q0_precondition_status": scan_report.q0_precondition_status.get(pairing_name, "missing_q0_status"),
            "row_count": len(rows),
            "max_ward_residual_norm": _round_float(max_residual),
            "max_closure_residual_norm": _round_float(max_closure_residual),
            "ward_identity_closed": bool(
                closure_rows and all(row.max_ward_residual_norm <= 1e-8 for row in closure_rows)
            ),
            "valid_for_casimir_input": False,
        }
    return summary


def _diagnostic_interpretation(scan_report: Any) -> dict[str, Any]:
    blockers: list[str] = []
    if not scan_report.ward_identity_closed:
        blockers.append("finite-q Ward residuals remain above tolerance for at least one closure response")
    if any(status == "diagnostic_only_not_passed" for status in scan_report.q0_precondition_status.values()):
        blockers.append("at least one q=0 precondition is diagnostic-only")
    if not blockers:
        blockers.append("Casimir gating remains intentionally closed for this diagnostic report")
    return {
        "main_observation": (
            "The finite-q diagnostic completed, but ward_identity_closed is False."
            if not scan_report.ward_identity_closed
            else "The finite-q diagnostic completed with closure at the configured tolerance."
        ),
        "suspected_blockers": blockers,
        "recommended_next_action": (
            "Investigate the largest finite-q residual rows before changing any Casimir input gate."
        ),
    }


def _ward_criterion_summary(ward_criterion: dict[str, Any]) -> dict[str, Any]:
    if not ward_criterion.get("evaluated", False):
        suspected_layer = "ward_criterion_incomplete"
        for payload in ward_criterion.get("by_pairing", {}).values():
            if isinstance(payload, dict) and payload.get("blocking_reason") == "missing_response_row":
                suspected_layer = "finite_q_response_reporting"
                break
    elif not ward_criterion.get("ward_identity_closed", False):
        suspected_layer = "bdg_collective_closure"
    else:
        suspected_layer = "none"
    return {
        "suspected_layer": suspected_layer,
        "recommended_next_fix": ward_criterion.get("summary", {}).get(
            "recommended_next_fix", "Inspect the formal Ward criterion rows."
        ),
        "valid_for_casimir_input": False,
    }


def _diagnostic_interpretation_from_criterion(scan_report: Any, ward_criterion: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if not ward_criterion.get("evaluated", False):
        blockers.append("finite-q Ward criterion is incomplete because required residual rows or vectors are missing")
    elif not ward_criterion.get("ward_identity_closed", False):
        blockers.append("contact-aware finite-q BdG Ward criterion failed for at least one requested pairing")
    if any(status == "diagnostic_only_not_passed" for status in scan_report.q0_precondition_status.values()):
        blockers.append("at least one q=0 precondition is diagnostic-only")
    if not blockers:
        blockers.append("Casimir gating remains intentionally closed for this diagnostic report")
    return {
        "main_observation": (
            "The contact-aware finite-q BdG Ward criterion is closed for the requested pairings."
            if ward_criterion.get("ward_identity_closed", False)
            else "The contact-aware finite-q BdG Ward criterion is not closed for the requested pairings."
        ),
        "suspected_blockers": blockers,
        "recommended_next_action": ward_criterion.get("summary", {}).get(
            "recommended_next_fix",
            "Inspect the largest finite-q contact-aware residual before changing any Casimir input gate.",
        ),
    }


def _run_ward_triage(
    *,
    model_name: str,
    pairings: tuple[str, ...],
    q_model: tuple[float, float],
    omega: float,
    nk: int,
    delta0: float,
    bubble_audit_nk_values: tuple[int, ...],
    bubble_audit_q_values: tuple[float, ...],
    bubble_audit_omega_values: tuple[float, ...],
    bubble_audit_mesh_shifts_enabled: bool = True,
    include_bubble_outlier_audit: bool = False,
    bubble_outlier_top_n: int = 12,
) -> dict[str, Any]:
    normal = run_normal_finite_q_ward_triage(
        model_name=model_name,
        q_model=q_model,
        omega_eV=omega,
        nk=nk,
    )
    operator = run_operator_identity_triage(
        model_name=model_name,
        pairings=pairings,
        q_model=q_model,
        nk=nk,
        delta0_eV=delta0,
    )
    contact = run_contact_cancellation_triage(
        model_name=model_name,
        pairings=pairings,
        q_model=q_model,
        omega_eV=omega,
        nk=nk,
        delta0_eV=delta0,
    )
    normal_contact = run_normal_contact_direct_audit(
        model_name=model_name,
        q_model=q_model,
        omega_eV=omega,
        nk=nk,
    )
    normal_ward_convention = run_normal_ward_convention_audit(
        model_name=model_name,
        q_model=q_model,
        omega_eV=omega,
        nk=nk,
    )
    normal_bubble_convergence = run_normal_bubble_convergence_audit(
        model_name=model_name,
        base_q_model=q_model,
        base_omega_eV=omega,
        base_nk=nk,
        nk_values=bubble_audit_nk_values,
        q_values=bubble_audit_q_values,
        omega_values=bubble_audit_omega_values,
        mesh_shifts_enabled=bubble_audit_mesh_shifts_enabled,
    )
    payload = {
        "normal_finite_q": normal,
        "operator_identity": operator,
        "contact_cancellation": contact,
        "normal_contact_direct_audit": normal_contact,
        "normal_ward_convention_audit": normal_ward_convention,
        "normal_bubble_convergence_audit": normal_bubble_convergence,
        "summary": summarize_ward_triage(normal, operator, contact, normal_ward_convention),
    }
    if include_bubble_outlier_audit:
        payload["normal_bubble_per_k_outlier_audit"] = run_normal_bubble_per_k_outlier_audit(
            model_name=model_name,
            q_model=q_model,
            omega_eV=omega,
            top_n=bubble_outlier_top_n,
        )
    return payload


def build_report(
    *,
    model_name: str,
    output_dir: Path,
    nk: int,
    omega: float,
    delta0: float,
    q_values: tuple[float, ...],
    pairings: tuple[str, ...],
    include_triage: bool = True,
    triage_q: tuple[float, float] = (0.01, 0.0),
    bubble_audit_nk_values: tuple[int, ...] = (7, 9, 11),
    bubble_audit_q_values: tuple[float, ...] = (0.005, 0.01, 0.02),
    bubble_audit_omega_values: tuple[float, ...] = (0.005, 0.01, 0.02),
    bubble_audit_mesh_shifts_enabled: bool = True,
    include_bubble_outlier_audit: bool = False,
    bubble_outlier_top_n: int = 12,
    ward_criterion: str = "contact_aware_v1",
    bdg_ward_closure_response: str = "amplitude_phase_schur",
    bdg_ward_absolute_tol: float = 1e-6,
    bdg_ward_relative_tol: float = 1e-6,
) -> dict[str, Any]:
    model = get_finite_q_validation_model(model_name)
    for pairing in pairings:
        model.require_pairing(pairing)
    pairing_params = model.build_pairing_params(delta0)
    q0_reports = run_q0_bdg_response_alignment_many(
        pairings,
        model_name=model.name,
        omega_eV=omega,
        nk=nk,
        pairing_params=pairing_params,
    )
    q0_status = {report.pairing_name: report.status for report in q0_reports}
    scan_report = run_finite_q_ward_scan(
        pairings,
        model_name=model.name,
        omega_eV=omega,
        q_values=q_values,
        nk=nk,
        pairing_params=pairing_params,
        q0_status=q0_status,
    )
    finite_q_rows = _compact_rows(scan_report)
    if ward_criterion != "contact_aware_v1":
        raise ValueError("only contact_aware_v1 Ward criterion is supported")
    ward_criterion_payload = evaluate_finite_q_bdg_ward_criterion(
        finite_q_rows=finite_q_rows,
        pairings=pairings,
        q_values=q_values,
        closure_response_name=bdg_ward_closure_response,
        absolute_tol=bdg_ward_absolute_tol,
        relative_tol=bdg_ward_relative_tol,
        q0_precondition_status=dict(scan_report.q0_precondition_status),
    )
    report = {
        "problem": "finite_q_ward",
        "model_name": scan_report.model_name,
        "primary_validation_model": bool(scan_report.primary_validation_model),
        "valid_for_casimir_input": False,
        "run_config": {
            "output_dir": str(output_dir),
            "nk": nk,
            "omega": omega,
            "delta0": delta0,
            "q_values": list(q_values),
            "pairings": list(pairings),
            "ward_criterion": ward_criterion,
            "bdg_ward_closure_response": bdg_ward_closure_response,
            "bdg_ward_absolute_tol": bdg_ward_absolute_tol,
            "bdg_ward_relative_tol": bdg_ward_relative_tol,
        },
        "q0_precondition_status": dict(scan_report.q0_precondition_status),
        "q0_comparator_summary": _compact_q0_summary(q0_reports),
        "finite_q_status": {
            "diagnostic_run_completed": bool(scan_report.diagnostic_run_completed),
            "ward_identity_closed": bool(ward_criterion_payload["ward_identity_closed"]),
            "workspace_evaluation": bool(scan_report.workspace_evaluation),
        },
        "pairing_summary": _pairing_summary(scan_report),
        "finite_q_rows": finite_q_rows,
        "ward_criterion": ward_criterion_payload,
        "diagnostic_interpretation": _diagnostic_interpretation_from_criterion(scan_report, ward_criterion_payload),
    }
    if include_triage:
        report["ward_triage_run_config"] = {
            "include_triage": True,
            "triage_q": [float(triage_q[0]), float(triage_q[1])],
            "triage_nk": int(nk),
            "triage_omega_eV": float(omega),
            "normal_bubble_convergence_audit": {
                "base_nk": int(nk),
                "base_q_model": [float(triage_q[0]), float(triage_q[1])],
                "base_omega_eV": float(omega),
                "nk_values": [int(value) for value in bubble_audit_nk_values],
                "q_values": [float(value) for value in bubble_audit_q_values],
                "omega_values": [float(value) for value in bubble_audit_omega_values],
                "mesh_shifts_enabled": bool(bubble_audit_mesh_shifts_enabled),
                "computed_in_current_run": True,
                "valid_for_casimir_input": False,
            },
            "normal_bubble_per_k_outlier_audit": {
                "enabled": bool(include_bubble_outlier_audit),
                "top_n": int(bubble_outlier_top_n),
                "cases": [
                    {"case_name": "nk9_unshifted", "nk": 9, "shift": [0.0, 0.0]},
                    {"case_name": "nk11_unshifted", "nk": 11, "shift": [0.0, 0.0]},
                    {"case_name": "nk11_shift_y", "nk": 11, "shift": [0.0, 1.0 / 22.0]},
                    {"case_name": "nk8_unshifted", "nk": 8, "shift": [0.0, 0.0]},
                ],
                "computed_in_current_run": bool(include_bubble_outlier_audit),
                "valid_for_casimir_input": False,
            },
            "computed_in_current_run": True,
            "valid_for_casimir_input": False,
        }
        report["ward_triage"] = _run_ward_triage(
            model_name=model.name,
            pairings=pairings,
            q_model=triage_q,
            omega=omega,
            nk=nk,
            delta0=delta0,
            bubble_audit_nk_values=bubble_audit_nk_values,
            bubble_audit_q_values=bubble_audit_q_values,
            bubble_audit_omega_values=bubble_audit_omega_values,
            bubble_audit_mesh_shifts_enabled=bubble_audit_mesh_shifts_enabled,
            include_bubble_outlier_audit=include_bubble_outlier_audit,
            bubble_outlier_top_n=bubble_outlier_top_n,
        )
        report["ward_triage"]["diagnostic_summary"] = report["ward_triage"].get("summary", {})
        report["ward_triage"]["summary"] = _ward_criterion_summary(ward_criterion_payload)
    else:
        report["ward_triage_run_config"] = {
            "include_triage": False,
            "computed_in_current_run": False,
            "valid_for_casimir_input": False,
        }
        report["ward_triage"] = {
            "available": False,
            "reason": "triage disabled by CLI",
            "summary": {
                "suspected_layer": "unavailable",
                "recommended_next_fix": "rerun report with --include-triage",
                "valid_for_casimir_input": False,
            },
            "valid_for_casimir_input": False,
        }
    return report


def _format_bool(value: bool) -> str:
    return "True" if value else "False"


def format_markdown(report: dict[str, Any]) -> str:
    pairing_summary = report["pairing_summary"]
    finite_q_status = report["finite_q_status"]
    q0_status = report["q0_precondition_status"]
    interpretation = report["diagnostic_interpretation"]
    triage = report.get("ward_triage", {})
    triage_config = report.get("ward_triage_run_config", {})
    triage_summary = triage.get("summary", {}) if isinstance(triage, dict) else {}
    normal_triage = triage.get("normal_finite_q", {}) if isinstance(triage, dict) else {}
    operator_triage = triage.get("operator_identity", {}) if isinstance(triage, dict) else {}
    contact_triage = triage.get("contact_cancellation", {}) if isinstance(triage, dict) else {}
    normal_contact_audit = triage.get("normal_contact_direct_audit", {}) if isinstance(triage, dict) else {}
    normal_contact_summary = normal_contact_audit.get("summary", {}) if isinstance(normal_contact_audit, dict) else {}
    normal_ward_audit = triage.get("normal_ward_convention_audit", {}) if isinstance(triage, dict) else {}
    normal_ward_summary = normal_ward_audit.get("summary", {}) if isinstance(normal_ward_audit, dict) else {}
    normal_bubble_audit = triage.get("normal_bubble_convergence_audit", {}) if isinstance(triage, dict) else {}
    normal_bubble_summary = normal_bubble_audit.get("summary", {}) if isinstance(normal_bubble_audit, dict) else {}
    bubble_outlier_audit = triage.get("normal_bubble_per_k_outlier_audit", {}) if isinstance(triage, dict) else {}
    bubble_outlier_summary = bubble_outlier_audit.get("summary", {}) if isinstance(bubble_outlier_audit, dict) else {}
    bubble_config = (
        triage_config.get("normal_bubble_convergence_audit", {})
        if isinstance(triage_config, dict)
        else {}
    )
    ward_criterion = report.get("ward_criterion", {})
    ward_summary = ward_criterion.get("summary", {}) if isinstance(ward_criterion, dict) else {}
    largest_blocker = ward_summary.get("largest_blocker") if isinstance(ward_summary, dict) else None
    by_pairing = ward_criterion.get("by_pairing", {}) if isinstance(ward_criterion, dict) else {}
    spm_criterion = by_pairing.get("spm", {}) if isinstance(by_pairing, dict) else {}
    dwave_criterion = by_pairing.get("dwave", {}) if isinstance(by_pairing, dict) else {}
    blocker_text = (
        "none"
        if not largest_blocker
        else (
            f"pairing={largest_blocker.get('pairing_name')}, q={largest_blocker.get('q_model')}, "
            f"response={largest_blocker.get('response_name')}, "
            f"residual={largest_blocker.get('contact_aware_residual_norm')}"
        )
    )
    lines = [
        "# finite-q Ward validation report",
        "",
        "## 当前状态",
        f"- diagnostic_run_completed: {_format_bool(finite_q_status['diagnostic_run_completed'])}",
        f"- ward_identity_closed: {_format_bool(finite_q_status['ward_identity_closed'])}",
        f"- valid_for_casimir_input: {_format_bool(report['valid_for_casimir_input'])}",
        "",
        "## q=0 前置结论",
    ]
    lines.extend(f"- {pairing}: {status}" for pairing, status in q0_status.items())
    for pairing in ("spm", "dwave"):
        summary = pairing_summary.get(pairing, {})
        lines.extend(
            [
                "",
                f"## {pairing} 结论",
                f"- q0_precondition_status: {summary.get('q0_precondition_status', 'not_requested')}",
                f"- max_closure_residual_norm: {summary.get('max_closure_residual_norm')}",
                f"- ward_identity_closed: {_format_bool(bool(summary.get('ward_identity_closed', False)))}",
            ]
        )
    lines.extend(
        [
            "",
            "## Ward criterion",
            f"- criterion_version: {ward_criterion.get('criterion_version', 'unavailable')}",
            f"- closure_response_name: {ward_criterion.get('closure_response_name', 'unavailable')}",
            f"- full_bdg_ward_closed: {_format_bool(bool(ward_criterion.get('ward_identity_closed', False)))}",
            f"- spm max contact-aware closure residual: {spm_criterion.get('max_closure_contact_aware_residual_norm')}",
            f"- dwave max contact-aware closure residual: {dwave_criterion.get('max_closure_contact_aware_residual_norm')}",
            f"- largest blocker: {blocker_text}",
            f"- recommended next fix: {ward_summary.get('recommended_next_fix', 'inspect finite-q Ward criterion rows')}",
            f"- valid_for_casimir_input: {_format_bool(bool(ward_criterion.get('valid_for_casimir_input', False)))}",
            "",
            "## Casimir gating",
            "- valid_for_casimir_input: False",
            "- This report is diagnostic-only and does not promote finite-q response data to Casimir input.",
            "",
            "## Ward triage",
            f"- normal finite-q triage conclusion: {normal_triage.get('suspected_layer', normal_triage.get('reason', 'unavailable'))}",
            f"- operator identity conclusion: {operator_triage.get('suspected_layer', operator_triage.get('reason', 'unavailable'))}",
            f"- contact cancellation conclusion: {_contact_conclusion(contact_triage)}",
            f"- normal contact/direct audit: {normal_contact_summary.get('suspected_issue', normal_contact_audit.get('reason', 'unavailable'))}",
            f"- recommended normal contact fix: {normal_contact_summary.get('recommended_next_fix', 'rerun report with triage enabled')}",
            f"- normal Ward convention audit: {normal_ward_summary.get('suspected_issue', normal_ward_audit.get('reason', 'unavailable'))}",
            f"- recommended Ward convention fix: {normal_ward_summary.get('recommended_next_fix', 'rerun report with triage enabled')}",
            f"- normal bubble convergence audit: {normal_bubble_summary.get('suspected_issue', normal_bubble_audit.get('reason', 'unavailable'))}",
            f"- recommended normal bubble fix: {normal_bubble_summary.get('recommended_next_fix', 'rerun report with triage enabled')}",
            *(
                [
                    f"- normal bubble per-k outlier audit: {bubble_outlier_summary.get('suspected_issue', bubble_outlier_audit.get('reason', 'unavailable'))}",
                    f"- recommended per-k outlier fix: {bubble_outlier_summary.get('recommended_next_fix', 'rerun report with --include-bubble-outlier-audit')}",
                ]
                if bubble_outlier_audit
                else []
            ),
            f"- normal bubble audit config: nk={bubble_config.get('nk_values', 'unavailable')}, q={bubble_config.get('q_values', 'unavailable')}, omega={bubble_config.get('omega_values', 'unavailable')}, mesh_shifts={bubble_config.get('mesh_shifts_enabled', 'unavailable')}",
            f"- suspected primary layer: {triage_summary.get('suspected_layer', 'unavailable')}",
            f"- recommended next fix: {triage_summary.get('recommended_next_fix', 'rerun report with triage enabled')}",
            f"- valid_for_casimir_input: {_format_bool(bool(triage_summary.get('valid_for_casimir_input', False)))}",
            "",
            "## 下一步建议",
            f"- {interpretation['recommended_next_action']}",
            "",
            "## 主要观察",
            f"- {interpretation['main_observation']}",
        ]
    )
    return "\n".join(lines) + "\n"


def _contact_conclusion(contact_triage: dict[str, Any]) -> str:
    if not contact_triage.get("available", False):
        return str(contact_triage.get("reason", "unavailable"))
    by_pairing = contact_triage.get("by_pairing", {})
    if not isinstance(by_pairing, dict):
        return "unavailable"
    pieces = []
    for pairing, payload in by_pairing.items():
        if isinstance(payload, dict):
            pieces.append(f"{pairing}: {payload.get('interpretation', payload.get('reason', 'unknown'))}")
    return "; ".join(pieces) if pieces else "unavailable"


def _command_text(args: argparse.Namespace) -> str:
    rel_script = "validation/scripts/bdg_finite_q/run_finite_q_ward_report.py"
    output_dir = args.output_dir
    try:
        output_dir_text = str(output_dir.resolve().relative_to(ROOT))
    except ValueError:
        output_dir_text = str(output_dir)
    parts = [
        "python",
        rel_script,
        "--model",
        args.model,
        "--output-dir",
        output_dir_text,
        "--nk",
        str(args.nk),
        "--omega",
        str(args.omega),
        "--delta0",
        str(args.delta0),
        "--q-values",
        *(str(value) for value in args.q_values),
        "--pairings",
        *args.pairings,
        "--ward-criterion",
        args.ward_criterion,
        "--bdg-ward-closure-response",
        args.bdg_ward_closure_response,
        "--bdg-ward-absolute-tol",
        str(args.bdg_ward_absolute_tol),
        "--bdg-ward-relative-tol",
        str(args.bdg_ward_relative_tol),
        "--triage-qx",
        str(args.triage_qx),
        "--triage-qy",
        str(args.triage_qy),
    ]
    if args.include_triage:
        parts.extend(
            [
                "--bubble-audit-nk-values",
                *(str(value) for value in args.bubble_audit_nk_values),
                "--bubble-audit-q-values",
                *(str(value) for value in args.bubble_audit_q_values),
                "--bubble-audit-omega-values",
                *(str(value) for value in args.bubble_audit_omega_values),
            ]
        )
        if args.disable_bubble_audit_mesh_shifts:
            parts.append("--disable-bubble-audit-mesh-shifts")
        if args.include_bubble_outlier_audit:
            parts.extend(["--include-bubble-outlier-audit", "--bubble-outlier-top-n", str(args.bubble_outlier_top_n)])
        parts.append("--include-triage")
    else:
        parts.append("--no-triage")
    return " ".join(shlex.quote(part) for part in parts) + "\n"


def write_report(report: dict[str, Any], output_dir: Path, command_text: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(format_markdown(report), encoding="utf-8")
    (output_dir / "command.sh").write_text(command_text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the unified finite-q Ward diagnostic report.")
    parser.add_argument("--model", choices=available_finite_q_validation_models(), default="symmetry_bdg_2band")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nk", type=int, default=9)
    parser.add_argument("--omega", type=float, default=0.01)
    parser.add_argument("--delta0", type=float, default=0.1)
    parser.add_argument("--q-values", nargs="+", type=float, default=list(DEFAULT_Q_VALUES))
    parser.add_argument("--pairings", nargs="+", default=list(DEFAULT_PAIRINGS))
    parser.add_argument("--ward-criterion", choices=["contact_aware_v1"], default="contact_aware_v1")
    parser.add_argument(
        "--bdg-ward-closure-response",
        choices=["minus_schur", "amplitude_phase_schur"],
        default="amplitude_phase_schur",
    )
    parser.add_argument("--bdg-ward-absolute-tol", type=float, default=1e-6)
    parser.add_argument("--bdg-ward-relative-tol", type=float, default=1e-6)
    triage_group = parser.add_mutually_exclusive_group()
    triage_group.add_argument("--include-triage", dest="include_triage", action="store_true", default=True)
    triage_group.add_argument("--no-triage", dest="include_triage", action="store_false")
    parser.add_argument("--triage-qx", type=float, default=0.01)
    parser.add_argument("--triage-qy", type=float, default=0.0)
    parser.add_argument("--bubble-audit-nk-values", nargs="+", type=int, default=[7, 9, 11])
    parser.add_argument("--bubble-audit-q-values", nargs="+", type=float, default=[0.005, 0.01, 0.02])
    parser.add_argument("--bubble-audit-omega-values", nargs="+", type=float, default=[0.005, 0.01, 0.02])
    parser.add_argument("--disable-bubble-audit-mesh-shifts", action="store_true", default=False)
    parser.add_argument("--include-bubble-outlier-audit", action="store_true", default=False)
    parser.add_argument("--bubble-outlier-top-n", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    report = build_report(
        model_name=args.model,
        output_dir=output_dir,
        nk=args.nk,
        omega=args.omega,
        delta0=args.delta0,
        q_values=tuple(args.q_values),
        pairings=tuple(args.pairings),
        include_triage=bool(args.include_triage),
        triage_q=(float(args.triage_qx), float(args.triage_qy)),
        bubble_audit_nk_values=tuple(args.bubble_audit_nk_values),
        bubble_audit_q_values=tuple(args.bubble_audit_q_values),
        bubble_audit_omega_values=tuple(args.bubble_audit_omega_values),
        bubble_audit_mesh_shifts_enabled=not bool(args.disable_bubble_audit_mesh_shifts),
        include_bubble_outlier_audit=bool(args.include_bubble_outlier_audit),
        bubble_outlier_top_n=int(args.bubble_outlier_top_n),
        ward_criterion=args.ward_criterion,
        bdg_ward_closure_response=args.bdg_ward_closure_response,
        bdg_ward_absolute_tol=float(args.bdg_ward_absolute_tol),
        bdg_ward_relative_tol=float(args.bdg_ward_relative_tol),
    )
    write_report(report, output_dir, _command_text(args))
    print(f"Wrote finite-q Ward report to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
