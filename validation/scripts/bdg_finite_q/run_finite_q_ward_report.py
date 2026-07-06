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
from validation.lib.finite_q_ward_criterion import evaluate_finite_q_bdg_ward_criterion  # noqa: E402
from validation.lib.finite_q_ward_triage import (  # noqa: E402
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
SUPPORTED_WARD_CRITERIA = ("full_hessian_v1",)


def _round_float(value: float | None) -> float | None:
    return None if value is None else float(value)


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
        closure_rows = [row for row in rows if row.response_name == "amplitude_phase_schur"]
        max_residual = max((row.max_ward_residual_norm for row in rows), default=None)
        max_closure_residual = max((row.max_ward_residual_norm for row in closure_rows), default=None)
        summary[pairing_name] = {
            "q0_precondition_status": scan_report.q0_precondition_status.get(pairing_name, "missing_q0_status"),
            "row_count": len(rows),
            "max_ward_residual_norm": _round_float(max_residual),
            "max_closure_residual_norm": _round_float(max_closure_residual),
            "ward_identity_closed": bool(closure_rows and all(row.max_ward_residual_norm <= 1e-8 for row in closure_rows)),
            "valid_for_casimir_input": False,
        }
    return summary


def _ward_criterion_summary(ward_criterion: dict[str, Any]) -> dict[str, Any]:
    if not ward_criterion.get("evaluated", False):
        suspected_layer = "ward_criterion_incomplete"
    elif not ward_criterion.get("ward_identity_closed", False):
        suspected_layer = "bdg_collective_closure"
    else:
        suspected_layer = "none"
    return {
        "suspected_layer": suspected_layer,
        "recommended_next_fix": ward_criterion.get("summary", {}).get(
            "recommended_next_fix", "Inspect the formal full-Hessian Ward criterion rows."
        ),
        "valid_for_casimir_input": False,
    }


def _diagnostic_interpretation_from_criterion(scan_report: Any, ward_criterion: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if not ward_criterion.get("evaluated", False):
        blockers.append("finite-q Ward criterion is incomplete because requested closure rows or vectors are missing")
    elif not ward_criterion.get("ward_identity_closed", False):
        blockers.append("finite-q BdG full-Hessian Schur Ward closure failed for at least one requested pairing")
    if any(status == "diagnostic_only_not_passed" for status in scan_report.q0_precondition_status.values()):
        blockers.append("at least one q=0 precondition is diagnostic-only")
    if not blockers:
        blockers.append("Casimir gating remains intentionally closed for this diagnostic report")
    return {
        "main_observation": (
            "The finite-q BdG full-Hessian Schur Ward criterion is closed for the requested pairings."
            if ward_criterion.get("ward_identity_closed", False)
            else "The finite-q BdG full-Hessian Schur Ward criterion is not closed for the requested pairings."
        ),
        "suspected_blockers": blockers,
        "recommended_next_action": ward_criterion.get("summary", {}).get(
            "recommended_next_fix",
            "Inspect the largest full-Hessian Schur Ward residual before changing any Casimir input gate.",
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
    normal = run_normal_finite_q_ward_triage(model_name=model_name, q_model=q_model, omega_eV=omega, nk=nk)
    operator = run_operator_identity_triage(
        model_name=model_name, pairings=pairings, q_model=q_model, nk=nk, delta0_eV=delta0
    )
    contact = run_contact_cancellation_triage(
        model_name=model_name, pairings=pairings, q_model=q_model, omega_eV=omega, nk=nk, delta0_eV=delta0
    )
    normal_contact = run_normal_contact_direct_audit(model_name=model_name, q_model=q_model, omega_eV=omega, nk=nk)
    normal_ward_convention = run_normal_ward_convention_audit(
        model_name=model_name, q_model=q_model, omega_eV=omega, nk=nk
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
        "diagnostic_summary": summarize_ward_triage(normal, operator, contact, normal_ward_convention),
    }
    if include_bubble_outlier_audit:
        payload["normal_bubble_per_k_outlier_audit"] = run_normal_bubble_per_k_outlier_audit(
            model_name=model_name, q_model=q_model, omega_eV=omega, top_n=bubble_outlier_top_n
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
    ward_criterion: str = "full_hessian_v1",
    bdg_ward_closure_response: str = "amplitude_phase_schur",
    bdg_ward_absolute_tol: float = 1e-6,
    bdg_ward_relative_tol: float = 1e-6,
) -> dict[str, Any]:
    model = get_finite_q_validation_model(model_name)
    for pairing in pairings:
        model.require_pairing(pairing)
    if ward_criterion not in SUPPORTED_WARD_CRITERIA:
        raise ValueError("Ward criterion must be full_hessian_v1")
    pairing_params = model.build_pairing_params(delta0)
    q0_reports = run_q0_bdg_response_alignment_many(
        pairings, model_name=model.name, omega_eV=omega, nk=nk, pairing_params=pairing_params
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
    ward_criterion_payload = evaluate_finite_q_bdg_ward_criterion(
        finite_q_rows=finite_q_rows,
        pairings=pairings,
        q_values=q_values,
        closure_response_name=bdg_ward_closure_response,
        absolute_tol=bdg_ward_absolute_tol,
        relative_tol=bdg_ward_relative_tol,
        q0_precondition_status=dict(scan_report.q0_precondition_status),
    )
    ward_criterion_payload["requested_criterion"] = ward_criterion
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
        report["ward_triage"]["summary"] = _ward_criterion_summary(ward_criterion_payload)
    else:
        report["ward_triage_run_config"] = {"include_triage": False, "computed_in_current_run": False, "valid_for_casimir_input": False}
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
    q0_status = report["q0_precondition_status"]
    finite_q_status = report["finite_q_status"]
    pairing_summary = report["pairing_summary"]
    ward_criterion = report.get("ward_criterion", {})
    ward_summary = ward_criterion.get("summary", {}) if isinstance(ward_criterion, dict) else {}
    largest_blocker = ward_summary.get("largest_blocker") if isinstance(ward_summary, dict) else None
    interpretation = report["diagnostic_interpretation"]
    blocker_text = "none" if not largest_blocker else (
        f"pairing={largest_blocker.get('pairing_name')}, q={largest_blocker.get('q_model')}, "
        f"response={largest_blocker.get('response_name')}, primary_residual={largest_blocker.get('primary_residual_norm')}"
    )
    lines = [
        "# finite-q Ward validation report",
        "",
        "## Current status",
        f"- diagnostic_run_completed: {_format_bool(finite_q_status['diagnostic_run_completed'])}",
        f"- ward_identity_closed: {_format_bool(finite_q_status['ward_identity_closed'])}",
        f"- valid_for_casimir_input: {_format_bool(report['valid_for_casimir_input'])}",
        "",
        "## q=0 preconditions",
        *(f"- {pairing}: {status}" for pairing, status in q0_status.items()),
    ]
    for pairing in report["run_config"].get("pairings", []):
        summary = pairing_summary.get(pairing, {})
        lines.extend([
            "",
            f"## {pairing} conclusion",
            f"- q0_precondition_status: {summary.get('q0_precondition_status', 'not_requested')}",
            f"- max_closure_residual_norm: {summary.get('max_closure_residual_norm')}",
            f"- ward_identity_closed: {_format_bool(bool(summary.get('ward_identity_closed', False)))}",
        ])
    lines.extend([
        "",
        "## Ward criterion",
        f"- criterion_version: {ward_criterion.get('criterion_version', 'unavailable')}",
        f"- criterion_formal_name: {ward_criterion.get('criterion_formal_name', 'unavailable')}",
        f"- closure_response_name: {ward_criterion.get('closure_response_name', 'unavailable')}",
        f"- full_bdg_ward_closed: {_format_bool(bool(ward_criterion.get('ward_identity_closed', False)))}",
        f"- largest blocker: {blocker_text}",
        f"- recommended next fix: {ward_summary.get('recommended_next_fix', 'inspect finite-q Ward criterion rows')}",
        f"- valid_for_casimir_input: {_format_bool(bool(ward_criterion.get('valid_for_casimir_input', False)))}",
        "",
        "## Casimir gating",
        "- valid_for_casimir_input: False",
        "- This report is diagnostic-only and does not promote finite-q response data to Casimir input.",
        "",
        "## Next action",
        f"- {interpretation['recommended_next_action']}",
        "",
        "## Main observation",
        f"- {interpretation['main_observation']}",
    ])
    return "\n".join(lines) + "\n"


def _command_text(args: argparse.Namespace) -> str:
    rel_script = "validation/scripts/bdg_finite_q/run_finite_q_ward_report.py"
    output_dir = args.output_dir
    try:
        output_dir_text = str(output_dir.resolve().relative_to(ROOT))
    except ValueError:
        output_dir_text = str(output_dir)
    parts = [
        "python", rel_script,
        "--model", args.model,
        "--output-dir", output_dir_text,
        "--nk", str(args.nk),
        "--omega", str(args.omega),
        "--delta0", str(args.delta0),
        "--q-values", *(str(value) for value in args.q_values),
        "--pairings", *args.pairings,
        "--ward-criterion", args.ward_criterion,
        "--bdg-ward-closure-response", args.bdg_ward_closure_response,
        "--bdg-ward-absolute-tol", str(args.bdg_ward_absolute_tol),
        "--bdg-ward-relative-tol", str(args.bdg_ward_relative_tol),
        "--triage-qx", str(args.triage_qx),
        "--triage-qy", str(args.triage_qy),
    ]
    if args.include_triage:
        parts.extend([
            "--bubble-audit-nk-values", *(str(value) for value in args.bubble_audit_nk_values),
            "--bubble-audit-q-values", *(str(value) for value in args.bubble_audit_q_values),
            "--bubble-audit-omega-values", *(str(value) for value in args.bubble_audit_omega_values),
        ])
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
    parser.add_argument("--ward-criterion", choices=list(SUPPORTED_WARD_CRITERIA), default="full_hessian_v1")
    parser.add_argument("--bdg-ward-closure-response", choices=["minus_schur", "amplitude_phase_schur"], default="amplitude_phase_schur")
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
    report = build_report(
        model_name=args.model,
        output_dir=args.output_dir,
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
    write_report(report, args.output_dir, _command_text(args))
    print(f"Wrote finite-q Ward report to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
