#!/usr/bin/env python3
"""Finite-q Ward residual scans for superconducting BdG diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from lno327 import KuboConfig, k_weights, uniform_bz_mesh  # noqa: E402
from lno327.collective.validation import validate_physical_ward_identity  # noqa: E402
from lno327.response.finite_q_bdg import precompute_finite_q_bdg_workspace_from_model_ansatz  # noqa: E402
from lno327.workflows.finite_q_engine import FiniteQEngineOptions, bdg_finite_q_response_imag_axis_from_workspace  # noqa: E402
from q0_bdg_response_alignment import run_q0_bdg_response_alignment_many  # noqa: E402
from validation.lib.finite_q_collective_ward_blocks import evaluate_collective_ward_blocks  # noqa: E402
from validation.lib.finite_q_integrated_ward_chain import evaluate_integrated_ward_chain  # noqa: E402
from validation.lib.finite_q_operator_ward_checks import evaluate_bdg_operator_ward_checks  # noqa: E402
from validation.lib.finite_q_shifted_average import (  # noqa: E402
    DIRECT_QUADRATURE_MODES,
    average_finite_q_bdg_response_over_shifts,
    shift_pairs_from_fractions,
)
from validation.lib.finite_q_validation_models import available_finite_q_validation_models, get_finite_q_validation_model  # noqa: E402

WardScanPairingName = str
WARD_COMPONENT_LABELS = ("density", "current_x", "current_y")
WARD_CLOSURE_RESPONSE_NAMES = ("amplitude_phase_schur",)
RESPONSE_NAMES = ("bare_bubble", "direct", "bare_total", "minus_schur", "plus_schur", "amplitude_phase_schur")


@dataclass(frozen=True)
class FiniteQWardScanRow:
    pairing_name: str
    response_name: str
    q_model: tuple[float, float]
    q_norm: float
    left_ward_residual_norm: float
    right_ward_residual_norm: float
    residual_component_labels: tuple[str, ...]
    left_ward_residual_vector: tuple[dict[str, float | str], ...]
    right_ward_residual_vector: tuple[dict[str, float | str], ...]
    max_ward_residual_norm: float
    residual_ratio_to_bare: float | None
    collective_matrix_condition_number: float | None
    inverse_method: str
    pinv_diagnostic_used: bool
    valid_for_casimir_input: bool = False


@dataclass(frozen=True)
class FiniteQWardScanReport:
    model_name: str
    model_metadata: dict[str, Any]
    primary_validation_model: bool
    pairing_names: tuple[str, ...]
    omega_eV: float
    nk: int | None
    mesh_size: int
    delta0_eV: float
    rows: tuple[FiniteQWardScanRow, ...]
    collective_ward_blocks: tuple[dict[str, Any], ...]
    operator_ward_checks: tuple[dict[str, Any], ...]
    integrated_ward_chains: tuple[dict[str, Any], ...]
    shifted_mesh_average: dict[str, Any]
    q0_alignment_prerequisite: dict[str, str]
    q0_precondition_status: dict[str, str]
    q_scaling_estimates: dict[str, float | None]
    schur_residual_differences: tuple[dict[str, Any], ...]
    diagnostic_run_completed: bool
    ward_identity_closed: bool
    workspace_evaluation: bool
    notes: tuple[str, ...]
    valid_for_casimir_input: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_metadata": self.model_metadata,
            "primary_validation_model": self.primary_validation_model,
            "pairing_names": list(self.pairing_names),
            "omega_eV": self.omega_eV,
            "nk": self.nk,
            "mesh_size": self.mesh_size,
            "delta0_eV": self.delta0_eV,
            "rows": [{**row.__dict__, "q_model": list(row.q_model), "valid_for_casimir_input": False} for row in self.rows],
            "collective_ward_blocks": list(self.collective_ward_blocks),
            "operator_ward_checks": list(self.operator_ward_checks),
            "integrated_ward_chains": list(self.integrated_ward_chains),
            "shifted_mesh_average": self.shifted_mesh_average,
            "q0_alignment_prerequisite": self.q0_alignment_prerequisite,
            "q0_precondition_status": self.q0_precondition_status,
            "q_scaling_estimates": self.q_scaling_estimates,
            "schur_residual_differences": list(self.schur_residual_differences),
            "diagnostic_run_completed": self.diagnostic_run_completed,
            "ward_identity_closed": self.ward_identity_closed,
            "workspace_evaluation": self.workspace_evaluation,
            "notes": list(self.notes),
            "valid_for_casimir_input": False,
        }

    def format_text(self) -> str:
        lines = [
            "finite-q Ward residual scan report",
            f"model_name: {self.model_name}",
            f"pairings: {', '.join(self.pairing_names)}",
            f"omega_eV: {self.omega_eV:.12g}",
            f"nk: {self.nk if self.nk is not None else 'external mesh'}",
            f"mesh_size: {self.mesh_size}",
            f"delta0_eV: {self.delta0_eV:.12g}",
            f"q0_precondition_status: {self.q0_precondition_status}",
            f"diagnostic_run_completed: {self.diagnostic_run_completed}",
            f"ward_identity_closed: {self.ward_identity_closed}",
            f"workspace_evaluation: {self.workspace_evaluation}",
            f"shifted_mesh_average: {self.shifted_mesh_average}",
        ]
        lines.extend(
            f"- {row.pairing_name} {row.response_name} q={row.q_model}: "
            f"max_residual={row.max_ward_residual_norm:.6e}, inverse={row.inverse_method}"
            for row in self.rows
        )
        for payload in self.collective_ward_blocks:
            dominant = payload.get("dominant_block_residual") or {}
            lines.append(
                f"- collective-block {payload.get('pairing_name')} q={payload.get('q_model')}: "
                f"dominant={dominant.get('block')} norm={dominant.get('norm')}"
            )
        for payload in self.operator_ward_checks:
            first = payload.get("first_order_bdg_identity", {})
            contact = payload.get("bdg_contact_identity", {})
            lines.append(
                f"- operator-check {payload.get('pairing_name')} q={payload.get('q_model')}: "
                f"first={first.get('max_error_norm')} contact={contact.get('max_error_norm')}"
            )
        for payload in self.integrated_ward_chains:
            lines.append(
                f"- integrated-chain {payload.get('pairing_name')} q={payload.get('q_model')}: "
                f"bubble_to_equal_time={payload.get('max_bubble_to_equal_time_difference_norm')} "
                f"equal_time_to_contact={payload.get('max_equal_time_to_contact_difference_norm')}"
            )
        lines.append("valid_for_casimir_input: False")
        return "\n".join(lines)


def _safe_ratio(value: float, reference: float) -> float | None:
    return None if reference <= 0.0 else float(value / reference)


def _complex_vector_components(vector: np.ndarray) -> tuple[dict[str, float | str], ...]:
    array = np.asarray(vector, dtype=complex)
    if array.shape != (3,):
        raise ValueError("Ward residual vector must have shape (3,)")
    return tuple(
        {"component": label, "real": float(np.real(value)), "imag": float(np.imag(value))}
        for label, value in zip(WARD_COMPONENT_LABELS, array, strict=True)
    )


def _residual_vectors_for_report(ward_report: Any) -> dict[str, Any]:
    return {
        "left_ward_residual_vector": _complex_vector_components(ward_report.left_residual),
        "right_ward_residual_vector": _complex_vector_components(ward_report.right_residual),
        "left_ward_residual_norm": float(ward_report.left_norm),
        "right_ward_residual_norm": float(ward_report.right_norm),
    }


def _complex_vector_difference(left: np.ndarray, right: np.ndarray) -> tuple[dict[str, float | str], ...]:
    return _complex_vector_components(np.asarray(left, dtype=complex) - np.asarray(right, dtype=complex))


def _schur_difference_report(pairing_name: str, q: np.ndarray, ward_reports: dict[str, Any]) -> dict[str, Any]:
    response_names = ("bare_total", "minus_schur", "amplitude_phase_schur")
    difference_pairs = (("bare_total", "minus_schur"),) if pairing_name in {"onsite_s", "spm"} else (
        ("bare_total", "minus_schur"),
        ("bare_total", "amplitude_phase_schur"),
        ("minus_schur", "amplitude_phase_schur"),
    )
    return {
        "pairing_name": pairing_name,
        "q_model": [float(q[0]), float(q[1])],
        "q_norm": float(np.linalg.norm(q)),
        "diagnostic_role": "residual_vector_decomposition",
        "residual_component_labels": list(WARD_COMPONENT_LABELS),
        "response_residual_vectors": {name: _residual_vectors_for_report(ward_reports[name]) for name in response_names},
        "differences": [
            {
                "formula": f"{left_name} - {right_name}",
                "left_ward_residual_vector_difference": _complex_vector_difference(
                    ward_reports[left_name].left_residual, ward_reports[right_name].left_residual
                ),
                "right_ward_residual_vector_difference": _complex_vector_difference(
                    ward_reports[left_name].right_residual, ward_reports[right_name].right_residual
                ),
                "valid_for_casimir_input": False,
            }
            for left_name, right_name in difference_pairs
        ],
        "valid_for_casimir_input": False,
    }


def _scaling_slope(q_values: list[float], residuals: list[float]) -> float | None:
    positive = [(q, r) for q, r in zip(q_values, residuals, strict=True) if q > 0.0 and r > 0.0]
    if len(positive) < 3:
        return None
    q_first, r_first = positive[0]
    q_last, r_last = positive[-1]
    return None if q_first == q_last else float((np.log(r_last) - np.log(r_first)) / (np.log(q_last) - np.log(q_first)))


def _q0_status_from_json(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    status = payload.get("status_by_pairing")
    if not isinstance(status, dict):
        raise ValueError("q0 status JSON must contain a status_by_pairing object")
    return {str(key): str(value) for key, value in status.items()}


def run_finite_q_ward_scan(
    pairing_names: tuple[WardScanPairingName, ...] | None = None,
    *,
    model_name: str = "symmetry_bdg_2band",
    omega_eV: float = 0.01,
    q_values: tuple[float, ...] = (0.005, 0.01, 0.02),
    q_directions: tuple[tuple[float, float], ...] = ((1.0, 0.0),),
    nk: int = 3,
    k_points: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    config: KuboConfig | None = None,
    pairing_params=None,
    tolerance: float = 1e-8,
    q0_status: dict[str, str] | None = None,
    average_shifted_meshes: bool = False,
    shift_fractions: tuple[float, ...] = (0.0,),
    direct_quadrature: str = "center",
) -> FiniteQWardScanReport:
    if direct_quadrature not in DIRECT_QUADRATURE_MODES:
        raise ValueError(f"direct_quadrature must be one of {DIRECT_QUADRATURE_MODES}")
    if average_shifted_meshes and k_points is not None:
        raise ValueError("average_shifted_meshes requires internally generated uniform meshes, not explicit k_points")
    if direct_quadrature != "center" and not average_shifted_meshes:
        raise ValueError("non-center direct_quadrature is currently implemented only with average_shifted_meshes")
    model = get_finite_q_validation_model(model_name)
    selected_pairings = tuple(model.default_pairings if pairing_names is None else pairing_names)
    for pairing_name in selected_pairings:
        model.require_pairing(pairing_name)
    points = uniform_bz_mesh(nk) if k_points is None else np.asarray(k_points, dtype=float)
    mesh_weights = k_weights(points) if weights is None else np.asarray(weights, dtype=float)
    kubo = config or KuboConfig.from_kelvin(omega_eV=omega_eV, temperature_K=10.0, eta_eV=1e-8, output_si=False)
    amp = pairing_params or model.build_pairing_params()
    options = FiniteQEngineOptions(
        current_vertex="peierls",
        collective_mode="amplitude_phase",
        collective_counterterm="goldstone_gap_equation",
        include_phase_phase_direct=True,
    )
    shift_pairs = shift_pairs_from_fractions(shift_fractions)
    shifted_mesh_average = {
        "enabled": bool(average_shifted_meshes),
        "shift_fractions": [float(value) for value in shift_fractions],
        "shift_pairs": [[float(x), float(y)] for x, y in shift_pairs],
        "num_shifted_meshes": len(shift_pairs) if average_shifted_meshes else 1,
        "direct_quadrature": direct_quadrature,
        "valid_for_casimir_input": False,
    }
    if q0_status is None:
        q0_reports = run_q0_bdg_response_alignment_many(
            tuple(selected_pairings),
            model_name=model.name,
            omega_eV=float(kubo.omega_eV),
            nk=nk,
            k_points=points if k_points is not None else None,
            weights=mesh_weights if weights is not None else None,
            config=kubo,
            pairing_params=amp,
        )
        q0_alignment = {report.pairing_name: report.status for report in q0_reports}
    else:
        q0_alignment = {pairing_name: q0_status.get(pairing_name, "missing_q0_status") for pairing_name in selected_pairings}

    rows: list[FiniteQWardScanRow] = []
    collective_ward_blocks: list[dict[str, Any]] = []
    operator_ward_checks: list[dict[str, Any]] = []
    integrated_ward_chains: list[dict[str, Any]] = []
    schur_residual_differences: list[dict[str, Any]] = []
    scaling_inputs: dict[str, tuple[list[float], list[float]]] = {}
    for pairing_name in selected_pairings:
        ansatz = model.build_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
        for q_value in q_values:
            for direction in q_directions:
                direction_array = np.asarray(direction, dtype=float)
                norm = float(np.linalg.norm(direction_array))
                if norm <= 0.0:
                    raise ValueError("q direction must be nonzero")
                q = float(q_value) * direction_array / norm
                operator_ward_checks.append(
                    evaluate_bdg_operator_ward_checks(
                        pairing_name=pairing_name,
                        q_model=q,
                        delta0_eV=float(amp.delta0_eV),
                        spec=model.spec,
                        ansatz=ansatz,
                        amp=amp,
                        k_points=points,
                        current_vertex=options.current_vertex,
                    )
                )
                if average_shifted_meshes:
                    response, integrated_chain = average_finite_q_bdg_response_over_shifts(
                        model=model,
                        ansatz=ansatz,
                        q_model=q,
                        nk=nk,
                        config=kubo,
                        pairing_params=amp,
                        options=options,
                        shift_fractions=shift_fractions,
                        direct_quadrature=direct_quadrature,
                    )
                else:
                    workspace = precompute_finite_q_bdg_workspace_from_model_ansatz(
                        model.spec, ansatz, q, points, mesh_weights, kubo, amp, options
                    )
                    response = bdg_finite_q_response_imag_axis_from_workspace(workspace, config=kubo)
                    integrated_chain = evaluate_integrated_ward_chain(workspace=workspace, response=response, delta0_eV=float(amp.delta0_eV))
                integrated_ward_chains.append(integrated_chain)
                matrices = {name: getattr(response, name) for name in RESPONSE_NAMES}
                collective_ward_blocks.append(
                    evaluate_collective_ward_blocks(
                        pairing_name=pairing_name,
                        q_model=q,
                        omega_eV=float(kubo.omega_eV),
                        delta0_eV=float(amp.delta0_eV),
                        k_aa_full=response.bare_total,
                        k_aeta=response.em_collective_left,
                        k_etaa=response.collective_em_right,
                        k_etaeta=response.collective_total,
                        schur_response=response.amplitude_phase_schur,
                        k_aa_bubble=response.bare_bubble,
                        k_aa_direct=response.direct,
                    )
                )
                bare_report = validate_physical_ward_identity(response.bare_total, kubo.omega_eV, q, tolerance=tolerance)
                bare_max = float(max(bare_report.left_norm, bare_report.right_norm))
                ward_reports: dict[str, Any] = {}
                for response_name, matrix in matrices.items():
                    ward = validate_physical_ward_identity(matrix, kubo.omega_eV, q, tolerance=tolerance)
                    ward_reports[response_name] = ward
                    max_norm = float(max(ward.left_norm, ward.right_norm))
                    scaling_q, scaling_residual = scaling_inputs.setdefault(f"{pairing_name}:{response_name}", ([], []))
                    scaling_q.append(float(np.linalg.norm(q)))
                    scaling_residual.append(max_norm)
                    inverse_method = str(response.metadata.get("collective_inverse_method", "not_used"))
                    rows.append(
                        FiniteQWardScanRow(
                            pairing_name=pairing_name,
                            response_name=response_name,
                            q_model=(float(q[0]), float(q[1])),
                            q_norm=float(np.linalg.norm(q)),
                            left_ward_residual_norm=float(ward.left_norm),
                            right_ward_residual_norm=float(ward.right_norm),
                            residual_component_labels=WARD_COMPONENT_LABELS,
                            left_ward_residual_vector=_complex_vector_components(ward.left_residual),
                            right_ward_residual_vector=_complex_vector_components(ward.right_residual),
                            max_ward_residual_norm=max_norm,
                            residual_ratio_to_bare=_safe_ratio(max_norm, bare_max),
                            collective_matrix_condition_number=response.metadata.get("collective_total_condition_number"),
                            inverse_method=inverse_method,
                            pinv_diagnostic_used=inverse_method == "pinv_diagnostic",
                            valid_for_casimir_input=False,
                        )
                    )
                schur_residual_differences.append(_schur_difference_report(pairing_name, q, ward_reports))

    slopes = {key: _scaling_slope(value[0], value[1]) for key, value in scaling_inputs.items()}
    diagnostic_completed = all(np.isfinite(row.max_ward_residual_norm) for row in rows)
    closure_rows = [row for row in rows if row.response_name in WARD_CLOSURE_RESPONSE_NAMES]
    ward_identity_closed = bool(closure_rows and all(row.max_ward_residual_norm <= tolerance for row in closure_rows))
    notes = (
        "q=0 response definition alignment is recorded as a precondition.",
        "ward_identity_closed checks only the requested final full-Hessian Schur response.",
        "bare_bubble, direct, bare_total, minus_schur, and plus_schur rows are decomposition outputs only.",
        "collective_ward_blocks reports the four block identities required by the Schur proof; it is diagnostic-only.",
        "operator_ward_checks reports matrix identities before Kubo integration; it is diagnostic-only.",
        "integrated_ward_chains reports denominator-cancelled Ward proof checks; it is diagnostic-only.",
        "shifted_mesh_average is validation-only and does not promote finite-q data to Casimir input.",
        "direct_quadrature endpoint_average translates only the contact integral quadrature to match finite-q endpoint routing; it is validation-only.",
        "finite-q outputs remain valid_for_casimir_input=False.",
        f"model_name={model.name}; primary_validation_model={model.primary_validation_model}.",
        "left/right Ward residual vectors are ordered as density,current_x,current_y.",
        "schur_residual_differences compares residual vectors only and does not affect closure or Casimir gating.",
    )
    return FiniteQWardScanReport(
        model_name=model.name,
        model_metadata=model.metadata(),
        primary_validation_model=model.primary_validation_model,
        pairing_names=tuple(selected_pairings),
        omega_eV=float(kubo.omega_eV),
        nk=nk if k_points is None else None,
        mesh_size=int(points.shape[0]),
        delta0_eV=float(amp.delta0_eV),
        rows=tuple(rows),
        collective_ward_blocks=tuple(collective_ward_blocks),
        operator_ward_checks=tuple(operator_ward_checks),
        integrated_ward_chains=tuple(integrated_ward_chains),
        shifted_mesh_average=shifted_mesh_average,
        q0_alignment_prerequisite=q0_alignment,
        q0_precondition_status=q0_alignment,
        q_scaling_estimates=slopes,
        schur_residual_differences=tuple(schur_residual_differences),
        diagnostic_run_completed=bool(diagnostic_completed),
        ward_identity_closed=ward_identity_closed,
        workspace_evaluation=True,
        notes=notes,
        valid_for_casimir_input=False,
    )


def _write_json(path: Path, report: FiniteQWardScanReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_text(path: Path, report: FiniteQWardScanReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.format_text() + "\n", encoding="utf-8")


def run_and_write_report(
    output_dir: Path,
    pairings: tuple[str, ...],
    omega_eV: float,
    q_values: tuple[float, ...],
    nk: int,
    model_name: str = "symmetry_bdg_2band",
    q0_status: dict[str, str] | None = None,
    average_shifted_meshes: bool = False,
    shift_fractions: tuple[float, ...] = (0.0,),
    direct_quadrature: str = "center",
) -> FiniteQWardScanReport:
    report = run_finite_q_ward_scan(
        pairings,
        model_name=model_name,
        omega_eV=omega_eV,
        q_values=q_values,
        nk=nk,
        q0_status=q0_status,
        average_shifted_meshes=average_shifted_meshes,
        shift_fractions=shift_fractions,
        direct_quadrature=direct_quadrature,
    )
    _write_json(output_dir / "finite_q_ward_scan.json", report)
    _write_text(output_dir / "finite_q_ward_scan.txt", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run finite-q Ward residual scans.")
    parser.add_argument("--model", choices=available_finite_q_validation_models(), default="symmetry_bdg_2band")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "validation" / "outputs" / "finite_q_ward_scan")
    parser.add_argument("--pairings", nargs="+", default=["spm", "dwave"])
    parser.add_argument("--omega", type=float, default=0.01)
    parser.add_argument("--q-values", nargs="+", type=float, default=[0.005, 0.01, 0.02])
    parser.add_argument("--nk", type=int, default=3)
    parser.add_argument("--q0-status-json", type=Path, default=None)
    parser.add_argument("--average-shifted-meshes", action="store_true", default=False)
    parser.add_argument("--shift-fractions", nargs="+", type=float, default=[0.0])
    parser.add_argument("--direct-quadrature", choices=DIRECT_QUADRATURE_MODES, default="center")
    args = parser.parse_args(argv)
    q0_status = _q0_status_from_json(args.q0_status_json) if args.q0_status_json is not None else None
    report = run_and_write_report(
        args.output_dir,
        tuple(args.pairings),
        args.omega,
        tuple(args.q_values),
        args.nk,
        model_name=args.model,
        q0_status=q0_status,
        average_shifted_meshes=bool(args.average_shifted_meshes),
        shift_fractions=tuple(args.shift_fractions),
        direct_quadrature=str(args.direct_quadrature),
    )
    print(report.format_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
