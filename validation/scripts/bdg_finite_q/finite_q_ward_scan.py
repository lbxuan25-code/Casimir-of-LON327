#!/usr/bin/env python3
"""Finite-q Ward residual scans for superconducting BdG diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from lno327 import KuboConfig, k_weights, uniform_bz_mesh  # noqa: E402
from lno327.workflows.finite_q_engine import FiniteQEngineOptions, bdg_finite_q_response_imag_axis_from_workspace  # noqa: E402
from lno327.response.finite_q_bdg import precompute_finite_q_bdg_workspace_from_model_ansatz  # noqa: E402
from lno327.collective.validation import validate_physical_ward_identity  # noqa: E402
from validation.lib.finite_q_validation_models import (  # noqa: E402
    available_finite_q_validation_models,
    get_finite_q_validation_model,
)
from q0_bdg_response_alignment import run_q0_bdg_response_alignment_many  # noqa: E402

WardScanPairingName = str
WARD_COMPONENT_LABELS = ("density", "current_x", "current_y")
WARD_CLOSURE_RESPONSE_NAMES = ("bare_total", "minus_schur", "amplitude_phase_schur")


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
            "rows": [
                {
                    **row.__dict__,
                    "q_model": list(row.q_model),
                    "valid_for_casimir_input": False,
                }
                for row in self.rows
            ],
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
            "有限 q Ward 残差扫描报告",
            f"model_name: {self.model_name}",
            f"配对: {', '.join(self.pairing_names)}",
            f"omega_eV: {self.omega_eV:.12g}",
            f"nk: {self.nk if self.nk is not None else '外部网格'}",
            f"网格点数: {self.mesh_size}",
            f"delta0_eV: {self.delta0_eV:.12g}",
            f"q0_precondition_status: {self.q0_precondition_status}",
            f"diagnostic_run_completed: {self.diagnostic_run_completed}",
            f"ward_identity_closed: {self.ward_identity_closed}",
            f"workspace_evaluation: {self.workspace_evaluation}",
        ]
        for row in self.rows:
            lines.append(
                f"- {row.pairing_name} {row.response_name} q={row.q_model}: "
                f"max残差={row.max_ward_residual_norm:.6e}, inverse={row.inverse_method}"
            )
        lines.append("valid_for_casimir_input: False")
        return "\n".join(lines)


def _safe_ratio(value: float, reference: float) -> float | None:
    if reference <= 0.0:
        return None
    return float(value / reference)


def _complex_vector_components(vector: np.ndarray) -> tuple[dict[str, float | str], ...]:
    array = np.asarray(vector, dtype=complex)
    if array.shape != (3,):
        raise ValueError("Ward residual vector must have shape (3,)")
    return tuple(
        {
            "component": label,
            "real": float(np.real(value)),
            "imag": float(np.imag(value)),
        }
        for label, value in zip(WARD_COMPONENT_LABELS, array, strict=True)
    )


def _complex_vector_difference(left: np.ndarray, right: np.ndarray) -> tuple[dict[str, float | str], ...]:
    return _complex_vector_components(np.asarray(left, dtype=complex) - np.asarray(right, dtype=complex))


def _residual_vectors_for_report(ward_report: Any) -> dict[str, Any]:
    return {
        "left_ward_residual_vector": _complex_vector_components(ward_report.left_residual),
        "right_ward_residual_vector": _complex_vector_components(ward_report.right_residual),
        "left_ward_residual_norm": float(ward_report.left_norm),
        "right_ward_residual_norm": float(ward_report.right_norm),
    }


def _schur_difference_report(
    pairing_name: str,
    q: np.ndarray,
    ward_reports: dict[str, Any],
) -> dict[str, Any]:
    response_names = ("bare_total", "minus_schur", "amplitude_phase_schur")
    if pairing_name in {"onsite_s", "spm"}:
        difference_pairs = (("bare_total", "minus_schur"),)
        diagnostic_role = "schur_correction_residual_delta"
    else:
        difference_pairs = (
            ("bare_total", "minus_schur"),
            ("bare_total", "amplitude_phase_schur"),
            ("minus_schur", "amplitude_phase_schur"),
        )
        diagnostic_role = "dwave_control_residual_delta"
    return {
        "pairing_name": pairing_name,
        "q_model": [float(q[0]), float(q[1])],
        "q_norm": float(np.linalg.norm(q)),
        "diagnostic_role": diagnostic_role,
        "residual_component_labels": list(WARD_COMPONENT_LABELS),
        "response_residual_vectors": {
            response_name: _residual_vectors_for_report(ward_reports[response_name])
            for response_name in response_names
        },
        "differences": [
            {
                "formula": f"{left_name} - {right_name}",
                "left_ward_residual_vector_difference": _complex_vector_difference(
                    ward_reports[left_name].left_residual,
                    ward_reports[right_name].left_residual,
                ),
                "right_ward_residual_vector_difference": _complex_vector_difference(
                    ward_reports[left_name].right_residual,
                    ward_reports[right_name].right_residual,
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
    if q_first == q_last:
        return None
    return float((np.log(r_last) - np.log(r_first)) / (np.log(q_last) - np.log(q_first)))


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
) -> FiniteQWardScanReport:
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
    rows: list[FiniteQWardScanRow] = []
    schur_residual_differences: list[dict[str, Any]] = []
    scaling_inputs: dict[str, tuple[list[float], list[float]]] = {}
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
    for pairing_name in selected_pairings:
        ansatz = model.build_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
        for q_value in q_values:
            for direction in q_directions:
                direction_array = np.asarray(direction, dtype=float)
                norm = float(np.linalg.norm(direction_array))
                if norm <= 0.0:
                    raise ValueError("q direction must be nonzero")
                q = float(q_value) * direction_array / norm
                workspace = precompute_finite_q_bdg_workspace_from_model_ansatz(
                    model.spec,
                    ansatz,
                    q,
                    points,
                    mesh_weights,
                    kubo,
                    amp,
                    options,
                )
                response = bdg_finite_q_response_imag_axis_from_workspace(workspace, config=kubo)
                matrices = {
                    "bare_bubble": response.bare_bubble,
                    "direct": response.direct,
                    "bare_total": response.bare_total,
                    "minus_schur": response.minus_schur,
                    "plus_schur": response.plus_schur,
                    "amplitude_phase_schur": response.amplitude_phase_schur,
                }
                bare_report = validate_physical_ward_identity(
                    response.bare_total,
                    kubo.omega_eV,
                    q,
                    tolerance=tolerance,
                )
                bare_max = float(max(bare_report.left_norm, bare_report.right_norm))
                ward_reports: dict[str, Any] = {}
                for response_name, matrix in matrices.items():
                    ward = validate_physical_ward_identity(matrix, kubo.omega_eV, q, tolerance=tolerance)
                    ward_reports[response_name] = ward
                    max_norm = float(max(ward.left_norm, ward.right_norm))
                    key = f"{pairing_name}:{response_name}"
                    scaling_q, scaling_residual = scaling_inputs.setdefault(key, ([], []))
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
                            collective_matrix_condition_number=response.metadata.get(
                                "collective_total_condition_number"
                            ),
                            inverse_method=inverse_method,
                            pinv_diagnostic_used=inverse_method == "pinv_diagnostic",
                            valid_for_casimir_input=False,
                        )
                    )
                schur_residual_differences.append(_schur_difference_report(pairing_name, q, ward_reports))
    slopes = {
        key: _scaling_slope(value[0], value[1])
        for key, value in scaling_inputs.items()
    }
    diagnostic_completed = all(np.isfinite(row.max_ward_residual_norm) for row in rows)
    closure_rows = [row for row in rows if row.response_name in WARD_CLOSURE_RESPONSE_NAMES]
    ward_identity_closed = bool(
        closure_rows and all(row.max_ward_residual_norm <= tolerance for row in closure_rows)
    )
    notes = (
        "本扫描在同一入口先记录 q=0 response definition alignment 前置结果。",
        "dwave 若显示 intraband_aware_pass，表示 q=0 raw bubble 对齐 local interband，raw-vs-total 差异由 intraband/-f'(E) local 项解释。",
        "diagnostic_run_completed 只表示扫描数值完成；ward_identity_closed 才表示 Ward identity 在给定 tolerance 下闭合。",
        "finite-q 输出保持 valid_for_casimir_input=False。",
        f"model_name={model.name}; primary_validation_model={model.primary_validation_model}.",
        "finite-q response 使用预计算 workspace evaluate。",
        (
            "q0_precondition_not_established; finite-q Ward result is diagnostic-only."
            if any(status == "diagnostic_only_not_passed" for status in q0_alignment.values())
            else "q0_precondition_status contains no diagnostic_only_not_passed entries."
        ),
        "残差比例为各响应 max residual 相对 bare_total 的比例。",
        "新增 bare_bubble/direct/plus_schur 行为 staged diagnostic-only 输出，不改变 ward_identity_closed 判据。",
        "left/right Ward residual vector 分量顺序为 density,current_x,current_y，并分别记录 real/imag。",
        "schur_residual_differences 只比较 residual vectors，不改变 Schur correction、Ward 判据或 Casimir gating。",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行有限 q Ward 残差扫描诊断。")
    parser.add_argument("--model", choices=available_finite_q_validation_models(), default="symmetry_bdg_2band")
    parser.add_argument("--pairings", nargs="+")
    parser.add_argument("--omega", type=float, default=0.01)
    parser.add_argument("--q-values", nargs="+", type=float, default=[0.005, 0.01, 0.02])
    parser.add_argument("--nk", type=int, default=3)
    parser.add_argument("--delta0", type=float)
    parser.add_argument("--q0-status-json", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    q0_status = _q0_status_from_json(args.q0_status_json) if args.q0_status_json is not None else None
    model = get_finite_q_validation_model(args.model)
    pairings = tuple(args.pairings) if args.pairings else None
    if pairings is not None:
        for pairing in pairings:
            model.require_pairing(pairing)
    report = run_finite_q_ward_scan(
        pairings,
        model_name=model.name,
        omega_eV=args.omega,
        q_values=tuple(args.q_values),
        nk=args.nk,
        pairing_params=model.build_pairing_params(args.delta0),
        q0_status=q0_status,
    )
    print(report.format_text())
    if args.json_output is not None:
        _write_json(args.json_output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
