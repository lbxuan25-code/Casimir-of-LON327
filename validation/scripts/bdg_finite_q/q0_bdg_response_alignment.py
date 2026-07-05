#!/usr/bin/env python3
"""Unified q=0 BdG response definition-alignment diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from lno327.workflows.finite_q_engine import FiniteQEngineOptions, bdg_finite_q_response_imag_axis_from_workspace  # noqa: E402
from lno327.diagnostics.bdg_q0_conventions import (  # noqa: E402
    BdGQ0Comparison,
    BdGQ0ConventionResult,
    current_block,
    evaluate_bdg_q0_convention,
    relative_norm,
)
from lno327 import KuboConfig, k_weights, uniform_bz_mesh  # noqa: E402
from lno327.response.finite_q_bdg import precompute_finite_q_bdg_workspace_from_model_ansatz  # noqa: E402
from lno327.response.local_bdg import (  # noqa: E402
    bdg_local_diamagnetic_kernel_from_workspace,
    bdg_local_paramagnetic_kernel_imag_axis_from_workspace,
    bdg_local_superconducting_response_imag_axis_from_workspace,
    bdg_local_total_kernel_imag_axis_from_workspace,
    precompute_bdg_local_workspace_from_model,
)
from lno327.response.normal_density_current import normal_physical_density_current_response_components_imag_axis_from_model  # noqa: E402
from validation.lib.finite_q_validation_models import (  # noqa: E402
    available_finite_q_validation_models,
    get_finite_q_validation_model,
)

AlignmentPairingName = Literal["normal", "onsite_s", "spm", "dwave"]


@dataclass(frozen=True)
class TransformedComparisonRow:
    finite_q_quantity: str
    transformed_local_quantity: str
    absolute_norm_difference: float
    relative_norm_difference: float
    finite_q_matrix_norm: float
    transformed_local_matrix_norm: float
    passes_tolerance: bool
    label: str = ""

    def to_dict(self) -> dict[str, float | str | bool]:
        return {
            "finite_q_quantity": self.finite_q_quantity,
            "transformed_local_quantity": self.transformed_local_quantity,
            "absolute_norm_difference": self.absolute_norm_difference,
            "relative_norm_difference": self.relative_norm_difference,
            "finite_q_matrix_norm": self.finite_q_matrix_norm,
            "transformed_local_matrix_norm": self.transformed_local_matrix_norm,
            "passes_tolerance": self.passes_tolerance,
            "label": self.label,
        }


@dataclass(frozen=True)
class Q0BdGAlignmentReport:
    model_name: str
    model_metadata: dict[str, Any]
    primary_validation_model: bool
    pairing_name: str
    omega_eV: float
    q_model: tuple[float, float]
    nk: int | None
    mesh_size: int
    delta0_eV: float
    status: str
    comparator_family: str
    q0_comparator_available: bool
    status_reason: str
    compared_quantity_names: tuple[str, ...]
    finite_q_matrices: dict[str, np.ndarray]
    local_matrices: dict[str, np.ndarray]
    matrix_norms: dict[str, float]
    pairwise_difference_norms: dict[str, float]
    relative_difference_norms: dict[str, float]
    best_matching_local_quantity: dict[str, str | None]
    transformed_comparison_rows: tuple[TransformedComparisonRow, ...]
    best_transformed_match: dict[str, str | None]
    convention_notes: tuple[str, ...]
    passed: bool
    pass_fail_notes: tuple[str, ...]
    q0_convention: BdGQ0ConventionResult | None = None
    valid_for_casimir_input: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_metadata": self.model_metadata,
            "primary_validation_model": self.primary_validation_model,
            "pairing_name": self.pairing_name,
            "omega_eV": self.omega_eV,
            "q_model": list(self.q_model),
            "nk": self.nk,
            "mesh_size": self.mesh_size,
            "delta0_eV": self.delta0_eV,
            "status": self.status,
            "comparator_family": self.comparator_family,
            "q0_comparator_available": self.q0_comparator_available,
            "status_reason": self.status_reason,
            "compared_quantity_names": list(self.compared_quantity_names),
            "matrix_norms": self.matrix_norms,
            "pairwise_difference_norms": self.pairwise_difference_norms,
            "relative_difference_norms": self.relative_difference_norms,
            "best_matching_local_quantity": self.best_matching_local_quantity,
            "transformed_comparison_rows": [row.to_dict() for row in self.transformed_comparison_rows],
            "best_transformed_match": self.best_transformed_match,
            "convention_notes": list(self.convention_notes),
            "passed": self.passed,
            "pass_fail_notes": list(self.pass_fail_notes),
            "q0_convention": self.q0_convention.to_dict() if self.q0_convention is not None else None,
            "valid_for_casimir_input": False,
        }

    def format_text(self, *, explain: bool = False) -> str:
        lines = [
            "q=0 BdG 响应定义对齐报告",
            f"model_name: {self.model_name}",
            f"配对名称: {self.pairing_name}",
            f"status: {self.status}",
            f"comparator_family: {self.comparator_family}",
            f"q0_comparator_available: {self.q0_comparator_available}",
            f"status_reason: {self.status_reason}",
            f"omega_eV: {self.omega_eV:.12g}",
            f"q_model: [{self.q_model[0]:.12g}, {self.q_model[1]:.12g}]",
            f"nk: {self.nk if self.nk is not None else '外部网格'}",
            f"网格点数: {self.mesh_size}",
            f"delta0_eV: {self.delta0_eV:.12g}",
            f"diagnostic_passed: {self.passed}",
            "最佳 local 匹配:",
        ]
        for name, best in self.best_matching_local_quantity.items():
            lines.append(f"- {name}: {best}")
        lines.append("最佳 transformed 匹配:")
        for name, best in self.best_transformed_match.items():
            lines.append(f"- {name}: {best}")
        lines.append("transformed comparison table（按 relative norm 升序）:")
        if self.transformed_comparison_rows:
            lines.append(
                "finite_q_quantity | transformed_local_quantity | label | abs_diff | rel_diff | "
                "finite_q_norm | local_norm | pass"
            )
            for row in sorted(self.transformed_comparison_rows, key=lambda item: item.relative_norm_difference):
                lines.append(
                    f"{row.finite_q_quantity} | {row.transformed_local_quantity} | {row.label} | "
                    f"{row.absolute_norm_difference:.6e} | {row.relative_norm_difference:.6e} | "
                    f"{row.finite_q_matrix_norm:.6e} | {row.transformed_local_matrix_norm:.6e} | "
                    f"{row.passes_tolerance}"
                )
        else:
            lines.append("- 无 transformed local comparator；仅报告 finite-q matrix norms。")
        lines.append("finite-q matrix norms:")
        for name in self.finite_q_matrices:
            lines.append(f"- {name}: {self.matrix_norms[name]:.6e}")
        if explain and self.q0_convention is not None:
            lines.append("intraband-aware explanation:")
            lines.append(f"- current_vertex_status: {self.q0_convention.current_vertex_status}")
            lines.append(f"- interpretation: {self.q0_convention.interpretation}")
            for comparison in self.q0_convention.comparisons:
                lines.append(
                    f"- {comparison.name}: rel={comparison.relative_norm_difference:.6e}, "
                    f"pass={comparison.passes_tolerance}"
                )
        lines.append("说明:")
        lines.extend(f"- {note}" for note in self.pass_fail_notes)
        lines.append("本报告只用于 convention 诊断；finite-q 输出不是 Casimir 输入，也不是 Ward closure proof。")
        lines.append("valid_for_casimir_input: False")
        return "\n".join(lines)


def _comparison_to_row(comparison: BdGQ0Comparison) -> TransformedComparisonRow:
    return TransformedComparisonRow(
        finite_q_quantity=comparison.left_name,
        transformed_local_quantity=comparison.right_name,
        absolute_norm_difference=comparison.absolute_norm_difference,
        relative_norm_difference=comparison.relative_norm_difference,
        finite_q_matrix_norm=comparison.left_norm,
        transformed_local_matrix_norm=comparison.right_norm,
        passes_tolerance=comparison.passes_tolerance,
        label=comparison.name,
    )


def _finite_q_q0_matrices(
    pairing_name: AlignmentPairingName,
    points: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    amp,
    model,
) -> dict[str, np.ndarray]:
    engine_amp = model.build_pairing_params(0.0) if pairing_name == "normal" else amp
    ansatz = model.build_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
    workspace = precompute_finite_q_bdg_workspace_from_model_ansatz(
        model.spec,
        ansatz,
        np.array([0.0, 0.0]),
        points,
        weights,
        config,
        engine_amp,
        FiniteQEngineOptions(
            current_vertex="peierls",
            collective_mode="amplitude_phase",
            collective_counterterm="goldstone_gap_equation",
            include_phase_phase_direct=True,
        ),
    )
    response = bdg_finite_q_response_imag_axis_from_workspace(workspace, config=config)
    return {
        "finite_q_raw_bubble_q0": response.bare_bubble,
        "finite_q_direct_q0": response.direct,
        "finite_q_total_q0": response.bare_total,
        "finite_q_minus_schur_q0": response.minus_schur,
        "finite_q_amplitude_phase_schur_q0": response.amplitude_phase_schur,
    }


def _normal_local_matrices(points: np.ndarray, weights: np.ndarray, config: KuboConfig, model) -> dict[str, np.ndarray]:
    normal_components = normal_physical_density_current_response_components_imag_axis_from_model(
        model.spec,
        points,
        config,
        np.array([0.0, 0.0]),
        weights,
    )
    return {
        "local_normal_density_current_total": normal_components["total"],
    }


def _local_bdg_matrices(
    pairing_name: AlignmentPairingName,
    points: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    model,
) -> dict[str, np.ndarray]:
    workspace = precompute_bdg_local_workspace_from_model(model.spec, pairing_name, points, config, weights)
    components = bdg_local_total_kernel_imag_axis_from_workspace(workspace, config)
    superconducting = bdg_local_superconducting_response_imag_axis_from_workspace(workspace, config)
    return {
        "local_BdG_paramagnetic_kernel": bdg_local_paramagnetic_kernel_imag_axis_from_workspace(workspace, config),
        "local_BdG_diamagnetic_kernel": bdg_local_diamagnetic_kernel_from_workspace(workspace, config),
        "local_BdG_total_kernel": components.total,
        "-local_BdG_paramagnetic_kernel": -components.paramagnetic,
        "-local_BdG_diamagnetic_kernel": -components.diamagnetic,
        "-local_BdG_total_kernel": -components.total,
        "local_BdG_sigma_like_response": superconducting.sigma_like_response,
        "omega * local_BdG_sigma_like_response": float(config.omega_eV) * superconducting.sigma_like_response,
    }


def _comparison_rows_against_local(
    finite_q: dict[str, np.ndarray],
    local: dict[str, np.ndarray],
    tolerance: float,
    label: str,
) -> tuple[TransformedComparisonRow, ...]:
    if not local:
        return ()
    rows: list[TransformedComparisonRow] = []
    for finite_name, finite_matrix in finite_q.items():
        finite_block = current_block(finite_matrix)
        finite_norm = float(np.linalg.norm(finite_block))
        for comparator_name, comparator_matrix in local.items():
            local_block = current_block(comparator_matrix)
            if finite_block.shape != local_block.shape:
                continue
            row_comparator_name = (
                f"{comparator_name}_current_block"
                if comparator_name.startswith("local_normal_density_current_")
                else comparator_name
            )
            diff = float(np.linalg.norm(finite_block - local_block))
            rel = relative_norm(diff, finite_block, local_block)
            rows.append(
                TransformedComparisonRow(
                    finite_name,
                    row_comparator_name,
                    diff,
                    rel,
                    finite_norm,
                    float(np.linalg.norm(local_block)),
                    bool(rel <= tolerance),
                    label,
                )
            )
    return tuple(rows)


def _best_transformed_matches(rows: tuple[TransformedComparisonRow, ...]) -> dict[str, str | None]:
    best: dict[str, tuple[str, float]] = {}
    for row in rows:
        current = best.get(row.finite_q_quantity)
        if current is None or row.relative_norm_difference < current[1]:
            best[row.finite_q_quantity] = (row.transformed_local_quantity, row.relative_norm_difference)
    return {name: value[0] for name, value in best.items()}


def _best_local_matches(
    finite_q: dict[str, np.ndarray],
    local: dict[str, np.ndarray],
    tolerance: float,
) -> tuple[dict[str, str | None], dict[str, float], dict[str, float]]:
    difference_norms: dict[str, float] = {}
    relative_norms: dict[str, float] = {}
    best_matches: dict[str, str | None] = {}
    for finite_name, finite_matrix in finite_q.items():
        finite_block = current_block(finite_matrix)
        best_name: str | None = None
        best_relative = float("inf")
        for local_name, local_matrix in local.items():
            local_block = current_block(local_matrix)
            if finite_block.shape != local_block.shape:
                continue
            diff = float(np.linalg.norm(finite_block - local_block))
            rel = relative_norm(diff, finite_block, local_block)
            difference_norms[f"{finite_name}__vs__{local_name}"] = diff
            relative_norms[f"{finite_name}__vs__{local_name}"] = rel
            if rel < best_relative:
                best_relative = rel
                best_name = local_name
        best_matches[finite_name] = best_name if best_relative <= tolerance else None
    return best_matches, difference_norms, relative_norms


def run_q0_bdg_response_alignment(
    pairing_name: AlignmentPairingName,
    *,
    model_name: str = "symmetry_bdg_2band",
    omega_eV: float = 0.01,
    nk: int = 3,
    k_points: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    config: KuboConfig | None = None,
    pairing_params=None,
    tolerance: float = 1e-6,
) -> Q0BdGAlignmentReport:
    model = get_finite_q_validation_model(model_name)
    model.require_pairing(pairing_name)
    points = uniform_bz_mesh(nk) if k_points is None else np.asarray(k_points, dtype=float)
    mesh_weights = k_weights(points) if weights is None else np.asarray(weights, dtype=float)
    kubo = config or KuboConfig.from_kelvin(omega_eV=omega_eV, temperature_K=10.0, eta_eV=1e-8, output_si=False)
    amp = pairing_params or model.build_pairing_params()

    q0_convention: BdGQ0ConventionResult | None = None
    comparator_family = "unavailable"
    q0_comparator_available = False
    status_reason = "no q0 comparator was selected"
    if model.name == "lno327_four_orbital" and pairing_name in {"spm", "dwave"}:
        q0_convention = evaluate_bdg_q0_convention(pairing_name, points, mesh_weights, kubo, amp, tolerance=tolerance)
        finite_q = {
            name: value
            for name, value in q0_convention.matrix_dict().items()
            if name.startswith("finite_q") or name == "local_K_para_total - finite_q_raw_bubble_q0"
        }
        local = {
            name: value
            for name, value in q0_convention.matrix_dict().items()
            if name.startswith("local_") and name != "local_K_para_total - finite_q_raw_bubble_q0"
        }
        transformed_rows = tuple(_comparison_to_row(comparison) for comparison in q0_convention.comparisons)
        status = q0_convention.status
        passed = status in {"convention_aware_pass", "intraband_aware_pass"}
        comparator_family = "local_bdg"
        q0_comparator_available = True
        status_reason = q0_convention.interpretation
        if status == "convention_aware_pass":
            notes = (
                "spm convention-aware q=0 pass: raw bubble aligns with local total/interband and intraband is negligible.",
            )
        elif status == "intraband_aware_pass":
            notes = (
                "dwave intraband-aware q=0 pass: raw bubble aligns with local interband.",
                "The old raw-vs-total mismatch remains visible and is explained by local intraband / -f'(E).",
            )
        else:
            notes = (q0_convention.interpretation,)
    else:
        finite_q = _finite_q_q0_matrices(pairing_name, points, mesh_weights, kubo, amp, model)
        if pairing_name == "normal":
            local = _normal_local_matrices(points, mesh_weights, kubo, model)
            comparator_family = "normal_local"
            q0_comparator_available = True
            transformed_rows = _comparison_rows_against_local(
                finite_q,
                local,
                tolerance,
                "normal_q0_convention_probe",
            )
            status_reason = "normal local density-current comparator is available"
        elif model.name == "symmetry_bdg_2band" and pairing_name in {"spm", "dwave"}:
            local = _local_bdg_matrices(pairing_name, points, mesh_weights, kubo, model)
            comparator_family = "local_bdg"
            q0_comparator_available = True
            transformed_rows = _comparison_rows_against_local(
                finite_q,
                local,
                tolerance,
                "two_band_local_bdg_convention_probe",
            )
            status_reason = (
                "two-band local BdG comparator is available; finite-q/local sign conventions are reported directly"
            )
        else:
            local = {}
            transformed_rows = ()
            status_reason = "no local public comparator is available for this pairing/model"
        passed_pairs = [row for row in transformed_rows if row.passes_tolerance]
        if q0_comparator_available and passed_pairs:
            status = "convention_aware_pass"
            passed = True
            status_reason += "; at least one finite/local current-block comparison passed tolerance"
        else:
            status = "diagnostic_only_not_passed"
            passed = False
        notes = (
            "normal q=0 alignment remains diagnostic-only." if pairing_name == "normal"
            else (
                "two-band superconducting local BdG comparator is reported without fitting or projection."
                if q0_comparator_available
                else "No local public comparator is available in this q=0 alignment report."
            ),
            status_reason,
        )

    matrix_norms = {name: float(np.linalg.norm(value)) for name, value in {**finite_q, **local}.items()}
    best_matches, difference_norms, relative_norms = _best_local_matches(finite_q, local, tolerance)
    best_transformed = _best_transformed_matches(transformed_rows)
    for finite_name in finite_q:
        best_transformed.setdefault(finite_name, None)

    common_notes = (
        "q=0 对齐是 finite-q Ward 诊断的前置定义检查，不是最终物理结论。",
        "spm/dwave q=0 约定由 lno327.diagnostics.bdg_q0_conventions 统一计算，避免脚本重复实现。",
        f"model_name={model.name}; primary_validation_model={model.primary_validation_model}.",
        "finite-q 输出保持 valid_for_casimir_input=False。",
    )
    return Q0BdGAlignmentReport(
        model_name=model.name,
        model_metadata=model.metadata(),
        primary_validation_model=model.primary_validation_model,
        pairing_name=pairing_name,
        omega_eV=float(kubo.omega_eV),
        q_model=(0.0, 0.0),
        nk=nk if k_points is None else None,
        mesh_size=int(points.shape[0]),
        delta0_eV=0.0 if pairing_name == "normal" else float(amp.delta0_eV),
        status=status,
        comparator_family=comparator_family,
        q0_comparator_available=q0_comparator_available,
        status_reason=status_reason,
        compared_quantity_names=tuple([*finite_q.keys(), *local.keys()]),
        finite_q_matrices=finite_q,
        local_matrices=local,
        matrix_norms=matrix_norms,
        pairwise_difference_norms=difference_norms,
        relative_difference_norms=relative_norms,
        best_matching_local_quantity=best_matches,
        transformed_comparison_rows=transformed_rows,
        best_transformed_match=best_transformed,
        convention_notes=tuple([*common_notes, *notes]),
        passed=passed,
        pass_fail_notes=tuple([*common_notes, *notes]),
        q0_convention=q0_convention,
        valid_for_casimir_input=False,
    )


def run_q0_bdg_response_alignment_many(
    pairings: tuple[AlignmentPairingName, ...],
    **kwargs: Any,
) -> tuple[Q0BdGAlignmentReport, ...]:
    return tuple(run_q0_bdg_response_alignment(pairing, **kwargs) for pairing in pairings)


def _write_json(path: Path, reports: tuple[Q0BdGAlignmentReport, ...]) -> None:
    payload = {
        "model_name": reports[0].model_name if reports else None,
        "model_metadata": reports[0].model_metadata if reports else None,
        "primary_validation_model": reports[0].primary_validation_model if reports else None,
        "pairings": [report.pairing_name for report in reports],
        "status_by_pairing": {report.pairing_name: report.status for report in reports},
        "valid_for_casimir_input": False,
        "reports": [report.to_dict() for report in reports],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行统一 q=0 BdG response definition alignment 诊断。")
    parser.add_argument("--model", choices=available_finite_q_validation_models(), default="symmetry_bdg_2band")
    parser.add_argument("pairing", nargs="?")
    parser.add_argument("--pairings", nargs="+")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--omega", type=float, default=0.01)
    parser.add_argument("--nk", type=int, default=3)
    parser.add_argument("--delta0", type=float)
    args = parser.parse_args(argv)
    model = get_finite_q_validation_model(args.model)
    pairings = tuple(args.pairings or ([args.pairing] if args.pairing else model.pairing_names))
    for pairing in pairings:
        model.require_pairing(pairing)
    reports = run_q0_bdg_response_alignment_many(
        pairings,
        model_name=model.name,
        omega_eV=args.omega,
        nk=args.nk,
        pairing_params=model.build_pairing_params(args.delta0),
    )
    for index, report in enumerate(reports):
        if index:
            print()
        print(report.format_text(explain=args.explain))
    if args.json_output is not None:
        _write_json(args.json_output, reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
