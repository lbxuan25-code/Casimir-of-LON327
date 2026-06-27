#!/usr/bin/env python3
"""q=0 BdG response definition-alignment diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from lno327.bdg_finite_q_response import bdg_finite_q_response_imag_axis  # noqa: E402
from lno327.bdg_response import bdg_superconducting_response_imag_axis, bdg_total_kernel_imag_axis  # noqa: E402
from lno327.conductivity import KuboConfig, k_weights, kubo_conductivity_imag_axis, uniform_bz_mesh  # noqa: E402
from lno327.pairing import PairingAmplitudes  # noqa: E402
from lno327.ward_response import normal_physical_density_current_response_components_imag_axis  # noqa: E402

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

    def to_dict(self) -> dict[str, float | str | bool]:
        return {
            "finite_q_quantity": self.finite_q_quantity,
            "transformed_local_quantity": self.transformed_local_quantity,
            "absolute_norm_difference": self.absolute_norm_difference,
            "relative_norm_difference": self.relative_norm_difference,
            "finite_q_matrix_norm": self.finite_q_matrix_norm,
            "transformed_local_matrix_norm": self.transformed_local_matrix_norm,
            "passes_tolerance": self.passes_tolerance,
        }


@dataclass(frozen=True)
class Q0BdGAlignmentReport:
    pairing_name: str
    omega_eV: float
    q_model: tuple[float, float]
    nk: int | None
    mesh_size: int
    delta0_eV: float
    compared_quantity_names: tuple[str, ...]
    finite_q_matrices: dict[str, np.ndarray]
    local_matrices: dict[str, np.ndarray]
    matrix_norms: dict[str, float]
    pairwise_difference_norms: dict[str, float]
    relative_difference_norms: dict[str, float]
    best_matching_local_quantity: dict[str, str | None]
    transformed_comparison_rows: tuple[TransformedComparisonRow, ...]
    best_transformed_match: dict[str, str | None]
    convention_table: dict[str, dict[str, str | bool]]
    convention_notes: tuple[str, ...]
    passed: bool
    pass_fail_notes: tuple[str, ...]
    valid_for_casimir_input: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "pairing_name": self.pairing_name,
            "omega_eV": self.omega_eV,
            "q_model": list(self.q_model),
            "nk": self.nk,
            "mesh_size": self.mesh_size,
            "delta0_eV": self.delta0_eV,
            "compared_quantity_names": list(self.compared_quantity_names),
            "matrix_norms": self.matrix_norms,
            "pairwise_difference_norms": self.pairwise_difference_norms,
            "relative_difference_norms": self.relative_difference_norms,
            "best_matching_local_quantity": self.best_matching_local_quantity,
            "transformed_comparison_rows": [row.to_dict() for row in self.transformed_comparison_rows],
            "best_transformed_match": self.best_transformed_match,
            "convention_table": self.convention_table,
            "convention_notes": list(self.convention_notes),
            "passed": self.passed,
            "pass_fail_notes": list(self.pass_fail_notes),
            "valid_for_casimir_input": False,
        }

    def format_text(self) -> str:
        lines = [
            "q=0 BdG 响应定义对齐报告",
            f"配对名称: {self.pairing_name}",
            f"omega_eV: {self.omega_eV:.12g}",
            f"q_model: [{self.q_model[0]:.12g}, {self.q_model[1]:.12g}]",
            f"nk: {self.nk if self.nk is not None else '外部网格'}",
            f"网格点数: {self.mesh_size}",
            f"delta0_eV: {self.delta0_eV:.12g}",
            f"通过: {self.passed}",
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
                "finite_q_quantity | transformed_local_quantity | abs_diff | rel_diff | "
                "finite_q_norm | local_norm | pass"
            )
            for row in sorted(self.transformed_comparison_rows, key=lambda item: item.relative_norm_difference):
                lines.append(
                    f"{row.finite_q_quantity} | {row.transformed_local_quantity} | "
                    f"{row.absolute_norm_difference:.6e} | {row.relative_norm_difference:.6e} | "
                    f"{row.finite_q_matrix_norm:.6e} | {row.transformed_local_matrix_norm:.6e} | "
                    f"{row.passes_tolerance}"
                )
        else:
            lines.append("- 无 transformed local comparator；仅报告 finite-q matrix norms。")
        lines.append("finite-q matrix norms:")
        for name in self.finite_q_matrices:
            lines.append(f"- {name}: {self.matrix_norms[name]:.6e}")
        lines.append("说明:")
        lines.extend(f"- {note}" for note in self.pass_fail_notes)
        lines.extend(f"- {note}" for note in self.convention_notes if note not in self.pass_fail_notes)
        lines.append("本报告只用于 convention 诊断；finite-q 输出不是 formal Casimir input。")
        lines.append("valid_for_casimir_input: False")
        return "\n".join(lines)


def _current_block(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=complex)
    return arr[1:, 1:] if arr.shape == (3, 3) else arr


def _relative_norm(diff: float, left: np.ndarray, right: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), 1e-30)
    return float(diff / scale)


def _finite_q_q0_matrices(
    pairing_name: AlignmentPairingName,
    points: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    amp: PairingAmplitudes,
) -> dict[str, np.ndarray]:
    engine_pairing = "onsite_s" if pairing_name == "normal" else pairing_name
    engine_amp = PairingAmplitudes(delta0_eV=0.0) if pairing_name == "normal" else amp
    response = bdg_finite_q_response_imag_axis(
        engine_pairing,
        config.omega_eV,
        np.array([0.0, 0.0]),
        points,
        weights,
        config,
        engine_amp,
        phase_vertex="bond_endpoint_gauge",
        current_vertex="peierls",
        collective_mode="amplitude_phase",
        collective_counterterm="goldstone_gap_equation",
        include_phase_phase_direct=True,
    )
    return {
        "finite_q_raw_bubble_q0": response.bare_bubble,
        "finite_q_direct_q0": response.direct,
        "finite_q_total_q0": response.bare_total,
        "finite_q_minus_schur_q0": response.minus_schur,
        "finite_q_amplitude_phase_schur_q0": response.amplitude_phase_schur,
    }


def _local_q0_matrices(
    pairing_name: AlignmentPairingName,
    points: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    amp: PairingAmplitudes,
) -> dict[str, np.ndarray]:
    if pairing_name == "normal":
        normal_components = normal_physical_density_current_response_components_imag_axis(
            points,
            config,
            np.array([0.0, 0.0]),
            weights,
        )
        normal_sigma = kubo_conductivity_imag_axis(points, config, weights).matrix()
        return {
            "local_normal_density_current_total": normal_components["total"],
            "local_normal_sigma_like": normal_sigma,
        }
    if pairing_name == "onsite_s":
        return {}
    kernels = bdg_total_kernel_imag_axis(points, config, pairing_name, amp, weights)
    response = bdg_superconducting_response_imag_axis(points, config, pairing_name, amp, weights)
    return {
        "local_K_para": kernels.paramagnetic,
        "local_K_total": kernels.total,
        "local_superconducting_response": response.sigma_like_response,
    }


def _transformed_local_comparators(
    pairing_name: AlignmentPairingName,
    local: dict[str, np.ndarray],
    omega_eV: float,
) -> dict[str, list[tuple[str, np.ndarray]]]:
    if pairing_name == "normal":
        density_total = local["local_normal_density_current_total"]
        sigma_like = local["local_normal_sigma_like"]
        normal_comparators = [
            ("local_normal_density_current_total_current_block", density_total),
            ("omega * local_normal_sigma_like", omega_eV * sigma_like),
            ("-omega * local_normal_sigma_like", -omega_eV * sigma_like),
        ]
        return {
            "finite_q_raw_bubble_q0": normal_comparators,
            "finite_q_direct_q0": normal_comparators,
            "finite_q_total_q0": normal_comparators,
            "finite_q_minus_schur_q0": normal_comparators,
            "finite_q_amplitude_phase_schur_q0": normal_comparators,
        }

    if pairing_name == "onsite_s":
        return {}

    local_k_para = local["local_K_para"]
    local_k_total = local["local_K_total"]
    local_response = local["local_superconducting_response"]
    return {
        "finite_q_raw_bubble_q0": [
            ("local_K_para", local_k_para),
            ("-local_K_para", -local_k_para),
        ],
        "finite_q_direct_q0": [
            ("local_K_total - local_K_para", local_k_total - local_k_para),
            ("local_K_para - local_K_total", local_k_para - local_k_total),
            ("-local_K_total - local_K_para", -local_k_total - local_k_para),
            ("local_K_total + local_K_para", local_k_total + local_k_para),
        ],
        "finite_q_total_q0": [
            ("local_K_total", local_k_total),
            ("-local_K_total", -local_k_total),
            ("omega * local_superconducting_response", omega_eV * local_response),
            ("-omega * local_superconducting_response", -omega_eV * local_response),
        ],
        "finite_q_minus_schur_q0": [
            ("local_K_total", local_k_total),
            ("-local_K_total", -local_k_total),
            ("omega * local_superconducting_response", omega_eV * local_response),
            ("-omega * local_superconducting_response", -omega_eV * local_response),
        ],
        "finite_q_amplitude_phase_schur_q0": [
            ("local_K_total", local_k_total),
            ("-local_K_total", -local_k_total),
            ("omega * local_superconducting_response", omega_eV * local_response),
            ("-omega * local_superconducting_response", -omega_eV * local_response),
        ],
    }


def _transformed_comparison_rows(
    finite_q: dict[str, np.ndarray],
    comparators: dict[str, list[tuple[str, np.ndarray]]],
    tolerance: float,
) -> tuple[TransformedComparisonRow, ...]:
    rows: list[TransformedComparisonRow] = []
    for finite_name, finite_matrix in finite_q.items():
        finite_block = _current_block(finite_matrix)
        finite_norm = float(np.linalg.norm(finite_block))
        for comparator_name, comparator_matrix in comparators.get(finite_name, []):
            local_block = _current_block(comparator_matrix)
            if finite_block.shape != local_block.shape:
                continue
            diff = float(np.linalg.norm(finite_block - local_block))
            local_norm = float(np.linalg.norm(local_block))
            rel = _relative_norm(diff, finite_block, local_block)
            rows.append(
                TransformedComparisonRow(
                    finite_q_quantity=finite_name,
                    transformed_local_quantity=comparator_name,
                    absolute_norm_difference=diff,
                    relative_norm_difference=rel,
                    finite_q_matrix_norm=finite_norm,
                    transformed_local_matrix_norm=local_norm,
                    passes_tolerance=bool(rel <= tolerance),
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


def _row_passes(
    rows: tuple[TransformedComparisonRow, ...],
    finite_q_quantity: str,
    transformed_local_quantity: str,
) -> bool:
    return any(
        row.finite_q_quantity == finite_q_quantity
        and row.transformed_local_quantity == transformed_local_quantity
        and row.passes_tolerance
        for row in rows
    )


def _convention_aware_pass_status(
    pairing_name: AlignmentPairingName,
    rows: tuple[TransformedComparisonRow, ...],
) -> tuple[bool, tuple[str, ...]]:
    notes: list[str] = []
    if pairing_name == "spm":
        raw_ok = _row_passes(rows, "finite_q_raw_bubble_q0", "local_K_para")
        direct_ok = _row_passes(rows, "finite_q_direct_q0", "-local_K_total - local_K_para")
        total_ok = _row_passes(rows, "finite_q_total_q0", "-local_K_total") or _row_passes(
            rows,
            "finite_q_total_q0",
            "-omega * local_superconducting_response",
        )
        minus_schur_ok = _row_passes(rows, "finite_q_minus_schur_q0", "-local_K_total") or _row_passes(
            rows,
            "finite_q_minus_schur_q0",
            "-omega * local_superconducting_response",
        )
        amplitude_phase_schur_ok = _row_passes(
            rows,
            "finite_q_amplitude_phase_schur_q0",
            "-local_K_total",
        ) or _row_passes(
            rows,
            "finite_q_amplitude_phase_schur_q0",
            "-omega * local_superconducting_response",
        )
        passed = raw_ok and direct_ok and total_ok and minus_schur_ok and amplitude_phase_schur_ok
        notes.append(
            "spm 使用 convention-aware q=0 判据：raw bubble 对齐 local_K_para，"
            "direct/contact 对齐 -local_K_total - local_K_para，total/Schur 对齐 "
            "-local_K_total 或 -omega * local_superconducting_response。"
        )
        if not raw_ok:
            notes.append("spm raw bubble 未对齐 local_K_para。")
        if not direct_ok:
            notes.append("spm direct/contact 未对齐 -local_K_total - local_K_para。")
        if not total_ok:
            notes.append("spm total 未对齐 -local_K_total 或 -omega * local_superconducting_response。")
        if not minus_schur_ok:
            notes.append("spm minus-Schur 未对齐 total 的 q=0 sign convention。")
        if not amplitude_phase_schur_ok:
            notes.append("spm amplitude/phase Schur 未对齐 total 的 q=0 sign convention。")
        return bool(passed), tuple(notes)

    if pairing_name == "dwave":
        return (
            False,
            (
                "dwave 保持保守 q=0 判据：raw-bubble 顶角/组装问题需由专门 raw-bubble audit 判定；"
                "本 alignment 报告不因 transformed total/contact 局部对齐而升级为通过。",
            ),
        )
    if pairing_name == "normal":
        return False, ("normal q=0 alignment 属于独立 convention 问题；本报告保持保守未通过。",)
    return False, ("onsite_s 当前没有既有 local BdG public API 作为直接对照，保持 diagnostic-only。",)


def _convention_table(local_names: tuple[str, ...]) -> dict[str, dict[str, str | bool]]:
    table: dict[str, dict[str, str | bool]] = {
        "finite_q_raw_bubble_q0": {
            "含BdG_Nambu_1_2因子": True,
            "仅bubble": True,
            "含direct_contact": False,
            "含collective_Schur": False,
            "已除以omega": False,
            "额外负号": "观测/源电流顶点采用有限q约定",
            "类型": "kernel-like density-current",
        },
        "finite_q_direct_q0": {
            "含BdG_Nambu_1_2因子": True,
            "仅bubble": False,
            "含direct_contact": True,
            "含collective_Schur": False,
            "已除以omega": False,
            "额外负号": "D_ij=-<M_ij>",
            "类型": "contact kernel",
        },
        "finite_q_total_q0": {
            "含BdG_Nambu_1_2因子": True,
            "仅bubble": False,
            "含direct_contact": True,
            "含collective_Schur": False,
            "已除以omega": False,
            "额外负号": "继承 finite-q density-current 约定",
            "类型": "kernel-like density-current",
        },
        "finite_q_minus_schur_q0": {
            "含BdG_Nambu_1_2因子": True,
            "仅bubble": False,
            "含direct_contact": True,
            "含collective_Schur": True,
            "已除以omega": False,
            "额外负号": "phase Schur 使用 minus 号",
            "类型": "kernel-like diagnostic",
        },
        "finite_q_amplitude_phase_schur_q0": {
            "含BdG_Nambu_1_2因子": True,
            "仅bubble": False,
            "含direct_contact": True,
            "含collective_Schur": True,
            "已除以omega": False,
            "额外负号": "amplitude-phase Schur 使用减号",
            "类型": "kernel-like diagnostic",
        },
    }
    for name in local_names:
        table[name] = {
            "含BdG_Nambu_1_2因子": "normal 项不适用；BdG local 项为 True",
            "仅bubble": name.endswith("K_para"),
            "含direct_contact": name.endswith("K_total") or name.endswith("superconducting_response"),
            "含collective_Schur": False,
            "已除以omega": name.endswith("response") or name.endswith("sigma_like"),
            "额外负号": "local BdG K_total=dia-para；normal sigma 使用既有 Kubo 约定",
            "类型": "conductivity-like" if name.endswith("response") or name.endswith("sigma_like") else "kernel-like",
        }
    return table


def run_q0_bdg_response_alignment(
    pairing_name: AlignmentPairingName,
    *,
    omega_eV: float = 0.01,
    nk: int = 3,
    k_points: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    config: KuboConfig | None = None,
    pairing_params: PairingAmplitudes | None = None,
    tolerance: float = 1e-6,
) -> Q0BdGAlignmentReport:
    if pairing_name not in {"normal", "onsite_s", "spm", "dwave"}:
        raise ValueError("pairing_name must be normal, onsite_s, spm, or dwave")
    points = uniform_bz_mesh(nk) if k_points is None else np.asarray(k_points, dtype=float)
    mesh_weights = k_weights(points) if weights is None else np.asarray(weights, dtype=float)
    kubo = config or KuboConfig.from_kelvin(omega_eV=omega_eV, temperature_K=10.0, eta_eV=1e-8, output_si=False)
    amp = pairing_params or PairingAmplitudes()
    finite_q = _finite_q_q0_matrices(pairing_name, points, mesh_weights, kubo, amp)
    local = _local_q0_matrices(pairing_name, points, mesh_weights, kubo, amp)

    matrix_norms = {name: float(np.linalg.norm(value)) for name, value in {**finite_q, **local}.items()}
    difference_norms: dict[str, float] = {}
    relative_norms: dict[str, float] = {}
    best_matches: dict[str, str | None] = {}
    for finite_name, finite_matrix in finite_q.items():
        finite_block = _current_block(finite_matrix)
        best_name: str | None = None
        best_relative = float("inf")
        for local_name, local_matrix in local.items():
            local_block = _current_block(local_matrix)
            if finite_block.shape != local_block.shape:
                continue
            diff = float(np.linalg.norm(finite_block - local_block))
            rel = _relative_norm(diff, finite_block, local_block)
            key = f"{finite_name}__vs__{local_name}"
            difference_norms[key] = diff
            relative_norms[key] = rel
            if rel < best_relative:
                best_relative = rel
                best_name = local_name
        best_matches[finite_name] = best_name if best_relative <= tolerance else None

    transformed_rows = _transformed_comparison_rows(
        finite_q,
        _transformed_local_comparators(pairing_name, local, float(kubo.omega_eV)) if local else {},
        tolerance,
    )
    best_transformed = _best_transformed_matches(transformed_rows)
    for finite_name in finite_q:
        best_transformed.setdefault(finite_name, None)

    passed, pass_rule_notes = _convention_aware_pass_status(pairing_name, transformed_rows)
    notes = [
        "q=0 对齐是有限 q Ward 诊断的前置检查，不是最终物理结论。",
        "有限 q 矩阵为 3x3 density-current 对象；与 2x2 local 对象比较时只使用 current-current 子块。",
        "transformed comparison table 显式比较符号、omega 因子和 direct/contact 组合约定。",
        "若没有明确 local 匹配，本报告保守标记为未通过。",
        *pass_rule_notes,
    ]
    if not local:
        notes.append("当前 onsite_s 没有既有 local BdG public API 作为直接对照。")
        notes.append("onsite_s 报告仅打印 finite-q matrix norms，并明确保持 diagnostic-only。")
    if not passed and pairing_name != "spm":
        notes.append("未找到满足容差的清楚 q=0 local 匹配。")
    return Q0BdGAlignmentReport(
        pairing_name=pairing_name,
        omega_eV=float(kubo.omega_eV),
        q_model=(0.0, 0.0),
        nk=nk if k_points is None else None,
        mesh_size=int(points.shape[0]),
        delta0_eV=0.0 if pairing_name == "normal" else float(amp.delta0_eV),
        compared_quantity_names=tuple([*finite_q.keys(), *local.keys()]),
        finite_q_matrices=finite_q,
        local_matrices=local,
        matrix_norms=matrix_norms,
        pairwise_difference_norms=difference_norms,
        relative_difference_norms=relative_norms,
        best_matching_local_quantity=best_matches,
        transformed_comparison_rows=transformed_rows,
        best_transformed_match=best_transformed,
        convention_table=_convention_table(tuple(local.keys())),
        convention_notes=tuple(notes),
        passed=passed,
        pass_fail_notes=tuple(notes),
        valid_for_casimir_input=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行 q=0 BdG response definition alignment 诊断。")
    parser.add_argument("pairing", choices=("normal", "onsite_s", "spm", "dwave"))
    parser.add_argument("--omega", type=float, default=0.01)
    parser.add_argument("--nk", type=int, default=3)
    parser.add_argument("--delta0", type=float, default=0.04)
    args = parser.parse_args(argv)
    report = run_q0_bdg_response_alignment(
        args.pairing,
        omega_eV=args.omega,
        nk=args.nk,
        pairing_params=PairingAmplitudes(delta0_eV=args.delta0),
    )
    print(report.format_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
