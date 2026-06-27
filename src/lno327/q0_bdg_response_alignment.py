"""q=0 BdG response definition-alignment diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .bdg_finite_q_response import bdg_finite_q_response_imag_axis
from .bdg_response import bdg_superconducting_response_imag_axis, bdg_total_kernel_imag_axis
from .conductivity import KuboConfig, k_weights, kubo_conductivity_imag_axis, uniform_bz_mesh
from .pairing import PairingAmplitudes
from .ward_response import normal_physical_density_current_response_components_imag_axis

AlignmentPairingName = Literal["normal", "onsite_s", "spm", "dwave"]


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
        lines.append("说明:")
        lines.extend(f"- {note}" for note in self.pass_fail_notes)
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

    passed = bool(local and all(best_matches[name] is not None for name in ("finite_q_total_q0",)))
    notes = [
        "q=0 对齐是有限 q Ward 诊断的前置检查，不是最终物理结论。",
        "有限 q 矩阵为 3x3 density-current 对象；与 2x2 local 对象比较时只使用 current-current 子块。",
        "若没有明确 local 匹配，本报告保守标记为未通过。",
    ]
    if not local:
        notes.append("当前 onsite_s 没有既有 local BdG public API 作为直接对照。")
    if not passed:
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
        convention_table=_convention_table(tuple(local.keys())),
        convention_notes=tuple(notes),
        passed=passed,
        pass_fail_notes=tuple(notes),
        valid_for_casimir_input=False,
    )
