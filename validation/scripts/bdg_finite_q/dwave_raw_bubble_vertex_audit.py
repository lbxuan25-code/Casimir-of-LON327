#!/usr/bin/env python3
"""d-wave q=0 raw-bubble and vertex audit for BdG finite-q diagnostics."""

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
from lno327.bdg_response import bdg_current_vertex, bdg_total_kernel_imag_axis  # noqa: E402
from lno327.conductivity import KuboConfig, k_weights, uniform_bz_mesh  # noqa: E402
from lno327.finite_q_primitives import bdg_finite_q_vector_vertex  # noqa: E402
from lno327.pairing import PairingAmplitudes  # noqa: E402

AuditPairingName = Literal["spm", "dwave"]


@dataclass(frozen=True)
class RawBubbleAuditRow:
    pairing_name: AuditPairingName
    finite_q_raw_bubble_norm: float
    local_k_para_norm: float
    raw_vs_local_abs: float
    raw_vs_local_rel: float
    finite_q_vs_local_vertex_max_abs: float
    finite_q_vs_local_vertex_max_rel: float
    vertex_abs_tolerance: float
    vertex_rel_tolerance: float
    vertex_status: str
    evidence: str
    valid_for_casimir_input: bool = False

    def to_dict(self) -> dict[str, float | str | bool]:
        return {
            "pairing_name": self.pairing_name,
            "finite_q_raw_bubble_norm": self.finite_q_raw_bubble_norm,
            "local_k_para_norm": self.local_k_para_norm,
            "raw_vs_local_abs": self.raw_vs_local_abs,
            "raw_vs_local_rel": self.raw_vs_local_rel,
            "finite_q_vs_local_vertex_max_abs": self.finite_q_vs_local_vertex_max_abs,
            "finite_q_vs_local_vertex_max_rel": self.finite_q_vs_local_vertex_max_rel,
            "vertex_abs_tolerance": self.vertex_abs_tolerance,
            "vertex_rel_tolerance": self.vertex_rel_tolerance,
            "vertex_status": self.vertex_status,
            "evidence": self.evidence,
            "valid_for_casimir_input": False,
        }


@dataclass(frozen=True)
class DWaveRawBubbleVertexAuditReport:
    omega_eV: float
    q_model: tuple[float, float]
    nk: int
    mesh_size: int
    delta0_eV: float
    rows: tuple[RawBubbleAuditRow, ...]
    dwave_specific_mismatch: bool
    interpretation: str
    notes: tuple[str, ...]
    valid_for_casimir_input: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "omega_eV": self.omega_eV,
            "q_model": list(self.q_model),
            "nk": self.nk,
            "mesh_size": self.mesh_size,
            "delta0_eV": self.delta0_eV,
            "rows": [row.to_dict() for row in self.rows],
            "dwave_specific_mismatch": self.dwave_specific_mismatch,
            "interpretation": self.interpretation,
            "notes": list(self.notes),
            "valid_for_casimir_input": False,
        }

    def format_text(self) -> str:
        lines = [
            "d-wave raw-bubble / vertex audit",
            f"omega_eV: {self.omega_eV:.12g}",
            f"q_model: [{self.q_model[0]:.12g}, {self.q_model[1]:.12g}]",
            f"nk: {self.nk}",
            f"网格点数: {self.mesh_size}",
            f"delta0_eV: {self.delta0_eV:.12g}",
            f"dwave_specific_mismatch: {self.dwave_specific_mismatch}",
            f"interpretation: {self.interpretation}",
            (
                "pairing | raw_norm | local_K_para_norm | raw_abs | raw_rel | "
                "vertex_abs_max | vertex_rel_max | vertex_status | evidence"
            ),
        ]
        for row in self.rows:
            lines.append(
                f"{row.pairing_name} | {row.finite_q_raw_bubble_norm:.6e} | "
                f"{row.local_k_para_norm:.6e} | {row.raw_vs_local_abs:.6e} | "
                f"{row.raw_vs_local_rel:.6e} | {row.finite_q_vs_local_vertex_max_abs:.6e} | "
                f"{row.finite_q_vs_local_vertex_max_rel:.6e} | {row.vertex_status} | {row.evidence}"
            )
        lines.append("说明:")
        lines.extend(f"- {note}" for note in self.notes)
        lines.append("valid_for_casimir_input: False")
        return "\n".join(lines)


def _current_block(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=complex)
    return arr[1:, 1:] if arr.shape == (3, 3) else arr


def _relative_norm(diff: float, left: np.ndarray, right: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), 1e-30)
    return float(diff / scale)


def _vertex_difference_max(
    points: np.ndarray,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[float, float, str]:
    max_abs = 0.0
    max_rel = 0.0
    for kx, ky in points:
        for direction in ("x", "y"):
            finite_q_vertex = bdg_finite_q_vector_vertex(float(kx), float(ky), 0.0, 0.0, direction)
            local_vertex = bdg_current_vertex(float(kx), float(ky), direction)
            diff = float(np.linalg.norm(finite_q_vertex - local_vertex))
            rel = _relative_norm(diff, finite_q_vertex, local_vertex)
            max_abs = max(max_abs, diff)
            max_rel = max(max_rel, rel)
    status = (
        "vertex_operator_q0_match"
        if max_abs <= absolute_tolerance or max_rel <= relative_tolerance
        else "vertex_operator_level_mismatch"
    )
    return max_abs, max_rel, status


def _audit_one_pairing(
    pairing_name: AuditPairingName,
    points: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    amp: PairingAmplitudes,
    raw_relative_tolerance: float,
    vertex_abs_tolerance: float,
    vertex_rel_tolerance: float,
) -> RawBubbleAuditRow:
    response = bdg_finite_q_response_imag_axis(
        pairing_name,
        config.omega_eV,
        np.array([0.0, 0.0]),
        points,
        weights,
        config,
        amp,
        phase_vertex="bond_endpoint_gauge",
        current_vertex="peierls",
        collective_mode="amplitude_phase",
        collective_counterterm="goldstone_gap_equation",
        include_phase_phase_direct=True,
    )
    local = bdg_total_kernel_imag_axis(points, config, pairing_name, amp, weights)
    finite_q_raw = _current_block(response.bare_bubble)
    local_k_para = np.asarray(local.paramagnetic, dtype=complex)
    raw_abs = float(np.linalg.norm(finite_q_raw - local_k_para))
    raw_rel = _relative_norm(raw_abs, finite_q_raw, local_k_para)
    vertex_abs, vertex_rel, vertex_status = _vertex_difference_max(
        points,
        absolute_tolerance=vertex_abs_tolerance,
        relative_tolerance=vertex_rel_tolerance,
    )
    if raw_rel <= raw_relative_tolerance:
        evidence = "raw_bubble_matches_local_K_para"
    elif vertex_status == "vertex_operator_level_mismatch":
        evidence = vertex_status
    else:
        evidence = "bubble_assembly_or_pairing_state_convention_mismatch"
    return RawBubbleAuditRow(
        pairing_name=pairing_name,
        finite_q_raw_bubble_norm=float(np.linalg.norm(finite_q_raw)),
        local_k_para_norm=float(np.linalg.norm(local_k_para)),
        raw_vs_local_abs=raw_abs,
        raw_vs_local_rel=raw_rel,
        finite_q_vs_local_vertex_max_abs=vertex_abs,
        finite_q_vs_local_vertex_max_rel=vertex_rel,
        vertex_abs_tolerance=vertex_abs_tolerance,
        vertex_rel_tolerance=vertex_rel_tolerance,
        vertex_status=vertex_status,
        evidence=evidence,
        valid_for_casimir_input=False,
    )


def run_dwave_raw_bubble_vertex_audit(
    *,
    nk: int = 3,
    omega_eV: float = 0.01,
    delta0_eV: float = 0.04,
    tolerance: float = 1e-6,
    vertex_abs_tolerance: float = 1e-12,
    vertex_rel_tolerance: float | None = None,
) -> DWaveRawBubbleVertexAuditReport:
    points = uniform_bz_mesh(nk)
    weights = k_weights(points)
    config = KuboConfig.from_kelvin(omega_eV=omega_eV, temperature_K=10.0, eta_eV=1e-8, output_si=False)
    amp = PairingAmplitudes(delta0_eV=delta0_eV)
    effective_vertex_rel_tolerance = tolerance if vertex_rel_tolerance is None else vertex_rel_tolerance
    rows = tuple(
        _audit_one_pairing(
            pairing_name,
            points,
            weights,
            config,
            amp,
            tolerance,
            vertex_abs_tolerance,
            effective_vertex_rel_tolerance,
        )
        for pairing_name in ("spm", "dwave")
    )
    row_by_pairing = {row.pairing_name: row for row in rows}
    spm_ok = row_by_pairing["spm"].raw_vs_local_rel <= tolerance
    dwave_ok = row_by_pairing["dwave"].raw_vs_local_rel <= tolerance
    dwave_specific_mismatch = bool(spm_ok and not dwave_ok)
    if dwave_specific_mismatch:
        interpretation = "dwave_specific_raw_bubble_mismatch"
    elif dwave_ok:
        interpretation = "no_dwave_raw_bubble_mismatch_on_this_grid"
    else:
        interpretation = "shared_or_inconclusive_raw_bubble_mismatch"
    notes = (
        "该脚本只读比较 q=0 finite-q raw bubble、local K_para 与 q=0 current vertex；不改变响应公式。",
        "q=0 current vertex 用绝对或相对容差判定；roundoff 级绝对差不会被标为 vertex operator mismatch。",
        "spm 作为同一有限 q 后端的控制样本，用来判断问题是否具有 d-wave 特异性。",
        "若 q=0 vertex 层面对齐但 raw bubble 不对齐，优先怀疑 bubble 组装、配对态约定或动量依赖配对电流贡献。",
        "本 audit 是 diagnostic-only，不作为 Casimir 输入或 production 通过条件。",
    )
    return DWaveRawBubbleVertexAuditReport(
        omega_eV=float(config.omega_eV),
        q_model=(0.0, 0.0),
        nk=nk,
        mesh_size=int(points.shape[0]),
        delta0_eV=float(amp.delta0_eV),
        rows=rows,
        dwave_specific_mismatch=dwave_specific_mismatch,
        interpretation=interpretation,
        notes=notes,
        valid_for_casimir_input=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行 d-wave raw-bubble / vertex audit。")
    parser.add_argument("--omega", type=float, default=0.01)
    parser.add_argument("--nk", type=int, default=3)
    parser.add_argument("--delta0", type=float, default=0.04)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--vertex-abs-tolerance", type=float, default=1e-12)
    parser.add_argument("--vertex-rel-tolerance", type=float, default=None)
    args = parser.parse_args(argv)
    report = run_dwave_raw_bubble_vertex_audit(
        nk=args.nk,
        omega_eV=args.omega,
        delta0_eV=args.delta0,
        tolerance=args.tolerance,
        vertex_abs_tolerance=args.vertex_abs_tolerance,
        vertex_rel_tolerance=args.vertex_rel_tolerance,
    )
    print(report.format_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
