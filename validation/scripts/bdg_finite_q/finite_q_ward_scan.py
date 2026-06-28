#!/usr/bin/env python3
"""Finite-q Ward residual scans for superconducting BdG diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from lno327.conductivity import KuboConfig, k_weights, uniform_bz_mesh  # noqa: E402
from lno327.finite_q_engine import FiniteQEngineOptions, finite_q_bdg_response_from_ansatz  # noqa: E402
from lno327.pairing import PairingAmplitudes  # noqa: E402
from lno327.pairing_ansatz import build_pairing_ansatz  # noqa: E402
from lno327.ward_validation import validate_physical_ward_identity  # noqa: E402
from q0_bdg_response_alignment import run_q0_bdg_response_alignment  # noqa: E402

WardScanPairingName = Literal["onsite_s", "spm", "dwave"]


@dataclass(frozen=True)
class FiniteQWardScanRow:
    pairing_name: str
    response_name: str
    q_model: tuple[float, float]
    q_norm: float
    left_ward_residual_norm: float
    right_ward_residual_norm: float
    max_ward_residual_norm: float
    residual_ratio_to_bare: float | None
    collective_matrix_condition_number: float | None
    inverse_method: str
    pinv_diagnostic_used: bool
    valid_for_casimir_input: bool = False


@dataclass(frozen=True)
class FiniteQWardScanReport:
    pairing_names: tuple[str, ...]
    omega_eV: float
    nk: int | None
    mesh_size: int
    delta0_eV: float
    rows: tuple[FiniteQWardScanRow, ...]
    q0_alignment_prerequisite: dict[str, str]
    q_scaling_estimates: dict[str, float | None]
    passed: bool
    notes: tuple[str, ...]
    valid_for_casimir_input: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
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
            "q_scaling_estimates": self.q_scaling_estimates,
            "passed": self.passed,
            "notes": list(self.notes),
            "valid_for_casimir_input": False,
        }

    def format_text(self) -> str:
        lines = [
            "有限 q Ward 残差扫描报告",
            f"配对: {', '.join(self.pairing_names)}",
            f"omega_eV: {self.omega_eV:.12g}",
            f"nk: {self.nk if self.nk is not None else '外部网格'}",
            f"网格点数: {self.mesh_size}",
            f"delta0_eV: {self.delta0_eV:.12g}",
            f"q=0 对齐前置结果: {self.q0_alignment_prerequisite}",
            f"通过: {self.passed}",
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


def _scaling_slope(q_values: list[float], residuals: list[float]) -> float | None:
    positive = [(q, r) for q, r in zip(q_values, residuals, strict=True) if q > 0.0 and r > 0.0]
    if len(positive) < 3:
        return None
    q_first, r_first = positive[0]
    q_last, r_last = positive[-1]
    if q_first == q_last:
        return None
    return float((np.log(r_last) - np.log(r_first)) / (np.log(q_last) - np.log(q_first)))


def _q0_alignment_status(pairing_name: str, passed: bool, notes: tuple[str, ...]) -> str:
    if pairing_name == "dwave" and passed and any("intraband-aware" in note for note in notes):
        return "intraband_aware_pass"
    if pairing_name == "spm" and passed:
        return "convention_aware_pass"
    if passed:
        return "pass"
    return "diagnostic_only_not_passed"


def run_finite_q_ward_scan(
    pairing_names: tuple[WardScanPairingName, ...] = ("onsite_s", "spm", "dwave"),
    *,
    omega_eV: float = 0.01,
    q_values: tuple[float, ...] = (0.005, 0.01, 0.02),
    q_directions: tuple[tuple[float, float], ...] = ((1.0, 0.0),),
    nk: int = 3,
    k_points: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    config: KuboConfig | None = None,
    pairing_params: PairingAmplitudes | None = None,
    tolerance: float = 1e-8,
) -> FiniteQWardScanReport:
    points = uniform_bz_mesh(nk) if k_points is None else np.asarray(k_points, dtype=float)
    mesh_weights = k_weights(points) if weights is None else np.asarray(weights, dtype=float)
    kubo = config or KuboConfig.from_kelvin(omega_eV=omega_eV, temperature_K=10.0, eta_eV=1e-8, output_si=False)
    amp = pairing_params or PairingAmplitudes()
    options = FiniteQEngineOptions(
        current_vertex="peierls",
        collective_mode="amplitude_phase",
        collective_counterterm="goldstone_gap_equation",
        include_phase_phase_direct=True,
    )
    rows: list[FiniteQWardScanRow] = []
    scaling_inputs: dict[str, tuple[list[float], list[float]]] = {}
    q0_alignment = {}
    for pairing_name in pairing_names:
        alignment_report = run_q0_bdg_response_alignment(
            pairing_name,
            omega_eV=float(kubo.omega_eV),
            nk=nk,
            k_points=points if k_points is not None else None,
            weights=mesh_weights if weights is not None else None,
            config=kubo,
            pairing_params=amp,
        )
        q0_alignment[pairing_name] = _q0_alignment_status(
            pairing_name,
            alignment_report.passed,
            alignment_report.pass_fail_notes,
        )
    for pairing_name in pairing_names:
        ansatz = build_pairing_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
        for q_value in q_values:
            for direction in q_directions:
                direction_array = np.asarray(direction, dtype=float)
                norm = float(np.linalg.norm(direction_array))
                if norm <= 0.0:
                    raise ValueError("q direction must be nonzero")
                q = float(q_value) * direction_array / norm
                response = finite_q_bdg_response_from_ansatz(
                    ansatz,
                    float(kubo.omega_eV),
                    q,
                    points,
                    mesh_weights,
                    kubo,
                    amp,
                    options,
                )
                matrices = {
                    "bare_total": response.bare_total,
                    "minus_schur": response.minus_schur,
                    "amplitude_phase_schur": response.amplitude_phase_schur,
                }
                bare_report = validate_physical_ward_identity(
                    response.bare_total,
                    kubo.omega_eV,
                    q,
                    tolerance=tolerance,
                )
                bare_max = float(max(bare_report.left_norm, bare_report.right_norm))
                for response_name, matrix in matrices.items():
                    ward = validate_physical_ward_identity(matrix, kubo.omega_eV, q, tolerance=tolerance)
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
    slopes = {
        key: _scaling_slope(value[0], value[1])
        for key, value in scaling_inputs.items()
    }
    finite = all(np.isfinite(row.max_ward_residual_norm) for row in rows)
    notes = (
        "本扫描在同一入口先记录 q=0 response definition alignment 前置结果。",
        "dwave 若显示 intraband_aware_pass，表示 q=0 raw bubble 对齐 local interband，raw-vs-total 差异由 intraband/-f'(E) local 项解释。",
        "本扫描只记录 Ward 残差，不解释为 Casimir 各向异性。",
        "finite-q 输出保持 valid_for_casimir_input=False。",
        "残差比例为各响应 max residual 相对 bare_total 的比例。",
    )
    return FiniteQWardScanReport(
        pairing_names=tuple(pairing_names),
        omega_eV=float(kubo.omega_eV),
        nk=nk if k_points is None else None,
        mesh_size=int(points.shape[0]),
        delta0_eV=float(amp.delta0_eV),
        rows=tuple(rows),
        q0_alignment_prerequisite=q0_alignment,
        q_scaling_estimates=slopes,
        passed=bool(finite),
        notes=notes,
        valid_for_casimir_input=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行有限 q Ward 残差扫描诊断。")
    parser.add_argument("--pairings", nargs="+", choices=("onsite_s", "spm", "dwave"), default=["onsite_s", "spm", "dwave"])
    parser.add_argument("--omega", type=float, default=0.01)
    parser.add_argument("--q-values", nargs="+", type=float, default=[0.005, 0.01, 0.02])
    parser.add_argument("--nk", type=int, default=3)
    parser.add_argument("--delta0", type=float, default=0.04)
    args = parser.parse_args(argv)
    report = run_finite_q_ward_scan(
        tuple(args.pairings),
        omega_eV=args.omega,
        q_values=tuple(args.q_values),
        nk=args.nk,
        pairing_params=PairingAmplitudes(delta0_eV=args.delta0),
    )
    print(report.format_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
