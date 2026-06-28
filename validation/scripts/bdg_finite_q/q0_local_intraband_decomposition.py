#!/usr/bin/env python3
"""q=0 local BdG K_para interband/intraband decomposition diagnostic."""

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
from lno327.bdg_response import bdg_eigensystem, bdg_total_kernel_imag_axis  # noqa: E402
from lno327.conductivity import KuboConfig, k_weights, uniform_bz_mesh  # noqa: E402
from lno327.pairing import PairingAmplitudes, pairing_matrix  # noqa: E402

PairingName = Literal["spm", "dwave"]


@dataclass(frozen=True)
class Q0IntrabandDecompositionRow:
    pairing_name: PairingName
    local_k_para_total: np.ndarray
    local_k_para_interband: np.ndarray
    local_k_para_intraband: np.ndarray
    finite_q_raw_bubble_q0_current_block: np.ndarray
    total_norm: float
    interband_norm: float
    intraband_norm: float
    finite_q_raw_norm: float
    decomposition_abs: float
    decomposition_rel: float
    raw_vs_interband_abs: float
    raw_vs_interband_rel: float
    missing_vs_intraband_abs: float
    missing_vs_intraband_rel: float
    raw_vs_total_abs: float
    raw_vs_total_rel: float
    hypothesis_supported: bool
    interpretation: str
    valid_for_casimir_input: bool = False

    def to_dict(self) -> dict[str, float | str | bool]:
        return {
            "pairing_name": self.pairing_name,
            "total_norm": self.total_norm,
            "interband_norm": self.interband_norm,
            "intraband_norm": self.intraband_norm,
            "finite_q_raw_norm": self.finite_q_raw_norm,
            "decomposition_abs": self.decomposition_abs,
            "decomposition_rel": self.decomposition_rel,
            "raw_vs_interband_abs": self.raw_vs_interband_abs,
            "raw_vs_interband_rel": self.raw_vs_interband_rel,
            "missing_vs_intraband_abs": self.missing_vs_intraband_abs,
            "missing_vs_intraband_rel": self.missing_vs_intraband_rel,
            "raw_vs_total_abs": self.raw_vs_total_abs,
            "raw_vs_total_rel": self.raw_vs_total_rel,
            "hypothesis_supported": self.hypothesis_supported,
            "interpretation": self.interpretation,
            "valid_for_casimir_input": False,
        }


@dataclass(frozen=True)
class Q0LocalIntrabandDecompositionReport:
    omega_eV: float
    q_model: tuple[float, float]
    nk: int
    mesh_size: int
    delta0_eV: float
    rows: tuple[Q0IntrabandDecompositionRow, ...]
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
            "notes": list(self.notes),
            "valid_for_casimir_input": False,
        }

    def format_text(self) -> str:
        lines = [
            "q=0 local K_para interband/intraband decomposition diagnostic",
            f"omega_eV: {self.omega_eV:.12g}",
            f"q_model: [{self.q_model[0]:.12g}, {self.q_model[1]:.12g}]",
            f"nk: {self.nk}",
            f"网格点数: {self.mesh_size}",
            f"delta0_eV: {self.delta0_eV:.12g}",
            (
                "pairing | hypothesis_supported | total_norm | interband_norm | intraband_norm | "
                "finite_q_raw_norm | raw_vs_interband_rel | missing_vs_intraband_rel | raw_vs_total_rel | "
                "interpretation"
            ),
        ]
        for row in self.rows:
            lines.append(
                f"{row.pairing_name} | {row.hypothesis_supported} | {row.total_norm:.6e} | "
                f"{row.interband_norm:.6e} | {row.intraband_norm:.6e} | {row.finite_q_raw_norm:.6e} | "
                f"{row.raw_vs_interband_rel:.6e} | {row.missing_vs_intraband_rel:.6e} | "
                f"{row.raw_vs_total_rel:.6e} | {row.interpretation}"
            )
        lines.append("说明:")
        lines.extend(f"- {note}" for note in self.notes)
        lines.append("valid_for_casimir_input: False")
        return "\n".join(lines)


def _relative_norm(diff: float, left: np.ndarray, right: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), 1e-30)
    return float(diff / scale)


def _current_block(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=complex)
    return arr[1:, 1:] if arr.shape == (3, 3) else arr


def _local_k_para_decomposition(
    pairing_name: PairingName,
    points: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    amp: PairingAmplitudes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    omega = float(config.omega_eV + config.eta_eV)
    interband = np.zeros((2, 2), dtype=complex)
    intraband = np.zeros((2, 2), dtype=complex)
    for weight, (kx, ky) in zip(weights, points, strict=True):
        delta = pairing_matrix(pairing_name, float(kx), float(ky), amp)
        bands = bdg_eigensystem(float(kx), float(ky), delta, config)
        currents = (bands.current_x_band, bands.current_y_band)
        for m, energy_m in enumerate(bands.energies_eV):
            for n, energy_n in enumerate(bands.energies_eV):
                if m == n:
                    response_factor = bands.negative_fermi_derivative[m]
                    target = intraband
                else:
                    occupation_diff = bands.occupations[m] - bands.occupations[n]
                    if np.isclose(occupation_diff, 0.0):
                        continue
                    energy_diff = float(energy_m - energy_n)
                    if abs(energy_diff) < config.eta_eV:
                        continue
                    response_factor = -occupation_diff * energy_diff / (energy_diff**2 + omega**2)
                    target = interband
                for alpha in range(2):
                    for beta in range(2):
                        target[alpha, beta] += (
                            float(weight)
                            * response_factor
                            * currents[alpha][m, n]
                            * currents[beta][n, m]
                        )
    interband *= 0.5
    intraband *= 0.5
    return interband + intraband, interband, intraband


def _finite_q_raw_current_block(
    pairing_name: PairingName,
    points: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    amp: PairingAmplitudes,
) -> np.ndarray:
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
    return _current_block(response.bare_bubble)


def _row(
    pairing_name: PairingName,
    points: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    amp: PairingAmplitudes,
    tolerance: float,
) -> Q0IntrabandDecompositionRow:
    decomposed_total, interband, intraband = _local_k_para_decomposition(pairing_name, points, weights, config, amp)
    local_total = bdg_total_kernel_imag_axis(points, config, pairing_name, amp, weights).paramagnetic
    finite_q_raw = _finite_q_raw_current_block(pairing_name, points, weights, config, amp)
    missing = local_total - finite_q_raw

    decomposition_abs = float(np.linalg.norm(local_total - decomposed_total))
    decomposition_rel = _relative_norm(decomposition_abs, local_total, decomposed_total)
    raw_vs_interband_abs = float(np.linalg.norm(finite_q_raw - interband))
    raw_vs_interband_rel = _relative_norm(raw_vs_interband_abs, finite_q_raw, interband)
    missing_vs_intraband_abs = float(np.linalg.norm(missing - intraband))
    missing_vs_intraband_rel = _relative_norm(missing_vs_intraband_abs, missing, intraband)
    raw_vs_total_abs = float(np.linalg.norm(finite_q_raw - local_total))
    raw_vs_total_rel = _relative_norm(raw_vs_total_abs, finite_q_raw, local_total)
    absolute_tolerance = 1e-10
    negligible_intraband_case = bool(
        raw_vs_total_rel <= tolerance
        and raw_vs_interband_rel <= tolerance
        and float(np.linalg.norm(intraband)) <= absolute_tolerance
        and raw_vs_total_abs <= absolute_tolerance
    )
    missing_intraband_case = bool(raw_vs_interband_rel <= tolerance and missing_vs_intraband_rel <= tolerance)
    hypothesis_supported = bool(decomposition_rel <= tolerance and (missing_intraband_case or negligible_intraband_case))
    if hypothesis_supported and raw_vs_total_rel > tolerance:
        interpretation = "q0_raw_bubble_mismatch_consistent_with_missing_local_intraband_contribution"
    elif hypothesis_supported:
        interpretation = "q0_raw_bubble_matches_total_and_interband_intraband_is_negligible"
    else:
        interpretation = (
            "intraband_hypothesis_not_supported_check_real_imag_sign_nambu_band_pair_or_pairing_state_convention"
        )
    return Q0IntrabandDecompositionRow(
        pairing_name=pairing_name,
        local_k_para_total=local_total,
        local_k_para_interband=interband,
        local_k_para_intraband=intraband,
        finite_q_raw_bubble_q0_current_block=finite_q_raw,
        total_norm=float(np.linalg.norm(local_total)),
        interband_norm=float(np.linalg.norm(interband)),
        intraband_norm=float(np.linalg.norm(intraband)),
        finite_q_raw_norm=float(np.linalg.norm(finite_q_raw)),
        decomposition_abs=decomposition_abs,
        decomposition_rel=decomposition_rel,
        raw_vs_interband_abs=raw_vs_interband_abs,
        raw_vs_interband_rel=raw_vs_interband_rel,
        missing_vs_intraband_abs=missing_vs_intraband_abs,
        missing_vs_intraband_rel=missing_vs_intraband_rel,
        raw_vs_total_abs=raw_vs_total_abs,
        raw_vs_total_rel=raw_vs_total_rel,
        hypothesis_supported=hypothesis_supported,
        interpretation=interpretation,
        valid_for_casimir_input=False,
    )


def run_q0_local_intraband_decomposition(
    *,
    nk: int = 3,
    omega_eV: float = 0.01,
    delta0_eV: float = 0.04,
    tolerance: float = 1e-6,
) -> Q0LocalIntrabandDecompositionReport:
    points = uniform_bz_mesh(nk)
    weights = k_weights(points)
    config = KuboConfig.from_kelvin(omega_eV=omega_eV, temperature_K=10.0, eta_eV=1e-8, output_si=False)
    amp = PairingAmplitudes(delta0_eV=delta0_eV)
    rows = tuple(_row(pairing_name, points, weights, config, amp, tolerance) for pairing_name in ("spm", "dwave"))
    notes = (
        "该诊断只拆分 local BdG K_para 的 interband 与 intraband/-f'(E) 项，不修改任何响应公式。",
        "finite-q raw bubble 只作为 q=0 diagnostic 对照；本报告不是 Ward closure proof。",
        "若 d-wave raw bubble 对齐 interband 且 local_total - raw 对齐 intraband，则 q=0 mismatch 支持 intraband 解释。",
        "本报告不作为 Casimir 输入，也不改变任何 gating 状态。",
    )
    return Q0LocalIntrabandDecompositionReport(
        omega_eV=float(config.omega_eV),
        q_model=(0.0, 0.0),
        nk=nk,
        mesh_size=int(points.shape[0]),
        delta0_eV=float(amp.delta0_eV),
        rows=rows,
        notes=notes,
        valid_for_casimir_input=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行 q=0 local K_para interband/intraband decomposition 诊断。")
    parser.add_argument("--omega", type=float, default=0.01)
    parser.add_argument("--nk", type=int, default=3)
    parser.add_argument("--delta0", type=float, default=0.04)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args(argv)
    report = run_q0_local_intraband_decomposition(
        nk=args.nk,
        omega_eV=args.omega,
        delta0_eV=args.delta0,
        tolerance=args.tolerance,
    )
    print(report.format_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
