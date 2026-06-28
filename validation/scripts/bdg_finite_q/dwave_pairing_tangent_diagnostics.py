#!/usr/bin/env python3
"""dwave pairing representation and endpoint-gauge tangent diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from lno327.finite_q_primitives import phase_vertex  # noqa: E402
from lno327.pairing import PairingAmplitudes, dwave_pairing_matrix  # noqa: E402
from lno327.pairing import build_pairing_ansatz  # noqa: E402
from lno327.pairing import bond_endpoint_gauge_form_factor, pairing_from_bonds  # noqa: E402


@dataclass(frozen=True)
class DWavePairingTangentReport:
    k_points: tuple[tuple[float, float], ...]
    reconstruction_errors: tuple[float, ...]
    max_reconstruction_error: float
    q0_tangent_errors: tuple[float, ...]
    max_q0_tangent_error: float
    separable_tangent_errors: tuple[float, ...]
    max_separable_tangent_error: float
    reconstruction_passed: bool
    q0_tangent_passed: bool
    separable_tangent_passed: bool
    passed: bool
    notes: tuple[str, ...]
    valid_for_casimir_input: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_points": [list(item) for item in self.k_points],
            "reconstruction_errors": list(self.reconstruction_errors),
            "max_reconstruction_error": self.max_reconstruction_error,
            "q0_tangent_errors": list(self.q0_tangent_errors),
            "max_q0_tangent_error": self.max_q0_tangent_error,
            "separable_tangent_errors": list(self.separable_tangent_errors),
            "max_separable_tangent_error": self.max_separable_tangent_error,
            "reconstruction_passed": self.reconstruction_passed,
            "q0_tangent_passed": self.q0_tangent_passed,
            "separable_tangent_passed": self.separable_tangent_passed,
            "passed": self.passed,
            "notes": list(self.notes),
            "valid_for_casimir_input": False,
        }

    def format_text(self) -> str:
        return "\n".join(
            [
                "dwave 表示与 tangent 诊断报告",
                f"最大重构误差: {self.max_reconstruction_error:.12e}",
                f"最大 q=0 tangent 误差: {self.max_q0_tangent_error:.12e}",
                f"最大 separable tangent 误差: {self.max_separable_tangent_error:.12e}",
                f"通过: {self.passed}",
                "valid_for_casimir_input: False",
            ]
        )


def run_dwave_pairing_tangent_diagnostics(
    *,
    k_points: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.2, -0.1), (1.1, 0.7), (np.pi / 2.0, np.pi / 3.0)),
    q_model: tuple[float, float] = (0.01, 0.0),
    pairing_params: PairingAmplitudes | None = None,
    tolerance: float = 1e-10,
) -> DWavePairingTangentReport:
    amp = pairing_params or PairingAmplitudes()
    ansatz = build_pairing_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    reconstruction_errors: list[float] = []
    q0_tangent_errors: list[float] = []
    separable_tangent_errors: list[float] = []
    for kx, ky in k_points:
        direct = dwave_pairing_matrix(float(kx), float(ky), amp)
        reconstructed = pairing_from_bonds("dwave", float(kx), float(ky), amp)
        reconstruction_errors.append(float(np.linalg.norm(direct - reconstructed)))

        q0_form = bond_endpoint_gauge_form_factor("dwave", float(kx), float(ky), 0.0, 0.0, amp)
        q0_delta = float(amp.delta0_eV) * q0_form
        q0_tangent_errors.append(float(np.linalg.norm(q0_delta - direct)))

        endpoint_delta = ansatz.phase_pairing_matrix(
            float(kx),
            float(ky),
            float(q_model[0]),
            float(q_model[1]),
            amp,
        )
        separable_delta = float(amp.delta0_eV) * ansatz.collective_form_factor(
            float(kx),
            float(ky),
            float(q_model[0]),
            float(q_model[1]),
            amp,
        )
        separable_tangent_errors.append(
            float(np.linalg.norm(phase_vertex(endpoint_delta) - phase_vertex(separable_delta)))
        )
    max_reconstruction = float(max(reconstruction_errors))
    max_q0_tangent = float(max(q0_tangent_errors))
    max_separable = float(max(separable_tangent_errors))
    reconstruction_passed = max_reconstruction <= tolerance
    q0_tangent_passed = max_q0_tangent <= tolerance
    separable_passed = max_separable <= tolerance
    notes = (
        "本诊断只检查当前代码表示的一致性，不改变 dwave 静态 ansatz。",
        "q=0 endpoint-gauge tangent 应回到全局相位 tangent。",
        "massive 内部 pairing-shape 模式若出现只作为诊断线索，不是新增 Goldstone 模式。",
    )
    return DWavePairingTangentReport(
        k_points=tuple((float(kx), float(ky)) for kx, ky in k_points),
        reconstruction_errors=tuple(reconstruction_errors),
        max_reconstruction_error=max_reconstruction,
        q0_tangent_errors=tuple(q0_tangent_errors),
        max_q0_tangent_error=max_q0_tangent,
        separable_tangent_errors=tuple(separable_tangent_errors),
        max_separable_tangent_error=max_separable,
        reconstruction_passed=bool(reconstruction_passed),
        q0_tangent_passed=bool(q0_tangent_passed),
        separable_tangent_passed=bool(separable_passed),
        passed=bool(reconstruction_passed and q0_tangent_passed and separable_passed),
        notes=notes,
        valid_for_casimir_input=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行 dwave pairing 重构与 tangent 诊断。")
    parser.add_argument("--delta0", type=float, default=0.04)
    parser.add_argument("--qx", type=float, default=0.01)
    parser.add_argument("--qy", type=float, default=0.0)
    args = parser.parse_args(argv)
    report = run_dwave_pairing_tangent_diagnostics(
        q_model=(args.qx, args.qy),
        pairing_params=PairingAmplitudes(delta0_eV=args.delta0),
    )
    print(report.format_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
