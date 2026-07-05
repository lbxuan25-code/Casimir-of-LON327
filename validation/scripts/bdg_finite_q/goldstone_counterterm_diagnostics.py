#!/usr/bin/env python3
"""Goldstone counterterm and eta2-normalization diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Literal
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from lno327 import KuboConfig, k_weights, uniform_bz_mesh  # noqa: E402
from lno327.workflows.finite_q_engine import FiniteQEngineOptions, bdg_finite_q_response_imag_axis_from_workspace  # noqa: E402
from lno327.response.finite_q_bdg import precompute_finite_q_bdg_workspace_from_model_ansatz  # noqa: E402
from validation.lib.finite_q_validation_models import available_finite_q_validation_models, get_finite_q_validation_model  # noqa: E402

GoldstonePairingName = Literal["onsite_s", "spm", "dwave"]


@dataclass(frozen=True)
class GoldstoneCountertermRow:
    pairing_name: str
    eta2_kernel_after_counterterm: complex
    eta2_kernel_abs: float
    goldstone_condition_passed: bool
    eta2_normalization_status: str
    counterterm_only_collective_kernel: bool
    collective_matrix_condition_number: float | None
    inverse_method: str
    valid_for_casimir_input: bool = False


@dataclass(frozen=True)
class GoldstoneCountertermReport:
    omega_eV: float
    q_model: tuple[float, float]
    nk: int | None
    mesh_size: int
    delta0_eV: float
    rows: tuple[GoldstoneCountertermRow, ...]
    passed: bool
    notes: tuple[str, ...]
    valid_for_casimir_input: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "omega_eV": self.omega_eV,
            "q_model": list(self.q_model),
            "nk": self.nk,
            "mesh_size": self.mesh_size,
            "delta0_eV": self.delta0_eV,
            "rows": [
                {
                    **row.__dict__,
                    "eta2_kernel_after_counterterm": [
                        float(np.real(row.eta2_kernel_after_counterterm)),
                        float(np.imag(row.eta2_kernel_after_counterterm)),
                    ],
                    "valid_for_casimir_input": False,
                }
                for row in self.rows
            ],
            "passed": self.passed,
            "notes": list(self.notes),
            "valid_for_casimir_input": False,
        }

    def format_text(self) -> str:
        lines = [
            "Goldstone counterterm 与 eta2 归一化诊断报告",
            f"omega_eV: {self.omega_eV:.12g}",
            f"q_model: [{self.q_model[0]:.12g}, {self.q_model[1]:.12g}]",
            f"nk: {self.nk if self.nk is not None else '外部网格'}",
            f"网格点数: {self.mesh_size}",
            f"delta0_eV: {self.delta0_eV:.12g}",
        ]
        for row in self.rows:
            lines.append(
                f"- {row.pairing_name}: |K_eta2_eta2|={row.eta2_kernel_abs:.6e}, "
                f"通过={row.goldstone_condition_passed}, inverse={row.inverse_method}"
            )
        lines.append(f"总体通过: {self.passed}")
        lines.append("valid_for_casimir_input: False")
        return "\n".join(lines)


def run_goldstone_counterterm_diagnostics(
    pairing_names: tuple[GoldstonePairingName, ...] | None = None,
    *,
    model_name: str = "lno327_four_orbital",
    nk: int = 3,
    k_points: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    pairing_params=None,
    tolerance: float = 1e-8,
) -> GoldstoneCountertermReport:
    points = uniform_bz_mesh(nk) if k_points is None else np.asarray(k_points, dtype=float)
    mesh_weights = k_weights(points) if weights is None else np.asarray(weights, dtype=float)
    config = KuboConfig.from_kelvin(omega_eV=0.0, temperature_K=10.0, eta_eV=1e-8, output_si=False)
    model = get_finite_q_validation_model(model_name)
    selected_pairings = tuple(model.default_pairings if pairing_names is None else pairing_names)
    for pairing_name in selected_pairings:
        model.require_pairing(pairing_name)
    amp = pairing_params or model.build_pairing_params()
    options = FiniteQEngineOptions(
        current_vertex="peierls",
        collective_mode="amplitude_phase",
        collective_counterterm="goldstone_gap_equation",
        include_phase_phase_direct=True,
    )
    rows: list[GoldstoneCountertermRow] = []
    for pairing_name in selected_pairings:
        ansatz = model.build_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            workspace = precompute_finite_q_bdg_workspace_from_model_ansatz(
                model.spec,
                ansatz,
                np.array([0.0, 0.0]),
                points,
                mesh_weights,
                config,
                amp,
                options,
            )
            response = bdg_finite_q_response_imag_axis_from_workspace(workspace, config=config)
        eta2_kernel = complex(response.collective_total[1, 1])
        eta2_abs = float(abs(eta2_kernel))
        normalization_status = str(response.metadata.get("eta2_phase_relation", "missing_eta2_metadata"))
        rows.append(
            GoldstoneCountertermRow(
                pairing_name=pairing_name,
                eta2_kernel_after_counterterm=eta2_kernel,
                eta2_kernel_abs=eta2_abs,
                goldstone_condition_passed=eta2_abs <= tolerance,
                eta2_normalization_status=normalization_status,
                counterterm_only_collective_kernel=response.collective_counterterm.shape == (2, 2)
                and response.bare_total.shape == (3, 3),
                collective_matrix_condition_number=response.metadata.get("collective_total_condition_number"),
                inverse_method=str(response.metadata.get("collective_inverse_method", "not_used")),
                valid_for_casimir_input=False,
            )
        )
    passed = all(
        row.goldstone_condition_passed
        and row.eta2_normalization_status == "eta2 = delta0 * theta"
        and row.counterterm_only_collective_kernel
        for row in rows
    )
    notes = (
        "本诊断只检查 counterterm 与 eta2 归一化，不重写 counterterm 物理。",
        "counterterm 应只进入 collective kernel，不混入 q=0 local response 对照。",
        "finite-q 输出保持 valid_for_casimir_input=False。",
    )
    return GoldstoneCountertermReport(
        omega_eV=0.0,
        q_model=(0.0, 0.0),
        nk=nk if k_points is None else None,
        mesh_size=int(points.shape[0]),
        delta0_eV=float(amp.delta0_eV),
        rows=tuple(rows),
        passed=bool(passed),
        notes=notes,
        valid_for_casimir_input=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行 Goldstone counterterm 与 eta2 归一化诊断。")
    parser.add_argument("--model", choices=available_finite_q_validation_models(), default="lno327_four_orbital")
    parser.add_argument("--pairings", nargs="+")
    parser.add_argument("--nk", type=int, default=3)
    parser.add_argument("--delta0", type=float)
    args = parser.parse_args(argv)
    model = get_finite_q_validation_model(args.model)
    report = run_goldstone_counterterm_diagnostics(
        tuple(args.pairings) if args.pairings else None,
        model_name=model.name,
        nk=args.nk,
        pairing_params=model.build_pairing_params(args.delta0),
    )
    print(report.format_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
