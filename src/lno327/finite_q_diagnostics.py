"""Finite-q Ward-closure diagnostic workflow.

This module is diagnostic-only. It evaluates finite-q response components from
an explicit ``PairingAnsatz`` and validates Ward residuals without promoting the
result to Casimir input.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .conductivity import KuboConfig, k_weights, uniform_bz_mesh
from .finite_q_engine import FiniteQEngineOptions, finite_q_bdg_response_from_ansatz
from .pairing import PairingAmplitudes
from .pairing_ansatz import PairingAnsatzName, build_pairing_ansatz
from .ward_validation import WardValidationReport, validate_physical_ward_identity


DIAGNOSTIC_PHASE_VERTEX = "bond_endpoint_gauge"
DIAGNOSTIC_CURRENT_VERTEX = "peierls"
DIAGNOSTIC_COLLECTIVE_MODE = "amplitude_phase"
DIAGNOSTIC_COLLECTIVE_COUNTERTERM = "goldstone_gap_equation"
DIAGNOSTIC_INCLUDE_PHASE_PHASE_DIRECT = True


@dataclass(frozen=True)
class FiniteQDiagnosticReport:
    pairing_name: str
    phase_vertex: str
    omega_eV: float
    q_model: tuple[float, float]
    nk: int | None
    mesh_size: int
    delta0_eV: float
    current_vertex: str
    collective_mode: str
    collective_counterterm: str
    bare_ward_residual_norm: float
    minus_schur_ward_residual_norm: float
    amplitude_phase_schur_ward_residual_norm: float
    collective_matrix_condition_number: float | None
    inverse_method: str
    selected_response_name: str
    valid_for_casimir_input: bool
    ward_passed: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["q_model"] = list(self.q_model)
        return data

    def format_text(self) -> str:
        lines = [
            "Finite-q diagnostic report",
            f"pairing_name: {self.pairing_name}",
            f"phase_vertex: {self.phase_vertex}",
            f"omega_eV: {self.omega_eV:.12g}",
            f"q_model: [{self.q_model[0]:.12g}, {self.q_model[1]:.12g}]",
            f"nk: {self.nk if self.nk is not None else 'provided_mesh'}",
            f"mesh_size: {self.mesh_size}",
            f"delta0_eV: {self.delta0_eV:.12g}",
            f"current_vertex: {self.current_vertex}",
            f"collective_mode: {self.collective_mode}",
            f"collective_counterterm: {self.collective_counterterm}",
            f"bare_ward_residual_norm: {self.bare_ward_residual_norm:.12e}",
            f"minus_schur_ward_residual_norm: {self.minus_schur_ward_residual_norm:.12e}",
            (
                "amplitude_phase_schur_ward_residual_norm: "
                f"{self.amplitude_phase_schur_ward_residual_norm:.12e}"
            ),
            f"collective_matrix_condition_number: {self.collective_matrix_condition_number}",
            f"inverse_method: {self.inverse_method}",
            f"selected_response_name: {self.selected_response_name}",
            f"valid_for_casimir_input: {self.valid_for_casimir_input}",
        ]
        return "\n".join(lines)


def _residual_norm(report: WardValidationReport) -> float:
    return float(max(report.left_norm, report.right_norm))


def run_finite_q_diagnostic(
    pairing_name: PairingAnsatzName,
    *,
    omega_eV: float = 0.01,
    q_model: np.ndarray | tuple[float, float] = (0.01, 0.0),
    nk: int = 3,
    k_points: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    config: KuboConfig | None = None,
    pairing_params: PairingAmplitudes | None = None,
    tolerance: float = 1e-8,
) -> FiniteQDiagnosticReport:
    """Run the explicit diagnostic finite-q Ward workflow for one ansatz."""

    points = uniform_bz_mesh(nk) if k_points is None else np.asarray(k_points, dtype=float)
    mesh_weights = k_weights(points) if weights is None else np.asarray(weights, dtype=float)
    kubo = config or KuboConfig.from_kelvin(
        omega_eV=float(omega_eV),
        temperature_K=10.0,
        eta_eV=1e-8,
        output_si=False,
    )
    amp = pairing_params or PairingAmplitudes()
    q = np.asarray(q_model, dtype=float)
    ansatz = build_pairing_ansatz(pairing_name, phase_vertex=DIAGNOSTIC_PHASE_VERTEX)
    options = FiniteQEngineOptions(
        current_vertex=DIAGNOSTIC_CURRENT_VERTEX,
        collective_mode=DIAGNOSTIC_COLLECTIVE_MODE,
        collective_counterterm=DIAGNOSTIC_COLLECTIVE_COUNTERTERM,
        include_phase_phase_direct=DIAGNOSTIC_INCLUDE_PHASE_PHASE_DIRECT,
    )
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
    ward_bare = validate_physical_ward_identity(response.bare_total, kubo.omega_eV, q, tolerance=tolerance)
    ward_minus = validate_physical_ward_identity(response.minus_schur, kubo.omega_eV, q, tolerance=tolerance)
    ward_amp_phase = validate_physical_ward_identity(
        response.amplitude_phase_schur,
        kubo.omega_eV,
        q,
        tolerance=tolerance,
    )
    valid_for_casimir = bool(response.metadata.get("valid_for_casimir_input", False))
    if valid_for_casimir:
        raise RuntimeError("finite-q diagnostic response must not be marked valid_for_casimir_input=True")
    inferred_nk = nk if k_points is None else None
    return FiniteQDiagnosticReport(
        pairing_name=ansatz.name,
        phase_vertex=ansatz.phase_vertex,
        omega_eV=float(kubo.omega_eV),
        q_model=(float(q[0]), float(q[1])),
        nk=inferred_nk,
        mesh_size=int(points.shape[0]),
        delta0_eV=float(amp.delta0_eV),
        current_vertex=DIAGNOSTIC_CURRENT_VERTEX,
        collective_mode=str(response.metadata.get("collective_mode", DIAGNOSTIC_COLLECTIVE_MODE)),
        collective_counterterm=DIAGNOSTIC_COLLECTIVE_COUNTERTERM,
        bare_ward_residual_norm=_residual_norm(ward_bare),
        minus_schur_ward_residual_norm=_residual_norm(ward_minus),
        amplitude_phase_schur_ward_residual_norm=_residual_norm(ward_amp_phase),
        collective_matrix_condition_number=response.metadata.get("collective_total_condition_number"),
        inverse_method=str(response.metadata.get("collective_inverse_method", "not_used")),
        selected_response_name=str(response.metadata.get("gauge_restored_selected", "bare_total")),
        valid_for_casimir_input=False,
        ward_passed={
            "bare_total": ward_bare.passed,
            "minus_schur": ward_minus.passed,
            "amplitude_phase_schur": ward_amp_phase.passed,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run finite-q Ward-closure diagnostics.")
    parser.add_argument("pairing", choices=("onsite_s", "spm", "dwave"))
    parser.add_argument("--omega", type=float, default=0.01)
    parser.add_argument("--qx", type=float, default=0.01)
    parser.add_argument("--qy", type=float, default=0.0)
    parser.add_argument("--nk", type=int, default=3)
    parser.add_argument("--delta0", type=float, default=0.04)
    args = parser.parse_args(argv)
    report = run_finite_q_diagnostic(
        args.pairing,
        omega_eV=args.omega,
        q_model=(args.qx, args.qy),
        nk=args.nk,
        pairing_params=PairingAmplitudes(delta0_eV=args.delta0),
    )
    print(report.format_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
