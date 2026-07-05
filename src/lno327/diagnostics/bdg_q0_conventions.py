"""Shared q=0 BdG convention diagnostics.

This module centralizes diagnostic-only q=0 comparison logic used by validation
scripts. It does not modify response formulas or promote finite-q results to
Casimir inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from lno327.bdg.finite_q import bdg_finite_q_vertex_from_normal_blocks
from lno327.bdg.nambu import charge_current_vertex_from_model
from lno327.models.lno327_four_orbital.parameters import PairingAmplitudes
from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.response.config import KuboConfig
from lno327.response.local_bdg import (
    bdg_local_eigensystem_from_model,
    bdg_local_superconducting_response_imag_axis,
    bdg_local_total_kernel_imag_axis,
)
from lno327.workflows.finite_q_engine import bdg_finite_q_response_imag_axis

Q0PairingName = Literal["spm", "dwave"]


@dataclass(frozen=True)
class BdGQ0Comparison:
    name: str
    left_name: str
    right_name: str
    absolute_norm_difference: float
    relative_norm_difference: float
    left_norm: float
    right_norm: float
    passes_tolerance: bool

    def to_dict(self) -> dict[str, float | str | bool]:
        return {
            "name": self.name,
            "left_name": self.left_name,
            "right_name": self.right_name,
            "absolute_norm_difference": self.absolute_norm_difference,
            "relative_norm_difference": self.relative_norm_difference,
            "left_norm": self.left_norm,
            "right_norm": self.right_norm,
            "passes_tolerance": self.passes_tolerance,
        }


@dataclass(frozen=True)
class BdGQ0ConventionResult:
    pairing_name: Q0PairingName
    status: str
    local_k_para_total: np.ndarray
    local_k_para_interband: np.ndarray
    local_k_para_intraband: np.ndarray
    local_k_total: np.ndarray
    local_superconducting_response: np.ndarray
    finite_q_raw_bubble_q0: np.ndarray
    finite_q_direct_q0: np.ndarray
    finite_q_total_q0: np.ndarray
    finite_q_minus_schur_q0: np.ndarray
    finite_q_amplitude_phase_schur_q0: np.ndarray
    current_vertex_max_abs: float
    current_vertex_max_rel: float
    current_vertex_status: str
    comparisons: tuple[BdGQ0Comparison, ...]
    interpretation: str
    valid_for_casimir_input: bool = False

    @property
    def comparison_by_name(self) -> dict[str, BdGQ0Comparison]:
        return {comparison.name: comparison for comparison in self.comparisons}

    def matrix_dict(self) -> dict[str, np.ndarray]:
        return {
            "finite_q_raw_bubble_q0": self.finite_q_raw_bubble_q0,
            "finite_q_direct_q0": self.finite_q_direct_q0,
            "finite_q_total_q0": self.finite_q_total_q0,
            "finite_q_minus_schur_q0": self.finite_q_minus_schur_q0,
            "finite_q_amplitude_phase_schur_q0": self.finite_q_amplitude_phase_schur_q0,
            "local_K_para": self.local_k_para_total,
            "local_K_para_total": self.local_k_para_total,
            "local_K_para_interband": self.local_k_para_interband,
            "local_K_para_intraband": self.local_k_para_intraband,
            "local_K_total": self.local_k_total,
            "local_superconducting_response": self.local_superconducting_response,
            "local_K_para_total - finite_q_raw_bubble_q0": self.local_k_para_total
            - self.finite_q_raw_bubble_q0,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "pairing_name": self.pairing_name,
            "status": self.status,
            "current_vertex_max_abs": self.current_vertex_max_abs,
            "current_vertex_max_rel": self.current_vertex_max_rel,
            "current_vertex_status": self.current_vertex_status,
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
            "interpretation": self.interpretation,
            "valid_for_casimir_input": False,
        }


def current_block(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=complex)
    return arr[1:, 1:] if arr.shape == (3, 3) else arr


def relative_norm(diff: float, left: np.ndarray, right: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), 1e-30)
    return float(diff / scale)


def local_k_para_decomposition(
    pairing_name: Q0PairingName,
    points: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    amp: PairingAmplitudes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = LNO327FourOrbitalSpec(pairing_amplitudes=amp)
    omega = float(config.omega_eV + config.eta_eV)
    interband = np.zeros((2, 2), dtype=complex)
    intraband = np.zeros((2, 2), dtype=complex)
    for weight, (kx, ky) in zip(weights, points, strict=True):
        bands = bdg_local_eigensystem_from_model(spec, float(kx), float(ky), pairing_name, config)
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


def _bdg_finite_q_vector_vertex_from_spec(
    spec: LNO327FourOrbitalSpec,
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    direction: str,
) -> np.ndarray:
    particle = spec.peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, direction)
    hole_normal = spec.peierls_hamiltonian_vector_vertex(-kx, -ky, -qx, -qy, direction)
    return bdg_finite_q_vertex_from_normal_blocks(particle, hole_normal)


def _comparison(name: str, left_name: str, left: np.ndarray, right_name: str, right: np.ndarray, tolerance: float) -> BdGQ0Comparison:
    diff = float(np.linalg.norm(left - right))
    return BdGQ0Comparison(
        name=name,
        left_name=left_name,
        right_name=right_name,
        absolute_norm_difference=diff,
        relative_norm_difference=relative_norm(diff, left, right),
        left_norm=float(np.linalg.norm(left)),
        right_norm=float(np.linalg.norm(right)),
        passes_tolerance=bool(relative_norm(diff, left, right) <= tolerance),
    )


def _negligible_match(comparison: BdGQ0Comparison, absolute_tolerance: float) -> bool:
    return (
        comparison.absolute_norm_difference <= absolute_tolerance
        and comparison.left_norm <= absolute_tolerance
        and comparison.right_norm <= absolute_tolerance
    )


def q0_current_vertex_status(
    points: np.ndarray,
    *,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-6,
) -> tuple[float, float, str]:
    spec = LNO327FourOrbitalSpec()
    max_abs = 0.0
    max_rel = 0.0
    for kx, ky in points:
        for direction in ("x", "y"):
            finite_q_vertex = _bdg_finite_q_vector_vertex_from_spec(
                spec,
                float(kx),
                float(ky),
                0.0,
                0.0,
                direction,
            )
            local_vertex = charge_current_vertex_from_model(spec, float(kx), float(ky), direction)
            diff = float(np.linalg.norm(finite_q_vertex - local_vertex))
            rel = relative_norm(diff, finite_q_vertex, local_vertex)
            max_abs = max(max_abs, diff)
            max_rel = max(max_rel, rel)
    status = (
        "vertex_operator_q0_match"
        if max_abs <= absolute_tolerance or max_rel <= relative_tolerance
        else "vertex_operator_level_mismatch"
    )
    return max_abs, max_rel, status


def evaluate_bdg_q0_convention(
    pairing_name: Q0PairingName,
    points: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    amp: PairingAmplitudes,
    *,
    tolerance: float = 1e-6,
    absolute_tolerance: float = 1e-10,
) -> BdGQ0ConventionResult:
    spec = LNO327FourOrbitalSpec(pairing_amplitudes=amp)
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
    local = bdg_local_total_kernel_imag_axis(spec, pairing_name, points, config, weights)
    superconducting = bdg_local_superconducting_response_imag_axis(spec, pairing_name, points, config, weights)
    decomposed_total, interband, intraband = local_k_para_decomposition(pairing_name, points, weights, config, amp)
    finite_raw = current_block(response.bare_bubble)
    finite_direct = current_block(response.direct)
    finite_total = current_block(response.bare_total)
    finite_minus = current_block(response.minus_schur)
    finite_amp_phase = current_block(response.amplitude_phase_schur)
    missing = local.paramagnetic - finite_raw
    comparisons = (
        _comparison("decomposition_identity", "local_K_para_total", local.paramagnetic, "interband + intraband", decomposed_total, tolerance),
        _comparison("raw_vs_total", "finite_q_raw_bubble_q0", finite_raw, "local_K_para_total", local.paramagnetic, tolerance),
        _comparison("raw_vs_interband", "finite_q_raw_bubble_q0", finite_raw, "local_K_para_interband", interband, tolerance),
        _comparison("total_minus_raw_vs_intraband", "local_K_para_total - finite_q_raw_bubble_q0", missing, "local_K_para_intraband", intraband, tolerance),
        _comparison("direct_vs_contact_convention", "finite_q_direct_q0", finite_direct, "-local_K_total - local_K_para_total", -local.total - local.paramagnetic, tolerance),
        _comparison("total_vs_minus_local_total", "finite_q_total_q0", finite_total, "-local_K_total", -local.total, tolerance),
        _comparison("minus_schur_vs_minus_local_total", "finite_q_minus_schur_q0", finite_minus, "-local_K_total", -local.total, tolerance),
        _comparison("amplitude_phase_schur_vs_minus_local_total", "finite_q_amplitude_phase_schur_q0", finite_amp_phase, "-local_K_total", -local.total, tolerance),
    )
    by_name = {comparison.name: comparison for comparison in comparisons}
    current_abs, current_rel, current_status = q0_current_vertex_status(points)

    if pairing_name == "spm":
        raw_total_ok = by_name["raw_vs_total"].passes_tolerance
        raw_interband_ok = by_name["raw_vs_interband"].passes_tolerance
        negligible_intraband = float(np.linalg.norm(intraband)) <= absolute_tolerance
        direct_ok = by_name["direct_vs_contact_convention"].passes_tolerance
        total_ok = by_name["total_vs_minus_local_total"].passes_tolerance
        status = (
            "convention_aware_pass"
            if raw_total_ok and raw_interband_ok and negligible_intraband and direct_ok and total_ok
            else "diagnostic_only_not_passed"
        )
        interpretation = (
            "spm finite_q_raw_bubble_q0 aligns with local_K_para_total and local_K_para_interband; "
            "local_K_para_intraband is negligible."
            if status == "convention_aware_pass"
            else "spm q=0 convention-aware checks did not all pass."
        )
    else:
        raw_interband_ok = by_name["raw_vs_interband"].passes_tolerance
        missing_ok = by_name["total_minus_raw_vs_intraband"].passes_tolerance or (
            by_name["raw_vs_total"].passes_tolerance
            and _negligible_match(by_name["total_minus_raw_vs_intraband"], absolute_tolerance)
        )
        status = (
            "intraband_aware_pass"
            if raw_interband_ok and missing_ok and current_status == "vertex_operator_q0_match"
            else "diagnostic_only_not_passed"
        )
        interpretation = (
            "dwave finite_q_raw_bubble_q0 aligns with local_K_para_interband; raw-vs-total mismatch "
            "is explained by local_K_para_intraband / -f'(E)."
            if status == "intraband_aware_pass" and not by_name["raw_vs_total"].passes_tolerance
            else (
                "dwave finite_q_raw_bubble_q0 aligns with local total and interband because intraband is negligible on this grid."
                if status == "intraband_aware_pass"
                else "dwave q=0 intraband-aware checks did not all pass."
            )
        )

    return BdGQ0ConventionResult(
        pairing_name=pairing_name,
        status=status,
        local_k_para_total=local.paramagnetic,
        local_k_para_interband=interband,
        local_k_para_intraband=intraband,
        local_k_total=local.total,
        local_superconducting_response=superconducting.sigma_like_response,
        finite_q_raw_bubble_q0=finite_raw,
        finite_q_direct_q0=finite_direct,
        finite_q_total_q0=finite_total,
        finite_q_minus_schur_q0=finite_minus,
        finite_q_amplitude_phase_schur_q0=finite_amp_phase,
        current_vertex_max_abs=current_abs,
        current_vertex_max_rel=current_rel,
        current_vertex_status=current_status,
        comparisons=comparisons,
        interpretation=interpretation,
        valid_for_casimir_input=False,
    )
