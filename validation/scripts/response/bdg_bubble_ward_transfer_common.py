"""Shared helpers for the StageSC-0b BdG bubble Ward transfer audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from lno327.bdg_finite_q_response import (  # noqa: E402
    _amplitude_vertex,
    _eta2_phase_vertex,
    bdg_finite_q_response_imag_axis,
    bdg_finite_q_vector_vertex,
    collective_form_factor,
)
from lno327.conductivity import KuboConfig, fermi_function  # noqa: E402
from lno327.pairing import PairingAmplitudes, bdg_hamiltonian, pairing_matrix  # noqa: E402


PAIRINGS = ("onsite_s", "spm", "dwave")
K_POINTS = (
    (0.13, 0.27),
    (0.41, -0.22),
    (1.11, 0.73),
    (-0.64, 1.37),
)
Q_MODEL_LIST = ((0.01, 0.0), (0.01, 0.01))
PHASE_VERTEX_BY_OPERATOR_BEST = {
    "onsite_s": "midpoint",
    "spm": "midpoint",
    "dwave": "symmetric_kpm",
}
BEST_CONVENTION = {
    "candidate": "A",
    "candidate_ordering": "rho_Hp_minus_Hm_rho",
    "qV_sign": -1,
}
CHANNELS = ("rho", "Vx", "Vy", "eta1", "eta2")
BAND_PAIR_PASS = 1e-10
BAND_PAIR_MONITOR = 1e-8
BUBBLE_TRANSFER_PASS = 1e-10
RIGHT_VERTEX_PASS = 1e-10
BUBBLE_DIRECT_PASS = 1e-8
BUBBLE_DIRECT_MONITOR = 1e-6
NambuPrefactor = 0.5


@dataclass(frozen=True)
class StageSC0bInputs:
    pairing: str
    delta0_eV: float = 0.04
    omega_eV: float = 0.01
    q_model_list: tuple[tuple[float, float], ...] = Q_MODEL_LIST
    k_points: tuple[tuple[float, float], ...] = K_POINTS
    phase_vertex: str | None = None
    temperature_K: float = 10.0
    eta_eV: float = 1e-8


def rho_vertex() -> np.ndarray:
    eye = np.eye(4, dtype=complex)
    zero = np.zeros((4, 4), dtype=complex)
    return np.block([[eye, zero], [zero, -eye]])


def pairing_delta(pairing: str, kx: float, ky: float, amp: PairingAmplitudes) -> np.ndarray:
    if pairing == "onsite_s":
        return amp.delta0_eV * np.eye(4, dtype=complex)
    return pairing_matrix(pairing, kx, ky, amp)  # type: ignore[arg-type]


def bdg_vertices(
    pairing: str,
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    amp: PairingAmplitudes,
    phase_vertex: str,
) -> dict[str, np.ndarray]:
    phi = collective_form_factor(pairing, kx, ky, qx, qy, amp, phase_vertex)  # type: ignore[arg-type]
    return {
        "rho": rho_vertex(),
        "Vx": bdg_finite_q_vector_vertex(kx, ky, qx, qy, "x"),
        "Vy": bdg_finite_q_vector_vertex(kx, ky, qx, qy, "y"),
        "eta1": _amplitude_vertex(phi),
        "eta2": _eta2_phase_vertex(phi),
    }


def config(omega_eV: float, temperature_K: float = 10.0, eta_eV: float = 1e-8) -> KuboConfig:
    return KuboConfig.from_kelvin(omega_eV=omega_eV, temperature_K=temperature_K, eta_eV=eta_eV, output_si=False)


def c_eta2(delta0_eV: float) -> complex:
    return 2j * float(delta0_eV)


def convention_summary(delta0_eV: float) -> dict[str, Any]:
    return {**BEST_CONVENTION, "C_eta2": c_eta2(delta0_eV)}


def _diag_bdg(pairing: str, kx: float, ky: float, amp: PairingAmplitudes) -> tuple[np.ndarray, np.ndarray]:
    return np.linalg.eigh(bdg_hamiltonian(kx, ky, pairing_delta(pairing, kx, ky, amp)))


def _complex_record(value: complex) -> dict[str, float]:
    return {"real": float(np.real(value)), "imag": float(np.imag(value))}


def _relative(abs_value: float, scale: float) -> float:
    return float(abs_value / max(scale, 1e-300))


def _update_worst(current: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
    if current is None or float(candidate["abs_residual"]) > float(current["abs_residual"]):
        return candidate
    return current


def _kubo_raw_factor(
    em: float,
    en: float,
    fm: float,
    fn: float,
    omega_eV: float,
) -> complex:
    return (float(fm) - float(fn)) / (1j * float(omega_eV) + float(em - en))


def audit_pairing(inputs: StageSC0bInputs) -> dict[str, Any]:
    pairing = inputs.pairing
    amp = PairingAmplitudes(delta0_eV=inputs.delta0_eV)
    cfg = config(inputs.omega_eV, inputs.temperature_K, inputs.eta_eV)
    phase_vertex = inputs.phase_vertex or PHASE_VERTEX_BY_OPERATOR_BEST[pairing]
    weights = np.full(len(inputs.k_points), 1.0 / len(inputs.k_points), dtype=float)
    c2 = c_eta2(inputs.delta0_eV)

    max_band_abs = 0.0
    max_band_rel = 0.0
    worst_band: dict[str, Any] | None = None
    right_by_channel = {channel: 0.0 for channel in CHANNELS}
    q_cases: list[dict[str, Any]] = []

    for qx, qy in inputs.q_model_list:
        bubble_direct = {channel: 0.0 + 0.0j for channel in CHANNELS}
        band_rhs = {channel: 0.0 + 0.0j for channel in CHANNELS}
        for weight, (kx, ky) in zip(weights, inputs.k_points, strict=True):
            kx_minus, ky_minus = kx - 0.5 * qx, ky - 0.5 * qy
            kx_plus, ky_plus = kx + 0.5 * qx, ky + 0.5 * qy
            e_minus, u_minus = _diag_bdg(pairing, kx_minus, ky_minus, amp)
            e_plus, u_plus = _diag_bdg(pairing, kx_plus, ky_plus, amp)
            f_minus = fermi_function(e_minus, cfg.fermi_level_eV, cfg.temperature_eV)
            f_plus = fermi_function(e_plus, cfg.fermi_level_eV, cfg.temperature_eV)
            vertices = bdg_vertices(pairing, kx, ky, qx, qy, amp, phase_vertex)
            reverse_vertices = bdg_vertices(pairing, kx, ky, -qx, -qy, amp, phase_vertex)
            band = {name: u_minus.conjugate().T @ vertex @ u_plus for name, vertex in vertices.items()}
            reverse_band = {
                name: u_plus.conjugate().T @ vertex @ u_minus for name, vertex in reverse_vertices.items()
            }
            for channel in CHANNELS:
                diff = np.max(np.abs(np.conjugate(band[channel]) - reverse_band[channel].T))
                right_by_channel[channel] = max(right_by_channel[channel], float(diff))

            row_combo = (
                1j * inputs.omega_eV * band["rho"]
                - qx * band["Vx"]
                - qy * band["Vy"]
                + c2 * band["eta2"]
            )
            for m, em in enumerate(e_minus):
                for n, en in enumerate(e_plus):
                    d_mn = 1j * inputs.omega_eV + float(em - en)
                    rho_mn = band["rho"][m, n]
                    d_rho = d_mn * rho_mn
                    residual = row_combo[m, n] - d_rho
                    scale = max(abs(row_combo[m, n]), abs(d_rho), 1.0)
                    abs_res = float(abs(residual))
                    max_band_abs = max(max_band_abs, abs_res)
                    max_band_rel = max(max_band_rel, _relative(abs_res, scale))
                    qv_mn = qx * band["Vx"][m, n] + qy * band["Vy"][m, n]
                    worst_band = _update_worst(
                        worst_band,
                        {
                            "abs_residual": abs_res,
                            "relative_residual": _relative(abs_res, scale),
                            "pairing": pairing,
                            "k": [float(kx), float(ky)],
                            "q_model": [float(qx), float(qy)],
                            "m": int(m),
                            "n": int(n),
                            "E_minus": float(em),
                            "E_plus": float(en),
                            "D_mn": d_mn,
                            "rho_mn": rho_mn,
                            "qV_mn": qv_mn,
                            "Gamma_eta2_mn": band["eta2"][m, n],
                            "row_combo_mn": row_combo[m, n],
                            "D_rho_mn": d_rho,
                            "residual": residual,
                        },
                    )
                    occ_diff = float(f_minus[m] - f_plus[n])
                    if occ_diff == 0.0:
                        continue
                    factor = NambuPrefactor * float(weight) * _kubo_raw_factor(
                        float(em),
                        float(en),
                        float(f_minus[m]),
                        float(f_plus[n]),
                        inputs.omega_eV,
                    )
                    rhs_factor = NambuPrefactor * float(weight) * occ_diff
                    for channel in CHANNELS:
                        right_impl = np.conjugate(band[channel][m, n])
                        bubble_direct[channel] += factor * row_combo[m, n] * right_impl
                        band_rhs[channel] += rhs_factor * rho_mn * right_impl

        transfer_diff = {channel: bubble_direct[channel] - band_rhs[channel] for channel in CHANNELS}
        direct_row, bubble_plus_direct, missing_contact = _contact_remainder(
            pairing,
            inputs,
            phase_vertex,
            weights,
            np.array([qx, qy], dtype=float),
            bubble_direct,
        )
        q_cases.append(
            {
                "q_model": [float(qx), float(qy)],
                "bubble_left_ward_direct_sum": bubble_direct,
                "band_identity_rhs_sum": band_rhs,
                "bubble_transfer_difference": transfer_diff,
                "bubble_transfer_difference_max_abs": max(float(abs(value)) for value in transfer_diff.values()),
                "expected_contact_or_commutator_remainder": band_rhs,
                "bubble_ward_remainder": bubble_direct,
                "direct_ward_remainder": direct_row,
                "bubble_plus_direct_ward_remainder": bubble_plus_direct,
                "bubble_plus_direct_ward_remainder_max_abs": max(float(abs(value)) for value in bubble_plus_direct.values()),
                "missing_contact_terms_suspected": missing_contact,
            }
        )

    max_transfer = max(float(case["bubble_transfer_difference_max_abs"]) for case in q_cases)
    max_right = max(right_by_channel.values())
    max_bubble_direct = max(float(case["bubble_plus_direct_ward_remainder_max_abs"]) for case in q_cases)
    representative = max(q_cases, key=lambda case: float(case["bubble_plus_direct_ward_remainder_max_abs"]))
    status, dominant = _status_and_dominant(max_band_abs, max_transfer, max_right, max_bubble_direct)
    return {
        "pairing": pairing,
        "status": status,
        "dominant_failure_stage": dominant,
        "phase_vertex": phase_vertex,
        "candidate": "A",
        "candidate_ordering": "rho_Hp_minus_Hm_rho",
        "qV_sign": -1,
        "C_eta2": c2,
        "max_band_pair_identity_abs": max_band_abs,
        "max_band_pair_identity_relative": max_band_rel,
        "worst_band_pair": worst_band or {},
        "bubble_left_ward_direct_sum": representative["bubble_left_ward_direct_sum"],
        "band_identity_rhs_sum": representative["band_identity_rhs_sum"],
        "bubble_transfer_difference": representative["bubble_transfer_difference"],
        "bubble_transfer_difference_max_abs": max_transfer,
        "expected_contact_or_commutator_remainder": representative["expected_contact_or_commutator_remainder"],
        "right_vertex_impl_vs_explicit_max_abs": max_right,
        "right_vertex_impl_vs_explicit_by_channel": right_by_channel,
        "direct_blocks_available": {
            "em_em": True,
            "eta_em": False,
            "eta_eta": True,
        },
        "bubble_ward_remainder": representative["bubble_ward_remainder"],
        "direct_ward_remainder": representative["direct_ward_remainder"],
        "bubble_plus_direct_ward_remainder": representative["bubble_plus_direct_ward_remainder"],
        "bubble_plus_direct_ward_remainder_max_abs": max_bubble_direct,
        "missing_contact_terms_suspected": any(bool(case["missing_contact_terms_suspected"]) for case in q_cases),
        "q_cases": q_cases,
    }


def _contact_remainder(
    pairing: str,
    inputs: StageSC0bInputs,
    phase_vertex: str,
    weights: np.ndarray,
    q: np.ndarray,
    bubble_direct: dict[str, complex],
) -> tuple[dict[str, complex], dict[str, complex], bool]:
    points = np.asarray(inputs.k_points, dtype=float)
    cfg = config(inputs.omega_eV, inputs.temperature_K, inputs.eta_eV)
    result = bdg_finite_q_response_imag_axis(
        pairing,  # type: ignore[arg-type]
        inputs.omega_eV,
        q,
        points,
        weights,
        cfg,
        PairingAmplitudes(delta0_eV=inputs.delta0_eV),
        include_phase_correction=False,
        phase_vertex=phase_vertex,  # type: ignore[arg-type]
        collective_mode="amplitude_phase",
        collective_counterterm="goldstone_gap_equation",
    )
    coeff = np.array([1j * inputs.omega_eV, q[0], q[1]], dtype=complex)
    direct = {channel: 0.0 + 0.0j for channel in CHANNELS}
    em_row = coeff @ result.direct
    for idx, channel in enumerate(CHANNELS[:3]):
        direct[channel] += em_row[idx]
    direct["eta1"] += c_eta2(inputs.delta0_eV) * result.collective_counterterm[1, 0]
    direct["eta2"] += c_eta2(inputs.delta0_eV) * result.collective_counterterm[1, 1]
    total = {channel: bubble_direct[channel] + direct[channel] for channel in CHANNELS}
    missing_contact = bool(max(abs(value) for value in total.values()) >= BUBBLE_DIRECT_PASS)
    return direct, total, missing_contact


def _status_and_dominant(
    band_abs: float,
    transfer_abs: float,
    right_abs: float,
    contact_abs: float,
) -> tuple[str, str]:
    if band_abs >= BAND_PAIR_MONITOR:
        return "FAILED", "band_pair_identity"
    if band_abs >= BAND_PAIR_PASS:
        return "MONITOR", "band_pair_identity"
    if transfer_abs >= BUBBLE_TRANSFER_PASS:
        return "FAILED", "bubble_transfer"
    if right_abs >= RIGHT_VERTEX_PASS:
        return "FAILED", "right_vertex_orientation"
    if contact_abs < BUBBLE_DIRECT_PASS:
        return "PASSED", "none"
    if contact_abs < BUBBLE_DIRECT_MONITOR:
        return "MONITOR", "contact_remainder"
    return "FAILED", "contact_remainder"


def overall_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_pairing = {case["pairing"]: case for case in cases}
    onsite = by_pairing["onsite_s"]
    status = onsite["status"]
    return {
        "status": status,
        "band_pair_identity_passed_by_pairing": {
            pairing: by_pairing[pairing]["max_band_pair_identity_abs"] < BAND_PAIR_PASS for pairing in PAIRINGS
        },
        "bubble_transfer_passed_by_pairing": {
            pairing: by_pairing[pairing]["bubble_transfer_difference_max_abs"] < BUBBLE_TRANSFER_PASS
            for pairing in PAIRINGS
        },
        "right_vertex_orientation_passed_by_pairing": {
            pairing: by_pairing[pairing]["right_vertex_impl_vs_explicit_max_abs"] < RIGHT_VERTEX_PASS
            for pairing in PAIRINGS
        },
        "bubble_plus_direct_ward_passed_by_pairing": {
            pairing: by_pairing[pairing]["bubble_plus_direct_ward_remainder_max_abs"] < BUBBLE_DIRECT_PASS
            for pairing in PAIRINGS
        },
        "dominant_failure_stage_by_pairing": {
            pairing: by_pairing[pairing]["dominant_failure_stage"] for pairing in PAIRINGS
        },
    }


def concise_metric(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "pairing": case["pairing"],
        "band_pair_identity": case["max_band_pair_identity_abs"],
        "bubble_transfer": case["bubble_transfer_difference_max_abs"],
        "right_vertex_orientation": case["right_vertex_impl_vs_explicit_max_abs"],
        "bubble_plus_direct_Ward": case["bubble_plus_direct_ward_remainder_max_abs"],
        "dominant_failure": case["dominant_failure_stage"],
        "status": case["status"],
    }
