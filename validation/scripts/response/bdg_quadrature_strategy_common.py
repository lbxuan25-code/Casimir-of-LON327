"""Diagnostic-only batch helpers for StageSC-0f quadrature comparisons."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from bdg_bubble_ward_transfer_common import PAIRINGS, PHASE_VERTEX_BY_OPERATOR_BEST  # noqa: E402
from lno327.conductivity import KuboConfig, fermi_function  # noqa: E402
from lno327.conductivity_conventions import (  # noqa: E402
    spatial_response_to_bilayer_sheet_conductivity_model,
)
from lno327.model import NormalStateParameters  # noqa: E402
from lno327.tb_fourier import normal_state_hopping_terms, sinc_stable  # noqa: E402
from lno327.ward_response import physical_ward_residuals  # noqa: E402


STRATEGIES = (
    "ordinary_uniform",
    "multi_origin_symmetric",
    "grid_step_commensurate_reference",
    "high_resolution_uniform",
    "multi_origin_dense",
)
N_ORBITAL = 4
N_NAMBU = 8
NAMBU_PREFACTOR = 0.5
RHO = np.diag([1.0] * N_ORBITAL + [-1.0] * N_ORBITAL).astype(complex)
_TERMS = normal_state_hopping_terms(NormalStateParameters())
_R = np.asarray([item[0] for item in _TERMS], dtype=float)
_T = np.asarray([item[1] for item in _TERMS], dtype=complex)


def strategy_origins(strategy: str, q_model: tuple[float, float] | np.ndarray) -> list[tuple[float, float]]:
    qx, qy = (float(value) for value in q_model)
    if strategy in {"ordinary_uniform", "grid_step_commensurate_reference", "high_resolution_uniform"}:
        return [(0.0, 0.0)]
    if strategy in {"multi_origin_symmetric", "multi_origin_dense"}:
        return [
            (0.0, 0.0),
            (0.5 * qx, 0.5 * qy),
            (-0.5 * qx, -0.5 * qy),
            (0.5 * qx, 0.0),
            (-0.5 * qx, 0.0),
            (0.0, 0.5 * qy),
            (0.0, -0.5 * qy),
        ]
    raise ValueError(f"unknown strategy {strategy}")


def composite_uniform_quadrature(
    n_grid: int,
    origins: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Combine all origin grids with weights normalized over the composite grid."""

    if n_grid <= 0 or not origins:
        raise ValueError("n_grid and origins must be non-empty/positive")
    values = 2.0 * np.pi * np.arange(n_grid, dtype=float) / float(n_grid)
    base = np.array([(kx, ky) for kx in values for ky in values], dtype=float)
    points = np.concatenate([base + np.asarray(origin, dtype=float) for origin in origins], axis=0)
    weights = np.full(points.shape[0], 1.0 / float(points.shape[0]), dtype=float)
    return points, weights


def single_composite_schur(
    bare_total: np.ndarray,
    em_collective_left: np.ndarray,
    collective_total: np.ndarray,
    collective_em_right: np.ndarray,
) -> tuple[np.ndarray, float, str]:
    """Apply one nonlinear Schur correction after all linear kernels are accumulated."""

    condition = float(np.linalg.cond(collective_total))
    if not np.isfinite(condition) or condition > 1e12:
        inverse = np.linalg.pinv(collective_total)
        method = "pinv_diagnostic"
    else:
        inverse = np.linalg.inv(collective_total)
        method = "inv"
    response = bare_total - em_collective_left @ inverse @ collective_em_right
    return response, condition, method


def _normal_batch(points: np.ndarray) -> np.ndarray:
    phase = np.exp(1j * (points[:, :1] * _R[None, :, 0] + points[:, 1:] * _R[None, :, 1]))
    return np.einsum("bt,tij->bij", phase, _T, optimize=True)


def _normal_vector_batch(points: np.ndarray, q: np.ndarray, direction: int) -> np.ndarray:
    q_dot_r = q[0] * _R[:, 0] + q[1] * _R[:, 1]
    coeff = 1j * _R[:, direction] * sinc_stable(0.5 * q_dot_r)
    phase = np.exp(1j * (points[:, :1] * _R[None, :, 0] + points[:, 1:] * _R[None, :, 1]))
    return np.einsum("bt,t,tij->bij", phase, coeff, _T, optimize=True)


def _normal_contact_batch(points: np.ndarray, q: np.ndarray, direction_i: int, direction_j: int) -> np.ndarray:
    q_dot_r = q[0] * _R[:, 0] + q[1] * _R[:, 1]
    coeff = -_R[:, direction_i] * _R[:, direction_j] * sinc_stable(0.5 * q_dot_r) ** 2
    phase = np.exp(1j * (points[:, :1] * _R[None, :, 0] + points[:, 1:] * _R[None, :, 1]))
    return np.einsum("bt,t,tij->bij", phase, coeff, _T, optimize=True)


def _pairing_batch(pairing: str, points: np.ndarray, delta0_eV: float) -> np.ndarray:
    count = points.shape[0]
    if pairing == "onsite_s":
        return np.broadcast_to(delta0_eV * np.eye(4, dtype=complex), (count, 4, 4)).copy()
    if pairing == "spm":
        matrix = np.zeros((4, 4), dtype=complex)
        matrix[0, 2] = matrix[2, 0] = delta0_eV
        return np.broadcast_to(matrix, (count, 4, 4)).copy()
    if pairing == "dwave":
        orbital = np.zeros((4, 4), dtype=complex)
        orbital[0, 1] = orbital[1, 0] = orbital[2, 3] = orbital[3, 2] = 1.0
        form = delta0_eV * (np.cos(points[:, 0]) + np.cos(points[:, 1]))
        return form[:, None, None] * orbital[None, :, :]
    raise ValueError(f"unknown pairing {pairing}")


def _bdg_hamiltonian_batch(pairing: str, points: np.ndarray, delta0_eV: float) -> np.ndarray:
    count = points.shape[0]
    h_k = _normal_batch(points)
    h_minus = _normal_batch(-points)
    delta = _pairing_batch(pairing, points, delta0_eV)
    output = np.zeros((count, N_NAMBU, N_NAMBU), dtype=complex)
    output[:, :4, :4] = h_k
    output[:, :4, 4:] = delta
    output[:, 4:, :4] = np.swapaxes(delta.conjugate(), 1, 2)
    output[:, 4:, 4:] = -np.swapaxes(h_minus, 1, 2)
    return output


def _bdg_vector_batch(points: np.ndarray, q: np.ndarray, direction: int) -> np.ndarray:
    count = points.shape[0]
    particle = _normal_vector_batch(points, q, direction)
    hole = -np.swapaxes(_normal_vector_batch(-points, -q, direction), 1, 2)
    output = np.zeros((count, N_NAMBU, N_NAMBU), dtype=complex)
    output[:, :4, :4] = particle
    output[:, 4:, 4:] = hole
    return output


def _bdg_contact_batch(points: np.ndarray, q: np.ndarray, direction_i: int, direction_j: int) -> np.ndarray:
    count = points.shape[0]
    particle = _normal_contact_batch(points, q, direction_i, direction_j)
    hole = -np.swapaxes(_normal_contact_batch(-points, -q, direction_i, direction_j), 1, 2)
    output = np.zeros((count, N_NAMBU, N_NAMBU), dtype=complex)
    output[:, :4, :4] = particle
    output[:, 4:, 4:] = hole
    return output


def _collective_phi_batch(pairing: str, points: np.ndarray, q: np.ndarray, phase_vertex: str) -> np.ndarray:
    count = points.shape[0]
    if pairing == "onsite_s":
        return np.broadcast_to(np.eye(4, dtype=complex), (count, 4, 4)).copy()
    if pairing == "spm":
        matrix = np.zeros((4, 4), dtype=complex)
        matrix[0, 2] = matrix[2, 0] = 1.0
        return np.broadcast_to(matrix, (count, 4, 4)).copy()
    orbital = np.zeros((4, 4), dtype=complex)
    orbital[0, 1] = orbital[1, 0] = orbital[2, 3] = orbital[3, 2] = 1.0
    if phase_vertex == "midpoint":
        form = np.cos(points[:, 0]) + np.cos(points[:, 1])
    else:
        minus = points - 0.5 * q
        plus = points + 0.5 * q
        form = 0.5 * (
            np.cos(minus[:, 0]) + np.cos(minus[:, 1])
            + np.cos(plus[:, 0]) + np.cos(plus[:, 1])
        )
    return form[:, None, None] * orbital[None, :, :]


def _collective_vertices(phi: np.ndarray) -> np.ndarray:
    count = phi.shape[0]
    eta1 = np.zeros((count, 8, 8), dtype=complex)
    eta2 = np.zeros_like(eta1)
    phi_dag = np.swapaxes(phi.conjugate(), 1, 2)
    eta1[:, :4, 4:] = phi
    eta1[:, 4:, :4] = phi_dag
    eta2[:, :4, 4:] = 1j * phi
    eta2[:, 4:, :4] = -1j * phi_dag
    return np.stack((eta1, eta2), axis=0)


def _transform(states_left: np.ndarray, vertices: np.ndarray, states_right: np.ndarray) -> np.ndarray:
    return np.einsum(
        "bim,sbij,bjn->sbmn",
        states_left.conjugate(),
        vertices,
        states_right,
        optimize=True,
    )


def _bubble_from_bands(
    left_band: np.ndarray,
    right_band: np.ndarray,
    raw_factor: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    return NAMBU_PREFACTOR * np.einsum(
        "b,bmn,sbmn,tbmn->st",
        weights,
        raw_factor,
        left_band,
        right_band.conjugate(),
        optimize=True,
    )


def _fermi_matrix(states: np.ndarray, occupations: np.ndarray) -> np.ndarray:
    return np.einsum("bim,bm,bjm->bij", states, occupations, states.conjugate(), optimize=True)


def _static_raw_factor(energies: np.ndarray, occupations: np.ndarray, cfg: KuboConfig) -> np.ndarray:
    delta = energies[:, :, None] - energies[:, None, :]
    numerator = occupations[:, :, None] - occupations[:, None, :]
    output = np.empty_like(delta, dtype=complex)
    mask = np.abs(delta) >= cfg.eta_eV
    output[mask] = numerator[mask] / delta[mask]
    shifted = energies - cfg.fermi_level_eV
    x = np.clip(shifted / (2.0 * cfg.temperature_eV), -350.0, 350.0)
    derivative = -1.0 / (4.0 * cfg.temperature_eV * np.cosh(x) ** 2)
    output[~mask] = np.broadcast_to(derivative[:, :, None], output.shape)[~mask]
    return output


def _ward_max(response: np.ndarray, omega_eV: float, q: np.ndarray) -> float:
    left, right = physical_ward_residuals(response, omega_eV, q)
    return float(max(np.max(np.abs(left)), np.max(np.abs(right))))


def compute_bdg_components_for_composite_grid(
    pairing: str,
    omega_eV: float,
    q_model: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray,
    cfg: KuboConfig,
    *,
    delta0_eV: float = 0.04,
    chunk_size: int = 512,
    phase_vertex: str | None = None,
) -> dict[str, Any]:
    """Accumulate composite linear kernels, then apply one amplitude-phase Schur."""

    q = np.asarray(q_model, dtype=float)
    bare_bubble = np.zeros((3, 3), dtype=complex)
    direct = np.zeros((3, 3), dtype=complex)
    em_collective_left = np.zeros((3, 2), dtype=complex)
    collective_em_right = np.zeros((2, 3), dtype=complex)
    collective_bubble = np.zeros((2, 2), dtype=complex)
    goldstone_bubble = 0.0 + 0.0j
    e_band = np.zeros(2, dtype=complex)
    e_shifted = np.zeros(2, dtype=complex)
    qd = np.zeros(2, dtype=complex)
    selected_phase_vertex = phase_vertex or PHASE_VERTEX_BY_OPERATOR_BEST[pairing]
    if selected_phase_vertex not in {"midpoint", "symmetric_kpm"}:
        raise ValueError("phase_vertex must be midpoint or symmetric_kpm")

    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        p = points[start:stop]
        w = weights[start:stop]
        p_minus = p - 0.5 * q
        p_plus = p + 0.5 * q
        h_minus = _bdg_hamiltonian_batch(pairing, p_minus, delta0_eV)
        h_plus = _bdg_hamiltonian_batch(pairing, p_plus, delta0_eV)
        h_mid = _bdg_hamiltonian_batch(pairing, p, delta0_eV)
        em, um = np.linalg.eigh(h_minus)
        ep, up = np.linalg.eigh(h_plus)
        ec, uc = np.linalg.eigh(h_mid)
        fm = fermi_function(em, cfg.fermi_level_eV, cfg.temperature_eV)
        fp = fermi_function(ep, cfg.fermi_level_eV, cfg.temperature_eV)
        fc = fermi_function(ec, cfg.fermi_level_eV, cfg.temperature_eV)
        raw = (fm[:, :, None] - fp[:, None, :]) / (
            1j * omega_eV + em[:, :, None] - ep[:, None, :]
        )
        occupation_difference = fm[:, :, None] - fp[:, None, :]
        f_mid = _fermi_matrix(uc, fc)

        vx = _bdg_vector_batch(p, q, 0)
        vy = _bdg_vector_batch(p, q, 1)
        rho_stack = np.broadcast_to(RHO, (p.shape[0], 8, 8))
        observable = np.stack((rho_stack, -vx, -vy), axis=0)
        source = np.stack((rho_stack, vx, vy), axis=0)
        collective = _collective_vertices(_collective_phi_batch(pairing, p, q, selected_phase_vertex))
        observable_band = _transform(um, observable, up)
        source_band = _transform(um, source, up)
        collective_band = _transform(um, collective, up)
        bare_bubble += _bubble_from_bands(observable_band, source_band, raw, w)
        em_collective_left += _bubble_from_bands(observable_band, collective_band, raw, w)
        collective_em_right += _bubble_from_bands(collective_band, source_band, raw, w)
        collective_bubble += _bubble_from_bands(collective_band, collective_band, raw, w)

        contacts = np.stack(
            [_bdg_contact_batch(p, q, i, j) for i in range(2) for j in range(2)],
            axis=0,
        ).reshape(2, 2, p.shape[0], 8, 8)
        contact_expectation = NAMBU_PREFACTOR * np.einsum(
            "bij,stbji->stb", f_mid, contacts, optimize=True
        )
        direct[1:, 1:] -= np.einsum("b,stb->st", w, contact_expectation, optimize=True)

        collective_zero = _collective_vertices(
            _collective_phi_batch(pairing, p, np.zeros(2), selected_phase_vertex)
        )
        eta2_zero_band = _transform(uc, collective_zero[1:2], uc)
        static_raw = _static_raw_factor(ec, fc, cfg)
        goldstone_bubble += _bubble_from_bands(
            eta2_zero_band,
            eta2_zero_band,
            static_raw,
            w,
        )[0, 0]

        rho_band = observable_band[0]
        for j in range(2):
            reverse = _bdg_vector_batch(p, -q, j)
            reverse_band = _transform(up, reverse[None, ...], um)[0]
            e_band[j] += NAMBU_PREFACTOR * np.einsum(
                "b,bmn,bmn,bnm->",
                w,
                occupation_difference,
                rho_band,
                reverse_band,
                optimize=True,
            )
            vector_plus = _bdg_vector_batch(p_plus, -q, j)
            vector_minus = _bdg_vector_batch(p_minus, -q, j)
            difference = np.einsum("ij,bjk->bik", RHO, vector_plus) - np.einsum(
                "bij,jk->bik", vector_minus, RHO
            )
            e_shifted[j] += NAMBU_PREFACTOR * np.einsum(
                "b,bij,bji->", w, f_mid, difference, optimize=True
            )
            q_contact = q[0] * contacts[0, j] + q[1] * contacts[1, j]
            qd[j] -= NAMBU_PREFACTOR * np.einsum(
                "b,bij,bji->", w, f_mid, q_contact, optimize=True
            )

    collective_counterterm = -goldstone_bubble * np.eye(2, dtype=complex)
    collective_total = collective_bubble + collective_counterterm
    bare_total = bare_bubble + direct
    amplitude_phase, condition, inverse_method = single_composite_schur(
        bare_total,
        em_collective_left,
        collective_total,
        collective_em_right,
    )
    closure = {}
    for j, channel in enumerate(("Vx", "Vy")):
        closure[channel] = {
            "E_band_minus_E_shifted_abs": float(abs(e_band[j] - e_shifted[j])),
            "E_shifted_plus_qD_abs": float(abs(e_shifted[j] + qd[j])),
            "E_band_plus_qD_abs": float(abs(e_band[j] + qd[j])),
        }
    sigma = spatial_response_to_bilayer_sheet_conductivity_model(amplitude_phase, omega_eV)
    bare_left, bare_right = physical_ward_residuals(bare_total, omega_eV, q)
    amplitude_left, amplitude_right = physical_ward_residuals(amplitude_phase, omega_eV, q)
    diag_norm = float(np.sqrt(abs(sigma[0, 0]) ** 2 + abs(sigma[1, 1]) ** 2))
    offdiag_norm = float(np.sqrt(abs(sigma[0, 1]) ** 2 + abs(sigma[1, 0]) ** 2))
    return {
        "bare_total": bare_total,
        "amplitude_phase_schur": amplitude_phase,
        "collective_total": collective_total,
        "em_collective_left": em_collective_left,
        "collective_em_right": collective_em_right,
        "phase_vertex": selected_phase_vertex,
        "contact_closure": closure,
        "bare_left_ward": bare_left,
        "bare_right_ward": bare_right,
        "amplitude_phase_left_ward": amplitude_left,
        "amplitude_phase_right_ward": amplitude_right,
        "bare_total_ward_max_abs": float(max(np.max(np.abs(bare_left)), np.max(np.abs(bare_right)))),
        "amplitude_phase_ward_max_abs": float(
            max(np.max(np.abs(amplitude_left)), np.max(np.abs(amplitude_right)))
        ),
        "collective_condition_number": condition,
        "collective_inverse_method": inverse_method,
        "collective_total_det_abs": float(abs(np.linalg.det(collective_total))),
        "sigma_diag_min_real": float(min(sigma[0, 0].real, sigma[1, 1].real)),
        "sigma_offdiag_rel": float(offdiag_norm / max(diag_norm, 1e-300)),
        "sigma_xx_yy_anisotropy": float(
            abs(sigma[0, 0] - sigma[1, 1]) / max(abs(sigma[0, 0]) + abs(sigma[1, 1]), 1e-300)
        ),
        "max_abs_sigma_tilde": float(np.max(np.abs(sigma))),
    }


def strategy_case_status(row: dict[str, Any]) -> tuple[str, str]:
    shifted_direct = max(float(row[channel]["E_shifted_plus_qD_abs"]) for channel in ("Vx", "Vy"))
    contact = max(float(row[channel]["E_band_plus_qD_abs"]) for channel in ("Vx", "Vy"))
    if shifted_direct >= 1e-10:
        return "FAILED", "direct_expectation_mismatch"
    if contact >= 1e-5:
        return "FAILED", "band_vs_shifted_remainder"
    ward_limit = 1e-6 if row["pairing"] == "onsite_s" else 1e-5
    if float(row["amplitude_phase_ward_max_abs"]) >= ward_limit:
        return "FAILED", "amplitude_phase_ward"
    if float(row["bare_total_ward_max_abs"]) >= ward_limit:
        return "FAILED", "bare_ward"
    values = [
        row["sigma_diag_min_real"],
        row["sigma_offdiag_rel"],
        row["sigma_xx_yy_anisotropy"],
        row["max_abs_sigma_tilde"],
    ]
    if not all(np.isfinite(float(value)) for value in values):
        return "FAILED", "conductivity_sanity"
    if contact >= 1e-6:
        return "MONITOR", "band_vs_shifted_remainder"
    return "PASSED", "none"


def recommend_strategy(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select an arbitrary-q strategy only when both onsite_s q cases pass core limits."""

    allowed = {"ordinary_uniform", "multi_origin_symmetric", "high_resolution_uniform", "multi_origin_dense"}
    candidates: list[dict[str, Any]] = []
    keys = sorted({(row["strategy"], int(row["N"])) for row in rows if row["strategy"] in allowed})
    for strategy, n_grid in keys:
        onsite = [
            row
            for row in rows
            if row["pairing"] == "onsite_s" and row["strategy"] == strategy and int(row["N"]) == n_grid
        ]
        if len(onsite) != 2:
            continue
        contact = max(
            float(row[channel]["E_band_plus_qD_abs"])
            for row in onsite
            for channel in ("Vx", "Vy")
        )
        ap_ward = max(float(row["amplitude_phase_ward_max_abs"]) for row in onsite)
        bare_ward = max(float(row["bare_total_ward_max_abs"]) for row in onsite)
        if contact >= 1e-6 or ap_ward >= 1e-6 or bare_ward >= 1e-6:
            continue
        material = [
            row
            for row in rows
            if row["pairing"] in {"spm", "dwave"}
            and row["strategy"] == strategy
            and int(row["N"]) == n_grid
        ]
        material_ward = max((float(row["amplitude_phase_ward_max_abs"]) for row in material), default=np.inf)
        candidates.append(
            {
                "strategy": strategy,
                "N": n_grid,
                "num_origins": int(onsite[0]["num_origins"]),
                "onsite_s_amplitude_phase_ward_max_abs": ap_ward,
                "onsite_s_contact_closure_max_abs": contact,
                "material_amplitude_phase_ward_max_abs": material_ward,
                "num_k_points_total": int(onsite[0]["num_k_points_total"]),
            }
        )
    if not candidates:
        return None
    best = min(
        candidates,
        key=lambda row: (
            row["onsite_s_amplitude_phase_ward_max_abs"],
            row["onsite_s_contact_closure_max_abs"],
            row["material_amplitude_phase_ward_max_abs"],
            row["num_k_points_total"],
        ),
    )
    best["reason"] = (
        "smallest passing onsite_s AP Ward residual, then contact closure/material monitor/cost ordering"
    )
    return best
