"""Material Casimir finite-grid candidate data helpers."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .casimir import CasimirSetup, casimir_energy_integrand, matsubara_frequency, reflection_matrix_weak_2d
from .conductivity import ConductivityTensor, k_weights, uniform_bz_mesh
from .constants import KB
from .material_response_cache import atomic_write_json, cache_path_for_point, load_reusable_point_cache, to_jsonable, write_point_cache
from .pairing import PairingAmplitudes
from .response_interface import local_response_imag_axis
from .response_units import require_sheet_conductivity_for_reflection

PAIRING_ALIASES = {"s_pm": "spm", "d_wave": "dwave"}
PAIRING_LABELS = {"s_pm": "s_pm", "d_wave": "d_wave"}
DEFAULT_PAIRINGS = ("s_pm", "d_wave")
DEFAULT_THETA_DEG = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
DEFAULT_DISTANCE_NM = (50.0, 75.0, 100.0, 150.0, 200.0)
DEFAULT_ZERO_MODE_OMEGA_EV = (1e-4, 3e-4, 1e-3, 3e-3)
DEFAULT_BOUNDARY = {
    "finite_grid_publication_style_candidate_result": True,
    "not_full_convergence_audit": True,
    "no_response_formula_change": True,
    "no_conductivity_unit_change": True,
    "no_reflection_formula_change": True,
    "no_te_tm_formula_change": True,
    "no_trace_log_formula_change": True,
}


@dataclass(frozen=True)
class MaterialCasimirConfig:
    n_max: int = 8
    N_Q: int = 16
    N_phi: int = 16
    Q_max_nm_inv: float = 0.25
    theta_deg: tuple[float, ...] = DEFAULT_THETA_DEG
    distance_nm: tuple[float, ...] = DEFAULT_DISTANCE_NM
    zero_mode_omega_eV: tuple[float, ...] = DEFAULT_ZERO_MODE_OMEGA_EV
    temperature_K: float = 10.0
    eta_eV: float = 1e-4
    bdg_nk: int = 32
    delta0_eV: float = 0.04

    @property
    def theta_rad(self) -> np.ndarray:
        return np.deg2rad(np.asarray(self.theta_deg, dtype=float))

    @property
    def distance_m(self) -> np.ndarray:
        return 1.0e-9 * np.asarray(self.distance_nm, dtype=float)


def canonical_pairing(pairing: str) -> str:
    if pairing not in PAIRING_ALIASES:
        raise ValueError("pairing must be one of: s_pm, d_wave")
    return PAIRING_ALIASES[pairing]


def pairing_cache_label(pairing: str) -> str:
    canonical_pairing(pairing)
    return PAIRING_LABELS[pairing]


def interior_q_nodes_nm_inv(Q_max_nm_inv: float, N_Q: int) -> np.ndarray:
    if Q_max_nm_inv <= 0.0:
        raise ValueError("Q_max_nm_inv must be positive")
    if N_Q <= 0:
        raise ValueError("N_Q must be positive")
    return (np.arange(int(N_Q), dtype=float) + 0.5) * float(Q_max_nm_inv) / int(N_Q)


def uniform_phi_nodes_deg(N_phi: int) -> np.ndarray:
    if N_phi <= 0:
        raise ValueError("N_phi must be positive")
    return np.linspace(0.0, 360.0, int(N_phi), endpoint=False)


def response_config_for_pairing(pairing: str, config: MaterialCasimirConfig) -> dict[str, Any]:
    return {
        "pairing": pairing_cache_label(pairing),
        "canonical_pairing": canonical_pairing(pairing),
        "n_max": int(config.n_max),
        "N_Q": int(config.N_Q),
        "N_phi": int(config.N_phi),
        "Q_max_nm_inv": float(config.Q_max_nm_inv),
        "theta_deg": [float(theta) for theta in config.theta_deg],
        "distance_nm": [float(distance) for distance in config.distance_nm],
        "zero_mode_omega_eV": [float(omega) for omega in config.zero_mode_omega_eV],
        "temperature_K": float(config.temperature_K),
        "eta_eV": float(config.eta_eV),
        "bdg_nk": int(config.bdg_nk),
        "delta0_eV": float(config.delta0_eV),
        "Q0_policy": "interior Q nodes; Q=0 is not a regular point",
        "n0_policy": "xi->0+ extrapolated R_TE_TM; no Omega=0 division",
    }


def lattice_convention_payload() -> dict[str, Any]:
    return {"response_layer": "local_BdG_sheet_response_candidate", "reflection_layer": "weak_2d_local_response"}


def point_id(pairing: str, n: int, Q_nm_inv: float, phi_deg: float) -> str:
    return f"{pairing_cache_label(pairing)}_n{int(n)}_Q{float(Q_nm_inv):.6f}_phi{float(phi_deg):.1f}"


def _q_phi_weights(config: MaterialCasimirConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_nm = interior_q_nodes_nm_inv(config.Q_max_nm_inv, config.N_Q)
    phi_deg = uniform_phi_nodes_deg(config.N_phi)
    dq_m = float(config.Q_max_nm_inv) * 1.0e9 / int(config.N_Q)
    dphi = 2.0 * np.pi / int(config.N_phi)
    weights = (q_nm * 1.0e9)[:, None] * dq_m * dphi / (2.0 * np.pi) ** 2
    weights = np.repeat(weights, int(config.N_phi), axis=1)
    return q_nm, phi_deg, weights


def _sheet_tensor_for_pairing(pairing: str, omega_eV: float, config: MaterialCasimirConfig) -> ConductivityTensor:
    mesh = uniform_bz_mesh(config.bdg_nk)
    response = local_response_imag_axis(
        canonical_pairing(pairing),
        omega_eV,
        mesh,
        temperature_K=config.temperature_K,
        eta_eV=config.eta_eV,
        pairing_params=PairingAmplitudes(delta0_eV=config.delta0_eV),
        k_weights=k_weights(mesh),
    )
    return require_sheet_conductivity_for_reflection(response.matrix).tensor


def _zero_mode_tensor(pairing: str, config: MaterialCasimirConfig) -> ConductivityTensor:
    matrices = [_sheet_tensor_for_pairing(pairing, float(omega), config).matrix() for omega in config.zero_mode_omega_eV]
    matrix = np.mean(np.asarray(matrices, dtype=complex), axis=0)
    return ConductivityTensor(matrix[0, 0], matrix[1, 1], matrix[0, 1], matrix[1, 0])


def compute_material_point(pairing: str, n: int, Q_nm_inv: float, phi_deg: float, config: MaterialCasimirConfig) -> dict[str, Any]:
    q_m_inv = float(Q_nm_inv) * 1.0e9
    phi_rad = float(np.deg2rad(phi_deg))
    if n == 0:
        xi = 0.0
        omega_eV = None
        tensor = _zero_mode_tensor(pairing, config)
        n_weight = 0.5
        n0_source = "xi_to_0_plus_average"
    else:
        xi = float(matsubara_frequency(int(n), config.temperature_K))
        omega_eV = float(2.0 * np.pi * int(n) * config.temperature_K * 8.617333262145e-5)
        tensor = _sheet_tensor_for_pairing(pairing, omega_eV, config)
        n_weight = 1.0
        n0_source = None

    response_matrix = tensor.matrix()
    reflections = []
    for theta in config.theta_rad:
        reflections.append(reflection_matrix_weak_2d(max(xi, 1e-300), q_m_inv, phi_rad + float(theta), tensor))
    integrand = np.empty((len(config.distance_m), len(config.theta_rad)), dtype=complex)
    for idist, distance in enumerate(config.distance_m):
        setup = CasimirSetup(temperature=config.temperature_K, distance=float(distance))
        for itheta, theta in enumerate(config.theta_rad):
            integrand[idist, itheta] = casimir_energy_integrand(setup, max(xi, 1e-300), q_m_inv, phi_rad, float(theta), tensor, tensor)
    return {
        "point_id": point_id(pairing, n, Q_nm_inv, phi_deg),
        "pairing": pairing_cache_label(pairing),
        "canonical_pairing": canonical_pairing(pairing),
        "n": int(n),
        "Q_nm_inv": float(Q_nm_inv),
        "Q_m_inv": q_m_inv,
        "phi_deg": float(phi_deg),
        "phi_rad": phi_rad,
        "xi_si": xi,
        "omega_eV": omega_eV,
        "n_weight": n_weight,
        "n0_source": n0_source,
        "response_matrix": response_matrix,
        "reflection_TE_TM_by_theta": reflections,
        "integrand_grid": integrand,
        "ward_residual": {"status": "NOT_EVALUATED_LOCAL_RESPONSE_CANDIDATE", "total_max": None},
        "status": "PASS" if np.all(np.isfinite(integrand)) else "FAIL",
    }


def _cached_point_job(args: tuple[str, int, float, float, MaterialCasimirConfig, Path, bool, bool, bool]) -> dict[str, Any]:
    pairing, n, q, phi, config, cache_dir, resume, skip_existing, force_recompute = args
    pid = point_id(pairing, n, q, phi)
    response_config = response_config_for_pairing(pairing, config)
    lattice = lattice_convention_payload()
    if (resume or skip_existing) and not force_recompute:
        cached = load_reusable_point_cache(cache_dir, point_id=pid, response_config=response_config, lattice_convention=lattice)
        if cached is not None:
            cached["cache"] = {"source": "hit", "path": str(cache_path_for_point(cache_dir, pid))}
            return cached
    row = compute_material_point(pairing, n, q, phi, config)
    row["cache"] = {"source": "computed", "path": str(cache_path_for_point(cache_dir, pid))}
    write_point_cache(cache_dir, row, response_config=response_config, lattice_convention=lattice)
    return row


PointProvider = Callable[[str, MaterialCasimirConfig], list[dict[str, Any]]]


def run_point_grid(
    pairings: list[str],
    config: MaterialCasimirConfig,
    *,
    cache_dir: Path,
    workers: int,
    resume: bool,
    skip_existing: bool,
    force_recompute: bool,
    provider: PointProvider | None = None,
) -> list[dict[str, Any]]:
    if provider is not None:
        return [row for pairing in pairings for row in provider(pairing, config)]
    if workers < 1:
        raise ValueError("workers must be >= 1")
    q_nodes = interior_q_nodes_nm_inv(config.Q_max_nm_inv, config.N_Q)
    phi_nodes = uniform_phi_nodes_deg(config.N_phi)
    jobs = [
        (pairing, n, float(q), float(phi), config, Path(cache_dir), resume, skip_existing, force_recompute)
        for pairing in pairings
        for n in range(0, config.n_max + 1)
        for q in q_nodes
        for phi in phi_nodes
    ]
    if workers == 1:
        return [_cached_point_job(job) for job in jobs]
    rows: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=min(int(workers), len(jobs))) as executor:
        futures = {executor.submit(_cached_point_job, job): index for index, job in enumerate(jobs)}
        for future in as_completed(futures):
            rows[futures[future]] = future.result()
    return [rows[index] for index in range(len(jobs))]


def assemble_energy_data(pairings: list[str], config: MaterialCasimirConfig, point_rows: list[dict[str, Any]]) -> dict[str, Any]:
    q_nodes, phi_nodes, weights = _q_phi_weights(config)
    weight_by_q_phi = {
        (round(float(q), 12), round(float(phi), 12)): float(weights[iq, iphi])
        for iq, q in enumerate(q_nodes)
        for iphi, phi in enumerate(phi_nodes)
    }
    energy = np.zeros((len(pairings), len(config.distance_m), len(config.theta_rad)), dtype=complex)
    point_count = {pairing: 0 for pairing in pairings}
    failed_points = 0
    cache_hits = 0
    computed_points = 0
    max_ward = 0.0
    for row in point_rows:
        if row.get("status") == "FAIL":
            failed_points += 1
            continue
        pairing = str(row["pairing"])
        if pairing not in pairings:
            continue
        cache_source = row.get("cache", {}).get("source")
        cache_hits += int(cache_source == "hit")
        computed_points += int(cache_source == "computed")
        ward_value = row.get("ward_residual", {}).get("total_max")
        if ward_value is not None:
            max_ward = max(max_ward, float(ward_value))
        pidx = pairings.index(pairing)
        weight = weight_by_q_phi[(round(float(row["Q_nm_inv"]), 12), round(float(row["phi_deg"]), 12))]
        factor = KB * config.temperature_K * float(row["n_weight"]) * weight
        energy[pidx] += factor * np.asarray(row["integrand_grid"], dtype=complex)
        point_count[pairing] += 1
    delta = energy - energy[:, :, [0]]
    torque = -np.gradient(energy.real, config.theta_rad, axis=2)
    imag_ratio = float(np.nanmax(np.abs(energy.imag) / np.maximum(np.abs(energy.real), 1e-300)))
    angular_variation = float(np.nanmax(np.abs(delta.real)))
    return {
        "pairings": pairings,
        "distance_nm": np.asarray(config.distance_nm, dtype=float),
        "theta_deg": np.asarray(config.theta_deg, dtype=float),
        "F_over_A_J_m2": energy,
        "delta_F_over_A_J_m2": delta,
        "tau_over_A_J_m2_rad": torque,
        "point_count": point_count,
        "diagnostics": {
            "max_Ward_residual": max_ward,
            "failed_points": failed_points,
            "cache_hits": cache_hits,
            "computed_points": computed_points,
            "max_imaginary_real_ratio": imag_ratio,
            "max_angular_energy_variation": angular_variation,
        },
        "report_label": "finite-grid publication-style candidate result; not full convergence audit",
    }


def save_material_casimir_outputs(output_dir: Path, config: MaterialCasimirConfig, point_rows: list[dict[str, Any]], energy_data: dict[str, Any]) -> dict[str, str]:
    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    grid_json = data_dir / "material_response_reflection_grid.json"
    integrand_json = data_dir / "material_integrand_grid.json"
    energy_json = data_dir / "material_energy_torque_data.json"
    grid_npz = data_dir / "material_response_reflection_grid.npz"
    integrand_npz = data_dir / "material_integrand_grid.npz"
    energy_npz = data_dir / "material_energy_torque_data.npz"
    atomic_write_json(grid_json, {"config": config.__dict__, "point_results": point_rows})
    atomic_write_json(
        integrand_json,
        {
            "config": config.__dict__,
            "point_integrands": [
                {
                    "point_id": row["point_id"],
                    "pairing": row["pairing"],
                    "n": row["n"],
                    "Q_nm_inv": row["Q_nm_inv"],
                    "phi_deg": row["phi_deg"],
                    "integrand_grid": row.get("integrand_grid"),
                }
                for row in point_rows
            ],
        },
    )
    atomic_write_json(energy_json, {"config": config.__dict__, **energy_data})
    np.savez(
        grid_npz,
        point_id=np.asarray([row["point_id"] for row in point_rows]),
        pairing=np.asarray([row["pairing"] for row in point_rows]),
        n=np.asarray([row["n"] for row in point_rows], dtype=int),
        Q_nm_inv=np.asarray([row["Q_nm_inv"] for row in point_rows], dtype=float),
        phi_deg=np.asarray([row["phi_deg"] for row in point_rows], dtype=float),
    )
    np.savez(
        integrand_npz,
        point_id=np.asarray([row["point_id"] for row in point_rows]),
        integrand_grid=np.asarray([np.asarray(row.get("integrand_grid", np.nan), dtype=complex) for row in point_rows]),
    )
    np.savez(
        energy_npz,
        pairings=np.asarray(energy_data["pairings"]),
        distance_nm=energy_data["distance_nm"],
        theta_deg=energy_data["theta_deg"],
        F_over_A_J_m2=energy_data["F_over_A_J_m2"],
        delta_F_over_A_J_m2=energy_data["delta_F_over_A_J_m2"],
        tau_over_A_J_m2_rad=energy_data["tau_over_A_J_m2_rad"],
        diagnostics=np.asarray(to_jsonable(energy_data["diagnostics"]), dtype=object),
        report_label=np.asarray(energy_data["report_label"]),
    )
    report_json = data_dir / "material_casimir_report.json"
    atomic_write_json(
        report_json,
        {
            "boundary": dict(DEFAULT_BOUNDARY),
            "config": config.__dict__,
            "diagnostics": energy_data["diagnostics"],
            "report_label": energy_data["report_label"],
            "data_files": {
                "grid_json": str(grid_json),
                "grid_npz": str(grid_npz),
                "integrand_json": str(integrand_json),
                "integrand_npz": str(integrand_npz),
                "energy_json": str(energy_json),
                "energy_npz": str(energy_npz),
            },
        },
    )
    return {
        "grid_json": str(grid_json),
        "grid_npz": str(grid_npz),
        "integrand_json": str(integrand_json),
        "integrand_npz": str(integrand_npz),
        "energy_json": str(energy_json),
        "energy_npz": str(energy_npz),
        "report_json": str(report_json),
    }
