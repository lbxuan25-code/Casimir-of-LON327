"""Stage-5 finite-q material Casimir candidate helpers."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .bdg_finite_q_response import bdg_finite_q_response_imag_axis
from .casimir_grid import matsubara_xi_grid
from .casimir_integrand import casimir_integrand_single_point
from .conductivity import KuboConfig, k_weights, uniform_bz_mesh
from .conductivity_conventions import spatial_response_to_bilayer_sheet_conductivity_model
from .conductivity_units import SheetConductivityUnitConvention, model_to_dimensionless_sheet_conductivity
from .constants import KB
from .material_response_cache import atomic_write_json, cache_path_for_point, load_reusable_point_cache, to_jsonable, write_point_cache
from .material_structure import LNO327_THIN_FILM_SLAO_IN_PLANE
from .pairing import PairingAmplitudes
from .reflection_input import sigma_tilde_xy_to_te_tm_reflection_matrix
from .ward_response import physical_ward_residuals

PAIRING_ALIASES = {"s_pm": "spm", "d_wave": "dwave"}
DEFAULT_PAIRINGS = ("s_pm", "d_wave")
DEFAULT_THETA_DEG = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
DEFAULT_DISTANCE_NM = (50.0, 75.0, 100.0, 150.0, 200.0)
DEFAULT_ZERO_MODE_OMEGA_EV = (1e-4, 3e-4, 1e-3, 3e-3)
REPORT_LABEL = "finite-grid publication-style candidate result; not full convergence audit"
MISSING_SC_BACKEND_MESSAGE = "finite-q superconducting s_pm/d_wave response is not available as a validated pipeline"
UNVALIDATED_BDG_RESPONSE_MESSAGE = (
    "finite-q BdG response validation marker is missing or not PASSED; "
    "rerun validation or pass --allow-unvalidated-bdg-response"
)
VALIDATION_MARKER_PATH = Path(__file__).resolve().parents[2] / "validation" / "outputs" / "response" / "bdg_finite_q" / "stageSC_5_bdg_reflection_input_audit.json"
BOUNDARY = {
    "finite_grid_publication_style_candidate_result": True,
    "not_full_convergence_audit": True,
    "uses_stage5_finite_q_response_pipeline_when_validated_backend_exists": True,
    "validated_finite_q_sc_backend_available": False,
    "no_local_q0_response_input": True,
    "no_weak_2d_reflection_path": True,
    "no_legacy_energy_integrand_path": True,
    "no_response_formula_change": True,
    "no_conductivity_formula_change": True,
    "no_unit_conversion_change": True,
    "no_reflection_formula_change": True,
    "no_trace_log_formula_change": True,
}


@dataclass(frozen=True)
class MaterialCasimirConfig:
    n_max: int = 8
    N_Q: int = 16
    N_phi: int = 24
    Q_max_nm_inv: float = 0.25
    theta_deg: tuple[float, ...] = DEFAULT_THETA_DEG
    distance_nm: tuple[float, ...] = DEFAULT_DISTANCE_NM
    zero_mode_omega_eV: tuple[float, ...] = DEFAULT_ZERO_MODE_OMEGA_EV
    temperature_K: float = 10.0
    adaptive_level: int = 4
    gauss_order: int = 7
    coarse_grid: int = 32
    fermi_window_eV: float = 0.05
    eta_eV: float = 1e-10
    delta0_eV: float = 0.04
    allow_unvalidated_bdg_response: bool = False

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


def validate_theta_on_phi_grid(theta_deg: tuple[float, ...], N_phi: int, *, atol: float = 1e-9) -> None:
    step = 360.0 / int(N_phi)
    for theta in theta_deg:
        nearest = round(float(theta) / step) * step
        if abs(float(theta) - nearest) > atol:
            raise ValueError("theta_deg must land on the phi grid; use periodic interpolation only after implementing it explicitly")


def response_config_for_pairing(pairing: str, config: MaterialCasimirConfig) -> dict[str, Any]:
    return {
        "pairing": pairing,
        "canonical_pairing": canonical_pairing(pairing),
        "adaptive_level": int(config.adaptive_level),
        "gauss_order": int(config.gauss_order),
        "coarse_grid": int(config.coarse_grid),
        "fermi_window_eV": float(config.fermi_window_eV),
        "eta_eV": float(config.eta_eV),
        "temperature_K": float(config.temperature_K),
        "zero_mode_omega_eV": [float(item) for item in config.zero_mode_omega_eV],
        "pairing_amplitude": {"delta0_eV": float(config.delta0_eV)},
        "lattice_convention": lattice_convention_payload(),
        "zero_mode_policy": "small-frequency finite-q reflection-level average",
    }


def point_response_config(pairing: str, n: int, Q_nm_inv: float, phi_deg: float, config: MaterialCasimirConfig) -> dict[str, Any]:
    return {
        **response_config_for_pairing(pairing, config),
        "n": int(n),
        "Q_nm_inv": float(Q_nm_inv),
        "phi_deg": float(phi_deg),
    }


def lattice_convention_payload() -> dict[str, Any]:
    return {
        "a_x_m": LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m,
        "a_y_m": LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_y_m,
        "source": LNO327_THIN_FILM_SLAO_IN_PLANE.source_note,
        "is_placeholder": LNO327_THIN_FILM_SLAO_IN_PLANE.is_placeholder,
    }


def point_id(pairing: str, n: int, Q_nm_inv: float, phi_deg: float) -> str:
    return f"{pairing}_n{int(n)}_Q{float(Q_nm_inv):.6f}_phi{float(phi_deg):.1f}"


def q_phi_weight_map(config: MaterialCasimirConfig) -> dict[tuple[float, float], float]:
    q_nodes = interior_q_nodes_nm_inv(config.Q_max_nm_inv, config.N_Q)
    phi_nodes = uniform_phi_nodes_deg(config.N_phi)
    dq_m = float(config.Q_max_nm_inv) * 1.0e9 / int(config.N_Q)
    dphi = 2.0 * np.pi / int(config.N_phi)
    return {
        (round(float(q), 12), round(float(phi), 12)): float((float(q) * 1.0e9) * dq_m * dphi / (2.0 * np.pi) ** 2)
        for q in q_nodes
        for phi in phi_nodes
    }


def finite_q_superconducting_response(
    pairing: str,
    omega_eV: float,
    q_model: np.ndarray,
    kubo_config: KuboConfig,
    config: MaterialCasimirConfig,
) -> np.ndarray:
    """Call the validated finite-q SC backend after checking the validation marker."""

    validation = _bdg_validation_status()
    if validation.get("status") != "PASSED" and not config.allow_unvalidated_bdg_response:
        raise RuntimeError(UNVALIDATED_BDG_RESPONSE_MESSAGE)
    canonical = canonical_pairing(pairing)
    points = uniform_bz_mesh(config.coarse_grid)
    weights = k_weights(points)
    components = bdg_finite_q_response_imag_axis(
        canonical,  # type: ignore[arg-type]
        omega_eV,
        q_model,
        points,
        weights,
        kubo_config,
        PairingAmplitudes(delta0_eV=config.delta0_eV),
        include_phase_correction=True,
    )
    return components.gauge_restored


def _bdg_validation_status(marker_path: Path = VALIDATION_MARKER_PATH) -> dict[str, Any]:
    if not marker_path.exists():
        return {"status": "MISSING", "path": str(marker_path)}
    try:
        import json

        data = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "INVALID", "path": str(marker_path), "error": str(exc)}
    return {"status": data.get("status"), "path": str(marker_path)}


def stage5_reflection_from_response(
    response: np.ndarray,
    *,
    omega_eV: float,
    q_model: np.ndarray,
    Q_m_inv: float,
    convention: SheetConductivityUnitConvention,
) -> dict[str, Any]:
    sigma_model = spatial_response_to_bilayer_sheet_conductivity_model(response, omega_eV)
    sigma_tilde = model_to_dimensionless_sheet_conductivity(sigma_model, convention)
    reflection_package = sigma_tilde_xy_to_te_tm_reflection_matrix(
        sigma_tilde,
        float(q_model[0]),
        float(q_model[1]),
        omega_eV,
        convention.lattice_a_x_m,
        convention.lattice_a_y_m,
    )
    return {
        "response_matrix": response,
        "sigma_model_xy": sigma_model,
        "sigma_tilde_xy": sigma_tilde,
        "reflection_TE_TM": reflection_package["reflection_TE_TM"],
        "kappa_m_inv": float(reflection_package.get("kappa_m_inv", Q_m_inv)),
    }


def _point_geometry(n: int, Q_nm_inv: float, phi_deg: float, config: MaterialCasimirConfig) -> dict[str, Any]:
    phi_rad = float(np.deg2rad(phi_deg))
    Q_m_inv = float(Q_nm_inv) * 1.0e9
    qx = Q_m_inv * float(np.cos(phi_rad))
    qy = Q_m_inv * float(np.sin(phi_rad))
    xi = float(matsubara_xi_grid(config.temperature_K, max(1, int(n)))[max(1, int(n))]) if n > 0 else 0.0
    omega_eV = float(2.0 * np.pi * int(n) * config.temperature_K * 8.617333262145e-5) if n > 0 else None
    return {
        "n": int(n),
        "Q_nm_inv": float(Q_nm_inv),
        "Q_m_inv": Q_m_inv,
        "phi_deg": float(phi_deg),
        "phi_rad": phi_rad,
        "Qx_m_inv": qx,
        "Qy_m_inv": qy,
        "q_model_x": qx * LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m,
        "q_model_y": qy * LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_y_m,
        "xi_si": xi,
        "omega_eV": omega_eV,
    }


def _convention() -> SheetConductivityUnitConvention:
    return SheetConductivityUnitConvention(
        lattice_a_x_m=LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m,
        lattice_a_y_m=LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_y_m,
        unit_cell_area_m2=LNO327_THIN_FILM_SLAO_IN_PLANE.unit_cell_area_m2,
    )


def compute_reflection_point(pairing: str, n: int, Q_nm_inv: float, phi_deg: float, config: MaterialCasimirConfig) -> dict[str, Any]:
    geometry = _point_geometry(n, Q_nm_inv, phi_deg, config)
    q_model = np.array([geometry["q_model_x"], geometry["q_model_y"]], dtype=float)
    convention = _convention()
    if n == 0:
        reflections = []
        sigmas = []
        responses = []
        for omega in config.zero_mode_omega_eV:
            kubo_config = KuboConfig.from_kelvin(float(omega), config.temperature_K, eta_eV=config.eta_eV, output_si=False)
            response = finite_q_superconducting_response(pairing, float(omega), q_model, kubo_config, config)
            package = stage5_reflection_from_response(response, omega_eV=float(omega), q_model=q_model, Q_m_inv=geometry["Q_m_inv"], convention=convention)
            reflections.append(package["reflection_TE_TM"])
            sigmas.append(package["sigma_tilde_xy"])
            responses.append(package["response_matrix"])
        reflection = np.mean(np.asarray(reflections, dtype=complex), axis=0)
        sigma = np.mean(np.asarray(sigmas, dtype=complex), axis=0)
        response_matrix = np.mean(np.asarray(responses, dtype=complex), axis=0)
        omega_eV = None
        n0_source = "reflection_level_average_over_zero_mode_omega_eV"
    else:
        omega_eV = float(geometry["omega_eV"])
        kubo_config = KuboConfig.from_kelvin(omega_eV, config.temperature_K, eta_eV=config.eta_eV, output_si=False)
        response_matrix = finite_q_superconducting_response(pairing, omega_eV, q_model, kubo_config, config)
        package = stage5_reflection_from_response(response_matrix, omega_eV=omega_eV, q_model=q_model, Q_m_inv=geometry["Q_m_inv"], convention=convention)
        reflection = package["reflection_TE_TM"]
        sigma = package["sigma_tilde_xy"]
        n0_source = None
    left_ward, right_ward = physical_ward_residuals(response_matrix, float(omega_eV or config.zero_mode_omega_eV[0]), q_model)
    ward_total = float(max(np.linalg.norm(left_ward), np.linalg.norm(right_ward)))
    ward_status = "PASS" if ward_total < 1e-6 else ("MONITOR" if ward_total < 1e-5 else "FAIL")
    finite_status = "PASS" if np.all(np.isfinite(reflection)) and np.all(np.isfinite(response_matrix)) and np.all(np.isfinite(sigma)) else "FAIL"
    point_status = "FAIL" if "FAIL" in {ward_status, finite_status} else ("MONITOR" if "MONITOR" in {ward_status, finite_status} else "PASS")
    return {
        "point_id": point_id(pairing, n, Q_nm_inv, phi_deg),
        "pairing": pairing,
        "canonical_pairing": canonical_pairing(pairing),
        **geometry,
        "omega_eV": omega_eV,
        "n0_source": n0_source,
        "response_matrix": response_matrix,
        "sigma_tilde_xy": sigma,
        "reflection_TE_TM": reflection,
        "kappa_m_inv": float(np.sqrt(geometry["Q_m_inv"] ** 2 + (geometry["xi_si"] / 299792458.0) ** 2)),
        "ward_residual": {
            "left_max": float(np.linalg.norm(left_ward)),
            "right_max": float(np.linalg.norm(right_ward)),
            "total_max": ward_total,
            "status": ward_status,
        },
        "unvalidated_bdg_response_used": bool(config.allow_unvalidated_bdg_response and _bdg_validation_status().get("status") != "PASSED"),
        "status": point_status,
    }


def _cached_point_job(args: tuple[str, int, float, float, MaterialCasimirConfig, Path, bool, bool, bool]) -> dict[str, Any]:
    pairing, n, q, phi, config, cache_dir, resume, skip_existing, force_recompute = args
    pid = point_id(pairing, n, q, phi)
    response_config = point_response_config(pairing, n, q, phi, config)
    lattice = lattice_convention_payload()
    if (resume or skip_existing) and not force_recompute:
        cached = load_reusable_point_cache(cache_dir, point_id=pid, response_config=response_config, lattice_convention=lattice)
        if cached is not None:
            cached["cache"] = {"source": "hit", "path": str(cache_path_for_point(cache_dir, pid))}
            return cached
    try:
        row = compute_reflection_point(pairing, n, q, phi, config)
    except RuntimeError as exc:
        if str(exc) in {MISSING_SC_BACKEND_MESSAGE, UNVALIDATED_BDG_RESPONSE_MESSAGE}:
            raise
        row = {
            "point_id": pid,
            "pairing": pairing,
            "canonical_pairing": canonical_pairing(pairing),
            "n": int(n),
            "Q_nm_inv": float(q),
            "phi_deg": float(phi),
            "status": "FAIL",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    except Exception as exc:
        row = {
            "point_id": pid,
            "pairing": pairing,
            "canonical_pairing": canonical_pairing(pairing),
            "n": int(n),
            "Q_nm_inv": float(q),
            "phi_deg": float(phi),
            "status": "FAIL",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
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
    validate_theta_on_phi_grid(config.theta_deg, config.N_phi)
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


def _row_lookup(point_rows: list[dict[str, Any]]) -> dict[tuple[str, int, float, float], dict[str, Any]]:
    return {
        (str(row["pairing"]), int(row["n"]), round(float(row["Q_nm_inv"]), 12), round(float(row["phi_deg"]) % 360.0, 12)): row
        for row in point_rows
        if row.get("status") in {"PASS", "MONITOR"}
    }


def validate_complete_reflection_grid(pairings: list[str], config: MaterialCasimirConfig, point_rows: list[dict[str, Any]]) -> None:
    failed = [row for row in point_rows if row.get("status") == "FAIL"]
    if failed:
        preview = [row.get("point_id", f"{row.get('pairing')} n={row.get('n')} Q={row.get('Q_nm_inv')} phi={row.get('phi_deg')}") for row in failed[:5]]
        raise ValueError(f"reflection grid contains FAIL points: {preview}")
    lookup = _row_lookup(point_rows)
    missing = []
    q_nodes = interior_q_nodes_nm_inv(config.Q_max_nm_inv, config.N_Q)
    phi_nodes = uniform_phi_nodes_deg(config.N_phi)
    for pairing in pairings:
        for n in range(config.n_max + 1):
            for q in q_nodes:
                for phi in phi_nodes:
                    key = (pairing, n, round(float(q), 12), round(float(phi), 12))
                    if key not in lookup:
                        missing.append({"pairing": pairing, "n": n, "Q_nm_inv": float(q), "phi_deg": float(phi)})
    if missing:
        raise ValueError(f"reflection grid is missing required points: {missing[:5]}")


def trace_log_grid(pairings: list[str], config: MaterialCasimirConfig, point_rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    validate_complete_reflection_grid(pairings, config, point_rows)
    lookup = _row_lookup(point_rows)
    q_nodes = interior_q_nodes_nm_inv(config.Q_max_nm_inv, config.N_Q)
    phi_nodes = uniform_phi_nodes_deg(config.N_phi)
    logdet = np.empty((len(pairings), config.n_max + 1, config.N_Q, config.N_phi, len(config.distance_m), len(config.theta_rad)), dtype=complex)
    records: list[dict[str, Any]] = []
    for ip, pairing in enumerate(pairings):
        for n in range(config.n_max + 1):
            for iq, q in enumerate(q_nodes):
                for iph, phi in enumerate(phi_nodes):
                    row1 = lookup[(pairing, n, round(float(q), 12), round(float(phi), 12))]
                    r1 = np.asarray(row1["reflection_TE_TM"], dtype=complex)
                    for idist, distance in enumerate(config.distance_m):
                        for itheta, theta in enumerate(config.theta_deg):
                            phi2 = (float(phi) - float(theta)) % 360.0
                            row2 = lookup[(pairing, n, round(float(q), 12), round(float(phi2), 12))]
                            r2 = np.asarray(row2["reflection_TE_TM"], dtype=complex)
                            value = complex(casimir_integrand_single_point(r1, r2, float(row1["kappa_m_inv"]), float(distance))["logdet_integrand"])
                            logdet[ip, n, iq, iph, idist, itheta] = value
                            records.append(
                                {
                                    "pairing": pairing,
                                    "n": n,
                                    "Q_nm_inv": float(q),
                                    "phi_lab_deg": float(phi),
                                    "phi2_crystal_deg": float(phi2),
                                    "distance_nm": float(config.distance_nm[idist]),
                                    "theta_deg": float(theta),
                                    "logdet": value,
                                }
                            )
    return logdet, records


def assemble_energy_data(pairings: list[str], config: MaterialCasimirConfig, point_rows: list[dict[str, Any]]) -> dict[str, Any]:
    weights = q_phi_weight_map(config)
    q_nodes = interior_q_nodes_nm_inv(config.Q_max_nm_inv, config.N_Q)
    phi_nodes = uniform_phi_nodes_deg(config.N_phi)
    logdet, records = trace_log_grid(pairings, config, point_rows)
    energy = np.zeros((len(pairings), len(config.distance_m), len(config.theta_rad)), dtype=complex)
    for ip in range(len(pairings)):
        for n in range(config.n_max + 1):
            n_weight = 0.5 if n == 0 else 1.0
            for iq, q in enumerate(q_nodes):
                for iph, phi in enumerate(phi_nodes):
                    energy[ip] += KB * config.temperature_K * n_weight * weights[(round(float(q), 12), round(float(phi), 12))] * logdet[ip, n, iq, iph]
    delta = energy - energy[:, :, [0]]
    torque = -np.gradient(energy.real, config.theta_rad, axis=2)
    diagnostics = diagnostics_from_rows(point_rows, energy, delta)
    return {
        "pairings": pairings,
        "distance_nm": np.asarray(config.distance_nm, dtype=float),
        "theta_deg": np.asarray(config.theta_deg, dtype=float),
        "logdet_grid": logdet,
        "logdet_records": records,
        "F_over_A_J_m2": energy,
        "delta_F_over_A_J_m2": delta,
        "tau_over_A_J_m2_rad": torque,
        "diagnostics": diagnostics,
        "report_label": REPORT_LABEL,
    }


def diagnostics_from_rows(point_rows: list[dict[str, Any]], energy: np.ndarray, delta: np.ndarray) -> dict[str, Any]:
    cache_hits = sum(row.get("cache", {}).get("source") == "hit" for row in point_rows)
    computed = sum(row.get("cache", {}).get("source") == "computed" for row in point_rows)
    failed = sum(row.get("status") == "FAIL" for row in point_rows)
    ward_values = [row.get("ward_residual", {}).get("total_max") for row in point_rows]
    ward_numeric = [float(value) for value in ward_values if value is not None]
    return {
        "max_Ward_residual": max(ward_numeric, default=None),
        "failed_points": int(failed),
        "cache_hits": int(cache_hits),
        "computed_points": int(computed),
        "max_imaginary_real_ratio": float(np.nanmax(np.abs(energy.imag) / np.maximum(np.abs(energy.real), 1e-300))),
        "max_angular_energy_variation": float(np.nanmax(np.abs(delta.real))),
        "unvalidated_bdg_response_used": bool(any(row.get("unvalidated_bdg_response_used") for row in point_rows)),
    }


def save_material_casimir_outputs(output_dir: Path, config: MaterialCasimirConfig, point_rows: list[dict[str, Any]], energy_data: dict[str, Any]) -> dict[str, str]:
    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    grid_json = data_dir / "material_response_reflection_grid.json"
    integrand_json = data_dir / "material_integrand_logdet_grid.json"
    energy_json = data_dir / "material_energy_torque_data.json"
    report_json = data_dir / "material_casimir_report.json"
    grid_npz = data_dir / "material_response_reflection_grid.npz"
    integrand_npz = data_dir / "material_integrand_logdet_grid.npz"
    energy_npz = data_dir / "material_energy_torque_data.npz"
    atomic_write_json(grid_json, {"config": config.__dict__, "point_results": point_rows})
    atomic_write_json(integrand_json, {"config": config.__dict__, "logdet_records": energy_data["logdet_records"]})
    atomic_write_json(energy_json, {"config": config.__dict__, **{k: v for k, v in energy_data.items() if k != "logdet_records"}})
    atomic_write_json(
        report_json,
        {
            "boundary": BOUNDARY,
            "config": config.__dict__,
            "diagnostics": energy_data["diagnostics"],
            "unvalidated_bdg_response_used": bool(energy_data["diagnostics"].get("unvalidated_bdg_response_used")),
            "report_label": REPORT_LABEL,
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
    np.savez_compressed(
        grid_npz,
        point_id=np.asarray([row["point_id"] for row in point_rows]),
        pairing=np.asarray([row["pairing"] for row in point_rows]),
        n=np.asarray([row["n"] for row in point_rows], dtype=int),
        Q_nm_inv=np.asarray([row["Q_nm_inv"] for row in point_rows], dtype=float),
        phi_deg=np.asarray([row["phi_deg"] for row in point_rows], dtype=float),
        reflection_TE_TM=np.asarray([np.asarray(row["reflection_TE_TM"], dtype=complex) for row in point_rows]),
        kappa_m_inv=np.asarray([row["kappa_m_inv"] for row in point_rows], dtype=float),
        sigma_tilde_xy=np.asarray([np.asarray(row["sigma_tilde_xy"], dtype=complex) for row in point_rows]),
        response_matrix=np.asarray([np.asarray(row["response_matrix"], dtype=complex) for row in point_rows]),
        ward_residual=np.asarray([to_jsonable(row.get("ward_residual", {})) for row in point_rows], dtype=object),
        status=np.asarray([row["status"] for row in point_rows]),
    )
    np.savez_compressed(integrand_npz, logdet_grid=energy_data["logdet_grid"])
    np.savez_compressed(
        energy_npz,
        pairings=np.asarray(energy_data["pairings"]),
        distance_nm=energy_data["distance_nm"],
        theta_deg=energy_data["theta_deg"],
        F_over_A_J_m2=energy_data["F_over_A_J_m2"],
        delta_F_over_A_J_m2=energy_data["delta_F_over_A_J_m2"],
        tau_over_A_J_m2_rad=energy_data["tau_over_A_J_m2_rad"],
        diagnostics=np.asarray(to_jsonable(energy_data["diagnostics"]), dtype=object),
        report_label=np.asarray(REPORT_LABEL),
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
