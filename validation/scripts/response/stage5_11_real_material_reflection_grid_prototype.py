#!/usr/bin/env python3
"""Stage 5.11 small real-material reflection-grid prototype."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lno327.casimir_integrand import casimir_integrand_single_point  # noqa: E402
from lno327.conductivity import KuboConfig  # noqa: E402
from lno327.conductivity_conventions import spatial_response_to_bilayer_sheet_conductivity_model  # noqa: E402
from lno327.conductivity_units import SheetConductivityUnitConvention, model_to_dimensionless_sheet_conductivity  # noqa: E402
from lno327.material_reflection_grid import (  # noqa: E402
    MaterialReflectionGridPoint,
    complex_matrix_to_jsonable,
    default_stage5_11_points,
    grid_point_to_si_and_model_q,
    material_reflection_grid_prototype_metadata,
)
from lno327.material_structure import LNO327_THIN_FILM_SLAO_IN_PLANE  # noqa: E402
from lno327.reflection_input import sigma_tilde_xy_to_te_tm_reflection_matrix  # noqa: E402
from lno327.ward_response import physical_ward_residuals  # noqa: E402

from stage4_15_fermi_window_adaptive_quadrature import build_adaptive_cells, quadrature_points_for_cells  # noqa: E402
from stage4_16_full_response_adaptive_ward_diagnostic import integrate_physical_components_on_points  # noqa: E402

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "material_reflection_grid"
DEFAULT_INPUT = ROOT / "validation" / "outputs" / "response" / "casimir_toy_integration" / "stage5_10_toy_casimir_integration_convergence_audit.json"
JSON_OUTPUT = OUTPUT_DIR / "stage5_11_real_material_reflection_grid_prototype.json"
MD_OUTPUT = OUTPUT_DIR / "stage5_11_real_material_reflection_grid_prototype.md"

BOUNDARY = {
    "no_main_response_change": True,
    "no_bubble_sign_change": True,
    "no_direct_contact_change": True,
    "no_source_observable_change": True,
    "no_residual_tuning": True,
    "no_fitted_contact": True,
    "no_E_ET_added": True,
    "real_material_discrete_points_only": True,
    "no_full_matsubara_sum": True,
    "no_full_Q_integral": True,
    "no_energy_output": True,
    "no_force_output": True,
    "no_torque_output": True,
    "not_production_run": True,
    "not_casimir_ready_claim": True,
}


def to_jsonable(value: Any) -> Any:
    if isinstance(value, complex | np.complexfloating):
        return {"re": float(np.real(value)), "im": float(np.imag(value)), "abs": float(abs(value))}
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def _response_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "adaptive_level": int(args.adaptive_level),
        "gauss_order": int(args.gauss_order),
        "fermi_window_eV": float(args.fermi_window_eV),
        "coarse_grid": int(args.coarse_grid),
        "eta_eV": float(args.eta_eV),
    }


def _point_id(point: dict[str, Any]) -> str:
    return f"n{point['n']}_Q{point['Q_nm_inv']:.3f}_phi{point['phi_deg']:.1f}"


def _ward_status(total: float) -> str:
    if total < 1e-6:
        return "PASS"
    if total < 1e-5:
        return "MONITOR"
    return "FAIL"


def _integrand_status(value: complex) -> str:
    if not np.isfinite(value.real) or not np.isfinite(value.imag):
        return "FAIL"
    if abs(value.imag) <= 1e-12 or abs(value.imag) <= 1e-8 * max(1.0, abs(value.real)):
        return "PASS"
    return "MONITOR"


def _conductivity_sanity(sigma_tilde: np.ndarray) -> tuple[str, dict[str, float]]:
    max_abs = float(np.max(np.abs(sigma_tilde)))
    offdiag = float(np.sqrt(abs(sigma_tilde[0, 1]) ** 2 + abs(sigma_tilde[1, 0]) ** 2))
    diag = float(np.sqrt(abs(sigma_tilde[0, 0]) ** 2 + abs(sigma_tilde[1, 1]) ** 2))
    ratio = offdiag / max(diag, 1e-300)
    status = "PASS"
    if not np.all(np.isfinite(sigma_tilde)):
        status = "FAIL"
    elif max_abs > 100.0 or ratio > 1.0:
        status = "MONITOR"
    return status, {
        "max_abs_sigma_tilde": max_abs,
        "sigma_tilde_xx_real": float(sigma_tilde[0, 0].real),
        "sigma_tilde_yy_real": float(sigma_tilde[1, 1].real),
        "offdiag_norm_ratio": float(ratio),
    }


def _reflection_status(reflection: np.ndarray) -> tuple[str, float]:
    max_abs = float(np.max(np.abs(reflection)))
    if not np.all(np.isfinite(reflection)):
        return "FAIL", max_abs
    if max_abs > 10.0:
        return "MONITOR", max_abs
    return "PASS", max_abs


def run_real_point(
    point: MaterialReflectionGridPoint,
    *,
    response_config: dict[str, Any],
    separation_m: float,
    convention: SheetConductivityUnitConvention,
) -> dict[str, Any]:
    start = time.perf_counter()
    converted = grid_point_to_si_and_model_q(point, convention.lattice_a_x_m, convention.lattice_a_y_m)
    q_model = np.array([converted["q_model_x"], converted["q_model_y"]], dtype=float)
    try:
        config = KuboConfig.from_kelvin(
            omega_eV=float(converted["omega_eV"]),
            temperature_K=float(point.temperature_K),
            eta_eV=float(response_config["eta_eV"]),
            output_si=False,
        )
        cells, refined_count, _flagged = build_adaptive_cells(
            q_model,
            coarse_grid=int(response_config["coarse_grid"]),
            refinement_level=int(response_config["adaptive_level"]),
            fermi_window_eV=float(response_config["fermi_window_eV"]),
            fermi_level_eV=config.fermi_level_eV,
        )
        points, weights = quadrature_points_for_cells(cells, int(response_config["gauss_order"]))
        response = integrate_physical_components_on_points(points, weights, q_model, config)["total"]
        sigma_model = spatial_response_to_bilayer_sheet_conductivity_model(response, float(converted["omega_eV"]))
        sigma_tilde = model_to_dimensionless_sheet_conductivity(sigma_model, convention)
        reflection_package = sigma_tilde_xy_to_te_tm_reflection_matrix(
            sigma_tilde,
            float(converted["q_model_x"]),
            float(converted["q_model_y"]),
            float(converted["omega_eV"]),
            convention.lattice_a_x_m,
            convention.lattice_a_y_m,
        )
        reflection_te_tm = reflection_package["reflection_TE_TM"]
        integrand_package = casimir_integrand_single_point(
            reflection_te_tm,
            reflection_te_tm,
            float(reflection_package["kappa_m_inv"]),
            separation_m,
        )
        logdet = complex(integrand_package["logdet_integrand"])
        left, right = physical_ward_residuals(response, float(converted["omega_eV"]), q_model)
        ward_left = float(np.linalg.norm(left))
        ward_right = float(np.linalg.norm(right))
        ward_total = max(ward_left, ward_right)
        ward_state = _ward_status(ward_total)
        cond_state, cond_metrics = _conductivity_sanity(sigma_tilde)
        refl_state, max_abs_r = _reflection_status(reflection_te_tm)
        integ_state = _integrand_status(logdet)
        statuses = [ward_state, cond_state, refl_state, integ_state]
        point_status = "FAIL" if "FAIL" in statuses else ("MONITOR" if "MONITOR" in statuses else "PASS")
        return {
            "point_id": _point_id(converted),
            **converted,
            "ward_residual": {
                "left_max": ward_left,
                "right_max": ward_right,
                "total_max": ward_total,
                "status": ward_state,
            },
            "response_matrix": response,
            "sigma_model_xy": sigma_model,
            "sigma_tilde_xy": sigma_tilde,
            "sigma_tilde_LT": reflection_package["sigma_tilde_LT_matrix"],
            "reflection_tangential_E_LT": reflection_package["reflection_tangential_E_LT"],
            "reflection_TE_TM": reflection_te_tm,
            "conductivity_sanity": {"status": cond_state, **cond_metrics},
            "reflection_sanity": {"status": refl_state, "max_abs_R_TE_TM": max_abs_r},
            "integrand_identical_sheet": {
                "separation_m": separation_m,
                "round_trip_factor": integrand_package["round_trip_factor"],
                "logdet": logdet,
                "status": integ_state,
            },
            "num_quadrature_points": int(len(points)),
            "refined_cell_count": int(refined_count),
            "runtime_seconds": float(time.perf_counter() - start),
            "status": point_status,
        }
    except Exception as exc:
        return {
            "point_id": _point_id(converted),
            **converted,
            "runtime_seconds": float(time.perf_counter() - start),
            "status": "FAIL",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def _dry_run_result(point: MaterialReflectionGridPoint, convention: SheetConductivityUnitConvention, separation_m: float) -> dict[str, Any]:
    converted = grid_point_to_si_and_model_q(point, convention.lattice_a_x_m, convention.lattice_a_y_m)
    return {
        "point_id": _point_id(converted),
        **converted,
        "integrand_identical_sheet": {"separation_m": separation_m, "status": "DRY_RUN_NOT_COMPUTED"},
        "status": "DRY_RUN",
    }


def _run_real_point_job(
    index: int,
    point: MaterialReflectionGridPoint,
    response_config: dict[str, Any],
    separation_m: float,
    convention: SheetConductivityUnitConvention,
) -> tuple[int, dict[str, Any]]:
    return index, run_real_point(point, response_config=response_config, separation_m=separation_m, convention=convention)


def _parallel_failure_result(
    point: MaterialReflectionGridPoint,
    convention: SheetConductivityUnitConvention,
    separation_m: float,
    exc: BaseException,
) -> dict[str, Any]:
    converted = grid_point_to_si_and_model_q(point, convention.lattice_a_x_m, convention.lattice_a_y_m)
    return {
        "point_id": _point_id(converted),
        **converted,
        "integrand_identical_sheet": {"separation_m": separation_m, "status": "NOT_COMPUTED"},
        "runtime_seconds": 0.0,
        "status": "FAIL",
        "error": {"type": type(exc).__name__, "message": str(exc)},
    }


def run_real_points(
    points: list[MaterialReflectionGridPoint],
    *,
    response_config: dict[str, Any],
    separation_m: float,
    convention: SheetConductivityUnitConvention,
    workers: int,
) -> list[dict[str, Any]]:
    """Run point-level real response calculations with stable output order."""

    if workers < 1:
        raise ValueError("workers must be >= 1")
    if workers == 1 or len(points) <= 1:
        return [
            run_real_point(point, response_config=response_config, separation_m=separation_m, convention=convention)
            for point in points
        ]

    indexed_rows: dict[int, dict[str, Any]] = {}
    actual_workers = min(int(workers), len(points))
    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        future_to_point = {
            executor.submit(_run_real_point_job, index, point, response_config, separation_m, convention): (index, point)
            for index, point in enumerate(points)
        }
        for future in as_completed(future_to_point):
            index, point = future_to_point[future]
            try:
                result_index, row = future.result()
                indexed_rows[result_index] = row
            except Exception as exc:
                indexed_rows[index] = _parallel_failure_result(point, convention, separation_m, exc)
    return [indexed_rows[index] for index in range(len(points))]


def _q_direction_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    grouped: dict[tuple[int, float], dict[float, dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") not in {"PASS", "MONITOR"}:
            continue
        grouped.setdefault((int(row["n"]), float(row["Q_nm_inv"])), {})[float(row["phi_deg"])] = row
    for (n, q), by_phi in grouped.items():
        for a, b in ((0.0, 90.0), (45.0, 135.0)):
            if a in by_phi and b in by_phi:
                sig_a = np.asarray(by_phi[a]["sigma_tilde_xy"], dtype=complex)
                sig_b = np.asarray(by_phi[b]["sigma_tilde_xy"], dtype=complex)
                diagnostics.append(
                    {
                        "n": n,
                        "Q_nm_inv": q,
                        "phi_pair_deg": [a, b],
                        "diag_difference_norm": float(np.linalg.norm(np.diag(sig_a) - np.diag(sig_b))),
                        "offdiag_difference_norm": float(np.linalg.norm([sig_a[0, 1] - sig_b[0, 1], sig_a[1, 0] - sig_b[1, 0]])),
                    }
                )
    return diagnostics


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed_rows = [row for row in rows if "sigma_tilde_xy" in row and "reflection_TE_TM" in row and "integrand_identical_sheet" in row]
    return {
        "num_success": len(completed_rows),
        "num_failed": sum("error" in row for row in rows),
        "num_monitor": sum(row.get("status") == "MONITOR" for row in rows),
        "num_diagnostic_fail": sum(row.get("status") == "FAIL" and "error" not in row for row in rows),
        "max_ward_residual": max((row.get("ward_residual", {}).get("total_max", 0.0) for row in completed_rows), default=0.0),
        "max_abs_sigma_tilde": max((float(np.max(np.abs(row["sigma_tilde_xy"]))) for row in completed_rows), default=0.0),
        "max_abs_R_TE_TM": max((float(np.max(np.abs(row["reflection_TE_TM"]))) for row in completed_rows), default=0.0),
        "max_abs_logdet": max((abs(complex(row["integrand_identical_sheet"]["logdet"])) for row in completed_rows), default=0.0),
        "max_abs_logdet_imag": max((abs(complex(row["integrand_identical_sheet"]["logdet"]).imag) for row in completed_rows), default=0.0),
    }


def _checks(rows: list[dict[str, Any]], *, input_ok: bool, dry_run: bool) -> dict[str, str]:
    if dry_run:
        return {
            "stage5_10_input": "PASS" if input_ok else "FAIL",
            "grid_conversion": "PASS" if all(row["Q_m_inv"] > 0.0 for row in rows) else "FAIL",
            "n0_exclusion": "PASS" if all(row["n"] > 0 for row in rows) else "FAIL",
            "Q0_exclusion": "PASS" if all(row["Q_nm_inv"] > 0.0 for row in rows) else "FAIL",
            "dry_run_grid_only": "PASS",
        }
    completed_rows = [row for row in rows if "sigma_tilde_xy" in row and "reflection_TE_TM" in row and "integrand_identical_sheet" in row]
    return {
        "stage5_10_input": "PASS" if input_ok else "FAIL",
        "grid_conversion": "PASS" if all(row["Q_m_inv"] > 0.0 for row in rows) else "FAIL",
        "n0_exclusion": "PASS" if all(row["n"] > 0 for row in rows) else "FAIL",
        "Q0_exclusion": "PASS" if all(row["Q_nm_inv"] > 0.0 for row in rows) else "FAIL",
        "response_success": "PASS" if len(completed_rows) == len(rows) else "FAIL",
        "ward_residual": _aggregate(completed_rows, "ward_residual"),
        "conductivity_sanity": _aggregate(completed_rows, "conductivity_sanity"),
        "reflection_sanity": _aggregate(completed_rows, "reflection_sanity"),
        "integrand_sanity": _aggregate_integrand(completed_rows),
    }


def _aggregate(rows: list[dict[str, Any]], key: str) -> str:
    states = [row.get(key, {}).get("status", "FAIL") for row in rows]
    if not states or "FAIL" in states:
        return "FAIL"
    return "MONITOR" if "MONITOR" in states else "PASS"


def _aggregate_integrand(rows: list[dict[str, Any]]) -> str:
    states = [row.get("integrand_identical_sheet", {}).get("status", "FAIL") for row in rows]
    if not states or "FAIL" in states:
        return "FAIL"
    return "MONITOR" if "MONITOR" in states else "PASS"


def run_prototype(
    input_json: Path,
    *,
    temperature_K: float,
    separation_nm: float,
    smoke: bool,
    dry_run_grid_only: bool,
    allow_stage5_10_monitor: bool,
    response_config: dict[str, Any],
    workers: int = 1,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    data = json.loads(input_json.read_text(encoding="utf-8"))
    input_status = data.get("diagnostic_status", {}).get("stage5_10_status")
    input_ok = input_status == "STAGE5_10_TOY_CASIMIR_INTEGRATION_CONVERGENCE_AUDIT_PASSED" or (
        allow_stage5_10_monitor and input_status == "STAGE5_10_TOY_CASIMIR_INTEGRATION_CONVERGENCE_AUDIT_MONITOR"
    )
    if not input_ok:
        raise ValueError("input must have passed Stage 5.10 status unless --allow-stage5-10-monitor is used")
    separation_m = float(separation_nm) * 1.0e-9
    points = default_stage5_11_points(smoke=smoke, temperature_K=temperature_K)
    convention = SheetConductivityUnitConvention(
        lattice_a_x_m=LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m,
        lattice_a_y_m=LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_y_m,
        unit_cell_area_m2=LNO327_THIN_FILM_SLAO_IN_PLANE.unit_cell_area_m2,
    )
    if dry_run_grid_only:
        rows = [_dry_run_result(point, convention, separation_m) for point in points]
    else:
        rows = run_real_points(
            points,
            response_config=response_config,
            separation_m=separation_m,
            convention=convention,
            workers=workers,
        )
    checks = _checks(rows, input_ok=input_ok, dry_run=dry_run_grid_only)
    summary = _summary(rows) if not dry_run_grid_only else {
        "num_success": 0,
        "num_failed": 0,
        "num_monitor": 0,
        "max_ward_residual": None,
        "max_abs_sigma_tilde": None,
        "max_abs_R_TE_TM": None,
        "max_abs_logdet": None,
        "max_abs_logdet_imag": None,
    }
    metadata = material_reflection_grid_prototype_metadata()
    if dry_run_grid_only:
        stage_status = "STAGE5_11_DRY_RUN_GRID_ONLY_PASSED"
    elif any(value == "FAIL" for value in checks.values()):
        stage_status = "STAGE5_11_REAL_MATERIAL_REFLECTION_GRID_PROTOTYPE_FAILED"
    elif any(value == "MONITOR" for value in checks.values()):
        stage_status = "STAGE5_11_REAL_MATERIAL_REFLECTION_GRID_PROTOTYPE_MONITOR"
    else:
        stage_status = "STAGE5_11_REAL_MATERIAL_REFLECTION_GRID_PROTOTYPE_PASSED"
    return {
        "stage": "Stage 5.11",
        "purpose": "Small real-material LNO327 reflection-grid prototype and Casimir-integrand hook-in",
        "boundary": dict(BOUNDARY),
        "input": {
            "input_json": str(input_json),
            "input_stage": data.get("stage"),
            "input_status": input_status,
        },
        "scope": {
            "real_material_discrete_points": metadata["real_material_discrete_points"],
            "full_integration_grid": metadata["full_integration_grid"],
            "no_full_matsubara_sum": metadata["no_full_matsubara_sum"],
            "no_full_Q_integral": metadata["no_full_Q_integral"],
            "no_energy_output": metadata["no_energy_output"],
            "no_force_output": metadata["no_force_output"],
            "no_torque_output": metadata["no_torque_output"],
            "not_production_run": metadata["not_production"],
            "dry_run_grid_only": bool(dry_run_grid_only),
        },
        "prototype_grid": {
            "temperature_K": float(temperature_K),
            "separation_nm": float(separation_nm),
            "n_values": sorted({point.n for point in points}),
            "Q_nm_inv_values": sorted({point.Q_nm_inv for point in points}),
            "phi_deg_values": sorted({point.phi_deg for point in points}),
            "num_requested_points": len(points),
            "smoke": bool(smoke),
            "workers": int(workers),
            "parallelism": "point-level multiprocessing" if workers > 1 and not dry_run_grid_only else "sequential",
            "n0_excluded": True,
            "Q0_excluded": True,
            "zero_mode_note": "n=0 excluded in Stage 5.11; zero-mode audit is deferred to a later stage.",
        },
        "lattice_convention": {
            "a_x_m": LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m,
            "a_y_m": LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_y_m,
            "source": LNO327_THIN_FILM_SLAO_IN_PLANE.source_note,
            "is_placeholder": LNO327_THIN_FILM_SLAO_IN_PLANE.is_placeholder,
        },
        "response_config": dict(response_config),
        "point_results": rows,
        "q_direction_diagnostics": [] if dry_run_grid_only else _q_direction_diagnostics(rows),
        "summary": summary,
        "checks": checks,
        "diagnostic_status": {
            "stage5_11_status": stage_status,
            "recommended_next_action": "Proceed to small real-material energy-integration prototype only after material grid convergence strategy is defined."
            if not dry_run_grid_only
            else "Dry-run only; run smoke real-response validation before any next material stage.",
        },
    }


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def render_markdown(data: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "# Stage 5.11 real-material reflection-grid prototype",
            "## 1. Boundary\n\n" + "\n".join(f"- {key}: {value}" for key, value in data["boundary"].items()),
            "## 2. Input source\n\n" + _table(("quantity", "value"), list(data["input"].items())),
            "## 3. Prototype scope\n\n" + _table(("quantity", "value"), list(data["scope"].items())),
            "## 4. Prototype grid\n\n" + _table(("quantity", "value"), list(data["prototype_grid"].items())),
            "## 5. Lattice convention\n\n" + _table(("quantity", "value"), list(data["lattice_convention"].items())),
            "## 6. Response numerical config\n\n" + _table(("quantity", "value"), list(data["response_config"].items())),
            "## 7. Pointwise results summary\n\n" + _table(("quantity", "value"), list(data["summary"].items())),
            "## 8. Ward residual summary\n\n"
            + "Corrected Ward residuals are recorded per point when real response is run.",
            "## 9. Conductivity summary\n\n"
            + "`sigma_tilde_xy`, `sigma_tilde_LT`, `R_E_LT`, and `R_TE_TM` are retained per point in JSON.",
            "## 10. Reflection matrix summary\n\n"
            + "TE/TM ordering is `['s', 'p']`; rows are reflected polarization and columns are incident polarization.",
            "## 11. Integrand hook-in summary\n\n"
            + "Identical-sheet `logdet` values are pointwise hook-in checks only, not an energy integral.",
            "## 12. q-direction diagnostic spot checks\n\n"
            + _table(("quantity", "value"), [("num_diagnostics", len(data["q_direction_diagnostics"]))]),
            "## 13. What this is not\n\n"
            + "This is not a production grid, not a full Matsubara sum, not a full Q integral, and not Casimir energy/force/torque.",
            "## 14. Diagnostic decision\n\n" + _table(("quantity", "value"), list(data["diagnostic_status"].items())),
            "## 15. Recommended next step\n\n" + data["diagnostic_status"]["recommended_next_action"],
        ]
    ) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=JSON_OUTPUT)
    parser.add_argument("--output-md", type=Path, default=MD_OUTPUT)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--separation-nm", type=float, default=100.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run-grid-only", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Point-level multiprocessing worker count; this is separate from BLAS/OpenMP threading.",
    )
    parser.add_argument("--allow-stage5-10-monitor", action="store_true")
    parser.add_argument("--adaptive-level", type=int, default=4)
    parser.add_argument("--gauss-order", type=int, default=5)
    parser.add_argument("--fermi-window-eV", type=float, default=0.05)
    parser.add_argument("--coarse-grid", type=int, default=32)
    parser.add_argument("--eta-eV", type=float, default=1e-10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = run_prototype(
        args.input_json,
        temperature_K=args.temperature_K,
        separation_nm=args.separation_nm,
        smoke=args.smoke,
        dry_run_grid_only=args.dry_run_grid_only,
        allow_stage5_10_monitor=args.allow_stage5_10_monitor,
        response_config=_response_config(args),
        workers=args.workers,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(data), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
